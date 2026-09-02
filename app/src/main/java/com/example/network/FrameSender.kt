package com.example.network

import android.os.SystemClock
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Result of sending a frame over HTTP to the Python server.
 */
sealed class FrameSendResult {
    data class Success(val httpCode: Int, val latencyMs: Long) : FrameSendResult()
    data class Error(val message: String, val httpCode: Int? = null) : FrameSendResult()
}

/**
 * Manages HTTP communication with the Python laptop server.
 * Ensures non-blocking execution, frame dropping when network is congested,
 * and robust error reporting.
 */
class FrameSender {

    private val client = OkHttpClient.Builder()
        .connectTimeout(2, TimeUnit.SECONDS)
        .writeTimeout(2, TimeUnit.SECONDS)
        .readTimeout(2, TimeUnit.SECONDS)
        .retryOnConnectionFailure(false)
        .build()

    private val jpegMediaType = "image/jpeg".toMediaType()

    // Enforce "latest frame wins" policy: drops new frame if previous upload is still in flight
    private val isUploading = AtomicBoolean(false)

    /**
     * Check if an upload is currently in progress.
     */
    fun isBusy(): Boolean = isUploading.get()

    /**
     * Test server connectivity via GET /.
     */
    suspend fun pingServer(serverUrl: String): FrameSendResult = withContext(Dispatchers.IO) {
        val baseUrl = normalizeUrl(serverUrl)
        val request = Request.Builder()
            .url(baseUrl)
            .get()
            .build()

        val startTime = SystemClock.elapsedRealtime()
        try {
            client.newCall(request).execute().use { response ->
                val latency = SystemClock.elapsedRealtime() - startTime
                if (response.isSuccessful) {
                    FrameSendResult.Success(response.code, latency)
                } else {
                    FrameSendResult.Error("HTTP ${response.code}: ${response.message}", response.code)
                }
            }
        } catch (e: IOException) {
            FrameSendResult.Error(formatNetworkError(e))
        } catch (e: Exception) {
            FrameSendResult.Error("Error: ${e.localizedMessage ?: "Unknown error"}")
        }
    }

    /**
     * Send a JPEG frame to POST /frame.
     * Drops immediately if an upload is already running.
     */
    suspend fun sendFrame(serverUrl: String, jpegBytes: ByteArray): FrameSendResult = withContext(Dispatchers.IO) {
        if (!isUploading.compareAndSet(false, true)) {
            // Already sending a frame; drop this one to prevent latency buildup
            return@withContext FrameSendResult.Error("Dropped: Network busy")
        }

        try {
            val endpoint = "${normalizeUrl(serverUrl)}/frame"
            val requestBody = jpegBytes.toRequestBody(jpegMediaType)
            val request = Request.Builder()
                .url(endpoint)
                .post(requestBody)
                .build()

            val startTime = SystemClock.elapsedRealtime()
            client.newCall(request).execute().use { response ->
                val latency = SystemClock.elapsedRealtime() - startTime
                if (response.isSuccessful) {
                    FrameSendResult.Success(response.code, latency)
                } else {
                    FrameSendResult.Error("Server error: HTTP ${response.code}", response.code)
                }
            }
        } catch (e: IOException) {
            FrameSendResult.Error(formatNetworkError(e))
        } catch (e: Exception) {
            FrameSendResult.Error(e.localizedMessage ?: "Network error")
        } finally {
            isUploading.set(false)
        }
    }

    private fun normalizeUrl(url: String): String {
        var clean = url.trim()
        if (!clean.startsWith("http://") && !clean.startsWith("https://")) {
            clean = "http://$clean"
        }
        return clean.trimEnd('/')
    }

    private fun formatNetworkError(e: IOException): String {
        val msg = e.message?.lowercase() ?: ""
        return when {
            msg.contains("failed to connect") || msg.contains("connection refused") ->
                "Connection refused. Ensure phone_server.py is running and laptop IP is correct."
            msg.contains("timeout") ->
                "Connection timed out. Check firewall and ensure both devices share Wi-Fi/Hotspot."
            msg.contains("no route to host") || msg.contains("network is unreachable") ->
                "Unreachable host. Verify laptop IPv4 via 'ipconfig' and check network connection."
            else ->
                e.localizedMessage ?: "Network connection failed"
        }
    }
}
