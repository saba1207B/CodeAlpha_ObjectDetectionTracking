package com.example

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Size
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.ui.theme.MyApplicationTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : ComponentActivity() {
    private var hasCameraPermission by mutableStateOf(false)
    private var serverUrl by mutableStateOf("http://192.168.1.100:5000")
    private var streaming by mutableStateOf(false)
    private var status by mutableStateOf("Enter your laptop IP and start the camera")
    private var cameraProvider: ProcessCameraProvider? = null
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val networkScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(2, TimeUnit.SECONDS)
        .writeTimeout(2, TimeUnit.SECONDS)
        .readTimeout(2, TimeUnit.SECONDS)
        .build()
    private val uploadInProgress = AtomicBoolean(false)
    private var cameraPreview: PreviewView? = null

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        status = if (granted) "Camera ready" else "Camera permission is required"
        if (granted) startCamera()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        hasCameraPermission = ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        setContent {
            MyApplicationTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    CameraClientScreen()
                }
            }
        }

        if (hasCameraPermission) startCamera()
        else permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    @androidx.compose.runtime.Composable
    private fun CameraClientScreen() {
        Column(modifier = Modifier.fillMaxSize()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .background(Color.Black)
            ) {
                AndroidView(
                    modifier = Modifier.fillMaxSize(),
                    factory = { context ->
                        PreviewView(context).also {
                            it.layoutParams = android.view.ViewGroup.LayoutParams(MATCH_PARENT, MATCH_PARENT)
                            it.scaleType = PreviewView.ScaleType.FILL_CENTER
                            cameraPreview = it
                            if (hasCameraPermission) startCamera()
                        }
                    }
                )

                Text(
                    text = if (streaming) "● STREAMING" else "● CAMERA PREVIEW",
                    color = if (streaming) Color(0xFF66BB6A) else Color.White,
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(16.dp)
                )
            }

            Column(modifier = Modifier.padding(16.dp)) {
                Text("CodeAlpha Object Tracking", style = MaterialTheme.typography.headlineSmall)
                Spacer(Modifier.height(6.dp))
                Text("Phone camera → Wi-Fi → Laptop YOLO + ByteTrack", style = MaterialTheme.typography.bodyMedium)
                Spacer(Modifier.height(12.dp))

                OutlinedTextField(
                    value = serverUrl,
                    onValueChange = { serverUrl = it.trimEnd('/') },
                    label = { Text("Laptop server URL") },
                    placeholder = { Text("http://192.168.1.100:5000") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(Modifier.height(8.dp))
                Text(status, style = MaterialTheme.typography.bodySmall)
                Spacer(Modifier.height(10.dp))

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
                    Button(
                        onClick = { if (streaming) stopStreaming() else startStreaming() },
                        enabled = hasCameraPermission,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(if (streaming) "Stop Streaming" else "Start Streaming")
                    }
                }
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                val provider = cameraProvider ?: return@addListener
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(cameraPreview?.surfaceProvider)
                }

                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { analyzer ->
                        analyzer.setAnalyzer(cameraExecutor) { image ->
                            if (streaming) sendFrame(image) else image.close()
                        }
                    }

                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )
            } catch (e: Exception) {
                status = "Camera error: ${e.message ?: "unknown error"}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startStreaming() {
        val normalized = serverUrl.trim().removeSuffix("/")
        if (!normalized.startsWith("http://") && !normalized.startsWith("https://")) {
            status = "Use a URL like http://192.168.1.100:5000"
            return
        }
        serverUrl = normalized
        streaming = true
        status = "Connecting to laptop..."
    }

    private fun stopStreaming() {
        streaming = false
        status = "Streaming stopped"
    }

    private fun sendFrame(image: ImageProxy) {
        try {
            if (!uploadInProgress.compareAndSet(false, true)) {
                image.close()
                return
            }

            val jpeg = imageProxyToJpeg(image)
            image.close()

            networkScope.launch {
                try {
                    val request = Request.Builder()
                        .url("$serverUrl/frame")
                        .post(jpeg.toRequestBody("image/jpeg".toMediaType()))
                        .build()

                    httpClient.newCall(request).execute().use { response ->
                        status = if (response.isSuccessful) {
                            "Connected • sending camera frames to laptop"
                        } else {
                            "Laptop returned HTTP ${response.code}"
                        }
                    }
                } catch (e: Exception) {
                    status = "Connection failed • check Wi-Fi and laptop IP"
                } finally {
                    uploadInProgress.set(false)
                }
            }
        } catch (e: Exception) {
            image.close()
            uploadInProgress.set(false)
            status = "Frame error: ${e.message ?: "unknown error"}"
        }
    }

    private fun imageProxyToJpeg(image: ImageProxy): ByteArray {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val output = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, image.width, image.height), 65, output)
        var jpeg = output.toByteArray()

        if (image.imageInfo.rotationDegrees != 0) {
            val bitmap = android.graphics.BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)
            val matrix = Matrix().apply { postRotate(image.imageInfo.rotationDegrees.toFloat()) }
            val rotated = android.graphics.Bitmap.createBitmap(
                bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
            )
            val rotatedOutput = ByteArrayOutputStream()
            rotated.compress(android.graphics.Bitmap.CompressFormat.JPEG, 65, rotatedOutput)
            jpeg = rotatedOutput.toByteArray()
            bitmap.recycle()
            rotated.recycle()
        }
        return jpeg
    }

    override fun onDestroy() {
        stopStreaming()
        cameraProvider?.unbindAll()
        cameraExecutor.shutdown()
        networkScope.coroutineContext[Job]?.cancel()
        httpClient.dispatcher.executorService.shutdown()
        super.onDestroy()
    }
}
