package com.example.camera

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

/**
 * Utility to convert CameraX ImageProxy frames into compressed JPEG byte arrays
 * with orientation correction.
 */
object FrameProcessor {

    @OptIn(ExperimentalGetImage::class)
    fun convertImageProxyToJpeg(
        imageProxy: ImageProxy,
        quality: Int = 70,
        maxWidth: Int = 960
    ): ByteArray? {
        return try {
            val originalBitmap = imageProxy.toBitmap()
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees

            // Check if rotation or downscaling is necessary to preserve bandwidth
            val matrix = Matrix()
            if (rotationDegrees != 0) {
                matrix.postRotate(rotationDegrees.toFloat())
            }

            // Downscale if width exceeds maxWidth to optimize network transfer
            val currentWidth = if (rotationDegrees == 90 || rotationDegrees == 270) originalBitmap.height else originalBitmap.width
            if (currentWidth > maxWidth) {
                val scale = maxWidth.toFloat() / currentWidth.toFloat()
                matrix.postScale(scale, scale)
            }

            val processedBitmap = if (!matrix.isIdentity) {
                Bitmap.createBitmap(
                    originalBitmap,
                    0,
                    0,
                    originalBitmap.width,
                    originalBitmap.height,
                    matrix,
                    true
                )
            } else {
                originalBitmap
            }

            val outputStream = ByteArrayOutputStream()
            processedBitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream)
            outputStream.toByteArray()
        } catch (e: Exception) {
            null
        }
    }
}
