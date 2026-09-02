package com.example

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.camera.FrameProcessor
import com.example.network.FrameSendResult
import com.example.network.FrameSender
import com.example.ui.theme.MyApplicationTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

enum class ConnectionStatus {
    DISCONNECTED,
    CONNECTING,
    STREAMING,
    ERROR
}

@OptIn(ExperimentalMaterial3Api::class)
class MainActivity : ComponentActivity() {

    private val frameSender = FrameSender()
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            MyApplicationTheme {
                Scaffold(
                    modifier = Modifier.fillMaxSize(),
                    topBar = {
                        CenterAlignedTopAppBar(
                            title = {
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text(
                                        text = "CodeAlpha AI",
                                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                                    )
                                    Text(
                                        text = "Camera Client • Task 4",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            },
                            colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                                containerColor = MaterialTheme.colorScheme.surfaceContainer
                            )
                        )
                    }
                ) { innerPadding ->
                    CameraClientScreen(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(innerPadding),
                        frameSender = frameSender,
                        cameraExecutor = cameraExecutor,
                        onBindCamera = { previewView, url, onStatusUpdate, onStatsUpdate ->
                            bindCamera(previewView, url, onStatusUpdate, onStatsUpdate)
                        },
                        onUnbindCamera = {
                            unbindCamera()
                        }
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        unbindCamera()
        cameraExecutor.shutdown()
    }

    private fun bindCamera(
        previewView: PreviewView,
        serverUrl: String,
        onStatusUpdate: (ConnectionStatus, String) -> Unit,
        onStatsUpdate: (Long, Float, Long) -> Unit
    ) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                val provider = cameraProvider ?: return@addListener

                provider.unbindAll()

                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

                var framesSent = 0L
                var lastSentTimestamp = 0L
                var lastFpsCalcTime = SystemClock.elapsedRealtime()
                var framesInInterval = 0
                var currentFps = 0f
                val minIntervalMs = 90L // ~11 FPS throttle for stable transmission

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    val now = SystemClock.elapsedRealtime()

                    // Throttle transmission to avoid network congestion
                    if (now - lastSentTimestamp < minIntervalMs || frameSender.isBusy()) {
                        imageProxy.close()
                        return@setAnalyzer
                    }

                    val jpegBytes = FrameProcessor.convertImageProxyToJpeg(imageProxy, quality = 68, maxWidth = 800)
                    imageProxy.close()

                    if (jpegBytes != null) {
                        lastSentTimestamp = now
                        // Send frame asynchronously
                        kotlinx.coroutines.CoroutineScope(Dispatchers.IO).launch {
                            val result = frameSender.sendFrame(serverUrl, jpegBytes)
                            withContext(Dispatchers.Main) {
                                when (result) {
                                    is FrameSendResult.Success -> {
                                        framesSent++
                                        framesInInterval++
                                        val timeDiff = SystemClock.elapsedRealtime() - lastFpsCalcTime
                                        if (timeDiff >= 1000L) {
                                            currentFps = (framesInInterval * 1000f) / timeDiff
                                            framesInInterval = 0
                                            lastFpsCalcTime = SystemClock.elapsedRealtime()
                                        }
                                        onStatusUpdate(ConnectionStatus.STREAMING, "Streaming frames to laptop")
                                        onStatsUpdate(framesSent, currentFps, result.latencyMs)
                                    }
                                    is FrameSendResult.Error -> {
                                        if (!result.message.contains("busy")) {
                                            onStatusUpdate(ConnectionStatus.ERROR, result.message)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                provider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
                onStatusUpdate(ConnectionStatus.CONNECTING, "Camera active. Connecting to laptop...")
            } catch (e: Exception) {
                onStatusUpdate(ConnectionStatus.ERROR, "Camera start failed: ${e.localizedMessage}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun unbindCamera() {
        cameraProvider?.unbindAll()
    }
}

@Composable
fun CameraClientScreen(
    modifier: Modifier = Modifier,
    frameSender: FrameSender,
    cameraExecutor: java.util.concurrent.ExecutorService,
    onBindCamera: (PreviewView, String, (ConnectionStatus, String) -> Unit, (Long, Float, Long) -> Unit) -> Unit,
    onUnbindCamera: () -> Unit
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    var serverUrl by remember { mutableStateOf("http://192.168.1.100:5000") }
    var isCameraActive by remember { mutableStateOf(false) }
    var connectionStatus by remember { mutableStateOf(ConnectionStatus.DISCONNECTED) }
    var statusMessage by remember { mutableStateOf("Ready. Enter laptop server URL and tap Start Camera.") }

    var totalFramesSent by remember { mutableLongStateOf(0L) }
    var currentFps by remember { mutableFloatStateOf(0f) }
    var lastLatencyMs by remember { mutableLongStateOf(0L) }
    var showHelpDialog by remember { mutableStateOf(false) }

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        if (!granted) {
            Toast.makeText(context, "Camera permission is required to stream video.", Toast.LENGTH_LONG).show()
        }
    }

    var previewViewRef by remember { mutableStateOf<PreviewView?>(null) }

    val scrollState = rememberScrollState()

    Column(
        modifier = modifier
            .verticalScroll(scrollState)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {

        // 1. Camera Preview Card
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .height(260.dp)
                .testTag("camera_preview_card"),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color.Black)
        ) {
            Box(modifier = Modifier.fillMaxSize()) {
                if (isCameraActive && hasCameraPermission) {
                    AndroidView(
                        factory = { ctx ->
                            PreviewView(ctx).apply {
                                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                                previewViewRef = this
                                onBindCamera(
                                    this,
                                    serverUrl,
                                    { status, msg ->
                                        connectionStatus = status
                                        statusMessage = msg
                                    },
                                    { frames, fps, latency ->
                                        totalFramesSent = frames
                                        currentFps = fps
                                        lastLatencyMs = latency
                                    }
                                )
                            }
                        },
                        modifier = Modifier.fillMaxSize()
                    )
                } else {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        Icon(
                            imageVector = Icons.Default.VideocamOff,
                            contentDescription = "Camera Inactive",
                            tint = Color.Gray,
                            modifier = Modifier.size(48.dp)
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = if (!hasCameraPermission) "Camera Permission Required" else "Camera Inactive",
                            color = Color.LightGray,
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Text(
                            text = "Tap 'Start Camera' below to stream",
                            color = Color.DarkGray,
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }

                // Live status overlay badge
                Surface(
                    modifier = Modifier
                        .padding(12.dp)
                        .align(Alignment.TopEnd),
                    shape = RoundedCornerShape(20.dp),
                    color = Color.Black.copy(alpha = 0.65f)
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        val statusColor = when (connectionStatus) {
                            ConnectionStatus.DISCONNECTED -> Color.Gray
                            ConnectionStatus.CONNECTING -> Color(0xFFFFA500)
                            ConnectionStatus.STREAMING -> Color(0xFF4CAF50)
                            ConnectionStatus.ERROR -> Color(0xFFF44336)
                        }
                        Box(
                            modifier = Modifier
                                .size(10.dp)
                                .clip(CircleShape)
                                .background(statusColor)
                        )
                        Text(
                            text = connectionStatus.name,
                            color = Color.White,
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                }
            }
        }

        // 2. Server URL Field
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Laptop Server URL",
                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                    )
                    IconButton(
                        onClick = { showHelpDialog = true },
                        modifier = Modifier.size(24.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.HelpOutline,
                            contentDescription = "Network setup help",
                            tint = MaterialTheme.colorScheme.primary
                        )
                    }
                }

                OutlinedTextField(
                    value = serverUrl,
                    onValueChange = { serverUrl = it },
                    label = { Text("Complete Server URL") },
                    placeholder = { Text("http://192.168.1.100:5000") },
                    singleLine = true,
                    enabled = !isCameraActive,
                    leadingIcon = {
                        Icon(imageVector = Icons.Default.Computer, contentDescription = null)
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("server_url_input"),
                    shape = RoundedCornerShape(12.dp)
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End
                ) {
                    OutlinedButton(
                        onClick = {
                            coroutineScope.launch {
                                connectionStatus = ConnectionStatus.CONNECTING
                                statusMessage = "Testing connection to $serverUrl..."
                                val ping = frameSender.pingServer(serverUrl)
                                when (ping) {
                                    is FrameSendResult.Success -> {
                                        connectionStatus = ConnectionStatus.STREAMING
                                        statusMessage = "Server reachable! (Latency: ${ping.latencyMs}ms)"
                                    }
                                    is FrameSendResult.Error -> {
                                        connectionStatus = ConnectionStatus.ERROR
                                        statusMessage = ping.message
                                    }
                                }
                            }
                        },
                        enabled = !isCameraActive,
                        modifier = Modifier.testTag("test_connection_button")
                    ) {
                        Icon(imageVector = Icons.Default.NetworkCheck, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Test Ping (GET /)")
                    }
                }
            }
        }

        // 3. Action Buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Button(
                onClick = {
                    if (!hasCameraPermission) {
                        permissionLauncher.launch(Manifest.permission.CAMERA)
                        return@Button
                    }
                    if (serverUrl.isBlank()) {
                        Toast.makeText(context, "Please enter a valid server URL", Toast.LENGTH_SHORT).show()
                        return@Button
                    }
                    isCameraActive = true
                    connectionStatus = ConnectionStatus.CONNECTING
                    statusMessage = "Starting camera..."
                },
                enabled = !isCameraActive,
                modifier = Modifier
                    .weight(1f)
                    .height(52.dp)
                    .testTag("start_camera_button"),
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Icon(imageVector = Icons.Default.PlayArrow, contentDescription = null)
                Spacer(modifier = Modifier.width(6.dp))
                Text("Start Camera", fontWeight = FontWeight.SemiBold)
            }

            FilledTonalButton(
                onClick = {
                    isCameraActive = false
                    onUnbindCamera()
                    connectionStatus = ConnectionStatus.DISCONNECTED
                    statusMessage = "Camera stopped. Ready to reconnect."
                    currentFps = 0f
                },
                enabled = isCameraActive,
                modifier = Modifier
                    .weight(1f)
                    .height(52.dp)
                    .testTag("stop_camera_button"),
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.filledTonalButtonColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer,
                    contentColor = MaterialTheme.colorScheme.onErrorContainer
                )
            ) {
                Icon(imageVector = Icons.Default.Stop, contentDescription = null)
                Spacer(modifier = Modifier.width(6.dp))
                Text("Stop Camera", fontWeight = FontWeight.SemiBold)
            }
        }

        // 4. Status & Diagnostics Card
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .testTag("status_card"),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerHigh),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(
                    text = "Connection Status",
                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                )

                // Message banner
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(8.dp),
                    color = when (connectionStatus) {
                        ConnectionStatus.DISCONNECTED -> MaterialTheme.colorScheme.surfaceVariant
                        ConnectionStatus.CONNECTING -> Color(0xFFFFF3E0)
                        ConnectionStatus.STREAMING -> Color(0xFFE8F5E9)
                        ConnectionStatus.ERROR -> Color(0xFFFFEBEE)
                    }
                ) {
                    Text(
                        text = statusMessage,
                        modifier = Modifier.padding(12.dp),
                        style = MaterialTheme.typography.bodyMedium,
                        color = when (connectionStatus) {
                            ConnectionStatus.DISCONNECTED -> MaterialTheme.colorScheme.onSurfaceVariant
                            ConnectionStatus.CONNECTING -> Color(0xFFE65100)
                            ConnectionStatus.STREAMING -> Color(0xFF2E7D32)
                            ConnectionStatus.ERROR -> Color(0xFFC62828)
                        }
                    )
                }

                // Streaming metrics
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    MetricBadge(label = "Frames Sent", value = totalFramesSent.toString())
                    MetricBadge(label = "Upload FPS", value = String.format("%.1f", currentFps))
                    MetricBadge(label = "Latency", value = "${lastLatencyMs} ms")
                }
            }
        }

        // Help dialog
        if (showHelpDialog) {
            AlertDialog(
                onDismissRequest = { showHelpDialog = false },
                title = { Text("Network Configuration Help") },
                text = {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            text = "Finding your laptop IPv4 address:",
                            fontWeight = FontWeight.Bold,
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Text(
                            text = "1. Open Command Prompt on your laptop.\n2. Type 'ipconfig' and press Enter.\n3. Find your IPv4 Address (e.g. 192.168.1.50 or 10.138.x.x).\n4. Enter the full URL: http://<LAPTOP_IP>:5000",
                            style = MaterialTheme.typography.bodySmall
                        )
                        HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
                        Text(
                            text = "Connection Modes:",
                            fontWeight = FontWeight.Bold,
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Text(
                            text = "• Wi-Fi: Connect phone and laptop to same Wi-Fi network.\n• Hotspot: Turn on Phone Hotspot, connect laptop to it.\n• USB Tethering: Connect USB cable and enable USB Tethering on phone.",
                            style = MaterialTheme.typography.bodySmall
                        )
                        Text(
                            text = "Note: Do not use 127.0.0.1 or localhost on your phone. Windows Firewall may need to allow Python.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                },
                confirmButton = {
                    TextButton(onClick = { showHelpDialog = false }) {
                        Text("Got it")
                    }
                }
            )
        }
    }
}

@Composable
fun MetricBadge(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = value,
            style = MaterialTheme.typography.titleMedium.copy(
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace
            ),
            color = MaterialTheme.colorScheme.primary
        )
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}
