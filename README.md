# CodeAlpha Object Detection & Tracking

Real-time object detection and multi-object tracking using an Android CameraX client, a Python local-network backend, Ultralytics YOLO, and ByteTrack / BoT-SORT.

## Architecture

```text
Android APK
    │  POST /frame (JPEG)
    ▼
Wi-Fi router / trusted LAN
    │
    ▼
Python backend :5000 (0.0.0.0)
    │
    ├── latest raw frame ───────────────► Laptop dashboard
    │
    ▼
Ultralytics YOLO (default: yolo11n.pt)
    │
    ▼
ByteTrack / BoT-SORT
    │
    ├── bounding boxes + labels + confidence
    ├── persistent tracking IDs
    └── motion trails / HUD
    │
    ▼
Processed frame + detection API ───────► Laptop dashboard
```

> **Network safety:** the server listens on `0.0.0.0:5000` so a phone on the same Wi-Fi network can connect. This is intended for a trusted LAN only. Do **not** expose port 5000 directly to the public Internet.

## Repository layout

```text
.
├── android-app/              # APK + Android client documentation
├── app/                      # Android Gradle application source
├── backend/
│   ├── server.py              # Unified HTTP server + live vision pipeline
│   ├── app.py                 # CLI detection pipeline
│   ├── requirements.txt
│   └── src/
│       ├── detector.py        # Ultralytics YOLO wrapper
│       └── tracker.py         # ByteTrack / BoT-SORT integration
├── laptop-dashboard/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── screenshots/
├── output/
└── .github/workflows/build-apk.yml
```

## 1. Clone or update

### Windows PowerShell

```powershell
git clone https://github.com/saba1207B/CodeAlpha_ObjectDetectionTracking.git
cd CodeAlpha_ObjectDetectionTracking
# Existing checkout:
git pull origin main
```

### Linux / macOS

```bash
git clone https://github.com/saba1207B/CodeAlpha_ObjectDetectionTracking.git
cd CodeAlpha_ObjectDetectionTracking
git pull origin main
```

## 2. Python virtual environment

Python 3.10–3.13 is recommended. Python 3.12 is a good choice.

### Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation:

```bat
.venv\Scripts\activate.bat
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Install backend requirements

```bash
python -m pip install -r backend/requirements.txt
```

This installs OpenCV, NumPy, PyTorch, Ultralytics, Pillow and Requests. YOLO weights such as `yolo11n.pt` are downloaded by Ultralytics when first required, so the first detection startup needs Internet access unless the weight is already cached.

## 4. Start the backend/server

From the **repository root**:

```bash
python backend/server.py
```

The server listens on `0.0.0.0:5000` and prints the laptop IP and Android URL.

Open the dashboard on the laptop:

```text
http://localhost:5000/
```

or:

```text
http://LAPTOP_IPV4:5000/
```

## 5. Find the laptop IPv4

**Windows:**

```powershell
ipconfig
```

Use the IPv4 address of the Wi-Fi adapter, for example `192.168.1.25`.

**Linux:**

```bash
ip -4 addr
# or
hostname -I
```

**macOS:**

```bash
ipconfig getifaddr en0
```

Use `ifconfig` if your Wi-Fi interface is different.

## 6. Connect the APK over Wi-Fi

Put the phone and laptop on the **same Wi-Fi network**.

In the Android CodeAlpha Camera app enter:

```text
http://LAPTOP_IPV4:5000
```

Example:

```text
http://192.168.1.25:5000
```

**Do not enter `localhost` or `127.0.0.1` in the Android connection field.** Those addresses refer to the phone itself.

Tap **Test Ping (GET /)** first. A successful ping confirms phone-to-laptop reachability. Then tap **Start Camera**. The APK sends JPEG frames using `POST /frame`.

## 7. Start real YOLO detection

In the laptop dashboard:

1. Confirm **Backend Running**.
2. Confirm the phone badge changes to **Streaming**.
3. Confirm the frame count and resolution are increasing.
4. Keep **YOLOv11 Nano (`yolo11n.pt`)** as the default for CPU-friendly operation.
5. Select **ByteTrack** or **BoT-SORT**.
6. Adjust confidence and IoU thresholds if needed.
7. Select object-class filters if needed.
8. Toggle motion trails and HUD as desired.
9. Click **Start Detection**.

The processed stream uses real incoming phone frames and real Ultralytics YOLO + ByteTrack/BoT-SORT output. There are no fake boxes, simulated metrics or mock detections.

## Dashboard features

- Backend health and Android phone connection status
- Laptop IPv4 and complete APK connection URL
- Copy-URL button
- Received frame count, stream FPS, API round-trip latency and resolution
- Last-frame age
- Raw camera frame view
- Processed YOLO tracking frame view
- Split raw/processed view
- Real bounding boxes, labels, confidence scores and tracking IDs
- Motion trails and HUD/statistics
- Current tracked-object table
- Start Detection / Stop Detection / Reset Results
- YOLO model selector (`yolo11n.pt` default)
- Confidence and IoU controls
- ByteTrack / BoT-SORT selector
- COCO class filters
- Motion-trails and HUD toggles
- Screenshot capture
- Optional annotated video recording
- Visible errors for missing frames, server/Wi-Fi problems, missing packages and YOLO/weight initialization failures

## Android-compatible API

These endpoints remain compatible with the Android client:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Health check / dashboard |
| POST | `/frame` | Receive Android JPEG frame |
| GET | `/latest_frame` | Latest raw camera frame |
| GET | `/status` | Server and frame metrics |

Dashboard/vision endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/processed_frame` | Latest YOLO annotated frame |
| GET | `/detections` | Current tracked objects |
| POST | `/start-detection` | Start detection |
| POST | `/stop-detection` | Stop detection |
| GET | `/settings` | Current settings |
| POST | `/settings` | Update model/tracker/thresholds/filters |
| POST | `/clear-results` | Reset tracker state |
| POST | `/screenshot` | Capture current frame |
| POST | `/recording/start` | Start annotated recording |
| POST | `/recording/stop` | Stop annotated recording |

## 8. Firewall and port 5000

If the phone cannot connect while the dashboard works locally, allow TCP port 5000 on the laptop's **trusted/private** network.

Windows PowerShell as Administrator:

```powershell
New-NetFirewallRule -DisplayName "CodeAlpha Object Detection 5000" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow -Profile Private
```

Linux/UFW example:

```bash
sudo ufw allow 5000/tcp
sudo ufw status
```

Use the macOS firewall settings if inbound connections are blocked. Never expose this development server directly to the public Internet.

## 9. Troubleshooting

### Connection refused / timeout

- Confirm `python backend/server.py` is running.
- Confirm it reports `0.0.0.0:5000`.
- Verify phone and laptop are on the same Wi-Fi.
- Use the laptop's Wi-Fi IPv4, not `127.0.0.1`.
- Test `http://LAPTOP_IPV4:5000/` from another LAN device.
- Check the firewall.
- Guest/campus Wi-Fi may isolate devices; use a network that permits device-to-device traffic.

### Test Ping fails on Android

Test Ping checks network access to the server; it does not require YOLO. Fix the IP, Wi-Fi isolation and firewall first.

### No frames received

- Start the Android camera after entering `http://LAPTOP_IPV4:5000`.
- Confirm the phone shows streaming.
- Check whether the laptop received a different IP after reconnecting to Wi-Fi.
- Never use `localhost` or `127.0.0.1` in the Android URL.

### Missing Python packages

```bash
python -m pip install -r backend/requirements.txt
```

Restart `backend/server.py` after installation.

### Missing YOLO weights / model error

Keep `yolo11n.pt` selected and allow Ultralytics to download it on first detection startup. If the machine is offline, restore an existing cached weight or provide the required local weight file, then restart the backend. The dashboard reports the actual engine error instead of generating fake results.

### Low FPS

- Keep `yolo11n.pt` for CPU operation.
- Reduce phone stream resolution if necessary.
- Reduce Android transmission rate if Wi-Fi is congested.
- Prefer a wired laptop network connection where possible.
- Close other CPU/GPU-heavy applications.
- Prefer ByteTrack for a lighter tracking workload.
- Increase confidence threshold when appropriate.

### Processed view is empty

The processed endpoint intentionally does not fall back to a raw frame. Start the phone stream, click **Start Detection**, and wait for the first real inference result.

## 10. GitHub Actions APK build

The **Build Android APK** workflow:

- uses JDK 17;
- makes `gradlew` executable;
- runs `./gradlew assembleDebug`;
- verifies `app/build/outputs/apk/debug/app-debug.apk` exists and is non-empty;
- uploads the artifact named `CodeAlpha_ObjectDetectionTracking-debug-apk`.

After a successful run on GitHub:

1. Open **Actions**.
2. Select **Build Android APK**.
3. Open the successful **Build Debug APK** run.
4. Scroll to **Artifacts**.
5. Open `CodeAlpha_ObjectDetectionTracking-debug-apk`.
6. Download the ZIP and extract `app-debug.apk`.

A known-good APK is preserved at `android-app/app-debug.apk` and is not replaced by a failed build.

## Local Android build

Linux/macOS:

```bash
./gradlew assembleDebug
```

Windows:

```bat
gradlew.bat assembleDebug
```

Output:

```text
app/build/outputs/apk/debug/app-debug.apk
```

The Android Gradle configuration preserves Java 17 using `compileOptions` and removes the obsolete Kotlin `kotlinOptions { jvmTarget = "17" }` block that caused the failed CI build.
