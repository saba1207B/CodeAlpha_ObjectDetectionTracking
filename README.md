# CodeAlpha Object Detection & Tracking

Real-time object detection and multi-object tracking using an Android CameraX client, a Python local-network backend, Ultralytics YOLO, and ByteTrack / BoT-SORT.

## 🎯 Project purpose

This project turns an Android phone into a wireless camera client for a laptop-based computer-vision system. The phone captures camera frames and sends them over a trusted local Wi-Fi network to the Python backend. The laptop performs real YOLO inference and multi-object tracking, while the browser provides the live dashboard.

**The laptop GUI is the web browser.** No separate desktop GUI application is required.

```text
Android Camera APK
       │
       │ JPEG frames: POST /frame
       ▼
Same Wi-Fi / trusted LAN
       │
       ▼
Python Flask backend :5000
       │
       ├── Raw frame buffer ───────────────► Browser dashboard
       │
       ▼
Ultralytics YOLO
       │
       ▼
ByteTrack / BoT-SORT
       │
       ├── Bounding boxes
       ├── Class labels
       ├── Confidence scores
       ├── Persistent tracking IDs
       └── Motion trails / HUD
       │
       ▼
Processed frames + detection API
       │
       ▼
Laptop web browser dashboard
```

## ✨ Current project status

- ✅ Android camera client implemented
- ✅ Python Flask backend implemented
- ✅ Real Ultralytics YOLO inference
- ✅ ByteTrack / BoT-SORT tracking
- ✅ Live browser dashboard
- ✅ Android-compatible `/frame`, `/`, `/latest_frame`, and `/status` endpoints
- ✅ LAN server binding on `0.0.0.0:5000`
- ✅ Dashboard controls, statistics, class filters and tracking visualization
- ✅ GitHub Actions Debug APK workflow
- ✅ JDK 17 Android build configuration
- ✅ One-click Windows startup and shutdown scripts
- ✅ Detailed setup and troubleshooting documentation
- ✅ No fake detections or simulated tracking results

The project is ready for **end-to-end Android + laptop LAN testing**. The remaining validation that depends on physical hardware is to run the phone and laptop on the same Wi-Fi network and test the complete camera → backend → YOLO → dashboard path.

## 🚀 One-click Windows startup

If you are using Windows, you do not need to type the Python/virtual-environment commands every time.

1. Clone or update the repository once.
2. Make sure Python 3.10–3.13 is installed. Python 3.12 is recommended.
3. Double-click **`START.bat`** in the project folder.
4. The launcher creates `.venv` when needed, installs/updates backend requirements, detects a private LAN IPv4, opens the dashboard and starts the backend.
5. The terminal displays the Android connection URL, for example `http://192.168.1.25:5000`.
6. Enter that URL in the Android APK.
7. Tap **Test Ping**, then **Start Camera**.
8. In the browser dashboard, click **Start Detection**.

Keep the `START.bat` terminal open while using the system.

To stop the backend, double-click **`STOP.bat`**. It targets the project's `backend/server.py` process instead of indiscriminately terminating all Python processes.

> **Important:** `START.bat` is Windows-only. Linux/macOS users should follow the manual setup below.

## 📁 Repository layout

```text
.
├── START.bat
├── STOP.bat
├── android-app/
│   └── app-debug.apk             # Preserved known-good APK
├── app/                          # Android Gradle application source
├── backend/
│   ├── server.py                 # Flask server + live vision pipeline
│   ├── app.py                    # CLI/backend pipeline entry point
│   ├── requirements.txt
│   └── src/
│       ├── detector.py           # YOLO wrapper
│       └── tracker.py             # Tracker integration
├── laptop-dashboard/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── screenshots/
├── output/
└── .github/workflows/
    └── build-apk.yml
```

## 🛠️ Manual installation

### 1. Clone or update

```bash
git clone https://github.com/saba1207B/CodeAlpha_ObjectDetectionTracking.git
cd CodeAlpha_ObjectDetectionTracking
```

For an existing checkout:

```bash
git pull origin main
```

### 2. Create the Python virtual environment

Python 3.10–3.13 is recommended; Python 3.12 is recommended for a consistent Windows setup.

**Windows:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell activation is restricted:

```bat
.venv\Scripts\activate.bat
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install requirements

```bash
python -m pip install -r backend/requirements.txt
```

The backend uses Flask, OpenCV, NumPy, Ultralytics and related computer-vision dependencies. On first YOLO startup, Ultralytics may download `yolo11n.pt`; therefore first-time detection normally needs Internet access unless the model is already cached locally.

## ▶️ Start the backend

From the repository root:

```bash
python backend/server.py
```

The server listens on:

```text
0.0.0.0:5000
```

Open the dashboard on the laptop:

```text
http://localhost:5000/
```

or, from another device on the same LAN:

```text
http://LAPTOP_IPV4:5000/
```

The backend is the processing engine; the browser is the user interface.

## 🌐 Find the laptop IPv4

**Windows:**

```powershell
ipconfig
```

Use the Wi-Fi adapter's IPv4 address, such as `192.168.1.25`.

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

Use the correct active Wi-Fi interface if `en0` is not applicable.

## 📱 Connect the Android APK

The phone and laptop must be on the **same Wi-Fi network**.

In the Android app, enter:

```text
http://LAPTOP_IPV4:5000
```

Example:

```text
http://192.168.1.25:5000
```

### Important networking rule

**Never enter `localhost` or `127.0.0.1` in the Android app for the laptop server.** Those addresses point back to the phone itself.

Use the laptop's actual LAN IPv4 address.

### Connection sequence

1. Start the Python backend.
2. Confirm the laptop dashboard opens.
3. Enter the laptop LAN URL in the Android app.
4. Tap **Test Ping**.
5. If ping succeeds, tap **Start Camera**.
6. Return to the laptop dashboard.
7. Confirm frames and phone/stream status are updating.
8. Click **Start Detection**.
9. Confirm YOLO boxes, labels, confidence values and tracking IDs appear.

## 🧠 Detection and tracking pipeline

The application uses real incoming phone frames. When detection is started:

1. The backend receives JPEG frames from Android.
2. Frames are stored in the live frame buffer.
3. Ultralytics YOLO performs object detection.
4. The selected tracker associates detections between frames.
5. Bounding boxes, class labels, confidence values and tracking IDs are generated.
6. Optional trails/HUD information is rendered.
7. The annotated frame is exposed to the dashboard.
8. The dashboard reads current detection/tracking data through the API.

There are **no fake boxes, mock detections, simulated tracking IDs or fabricated performance metrics**.

### Recommended initial configuration

- Model: `yolo11n.pt` / YOLOv11 Nano
- Tracker: ByteTrack
- Confidence: use the dashboard default initially
- IoU: use the dashboard default initially
- Class filters: all classes initially
- Trails: optional
- HUD: optional

`yolo11n.pt` is recommended for CPU-friendly laptop testing. Larger models may improve detection quality but can reduce FPS and increase resource usage.

## 🖥️ Dashboard features

The browser dashboard provides:

- Backend health status
- Android/stream status
- Laptop IPv4
- Complete Android connection URL
- Copy URL control
- Received frame count
- Stream FPS
- API round-trip latency
- Frame resolution
- Last-frame age
- Raw camera frame
- Processed YOLO frame
- Raw/processed split view
- Bounding boxes
- Class labels
- Confidence scores
- Persistent tracking IDs
- Current tracked-object table
- Motion trails
- HUD/statistics overlay
- Start Detection
- Stop Detection
- Reset/Clear Results
- YOLO model selection
- Confidence threshold
- IoU threshold
- ByteTrack / BoT-SORT selection
- COCO class filtering
- Screenshot capture
- Optional annotated video recording
- Visible server, Wi-Fi, frame, package and YOLO errors

## 🔌 API reference

### Android-compatible endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Health check and dashboard |
| POST | `/frame` | Receive Android JPEG frame |
| GET | `/latest_frame` | Return latest raw frame |
| GET | `/status` | Return server/stream metrics |

### Dashboard and vision endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/processed_frame` | Latest annotated YOLO frame |
| GET | `/detections` | Current tracked objects |
| POST | `/start-detection` | Start real detection/tracking |
| POST | `/stop-detection` | Stop detection/tracking |
| GET | `/settings` | Read current settings |
| POST | `/settings` | Update model/tracker/threshold/filter settings |
| POST | `/clear-results` | Reset tracking/results state |
| POST | `/screenshot` | Capture a frame |
| POST | `/recording/start` | Start annotated recording |
| POST | `/recording/stop` | Stop annotated recording |

## 🔥 Firewall and LAN access

If the dashboard works on the laptop but the Android **Test Ping** fails, the firewall is a likely cause.

### Windows

Run PowerShell as Administrator:

```powershell
New-NetFirewallRule -DisplayName "CodeAlpha Object Detection 5000" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow -Profile Private
```

Only use this for a trusted/private network profile.

### Linux/UFW

```bash
sudo ufw allow 5000/tcp
sudo ufw status
```

### Network isolation

Some guest, campus, office and public Wi-Fi networks prevent devices from communicating with one another. If both devices are connected but the phone cannot reach the laptop, test using a trusted network that permits device-to-device traffic.

> **Security:** this is a local development server. Do not port-forward or expose TCP port 5000 directly to the public Internet.

## 🧪 End-to-end test checklist

Use this checklist after cloning/updating the repository.

### Laptop

- [ ] Python installed
- [ ] Virtual environment created
- [ ] Requirements installed
- [ ] Backend starts without a fatal error
- [ ] Server reports/listens on port 5000
- [ ] Dashboard opens at `http://localhost:5000/`
- [ ] Laptop Wi-Fi IPv4 identified

### Android

- [ ] APK installed
- [ ] Phone and laptop are on the same Wi-Fi
- [ ] Android connection URL uses laptop IPv4
- [ ] No `localhost`/`127.0.0.1` used
- [ ] **Test Ping** succeeds
- [ ] Camera starts
- [ ] Frames are transmitted

### Dashboard

- [ ] Android/stream status becomes active
- [ ] Frame count increases
- [ ] Resolution is reported
- [ ] Stream FPS updates
- [ ] Raw frame appears
- [ ] **Start Detection** succeeds
- [ ] Processed frame appears
- [ ] Real YOLO detections appear when objects are visible
- [ ] Tracking IDs persist across suitable frames
- [ ] Tracker/class/filter controls respond
- [ ] Screenshot works
- [ ] Recording works if enabled

## 🐞 Troubleshooting

### Android cannot connect

Check, in order:

1. Backend is running.
2. Backend is bound to `0.0.0.0:5000`.
3. Phone and laptop are on the same Wi-Fi.
4. Android uses the laptop's Wi-Fi IPv4.
5. Port 5000 is allowed through the private-network firewall.
6. The Wi-Fi network is not using client isolation.

### Test Ping fails

**Test Ping only checks phone-to-backend connectivity.** YOLO does not need to be running for this test. Fix IP, Wi-Fi and firewall problems before troubleshooting detection.

### No frames received

- Start the Android camera.
- Confirm the connection URL is correct.
- Check the laptop IP again if Wi-Fi was disconnected/reconnected.
- Confirm the dashboard is reachable from the phone's network.
- Check the backend terminal for incoming requests/errors.

### YOLO model/weights error

Keep `yolo11n.pt` selected for the initial test and allow Ultralytics to download the model if it is not cached. If the laptop is offline, a locally available compatible weight file is required.

The application surfaces the actual engine error instead of pretending that inference succeeded.

### Processed frame is empty

The processed endpoint does not silently substitute the raw frame. Confirm:

1. Phone camera is streaming.
2. Frames are increasing.
3. **Start Detection** has been clicked.
4. YOLO initialized successfully.
5. The backend terminal contains no inference error.

### Low FPS / high latency

- Start with `yolo11n.pt`.
- Prefer ByteTrack for lighter tracking.
- Reduce camera resolution if necessary.
- Reduce transmission rate if Wi-Fi is congested.
- Close other CPU/GPU-heavy applications.
- Use a stable Wi-Fi connection.
- A wired laptop connection can reduce LAN variability.

## 📦 GitHub Actions — Android APK

The repository contains a **Build Android APK** workflow.

The workflow is configured to:

- use JDK 17;
- make `gradlew` executable;
- run `./gradlew assembleDebug`;
- verify the generated Debug APK exists and is non-empty;
- upload the APK as `CodeAlpha_ObjectDetectionTracking-debug-apk`.

### Latest known-good build

The latest verified GitHub Actions build completed successfully as **Build Android APK run #22**.

Successful run:

https://github.com/saba1207B/CodeAlpha_ObjectDetectionTracking/actions/runs/33707787292

Build commit:

```text
11ba43102bb0f51c1fe3620ebf73904f9767823d
```

Artifact:

```text
CodeAlpha_ObjectDetectionTracking-debug-apk
```

The APK generated by that successful build is suitable for the current Android source. Laptop-only changes such as `START.bat`, `STOP.bat` and README documentation do **not** require rebuilding the APK.

### Downloading a new APK from Actions

1. Open the repository's **Actions** page.
2. Select **Build Android APK**.
3. Open a successful build run.
4. Scroll to **Artifacts**.
5. Select `CodeAlpha_ObjectDetectionTracking-debug-apk`.
6. Extract `app-debug.apk` from the downloaded ZIP.

A known-good APK is also preserved at:

```text
android-app/app-debug.apk
```

It should not be replaced by a failed CI output.

## 🔧 Android build configuration

The Android Gradle configuration uses Java 17 through `compileOptions`.

The obsolete Kotlin `kotlinOptions { jvmTarget = "17" }` configuration that caused the earlier CI failure was removed. The current build workflow uses JDK 17 and the Gradle wrapper.

Local build commands:

**Linux/macOS:**

```bash
./gradlew assembleDebug
```

**Windows:**

```bat
gradlew.bat assembleDebug
```

Generated APK:

```text
app/build/outputs/apk/debug/app-debug.apk
```

## 🧩 One-click workflow summary

### Windows

```text
Double-click START.bat
        ↓
Virtual environment + dependencies
        ↓
LAN IPv4 detected
        ↓
Browser dashboard opens
        ↓
Backend starts on port 5000
        ↓
Enter displayed URL in Android APK
        ↓
Test Ping
        ↓
Start Camera
        ↓
Start Detection in browser
        ↓
YOLO + tracking + live dashboard
```

### Stopping

```text
Double-click STOP.bat
        ↓
backend/server.py process stops
        ↓
Close browser tab normally
```

## 🤖 Review / recheck request

This repository is also intended to be independently reviewed after implementation changes. A reviewer should inspect the actual source rather than relying only on this README, with particular attention to:

- Android-to-laptop IP handling
- `/frame` compatibility
- Flask binding and LAN accessibility
- YOLO initialization and real inference
- ByteTrack / BoT-SORT integration
- frame buffering and concurrency
- dashboard API compatibility
- error handling and non-fake status reporting
- Gradle/JDK compatibility
- GitHub Actions APK artifact generation
- Windows launcher behavior
- firewall and trusted-LAN assumptions

The goal is to keep the implementation simple, reproducible and testable while avoiding unnecessary changes that do not address a real issue.

## 📄 License / project note

This is a CodeAlpha project implementation for learning and demonstration of real-time computer vision, Android networking, object detection and multi-object tracking.
