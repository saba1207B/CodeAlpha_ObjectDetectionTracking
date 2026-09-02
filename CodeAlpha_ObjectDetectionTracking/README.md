# CodeAlpha Object Detection & Tracking

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Android](https://img.shields.io/badge/Android-Kotlin%20%7C%20CameraX-3DDC84?logo=android&logoColor=white)](https://developer.android.com/)
[![YOLO](https://img.shields.io/badge/YOLO-v11%20%2F%20v8-00FFFF?logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![Tracking](https://img.shields.io/badge/Tracker-ByteTrack%20%7C%20BoT--SORT-brightgreen)](https://github.com/ifzhang/ByteTrack)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red?logo=opencv&logoColor=white)](https://opencv.org/)
[![Build APK](https://github.com/your-username/CodeAlpha_ObjectDetectionTracking/actions/workflows/build-apk.yml/badge.svg)](https://github.com/your-username/CodeAlpha_ObjectDetectionTracking/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview & CodeAlpha Internship

- **Organization:** CodeAlpha
- **Domain:** Artificial Intelligence Internship
- **Task Number:** Task 4
- **Task Title:** Object Detection and Tracking
- **Repository:** `CodeAlpha_ObjectDetectionTracking`

This project is a complete, production-grade system implementing real-time **Multi-Object Detection and Tracking (MOT)** across sequential video frames. The solution features an end-to-end distributed architecture: an **Android smartphone (CameraX client)** streams live camera frames over HTTP to a multi-threaded **Python server on a laptop**, which feeds the frames into **Ultralytics YOLO (v11/v8)** and **ByteTrack / BoT-SORT** to maintain persistent object identities, draw bounding boxes, compute rolling FPS, and render motion trajectory trails in real time.

In addition to phone camera streaming, the pipeline natively supports local laptop webcams (`--source 0`) and pre-recorded video files (`--source sample/video.mp4`).

---

## 2. Key Features

- 📱 **Real Android Camera Client:** Native Kotlin app powered by CameraX and OkHttp streaming low-latency JPEG frames with auto-rotation and live preview.
- 🚀 **Zero-Lag Frame Buffer:** Multi-threaded HTTP receiver (`phone_server.py`) enforcing a "latest frame wins" policy to eliminate network lag accumulation.
- 🧠 **Pretrained Deep Neural Networks:** Native inference with Ultralytics YOLOv11 and YOLOv8 models (`yolo11n.pt`, `yolov8n.pt`, etc.).
- 🎯 **Multi-Class Detection:** Detects up to 80 COCO classes with configurable confidence, IoU, and class filters.
- 🆔 **Persistent Multi-Object Tracking:** ByteTrack and BoT-SORT algorithms maintain persistent track IDs across frames and through partial occlusions.
- 🌈 **Motion Trajectory Trails:** Displays fading centroid history lines showing historical object movement.
- 📊 **Real-Time Heads-Up Display (HUD):** Shows rolling FPS, active tracks, cumulative unique IDs, and source metrics.
- 💻 **Cross-Source Flexibility:** Switch effortlessly between phone camera (`--source phone`), laptop webcam (`--source 0`), and video files (`--source video.mp4`).
- 🤖 **Automated CI/CD:** GitHub Actions workflow compiles the Android debug APK on push without requiring Android Studio on your development machine.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        ANDROID PHONE                        │
│                                                             │
│   CameraX Preview  ───>  ImageAnalysis Loop                 │
│                                │                            │
│                                ▼                            │
│                        Rotation & Resize                    │
│                        (640x480 JPEG)                       │
│                                │                            │
│                                ▼                            │
│                     OkHttp FrameSender Client               │
│                  (Atomic dropped frame throttle)            │
└────────────────────────────────┬────────────────────────────┘
                                 │
                                 │ HTTP POST /frame (JPEG)
                                 │ Over Wi-Fi / Hotspot / USB
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    LAPTOP (PYTHON SERVER)                   │
│                                                             │
│   phone_server.py (ThreadingHTTPServer on 0.0.0.0:5000)     │
│   ├── POST /frame        : Thread-safe FrameBuffer update   │
│   ├── GET  /             : Health check                     │
│   ├── GET  /latest_frame : Instant frame retrieval          │
│   └── GET  /status       : Diagnostic metrics (FPS, stats)  │
│                                │                            │
│                                │ Thread-Safe Memory / HTTP  │
│                                ▼                            │
│   app.py --source phone                                     │
│   ├── YOLOv11/v8 Object Detector (Classes + Confidence)     │
│   ├── ByteTrack / BoT-SORT Tracker (Persistent IDs)         │
│   ├── Motion Trajectory Trails & HUD Overlay                │
│   └── OpenCV HighGUI Real-Time Window                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Technology Stack

| Layer | Component | Technologies |
| :--- | :--- | :--- |
| **Android Client** | Camera Pipeline | Kotlin, AndroidX CameraX (1.4.1), Coroutines |
| **Android Client** | UI & Networking | Jetpack Compose, Material Design 3, OkHttp 4.12 |
| **Server Backend** | Ingestion & Storage | Python 3.12, ThreadingHTTPServer, Thread-safe Lock |
| **AI / Vision** | Detection Engine | PyTorch 2.2+, Ultralytics YOLOv11 / YOLOv8 |
| **AI / Vision** | Tracking Engine | ByteTrack, BoT-SORT, Lapx, FilterPy |
| **Rendering** | Computer Vision GUI | OpenCV (`cv2`), NumPy, Pillow |
| **CI / CD** | Automated Build | GitHub Actions, Gradle 9.3.1, OpenJDK 17 |

---

## 5. Repository Structure

```
CodeAlpha_ObjectDetectionTracking/
│
├── .github/
│   └── workflows/
│       └── build-apk.yml               # Automated APK compilation workflow
│
├── app/                                # Android Camera Client application
│   ├── build.gradle.kts                # Android build script (CameraX, OkHttp)
│   └── src/main/
│       ├── AndroidManifest.xml         # Camera & network permissions
│       ├── java/com/example/
│       │   ├── MainActivity.kt         # Compose UI & CameraX lifecycle
│       │   ├── camera/
│       │   │   └── FrameProcessor.kt   # JPEG compression & rotation
│       │   └── network/
│       │       └── FrameSender.kt      # Non-blocking HTTP streaming
│       └── res/
│           └── xml/
│               └── network_security_config.xml # Allows local HTTP cleartext
│
├── CodeAlpha_ObjectDetectionTracking/  # Core Python Vision System
│   ├── app.py                          # Main YOLO + ByteTrack runner
│   ├── phone_server.py                 # Multi-threaded HTTP phone frame server
│   ├── requirements.txt                # Python package specifications
│   ├── src/
│   │   ├── __init__.py
│   │   ├── detector.py                 # YOLODetector class
│   │   └── tracker.py                  # ObjectTracker class
│   ├── sample/                         # Folder for sample video files
│   ├── screenshots/                    # Output directory for saved captures
│   └── output/                         # Output directory for saved videos
│
├── app.py                              # Root runner convenience wrapper
├── phone_server.py                     # Root server convenience wrapper
├── requirements.txt                    # Root requirements file
├── build.gradle.kts                    # Root Gradle build configuration
├── settings.gradle.kts                 # Project modules & repository settings
├── gradle.properties                   # JVM memory & configuration cache flags
├── gradlew / gradlew.bat               # Gradle wrapper executables
└── README.md                           # Master documentation
```

---

## 6. Python Environment Setup

Target Environment: **Python 3.10 – 3.12** (Tested on Python 3.12.10).

### Step 1: Create and Activate Virtual Environment

**On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Upgrade PIP & Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Verify Installation

Run the verification commands to confirm all modules are operational:
```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
python -c "import requests; print('Requests:', requests.__version__)"
```

---

## 7. Android Client Setup & APK Compilation

### Option A: Download Automated Build Artifacts (No Android Studio Required)
1. Push your repository to GitHub.
2. In your repository, click the **Actions** tab.
3. Select the latest **Build Android APK** workflow run.
4. Under **Artifacts**, download `codealpha-camera-client-apk.zip`.
5. Unzip and install `app-debug.apk` onto your Android phone via USB or Google Drive.

### Option B: Local Command-Line Compilation
Build the APK locally using the Gradle wrapper (requires Java 17+):
```bash
# On Linux / macOS
./gradlew assembleDebug

# On Windows
gradlew.bat assembleDebug
```
The compiled APK will be generated at:
```
app/build/outputs/apk/debug/app-debug.apk
```

---

## 8. Network Configuration & Phone Connection Guide

The Android phone must be able to reach your laptop over your local network.

### How to Find Your Laptop's IP Address

**On Windows:**
1. Open Command Prompt (`cmd`).
2. Type `ipconfig` and press Enter.
3. Look for your active adapter (**Wireless LAN adapter Wi-Fi** or **Ethernet**).
4. Find the **IPv4 Address** (e.g., `192.168.1.150` or `10.138.211.159`).

> [!IMPORTANT]
> **Understanding Ports vs. IP Addresses:**
> `ipconfig` displays your device's **IP addresses only**. Port numbers (`5000`) will **NOT** appear in `ipconfig`.
> Port 5000 is opened by `phone_server.py` when it runs. You combine the IP from `ipconfig` with the port to form the server URL:
> `http://<YOUR_IPV4_ADDRESS>:5000`

**On Linux / macOS:**
```bash
# Linux
ip a | grep inet

# macOS
ifconfig | grep "inet "
```

---

## 9. Step-by-Step Running Guide

### Architecture: Android Phone Camera Streaming

#### Step 1: Start the Phone Server (Laptop - Terminal 1)
```bash
python phone_server.py
```
**Expected Output:**
```
==================================================
CodeAlpha Object Detection & Tracking
Phone Camera Server
==================================================
Listening on:
http://0.0.0.0:5000

Waiting for phone camera frames...
==================================================
```

#### Step 2: Start the Detection & Tracking Pipeline (Laptop - Terminal 2)
```bash
python app.py --source phone
```
The system will display:
```
[i] Waiting for phone camera frames...
    1. Open the CodeAlpha Camera app on your Android phone.
    2. Enter your laptop's IP address (e.g. http://<LAPTOP_IP>:5000).
    3. Tap 'Start Camera'.
```

#### Step 3: Stream from the Android App
1. Launch the **CodeAlpha Object Tracking** app on your phone.
2. Grant camera permissions when prompted.
3. In the **Laptop Server URL** input field, type:
   ```
   http://<YOUR_LAPTOP_IP>:5000
   ```
   *(Example: `http://192.168.1.150:5000` or `http://10.138.211.159:5000`)*
4. Tap **Start Camera**.
5. The Android app will display live preview and transmission metrics. The laptop window will immediately transition into live YOLO detection and tracking!

---

### Alternative Modes: Webcam & Video Files

#### Run with Laptop Webcam
```bash
python app.py --source 0
```

#### Run with Pre-recorded Video File
```bash
python app.py --source sample/video.mp4
```

#### Save Processed Output Video
```bash
python app.py --source phone --save output/tracked_phone.mp4
```

---

## 10. Interactive Keyboard Controls (Active Video Window)

| Key | Action | Description |
| :---: | :--- | :--- |
| `Q` / `ESC` | **Quit** | Gracefully releases all hardware resources and prints session summary. |
| `P` / `SPACE` | **Pause / Resume** | Pauses or resumes detection and tracking. |
| `S` | **Screenshot** | Saves a high-resolution screenshot to `screenshots/`. |
| `T` | **Toggle Trails** | Turns motion trajectory lines ON / OFF. |
| `H` | **Toggle HUD** | Shows or hides the Heads-Up Display banner. |

---

## 11. Command-Line Options Reference

```bash
python app.py --help
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source` | `str` | `0` | `'phone'`, webcam index (`0`, `1`), or video file path. |
| `--phone-server` | `str` | `http://127.0.0.1:5000` | URL of the running `phone_server.py` instance. |
| `--model` | `str` | `yolo11n.pt` | Pretrained YOLO checkpoint (`yolo11n.pt`, `yolov8n.pt`, `yolo11s.pt`). |
| `--conf` | `float` | `0.35` | Minimum confidence threshold for detection (0.01 – 1.0). |
| `--iou` | `float` | `0.45` | IoU threshold for Non-Maximum Suppression and tracker association. |
| `--tracker` | `str` | `bytetrack.yaml` | Multi-object tracking algorithm (`bytetrack.yaml` or `botsort.yaml`). |
| `--classes` | `int ...` | `None` | Filter specific classes (e.g. `--classes 0` for person, `0 2` for person/car). |
| `--save` | `str` | `None` | File path to export annotated video output (`output/result.mp4`). |
| `--no-trails` | `flag` | `False` | Disables rendering of trajectory history lines. |
| `--no-hud` | `flag` | `False` | Hides the top and bottom HUD statistics banner. |
| `--no-display` | `flag` | `False` | Runs headless without opening an OpenCV GUI window. |
| `--device` | `str` | `None` | Device compute target (`'cpu'`, `'cuda'`, `'mps'`). |

---

## 12. Troubleshooting Guide

### 1. Android App shows "Hello Android" / Old Screen
- **Cause:** An old placeholder build was installed on the device.
- **Solution:** Reinstall the app using the fresh `app-debug.apk` built from this codebase. The updated app has a full CameraX preview, URL input field, and stream stats.

### 2. "No module named cv2" / "No module named ultralytics"
- **Cause:** Virtual environment is not active.
- **Solution:** Activate your virtual environment first (`venv\Scripts\activate` on Windows or `source venv/bin/activate` on Linux) and run:
  ```bash
  pip install -r requirements.txt
  ```

### 3. Phone App shows "Server Connection Failed" / "Connection Refused"
- **Verify server:** Ensure `python phone_server.py` is running in Terminal 1.
- **Verify URL:** Test the URL on your laptop: `curl http://127.0.0.1:5000/`
- **Do NOT use `localhost` or `127.0.0.1` on the phone:** The phone is an independent device. You must use the laptop's LAN IPv4 address (e.g. `http://192.168.1.150:5000`).
- **Windows Firewall:** Ensure Python is allowed through Windows Firewall or add an inbound rule for TCP port 5000.

### 4. "Port 5000 is not visible in ipconfig"
- **Explanation:** `ipconfig` **only displays network interfaces and IP addresses**, never active port numbers. Port 5000 is created dynamically by `phone_server.py`.

---

## 13. Privacy & Security Statement

- **100% Local Processing:** All camera frames captured by the Android phone and transmitted to the laptop are processed entirely in local memory.
- **Zero Cloud Uploads:** No video footage, frame data, or analytical telemetry is ever transmitted to external servers or third-party APIs.
- **No Biometrics or Facial Recognition:** The system classifies generic object categories (e.g. person, car, bicycle) defined in the COCO dataset. It does not perform facial recognition or store biometric identities.
- **Explicit Storage Only:** Video recordings are only saved to disk when the user explicitly provides the `--save` parameter.

---

## 14. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Developed for the **CodeAlpha Artificial Intelligence Internship — Task 4: Object Detection and Tracking**.
