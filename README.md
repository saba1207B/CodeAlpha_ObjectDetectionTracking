# CodeAlpha Object Detection & Tracking 👁️

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Android](https://img.shields.io/badge/Android-Kotlin%20%7C%20CameraX-3DDC84?logo=android&logoColor=white)](https://developer.android.com/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF?logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red?logo=opencv&logoColor=white)](https://opencv.org/)
[![Tracking](https://img.shields.io/badge/Tracking-ByteTrack%20%7C%20BoT--SORT-brightgreen)](https://github.com/ifzhang/ByteTrack)
[![Build APK](https://github.com/saba1207B/CodeAlpha_ObjectDetectionTracking/actions/workflows/build-apk.yml/badge.svg)](https://github.com/saba1207B/CodeAlpha_ObjectDetectionTracking/actions/workflows/build-apk.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**CodeAlpha Object Detection & Tracking** is an end-to-end computer-vision project developed for the **CodeAlpha Artificial Intelligence Internship — Task 4: Object Detection and Tracking**.

The project combines a native Android CameraX client with a Python computer-vision pipeline. The phone streams JPEG camera frames over a local network to a laptop, where **Ultralytics YOLO** performs object detection and **ByteTrack / BoT-SORT** maintains persistent object IDs.

The same Python pipeline can also process a laptop webcam or a prerecorded video file.

## Architecture

```text
📱 Android Phone
   CameraX + OkHttp
          │
          │ HTTP POST /frame
          ▼
💻 Laptop — Python
   phone_server.py
          │
          ▼
   OpenCV frame buffer
          │
          ▼
   YOLO object detection
          │
          ▼
   ByteTrack / BoT-SORT
          │
          ▼
   Bounding boxes + IDs + trails + HUD
          │
          ▼
   OpenCV live display / saved output
```

## Features

- 📱 Native Android camera client using Kotlin + CameraX
- 🌐 Local HTTP streaming from phone to laptop
- 🚀 Latest-frame-wins buffering to prevent latency buildup
- 🧠 YOLO object detection with configurable confidence and IoU
- 🎯 ByteTrack / BoT-SORT multi-object tracking
- 🆔 Persistent tracking IDs across frames
- 🌈 Optional motion trajectory trails
- 📊 Live FPS, active-track and session statistics
- 💻 Laptop webcam, phone camera and video-file input modes
- 💾 Optional annotated video and screenshot output
- 🔧 Built-in server health and status endpoints
- 🤖 GitHub Actions APK build without Android Studio

## Technology Stack

| Layer | Technologies |
|---|---|
| Android | Kotlin, Jetpack Compose, CameraX, OkHttp |
| Python | Python 3.12, HTTP server, threading |
| Detection | Ultralytics YOLO |
| Tracking | ByteTrack, BoT-SORT, Lapx, FilterPy |
| Vision | OpenCV, NumPy, Pillow |
| Build | Gradle, Android Gradle Plugin, JDK 17 |
| CI/CD | GitHub Actions |

## Project Structure

```text
CodeAlpha_ObjectDetectionTracking/
├── .github/workflows/
│   └── build-apk.yml
├── app/
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── java/com/example/
│       │   ├── MainActivity.kt
│       │   ├── camera/FrameProcessor.kt
│       │   └── network/FrameSender.kt
│       └── res/
├── CodeAlpha_ObjectDetectionTracking/
│   ├── app.py
│   ├── phone_server.py
│   ├── requirements.txt
│   ├── src/
│   │   ├── detector.py
│   │   └── tracker.py
│   ├── sample/
│   ├── screenshots/
│   └── output/
├── gradle/
├── gradlew
├── gradlew.bat
├── build.gradle.kts
├── settings.gradle.kts
├── gradle.properties
├── metadata.json
├── .gitignore
└── README.md
```

## Python Setup

The tested environment is **Python 3.12.10**.

From the Python project directory:

```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the environment:

```cmd
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
```

The YOLO model weights are downloaded automatically on first use when required.

## Phone Camera Setup

### 1. Start the phone server

On the laptop, inside `CodeAlpha_ObjectDetectionTracking/`:

```cmd
python phone_server.py
```

Expected output:

```text
==================================================
CodeAlpha Object Detection & Tracking
Phone Camera Server
==================================================
Listening on:
http://0.0.0.0:5000

Waiting for phone camera frames...
```

The server uses **port 5000** by default.

### 2. Start the detection pipeline

Open another terminal:

```cmd
python app.py --source phone
```

The application waits for frames from the Android client and then sends them through YOLO and the tracker.

### 3. Find the laptop IP

On Windows:

```cmd
ipconfig
```

Find the **IPv4 Address** of the active network adapter connecting the phone and laptop.

> **Important:** `ipconfig` shows IP addresses, not port numbers. You will not see `:5000` there.

### 4. Configure the Android app

Open the Android APK and enter the complete URL in **Laptop Server URL**:

```text
http://YOUR_LAPTOP_IPV4:5000
```

Example:

```text
http://10.138.211.159:5000
```

Then use **Test Ping (GET /)** and, once reachable, tap **Start Camera**.

Do **not** use `127.0.0.1` or `localhost` on the phone; those addresses refer to the phone itself.

## Network Options

The phone and laptop can communicate over:

1. **Same Wi-Fi network** — both devices connected to the same router.
2. **Phone hotspot** — laptop connected to the phone's hotspot.
3. **USB tethering** — phone connected to the laptop by USB with USB tethering enabled.

The laptop IP can change when the connection method changes, so always check the current active IPv4 address.

## Connectivity Testing

The Python server provides:

```text
GET  /
POST /frame
GET  /latest_frame
GET  /status
```

Test locally on the laptop:

```cmd
curl http://127.0.0.1:5000
```

You should receive:

```text
CodeAlpha Object Detection & Tracking Server
Status: Running
Endpoint: POST /frame
```

You can also test the laptop's network address:

```cmd
curl http://YOUR_LAPTOP_IPV4:5000
```

If the phone cannot connect, check the server, IP address, network path and Windows Firewall. Do not disable Windows Firewall completely.

## Alternative Input Modes

### Laptop webcam

```cmd
python app.py --source 0
```

For another webcam:

```cmd
python app.py --source 1
```

### Video file

```cmd
python app.py --source sample/video.mp4
```

### Save processed output

```cmd
python app.py --source phone --save output/tracked_phone.mp4
```

## Controls

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `P` / `SPACE` | Pause / resume |
| `S` | Save screenshot |
| `T` | Toggle tracking trails |
| `H` | Toggle HUD |

## Android APK

The repository includes a GitHub Actions workflow that builds the debug APK automatically.

**Artifact:**

```text
CodeAlpha_ObjectDetectionTracking-debug-apk
```

The workflow uses JDK 17 and the Gradle wrapper. No Android Studio installation is required for the GitHub Actions build.

The APK is generated under:

```text
app/build/outputs/apk/debug/app-debug.apk
```

## Troubleshooting

### APK shows "Hello Android"

An old placeholder APK was installed. Build and install the latest GitHub Actions artifact.

### `ModuleNotFoundError: No module named 'cv2'`

Activate the Python virtual environment and install the requirements:

```cmd
venv\Scripts\activate
pip install -r requirements.txt
```

### `python phone_server.py` returns immediately

The current implementation contains an explicit server entry point and should print its listening address. If it exits, read the displayed Python exception rather than assuming port 5000 is unavailable.

### Android keeps connecting

First test:

```cmd
curl http://127.0.0.1:5000
```

Then test:

```cmd
curl http://YOUR_LAPTOP_IPV4:5000
```

If local access works but the phone cannot connect, check the laptop IP, network path and Windows Firewall.

### Connection refused

Make sure `phone_server.py` is running and that the Android URL uses the laptop's current IPv4 address and port `5000`.

## CodeAlpha Internship

- **Organization:** CodeAlpha
- **Program:** Artificial Intelligence Internship
- **Task:** Task 4 — Object Detection and Tracking
- **Project:** CodeAlpha Object Detection & Tracking
- **Domain:** Computer Vision / Artificial Intelligence

## Privacy

Camera frames are intended to be processed locally between the Android phone and the user's laptop. This project does not implement face recognition or biometric identification. Video output is saved only when the user requests it with the appropriate option.

## Future Improvements

- WebSocket-based streaming for more efficient transport
- Adaptive bitrate and frame-rate control
- Remote processed-frame preview on the Android client
- GPU acceleration options and performance profiling
- Improved reconnection handling
- Optional secure local-network transport

## License

This project is licensed under the MIT License. See `LICENSE` for details.
