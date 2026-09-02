# CodeAlpha Object Detection & Tracking 👁️

> A real-time computer vision application for detecting and tracking multiple objects using Ultralytics YOLO, ByteTrack / BoT-SORT, OpenCV, and an Android phone camera client.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/AI-Ultralytics%20YOLO-orange.svg)](https://www.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-green.svg)](https://opencv.org/)
[![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack-purple.svg)](https://github.com/ifzhang/ByteTrack)
[![Android](https://img.shields.io/badge/Android-Kotlin%20%2B%20CameraX-brightgreen.svg)](https://developer.android.com/)
[![CodeAlpha](https://img.shields.io/badge/CodeAlpha-AI%20Internship-black.svg)](https://www.codealpha.tech/)

## 📱 Overview

**CodeAlpha Object Detection & Tracking** is a computer vision project that detects objects in real time and assigns persistent tracking IDs as they move between video frames.

The project uses **Ultralytics YOLO** for object detection, **ByteTrack / BoT-SORT** for multi-object tracking, and OpenCV for video processing and visualisation.

The Android companion application turns an Android phone into a network camera. Camera frames are sent over the same local Wi-Fi network to the Python application running on a laptop, where the complete AI pipeline performs detection and tracking.

This project was developed as part of the **CodeAlpha Artificial Intelligence Internship — Task 4: Object Detection and Tracking**.

## ✨ Features

### 🎯 Object Detection

- Real-time object detection with YOLO.
- Bounding boxes with class labels.
- Detection confidence scores.
- Configurable confidence and IoU thresholds.
- Optional class filtering.

### 👁️ Object Tracking

- Multi-object tracking with ByteTrack.
- Optional BoT-SORT tracking.
- Persistent tracking IDs across video frames.
- Motion trajectory trails.
- Active and unique object statistics.

### 🎥 Video Processing

- Laptop webcam input.
- Video-file input.
- Android phone camera input over local Wi-Fi.
- Annotated video export.
- Screenshot capture.
- Headless processing.
- FPS and tracking statistics HUD.

### 📱 Android Camera Client

- CameraX-based live camera preview.
- Laptop IP and port configuration.
- JPEG frame transmission over local HTTP.
- Start / stop streaming controls.
- Connection status feedback.
- The phone performs camera capture; YOLO and tracking remain on the laptop.

## 🧠 System Architecture

```text
                         LOCAL WI-FI
┌──────────────────┐                         ┌─────────────────────────┐
│  Android Phone   │                         │       Laptop            │
│                  │     JPEG / HTTP         │                         │
│  CameraX Camera  │ ──────────────────────> │  Python Receiver        │
│       │          │                         │        │                │
│       ▼          │                         │        ▼                │
│  Camera Preview  │                         │      OpenCV             │
└──────────────────┘                         │        │                │
                                             │        ▼                │
                                             │       YOLO              │
                                             │        │                │
                                             │        ▼                │
                                             │ ByteTrack / BoT-SORT    │
                                             │        │                │
                                             │        ▼                │
                                             │ Detection + Tracking    │
                                             │        │                │
                                             │        ▼                │
                                             │ Annotated Laptop View   │
                                             └─────────────────────────┘
```

### AI Pipeline

```text
Camera Frame
    ↓
OpenCV Frame Decode
    ↓
YOLO Object Detection
    ↓
Bounding Boxes + Classes + Confidence
    ↓
ByteTrack / BoT-SORT
    ↓
Persistent Tracking IDs
    ↓
Trails + HUD + Statistics
    ↓
Annotated Output
```

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Object Detection | Ultralytics YOLO |
| Computer Vision | OpenCV |
| Object Tracking | ByteTrack / BoT-SORT |
| Numerical Processing | NumPy |
| Deep Learning | PyTorch |
| Android | Kotlin + Jetpack Compose |
| Camera | Android CameraX |
| Network Transport | Local HTTP + JPEG |
| Build Automation | GitHub Actions |

## 🏗️ Project Structure

```text
CodeAlpha_ObjectDetectionTracking/
├── app/                              # Android camera client
│   ├── src/main/AndroidManifest.xml  # Camera + network permissions
│   └── src/main/java/com/example/
│       └── MainActivity.kt           # CameraX + streaming UI
│
├── CodeAlpha_ObjectDetectionTracking/
│   ├── app.py                        # Main AI application
│   ├── phone_server.py               # Android frame receiver
│   ├── requirements.txt              # Python dependencies
│   ├── src/
│   │   ├── detector.py               # YOLO detection
│   │   └── tracker.py                # Tracking and rendering
│   ├── sample/                        # Sample-video resources
│   ├── screenshots/                  # Screenshot resources
│   └── output/                       # Output-video resources
│
├── .github/
│   └── workflows/
│       └── build-apk.yml             # Android APK workflow
├── gradle/
├── build.gradle.kts
├── settings.gradle.kts
├── gradle.properties
├── metadata.json
└── README.md
```

## 🚀 Run the Python Application

### Requirements

- Python 3.9 or later.
- A compatible environment for PyTorch and Ultralytics.
- Laptop connected to the same Wi-Fi network as the Android phone when using phone mode.

### Installation

```bash
cd CodeAlpha_ObjectDetectionTracking
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Laptop Webcam

```bash
python app.py --source 0
```

### Video File

```bash
python app.py --source sample/traffic.mp4
```

### Android Phone Camera

1. Connect the phone and laptop to the **same Wi-Fi network**.
2. On the laptop, start the Python receiver and AI pipeline:

```bash
python app.py --source phone
```

3. The terminal prints the laptop's local IP address, for example:

```text
[PHONE] Receiver listening on http://192.168.1.105:5000
[PHONE] Android endpoint: http://<LAPTOP-IP>:5000/frame
```

4. Install and open the **Object Tracking** Android APK.
5. Enter the printed laptop URL, for example:

```text
http://192.168.1.105:5000
```

6. Tap **Start Streaming**.
7. The phone camera frames are sent to the laptop and processed by **YOLO + ByteTrack**.

> **Important:** The Android app is a camera client, not the AI inference engine. The YOLO model and tracker run in the Python application on the laptop.

### Save Annotated Video

```bash
python app.py --source phone --save output/phone_tracking.mp4
```

## 🔌 Network Notes

- Both devices must be on the same local network.
- The laptop firewall must allow the configured TCP port (default `5000`) for local-network connections.
- The Android client uses HTTP because the receiver is intended for a trusted local network.
- No cloud service is required for the camera-to-laptop connection.
- Frames are processed by the local Python application and are not intentionally uploaded to a remote service.

## ⌨️ Controls

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `P` / `SPACE` | Pause / resume |
| `T` | Toggle tracking trails |
| `H` | Toggle HUD |
| `S` | Save screenshot |

## 📱 Android APK

The Android module is now a functional **phone-camera companion client** rather than a placeholder screen.

The GitHub Actions workflow builds a debug APK using JDK 17 and Gradle 9.3.1.

Artifact name:

```text
CodeAlpha_ObjectDetectionTracking-debug-apk
```

The APK provides:

- CameraX live preview.
- Laptop server URL configuration.
- Start / Stop streaming.
- Connection status.
- JPEG camera-frame transmission over Wi-Fi.

## 📸 Screenshots

Recommended project screenshots:

```text
screenshots/
├── object-detection.png
├── multi-object-tracking.png
├── tracking-ids.png
├── statistics-hud.png
├── android-camera-client.png
└── phone-to-laptop-demo.png
```

## 🎥 Project Demonstration

A strong demonstration can show:

1. The CodeAlpha internship project introduction.
2. The Android camera client opening on the phone.
3. The laptop Python receiver starting.
4. The phone and laptop connected to the same Wi-Fi network.
5. Live camera frames reaching the laptop.
6. YOLO detecting multiple objects.
7. ByteTrack assigning persistent IDs.
8. Motion trails and performance statistics.
9. The GitHub repository and project structure.

## 🎯 Internship Project

**Program:** CodeAlpha Artificial Intelligence Internship  
**Task:** Task 4 — Object Detection and Tracking  
**Project Type:** AI / Computer Vision  
**Primary Technologies:** YOLO + ByteTrack + OpenCV + Android CameraX

The project demonstrates practical computer vision, multi-object tracking, network camera integration, and Android/Python interoperability.

## ⚠️ Limitations

- Detection and tracking performance depends on laptop hardware and model size.
- CPU-only systems may provide lower real-time FPS.
- Wi-Fi quality affects camera-frame latency and throughput.
- Tracking IDs can change after prolonged disappearance or difficult occlusion.
- The current Android client sends compressed JPEG frames rather than using a full RTSP/WebRTC video stream.

## 🔮 Future Improvements

- WebRTC or RTSP low-latency streaming.
- On-device YOLO inference as an optional mode.
- GPU/NPU acceleration.
- Object counting and analytics.
- Zone-based detection alerts.
- Mobile viewing of annotated results.
- Configurable frame rate and JPEG quality.

## 👤 Author

**Sabareesh**

GitHub: [@saba1207B](https://github.com/saba1207B)

## 📄 License

MIT License

---

⭐ If you find this project useful, consider starring the repository.
