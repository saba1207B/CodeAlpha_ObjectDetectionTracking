# CodeAlpha Object Detection & Tracking 👁️

> A real-time computer vision application for detecting and tracking multiple objects using YOLO and ByteTrack.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/AI-Ultralytics%20YOLO-orange.svg)](https://www.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-green.svg)](https://opencv.org/)
[![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack-purple.svg)](https://github.com/ifzhang/ByteTrack)
[![CodeAlpha](https://img.shields.io/badge/CodeAlpha-AI%20Internship-black.svg)](https://www.codealpha.tech/)

## 📱 Overview

**CodeAlpha Object Detection & Tracking** is a computer vision project that detects objects in real time and assigns persistent tracking IDs as they move between video frames.

The project uses **Ultralytics YOLO** for object detection and **ByteTrack** for multi-object tracking, with OpenCV providing video processing and visualisation.

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
- Persistent tracking IDs across video frames.
- Optional BoT-SORT tracking.
- Motion trajectory trails.
- Active and unique object statistics.

### 🎥 Video Processing

- Webcam input.
- Video-file input.
- Annotated video export.
- Screenshot capture.
- Headless batch processing.
- FPS and tracking statistics HUD.

## 🧠 How It Works

```text
Webcam / Video
      ↓
OpenCV Frame Processing
      ↓
YOLO Object Detection
      ↓
Bounding Boxes + Classes + Confidence
      ↓
ByteTrack / BoT-SORT
      ↓
Persistent Tracking IDs
      ↓
Annotations + Trails + Statistics
```

YOLO identifies objects in each frame, while the tracking stage associates detections across consecutive frames so that moving objects can retain their identities.

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Object Detection | Ultralytics YOLO |
| Computer Vision | OpenCV |
| Object Tracking | ByteTrack / BoT-SORT |
| Numerical Processing | NumPy |
| Deep Learning | PyTorch |
| Android Module | Kotlin + Jetpack Compose |
| Build Automation | GitHub Actions |

## 🏗️ Project Structure

```text
CodeAlpha_ObjectDetectionTracking/
├── app/                              # Android application module
├── CodeAlpha_ObjectDetectionTracking/
│   ├── app.py                        # Main Python application
│   ├── requirements.txt              # Python dependencies
│   ├── src/
│   │   ├── detector.py               # YOLO detection
│   │   └── tracker.py                # Tracking and rendering
│   ├── sample/                        # Sample-video instructions
│   ├── screenshots/                  # Screenshot resources
│   └── output/                       # Output-video resources
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
- Webcam for live-camera detection, if desired.

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

### Webcam

```bash
python app.py --source 0
```

### Video File

```bash
python app.py --source sample/traffic.mp4
```

### Save Annotated Video

```bash
python app.py --source sample/traffic.mp4 --save output/tracked_traffic.mp4
```

## ⌨️ Controls

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `P` / `SPACE` | Pause / resume |
| `T` | Toggle tracking trails |
| `H` | Toggle HUD |
| `S` | Save screenshot |

## 📱 Android APK

The repository also contains an Android application module and a GitHub Actions workflow for building a debug APK.

The APK workflow uses JDK 17 and Gradle 9.3.1 and creates an isolated CI debug keystore for the build.

Artifact name:

```text
CodeAlpha_ObjectDetectionTracking-debug-apk
```

> **Note:** The Python YOLO/ByteTrack implementation is the main CodeAlpha Task 4 implementation. The Android module currently serves as a companion application and does not directly execute the Python detection/tracking pipeline.

## 📸 Screenshots

Recommended project screenshots:

```text
screenshots/
├── object-detection.png
├── multi-object-tracking.png
├── tracking-ids.png
├── statistics-hud.png
└── video-output.png
```

## 🎥 Project Demonstration

A short demonstration can show:

1. The CodeAlpha internship project introduction.
2. Live object detection from a webcam or video.
3. Multiple detected objects.
4. Persistent tracking IDs while objects move.
5. Tracking trails and performance statistics.
6. The YOLO + ByteTrack pipeline.
7. The GitHub repository and project structure.

## 🎯 Internship Project

**Program:** CodeAlpha Artificial Intelligence Internship  
**Task:** Task 4 — Object Detection and Tracking  
**Project Type:** AI / Computer Vision  
**Primary Technologies:** YOLO + ByteTrack + OpenCV

The project is intended to demonstrate practical computer vision and AI implementation as part of internship evaluation and portfolio presentation.

## ⚠️ Limitations

- Detection and tracking performance depends on hardware and model size.
- CPU-only systems may provide lower real-time FPS.
- Tracking IDs can change after prolonged disappearance or difficult occlusion.
- Android on-device YOLO inference is not yet connected to the Python pipeline.

## 🔮 Future Improvements

- On-device YOLO inference.
- CameraX integration for Android.
- GPU/NPU acceleration.
- Object counting and analytics.
- Zone-based detection alerts.
- Mobile video export.

## 👤 Author

**Sabareesh**

GitHub: [@saba1207B](https://github.com/saba1207B)

## 📄 License

MIT License

---

⭐ If you find this project useful, consider starring the repository.