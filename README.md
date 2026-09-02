# CodeAlpha Object Detection & Tracking 👁️

**CodeAlpha Artificial Intelligence Internship — Task 4: Object Detection and Tracking**

A computer-vision project built around **Ultralytics YOLO + ByteTrack** for real-time multi-object detection and tracking. The complete CodeAlpha implementation is maintained in `CodeAlpha_ObjectDetectionTracking/`.

## 🎯 Task 4 Implementation

The Python application provides:

- Pretrained YOLO object detection
- Real multi-object tracking with ByteTrack
- Optional BoT-SORT tracking
- Persistent tracking IDs across frames
- Bounding boxes, labels and confidence scores
- Motion trajectory trails
- Webcam and video-file input
- Configurable confidence and IoU thresholds
- Class filtering
- FPS/track statistics HUD
- Screenshot capture
- Annotated video export
- Headless batch processing
- Graceful resource cleanup

## 🧠 Pipeline

```text
Webcam / Video
      ↓
 OpenCV Frames
      ↓
 YOLO Detection
      ↓
Bounding Boxes + Classes + Confidence
      ↓
ByteTrack / BoT-SORT
      ↓
Persistent Track IDs
      ↓
Annotations + Trails + Statistics
```

The tracker uses the Ultralytics tracking interface with `persist=True`, so IDs are produced by a real multi-object tracking algorithm rather than by a simple counter or hardcoded logic.

## 🛠️ Technology Stack

- Python 3.9+
- Ultralytics YOLO
- OpenCV
- NumPy
- ByteTrack / BoT-SORT
- Kotlin + Jetpack Compose for the Android module

## 📁 Repository Structure

```text
CodeAlpha_ObjectDetectionTracking/
├── app/                              # Android application module
├── CodeAlpha_ObjectDetectionTracking/
│   ├── app.py                        # Python CLI application
│   ├── requirements.txt              # Python dependencies
│   ├── README.md                     # Detailed Python documentation
│   ├── src/
│   │   ├── __init__.py
│   │   ├── detector.py               # YOLO detection wrapper
│   │   └── tracker.py                # Tracking and rendering
│   ├── sample/                       # Sample-video instructions
│   ├── screenshots/                  # Screenshot instructions
│   └── output/                       # Output-video instructions
├── .github/workflows/build-apk.yml   # Android APK CI build
├── gradle/
├── build.gradle.kts
├── settings.gradle.kts
├── gradle.properties
├── metadata.json
└── README.md
```

## 🚀 Run the Python Application

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

### Video file

```bash
python app.py --source sample/traffic.mp4
```

### Save annotated video

```bash
python app.py --source sample/traffic.mp4 --save output/tracked_traffic.mp4
```

### Custom confidence

```bash
python app.py --source 0 --conf 0.50
```

### Selected classes

```bash
python app.py --source 0 --classes 0 2
```

### BoT-SORT

```bash
python app.py --source sample/traffic.mp4 --tracker botsort.yaml
```

## ⌨️ Controls

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `P` / `SPACE` | Pause/resume |
| `T` | Toggle trails |
| `H` | Toggle HUD |
| `S` | Save screenshot |

## 📱 Android APK

The repository also contains an Android application module. GitHub Actions builds a **debug APK** automatically on every push to `main` and through manual workflow dispatch.

The workflow uses JDK 17 and Gradle 9.3.1 and generates an isolated CI debug keystore, so no private signing key is committed to the repository.

Artifact name:

```text
CodeAlpha_ObjectDetectionTracking-debug-apk
```

APK output inside the runner:

```text
app/build/outputs/apk/debug/app-debug.apk
```

> **Important:** The Python YOLO/ByteTrack implementation is the CodeAlpha Task 4 implementation. The Android module is currently a companion application and does not execute the Python YOLO/ByteTrack pipeline directly. Future work can add on-device YOLO inference and CameraX tracking.

## 📸 Submission Screenshots

Recommended captures:

1. Person detection with a persistent tracking ID
2. Multiple objects with different IDs
3. Vehicle detection/tracking with confidence scores
4. HUD showing FPS and active tracks
5. Terminal session summary

## 🎥 LinkedIn Demo

A 60–90 second demonstration should show:

1. Task 4 introduction
2. Live webcam detection
3. Persistent object IDs while objects move
4. Multiple-object tracking
5. YOLO + ByteTrack explanation
6. GitHub architecture
7. Final repository link

## ⚠️ Limitations

- Performance depends on hardware, resolution and model size.
- CPU-only systems may have lower FPS.
- IDs can change after prolonged disappearance or difficult occlusion.
- Android on-device YOLO/tracking is not yet connected to the Python pipeline.

## 🔮 Future Improvements

- On-device YOLO inference
- CameraX integration
- GPU/NPU acceleration
- Object counting and analytics
- Zone-based alerts
- Mobile video export

## 👤 Author

**Sabareesh**

CodeAlpha Artificial Intelligence Internship — Task 4

## 📄 License

MIT License
