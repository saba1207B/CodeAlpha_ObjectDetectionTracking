# CodeAlpha_ObjectDetectionTracking: Real-Time Multi-Object Detection & Tracking

![Python Version](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue?logo=python)
![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8%20%2F%20v11-00FFFF?logo=ultralytics)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red?logo=opencv)
![Tracking](https://img.shields.io/badge/Tracker-ByteTrack%20%7C%20BoT--SORT-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 1. Project Title
**CodeAlpha_ObjectDetectionTracking** — Real-Time Deep Learning Multi-Object Detection and Visual Trajectory Tracking.

---

## 2. CodeAlpha Internship Task
- **Organization:** CodeAlpha
- **Domain:** Artificial Intelligence Internship
- **Task Number:** Task 4
- **Task Title:** Object Detection and Tracking

---

## 3. Project Overview
This project is an end-to-end, production-ready computer vision application designed to perform high-accuracy **Multi-Object Detection and Multi-Object Tracking (MOT)** in real-time. 

Using pretrained **Ultralytics YOLO (You Only Look Once)** deep neural networks coupled with the **ByteTrack** and **BoT-SORT** association algorithms, the system detects multiple object classes (such as people, cars, motorcycles, dogs, backpacks), assigns persistent identification numbers (Track IDs), tracks objects through occlusions, and renders visual trajectory trails across sequential video frames.

---

## ⚠️ Important Disclaimer & Pre-Flight Notes
Before launching the application, please review the following essential notes:

1. **Hardware & Real-Time Performance:**
   - The default model (`yolo11n.pt`) is optimized for standard laptop/desktop CPUs, achieving approximately **25–45 FPS**.
   - If an NVIDIA GPU with CUDA is detected, PyTorch will automatically accelerate inference up to **60–120+ FPS**.
2. **Webcam & Lighting Recommendations:**
   - Ensure adequate ambient lighting and avoid intense backlight or direct lens glare for optimal tracking confidence.
   - Close other applications (Zoom, Teams, Discord, browser camera tabs) that might lock your webcam device before running the script.
3. **100% Local Processing & Privacy Guarantee:**
   - All computer vision, inference, and video rendering occur **strictly locally** on your device.
   - No video frames, biometric data, or telemetry are transmitted to any external server or cloud API.
4. **First-Time Model Weights Download:**
   - On the very first run, PyTorch/Ultralytics will automatically download the lightweight pretrained weights (`yolo11n.pt`, ~6 MB) from official releases. No API keys or manual setup required.

---

## 4. Key Features
- 🎥 **Dual Input Flexibility:** Real-time webcam streaming (built-in/USB) and pre-recorded video file processing.
- 🧠 **Pretrained Deep Neural Networks:** Native support for Ultralytics YOLOv11 and YOLOv8 models (`yolo11n.pt`, `yolov8n.pt`, `yolo11s.pt`, etc.).
- 🎯 **Multi-Class Detection:** Detects up to 80 standard COCO classes with configurable class filtering.
- 🆔 **True Computer Vision Tracking:** Real ByteTrack/BoT-SORT tracking with continuous ID maintenance across frames.
- 🌈 **Motion Trajectory Trails:** Smooth, fading history lines tracing the exact movement path of each tracked object.
- 📊 **Dynamic Heads-Up Display (HUD):** Real-time rolling FPS counter, active track counter, cumulative unique object statistics, and system statuses.
- 🎮 **Interactive Runtime Controls:** Pause/resume playback, toggle motion trails, toggle HUD, take high-resolution screenshots, and quit cleanly.
- 💾 **Output Video Export:** Optional export of fully rendered, annotated videos in MP4/AVI formats.
- 🛡️ **Fault Tolerance & Safety:** Graceful error handling for disconnected webcams, missing files, and clean OpenCV window destruction.

---

## 5. How Object Detection Works
Object detection is a computer vision task that combines **image classification** (what is in the image) and **object localization** (where is it located):
1. **Feature Extraction:** Deep convolutional neural network (CNN) backbones and attention modules extract multi-scale spatial features from the input frame.
2. **Bounding Box Regression:** The network predicts continuous bounding box coordinates $(x_1, y_1, x_2, y_2)$ or $(c_x, c_y, w, h)$ indicating object boundaries.
3. **Class Confidence Scoring:** The network calculates probability distributions across target classes for each candidate box.
4. **Non-Maximum Suppression (NMS):** Eliminates redundant, overlapping candidate boxes by comparing their Intersection over Union (IoU) scores, keeping only the highest-scoring bounding boxes.

---

## 6. How Object Tracking Works
Object tracking extends detection across the time domain (video frames). Unlike detection which operates independently on each frame, tracking solves the **data association problem**:
1. **State Prediction:** Uses a mathematical motion model (e.g., Kalman Filter) to predict where an existing object will move in the next frame based on velocity and position.
2. **Feature Similarity / Spatial Overlap:** Measures the similarity between predicted bounding boxes and newly detected bounding boxes using spatial IoU distance or visual re-identification (ReID) embeddings.
3. **Hungarian Algorithm / Linear Assignment:** Optimally matches existing track IDs with new detections to ensure persistent identity.
4. **Track Lifecycle Management:** Initializes new tracks for entering objects and terminates tracks for objects that have left the scene or remained occluded past a threshold.

```
[ Video Frame ] ──> [ YOLO Detection ] ──> [ Kalman Filter Prediction ]
                            │                           │
                            └───────> [ IoU Association / Hungarian Matching ]
                                                    │
                                                    ▼
                                    [ Assigned Persistent Track IDs ]
                                                    │
                                                    ▼
                                    [ Motion Trail & HUD Rendering ]
```

---

## 7. YOLO (You Only Look Once) Explanation
YOLO is a single-stage object detection architecture introduced to achieve real-time inference speeds without sacrificing accuracy:
- **Single Forward Pass:** Unlike two-stage detectors (like Faster R-CNN) that generate region proposals first, YOLO processes the entire image in a single forward pass through the neural network.
- **Anchor-Free Design (YOLOv8 / YOLO11):** Direct prediction of bounding box center offsets and dimensions reduces hyperparameters and boosts generalization.
- **C3k2 & SPPF Blocks:** Incorporates optimized cross-stage partial bottleneck networks and Spatial Pyramid Pooling Fast layers for multi-scale context aggregation.
- **Lightweight Variants:** Models like `yolo11n.pt` (Nano) and `yolov8n.pt` contain fewer than 3 million parameters, allowing real-time processing (30–60+ FPS) on standard consumer laptops and CPUs.

---

## 8. Tracking Algorithm Explanation (ByteTrack & BoT-SORT)
This project leverages **ByteTrack**, a tracking-by-detection algorithm:
- **The Low-Confidence Dilemma:** Traditional trackers discard low-confidence detections ($< 0.5$) to filter noise, which causes ID switching when objects become blurred or partially occluded.
- **ByteTrack Dual Association:**
  - *First Association:* Matches high-confidence detections ($> \text{conf}$) with existing tracklets using Kalman-predicted IoU.
  - *Second Association:* Matches remaining unmatched tracks with **low-confidence** detections, recovering occluded or blurred objects without introducing false positives.
- **BoT-SORT Alternative:** Combines ByteTrack's association strategy with camera motion compensation (CMC) and deep appearance Re-ID features for challenging surveillance scenes.

---

## 9. Technology Stack
- **Language:** Python 3.8+
- **Deep Learning Framework:** PyTorch & Torchvision
- **Object Detection Engine:** Ultralytics YOLO (v8 / 11)
- **Computer Vision & GUI:** OpenCV (`cv2`)
- **Linear Algebra & Image Operations:** NumPy
- **Tracker Association:** Lapx (Linear Assignment Problem Solver), FilterPy (Kalman Filtering)

---

## 10. Project Structure
```
CodeAlpha_ObjectDetectionTracking/
├── app.py                     # Main CLI application runner & event loop
├── requirements.txt           # Python dependencies and versions
├── README.md                  # Comprehensive documentation and guide
├── .gitignore                 # Excludes weights, temp files, and media
├── src/
│   ├── __init__.py            # Package initialization & exports
│   ├── detector.py            # YOLODetector class (inference & bounding boxes)
│   └── tracker.py             # ObjectTracker class (ByteTrack, trails, HUD)
├── sample/
│   └── README.md              # Instructions for test video samples
├── screenshots/
│   └── README.md              # Captured application screenshot showcase
└── output/
    └── README.md              # Saved output video destination
```

---

## 11. Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/CodeAlpha_ObjectDetectionTracking.git
cd CodeAlpha_ObjectDetectionTracking
```

### Step 2: Create and Activate a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (Command Prompt):
python -m venv venv
venv\Scripts\activate

# On Windows (PowerShell):
python -m venv venv
venv\Scripts\Activate.ps1
```

### Step 3: Upgrade PIP and Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Note: Pretrained weights like `yolo11n.pt` will automatically download from official Ultralytics releases on first launch.)*

---

## 12. Requirements
The system requires standard Python packages specified in `requirements.txt`:
- `ultralytics>=8.3.0`
- `opencv-python>=4.8.0`
- `numpy>=1.24.0,<2.0.0`
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `lapx>=0.5.5`
- `filterpy>=1.4.5`

Hardware recommendations:
- **CPU:** Intel Core i3/i5/i7 (8th Gen+) or AMD Ryzen (Runs at 25–45 FPS with Nano model)
- **GPU (Optional):** NVIDIA GPU with CUDA 11.8+ for 100+ FPS processing
- **RAM:** Minimum 4 GB (8 GB recommended)

---

## 13. How to Run Webcam Detection
To start real-time tracking with your default webcam (Camera Index 0):
```bash
python app.py --source 0
```
For external USB webcams:
```bash
python app.py --source 1
```

---

## 14. How to Run Video Detection
To run object tracking on a pre-recorded video file:
```bash
python app.py --source sample/video.mp4
```

To save the processed video result with bounding boxes and trails:
```bash
python app.py --source sample/video.mp4 --save output/tracked_traffic.mp4
```

---

## 15. Command-Line Options
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source` | `str` | `0` | Camera index (`0`) or path to video file (`sample/video.mp4`). |
| `--model` | `str` | `yolo11n.pt` | YOLO model weights (`yolo11n.pt`, `yolov8n.pt`, `yolo11s.pt`). |
| `--conf` | `float` | `0.35` | Confidence threshold for object detection (0.01 to 1.0). |
| `--iou` | `float` | `0.45` | IoU threshold for Non-Maximum Suppression and tracking. |
| `--tracker` | `str` | `bytetrack.yaml` | Tracker algorithm: `bytetrack.yaml` or `botsort.yaml`. |
| `--classes` | `int ...` | `None` | Filter specific classes (e.g. `--classes 0` for persons, `0 2 3` for people/cars/motorcycles). |
| `--save` | `str` | `None` | Path to export annotated output video (e.g., `output/result.mp4`). |
| `--no-trails` | `flag` | `False` | Disable rendering of movement trajectory trails. |
| `--no-hud` | `flag` | `False` | Hide top and bottom HUD statistics overlay. |
| `--no-display`| `flag` | `False` | Headless execution without opening an OpenCV window. |
| `--device` | `str` | `None` | Force device compute: `'cpu'`, `'cuda'`, or `'mps'`. |

---

## 16. Example Output
During live execution, bounding boxes and labels appear formatted with high readability:
```
person | ID: 3 | 0.91
car    | ID: 7 | 0.88
dog    | ID: 1 | 0.79
```

Terminal statistics summary upon exiting:
```text
======================================================================
                  SESSION SUMMARY & STATISTICS
======================================================================
Total Frames Processed : 420
Total Elapsed Time     : 14.12 seconds
Average Processing FPS : 29.74 FPS
Total Unique Track IDs : 12
Class Breakdown:
  - person: 4 active
  - car: 2 active
======================================================================
[INFO] Cleanup complete. Project closed safely.
```

---

## 17. Screenshots Section
*(Captured frames saved via the `s` key during execution)*

| Live Webcam Tracking | Multi-Class Vehicle Tracking |
| :---: | :---: |
| ![Webcam Demo](screenshots/README.md) <br> *Real-time Person Tracking with ID and Confidence* | ![Traffic Demo](screenshots/README.md) <br> *Urban Traffic Tracking with Trajectory Trails* |

---

## 18. Performance Notes
- **Nano Model Efficiency:** `yolo11n.pt` and `yolov8n.pt` provide the optimal balance for consumer hardware, running at ~30 FPS on standard CPUs.
- **Hardware Acceleration:** PyTorch automatically leverages CUDA on NVIDIA systems or Apple MPS on macOS devices when available.
- **Memory Optimization:** Bounded deques prevent memory leaks during long-running 24/7 video streams.

---

## 19. Limitations
- Extreme low-light conditions may degrade detection confidence.
- Total, prolonged visual occlusions (e.g., an object hidden behind a large wall for >30 frames) will reset the track ID once the object reappears.
- Extremely rapid camera panning can increase tracker association ambiguity unless BoT-SORT camera motion compensation is active.

---

## 20. Future Improvements
- [ ] Integration of deep visual appearance embeddings (ReID) for cross-camera tracking.
- [ ] Speed estimation and virtual tripwire line crossing counters.
- [ ] Web dashboard deployment using Streamlit or FastAPI + WebRTC.
- [ ] Automated email/SMS alert triggering when restricted objects enter designated zones.

---

## 21. Author & Acknowledgements
- **Intern:** CodeAlpha Artificial Intelligence Intern
- **Program:** CodeAlpha AI Internship Program
- **Task:** Task 4 — Object Detection and Tracking
- **Repository:** `CodeAlpha_ObjectDetectionTracking`
- **Frameworks:** Ultralytics YOLO, OpenCV Community, PyTorch
