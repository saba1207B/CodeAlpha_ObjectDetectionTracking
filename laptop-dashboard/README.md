# CodeAlpha Object Detection & Tracking — Laptop Web Dashboard

The **Laptop Web Dashboard** provides a responsive, low-latency, browser-based command center for monitoring real-time camera streams from the **CodeAlpha Android APK**, viewing **YOLOv11/v8 deep neural network detections**, controlling **ByteTrack / BoT-SORT persistent tracking**, inspecting motion trajectory trails, and exporting snapshots and recordings.

---

## Key Features

1. **Dashboard Home Page & Connectivity Status:**
   - Real-time connection indicators: `Connected`, `Waiting`, `Disconnected`, `Error`.
   - Automatic local IPv4 address detection displaying the exact APK URL: `http://<LAPTOP_IP>:5000` with 1-click clipboard copy.
   - Real-time Heads-Up Display (HUD) metrics: **Stream FPS**, **Latency (ms)**, **Resolution**, **Total Ingested Frames**, **Active Tracked Objects**, and **Cumulative Unique IDs**.

2. **Responsive Live Camera & Detection View:**
   - **YOLO Tracking View:** Full processed canvas with bounding boxes, class labels, confidence scores, track ID badges (`#1`, `#2`), and fading trajectory trails.
   - **Split Side-by-Side View:** Synchronized view showing the raw CameraX frame on the left and the processed MOT frame on the right.
   - **Raw Phone Camera View:** Direct unprocessed camera stream.
   - Animated radar waiting screen when awaiting connection from the Android phone.

3. **Real-Time Detection & Tracking Controls:**
   - **Model Selector:** Seamlessly switch between `yolo11n.pt` (default), `yolo11s.pt`, `yolov8n.pt`, and `yolov8s.pt`.
   - **Tracker Algorithm:** Switch between **ByteTrack** (fastest, low-confidence recovery) and **BoT-SORT** (camera motion compensation).
   - **Confidence Threshold Slider:** Dynamically filter low-confidence predictions (0.05 to 0.95).
   - **IoU Association Slider:** Adjust Non-Maximum Suppression (NMS) overlap matching (0.10 to 0.90).
   - **Class Filtering:** Filter specific COCO classes (Person, Car, Bicycle, Dog, Cat, Cell phone, Laptop, Bottle, Backpack, etc.) or track all classes.
   - **Motion Trajectory Trails Toggle:** Enable or disable fading movement history lines.
   - **In-Frame HUD Toggle:** Enable or disable in-frame diagnostic stats.
   - **Action Controls:** `Start Detection`, `Stop Detection`, `Reset Results`.

4. **Tracked Objects Analytics & Table:**
   - Live table of currently visible objects with tracking ID, class name, confidence %, centroid coordinates (X, Y), and last-seen timestamp.

5. **Export & Media Actions:**
   - **Capture Snapshot:** Instantly saves high-resolution frame to `screenshots/` and triggers browser download.
   - **Video Recording:** Records annotated output video directly to `output/`.

---

## Running the Dashboard

The dashboard is served directly by the unified Python backend on port 5000:

```bash
# 1. Start the Python server on your laptop
python phone_server.py
# or
python backend/server.py

# 2. Open in any browser on your laptop
http://localhost:5000/
# or
http://<YOUR_LAPTOP_IP>:5000/
```

Alternatively, open `laptop-dashboard/index.html` directly in Chrome, Edge, or Firefox.
