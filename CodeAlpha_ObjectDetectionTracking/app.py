#!/usr/bin/env python3
"""
CodeAlpha AI Internship - Task 4: Object Detection & Tracking System
====================================================================
Main application pipeline integrating Ultralytics YOLOv11/v8 with
ByteTrack / BoT-SORT multi-object tracking.

Supports three flexible input video sources:
    1. Phone Camera (Android Camera Client over HTTP via phone_server.py)
    2. Local Laptop / USB Webcam (e.g. index 0, 1)
    3. Pre-recorded Video Files (.mp4, .avi, .mkv, .mov)

Usage Examples:
    # 1. Android Phone Camera Tracking (Preferred Task 4 Architecture):
    python app.py --source phone

    # 2. Local Laptop Webcam Tracking:
    python app.py --source 0

    # 3. Pre-recorded Video File Tracking:
    python app.py --source sample/video.mp4

    # 4. Custom Model, Confidence & Tracker (BoT-SORT):
    python app.py --source phone --model yolo11n.pt --conf 0.4 --tracker botsort.yaml

    # 5. Save Tracked Output Video:
    python app.py --source phone --save output/tracked_phone.mp4
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional, List, Tuple, Any


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments. Works immediately for --help."""
    parser = argparse.ArgumentParser(
        description="CodeAlpha AI Internship - Object Detection and Tracking with YOLO & ByteTrack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input video source: 'phone' (Android camera client), webcam index ('0', '1'), or video file path ('sample/video.mp4').",
    )
    parser.add_argument(
        "--phone-server",
        type=str,
        default="http://127.0.0.1:5000",
        help="URL of the running phone_server.py instance when using --source phone.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained YOLO model checkpoint ('yolo11n.pt', 'yolov8n.pt', 'yolo11s.pt', etc.).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Minimum confidence threshold for object detection (0.01 - 1.0).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Intersection-over-Union (IoU) threshold for Non-Maximum Suppression (NMS).",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Multi-object tracking algorithm configuration.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=None,
        help="Filter specific COCO class IDs (e.g. --classes 0 for person, 2 for car).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional file path to save the processed video output (e.g. 'output/result.mp4').",
    )
    parser.add_argument(
        "--show-trails",
        action="store_true",
        default=True,
        help="Draw motion trajectory lines showing object motion history.",
    )
    parser.add_argument(
        "--no-trails",
        dest="show_trails",
        action="store_false",
        help="Disable motion trajectory lines.",
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Hide the Heads-Up Display (HUD) statistics banner.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run in headless mode without opening an OpenCV GUI window.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cpu', 'cuda', 'mps', or auto-detect).",
    )

    return parser.parse_args()


# Validate runtime dependencies when execution starts
def ensure_dependencies():
    """Verify that all core libraries are available with helpful instructions if missing."""
    missing = []
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import requests
    except ImportError:
        missing.append("requests")

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import ultralytics
    except ImportError:
        missing.append("ultralytics")

    if missing:
        print("\n" + "=" * 70)
        print("[!] ERROR: Required dependencies are not installed in this environment:")
        for pkg in missing:
            print(f"    - {pkg}")
        print("=" * 70)
        print("Please activate your Python virtual environment and run:")
        print()
        print("    pip install -r requirements.txt")
        print()
        print("For Windows venv setup:")
        print("    python -m venv venv")
        print("    venv\\Scripts\\activate")
        print("    pip install -r requirements.txt")
        print("=" * 70 + "\n")
        sys.exit(1)


class PhoneCameraStream:
    """
    Client for receiving live video frames from phone_server.py.
    Provides a cv2.VideoCapture-like interface for the tracking pipeline.
    """

    def __init__(self, server_url: str = "http://127.0.0.1:5000", timeout: float = 1.5):
        import requests
        self.requests = requests
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._is_opened = True
        self._last_frame: Optional[Any] = None
        self._last_timestamp: float = 0.0
        self._waiting_logged: bool = False
        self._disconnected_logged: bool = False

        self._verify_server_connection()

    def _verify_server_connection(self) -> None:
        """Verify that phone_server.py is running and accessible."""
        try:
            resp = self.requests.get(f"{self.server_url}/", timeout=2.0)
            if resp.status_code != 200:
                print(f"[WARNING] Server at {self.server_url} returned HTTP {resp.status_code}")
        except self.requests.exceptions.ConnectionError:
            print("\n" + "=" * 70)
            print("[!] ERROR: Phone server is not reachable/running.")
            print("=" * 70)
            print(f"    Target Server URL: {self.server_url}")
            print("    Please start the phone server in another terminal first:")
            print()
            print("        python phone_server.py")
            print()
            print("    Once the server is running, re-run this command:")
            print("        python app.py --source phone")
            print("=" * 70 + "\n")
            sys.exit(1)
        except Exception as e:
            print(f"[WARNING] Could not check phone server status: {e}")

    def isOpened(self) -> bool:
        return self._is_opened

    def read(self) -> Tuple[bool, Optional[Any]]:
        """
        Fetch the most recent camera frame from the phone server.
        Implements 'latest frame wins' policy to eliminate queue latency.
        """
        import cv2
        import numpy as np

        if not self._is_opened:
            return False, None

        endpoint = f"{self.server_url}/latest_frame"

        try:
            resp = self.requests.get(endpoint, timeout=self.timeout)
            if resp.status_code == 200 and resp.content:
                np_arr = np.frombuffer(resp.content, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None and frame.size > 0:
                    self._last_frame = frame
                    self._last_timestamp = time.time()
                    self._waiting_logged = False
                    self._disconnected_logged = False
                    return True, frame

            elif resp.status_code == 503:
                # Server is alive, but no phone frame has arrived yet
                if not self._waiting_logged:
                    print("\n[i] Waiting for phone camera frames...")
                    print("    1. Open the CodeAlpha Camera app on your Android phone.")
                    print("    2. Enter your laptop's IP address (e.g. http://<LAPTOP_IP>:5000).")
                    print("    3. Tap 'Start Camera'.\n")
                    self._waiting_logged = True

                waiting_frame = self._create_placeholder_canvas(
                    "WAITING FOR PHONE CAMERA FRAMES...",
                    "Connect Android app to: http://<LAPTOP_IP>:5000 and tap 'Start Camera'",
                    Color=(0, 165, 255)
                )
                time.sleep(0.08)
                return True, waiting_frame

        except self.requests.exceptions.RequestException:
            now = time.time()
            if self._last_frame is not None and (now - self._last_timestamp > 3.0):
                if not self._disconnected_logged:
                    print("[!] Phone camera disconnected / waiting for frames...")
                    self._disconnected_logged = True

                disconnected_frame = self._create_placeholder_canvas(
                    "PHONE CAMERA DISCONNECTED",
                    "Waiting for camera frames to resume...",
                    Color=(0, 0, 255)
                )
                time.sleep(0.1)
                return True, disconnected_frame

        time.sleep(0.05)
        if self._last_frame is not None:
            return True, self._last_frame.copy()

        return True, self._create_placeholder_canvas("CONNECTING TO PHONE SERVER...", "http://127.0.0.1:5000")

    def _create_placeholder_canvas(self, title: str, subtitle: str, Color=(0, 200, 255)) -> Any:
        """Render an informative splash canvas when waiting for frames."""
        import cv2
        import numpy as np

        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        canvas[:] = (28, 28, 30)

        cv2.putText(canvas, "CodeAlpha AI - Object Detection & Tracking", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (20, 100), (620, 380), (50, 50, 50), -1)
        cv2.rectangle(canvas, (20, 100), (620, 380), Color, 2)

        cv2.putText(canvas, title, (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, Color, 2, cv2.LINE_AA)
        cv2.putText(canvas, subtitle, (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Press 'Q' or 'ESC' in this window to quit", (50, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

        return canvas

    def release(self) -> None:
        self._is_opened = False

    def get(self, prop_id: int) -> float:
        import cv2
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._last_frame.shape[1]) if self._last_frame is not None else 640.0
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._last_frame.shape[0]) if self._last_frame is not None else 480.0
        elif prop_id == cv2.CAP_PROP_FPS:
            return 15.0
        return 0.0


def open_video_source(source_arg: str, phone_server_url: str = "http://127.0.0.1:5000") -> Tuple[object, bool, str]:
    """Open the video capture source gracefully, supporting phone camera, webcam, or video files."""
    import cv2
    clean_src = source_arg.strip().lower()

    if clean_src == "phone":
        print(f"[INFO] Connecting to Android Phone Camera via {phone_server_url}...")
        cap = PhoneCameraStream(server_url=phone_server_url)
        return cap, True, "Android Phone Camera (HTTP)"

    if source_arg.isdigit():
        camera_idx = int(source_arg)
        source_name = f"Webcam (Device {camera_idx})"
        print(f"[INFO] Connecting to {source_name}...")
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            print(f"\n[ERROR] Failed to open webcam device: '{source_arg}'")
            print("[TROUBLESHOOTING]")
            print("  1. Ensure your webcam is connected and not locked by another app (Zoom, Teams, Discord).")
            print("  2. If using an external camera, try '--source 1' or '--source 2'.")
            print("  3. For phone streaming, run 'python phone_server.py' and use '--source phone'.")
            sys.exit(1)
        return cap, True, source_name

    if not os.path.exists(source_arg):
        print(f"\n[ERROR] Video file not found: '{source_arg}'")
        print("[HELP] Please verify the path or provide a valid video file.")
        print("[EXAMPLES]")
        print("  - Phone stream : python app.py --source phone")
        print("  - Laptop webcam: python app.py --source 0")
        print("  - Video sample : python app.py --source sample/video.mp4")
        sys.exit(1)

    source_name = f"File ({os.path.basename(source_arg)})"
    print(f"[INFO] Opening video file: {source_arg}...")
    cap = cv2.VideoCapture(source_arg)
    if not cap.isOpened():
        print(f"\n[ERROR] Failed to open video file: '{source_arg}'")
        sys.exit(1)

    return cap, False, source_name


def setup_video_writer(save_path: str, fps: float, width: int, height: int) -> Optional[Any]:
    """Initialize cv2.VideoWriter with fallback codecs."""
    import cv2
    if not save_path:
        return None

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fps = fps if fps > 0 else 30.0
    codecs_to_try = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("XVID", ".avi"),
    ]

    for codec_str, ext in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec_str)
        target_path = save_path
        if not target_path.endswith(ext) and not target_path.endswith(".mp4"):
            target_path = f"{target_path}{ext}"

        writer = cv2.VideoWriter(target_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"[INFO] Saving processed video output to: '{target_path}' (Codec: {codec_str}, FPS: {fps:.1f})")
            return writer

    print(f"[WARNING] Could not initialize VideoWriter for '{save_path}'. Continuing without saving.")
    return None


def print_startup_banner_and_disclaimer() -> None:
    """Print project header, pre-flight checklist, usage notes, and privacy disclaimer."""
    print("=" * 78)
    print("   CodeAlpha AI Internship - Task 4: Object Detection & Tracking System")
    print("=" * 78)
    print(" [!] PRE-FLIGHT NOTES & USAGE GUIDELINES:")
    print("  1. HARDWARE & PERFORMANCE:")
    print("     - Default model (yolo11n.pt) is optimized for standard laptop CPUs (~25-45 FPS).")
    print("     - NVIDIA GPU users with CUDA installed will automatically achieve 60-120+ FPS.")
    print("  2. PHONE STREAMING SETUP (--source phone):")
    print("     - Step 1: In Terminal 1, run 'python phone_server.py'")
    print("     - Step 2: In Terminal 2, run 'python app.py --source phone'")
    print("     - Step 3: On Android phone, enter http://<LAPTOP_IP>:5000 and tap 'Start Camera'")
    print("  3. PRIVACY & SECURITY GUARANTEE:")
    print("     - All video processing runs 100% LOCALLY on your laptop.")
    print("     - No video streams, biometric signatures, or telemetry data are uploaded.")
    print("  4. FIRST-TIME INITIALIZATION:")
    print("     - Pretrained YOLO weights will automatically download on first execution (~6MB).")
    print("=" * 78)
    print(" [?] QUICK KEYBOARD CONTROLS (Active Video Window):")
    print("     [Q] / [ESC]  : Quit application safely")
    print("     [P] / [SPACE]: Pause / Resume playback")
    print("     [S]          : Save instant screenshot to screenshots/ folder")
    print("     [T]          : Toggle motion trajectory trails (ON/OFF)")
    print("     [H]          : Toggle Heads-Up Display (HUD) banner (ON/OFF)")
    print("=" * 78 + "\n")


def run_application() -> None:
    """Execute the object detection and tracking pipeline."""
    args = parse_arguments()

    # Pre-flight check dependencies before running
    ensure_dependencies()

    import cv2
    from src.detector import YOLODetector
    from src.tracker import ObjectTracker

    print_startup_banner_and_disclaimer()

    # 1. Initialize YOLO Detector
    try:
        detector = YOLODetector(
            model_name=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            classes=args.classes,
        )
    except Exception as e:
        print(f"[FATAL] Could not initialize YOLO model: {e}")
        print("        YOLO model weights will be downloaded on first run. Check internet connectivity.")
        sys.exit(1)

    # 2. Initialize Object Tracker
    tracker = ObjectTracker(
        model=detector.model,
        tracker_type=args.tracker,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        classes=args.classes,
        max_trail_length=30,
    )

    # 3. Open Video Source
    cap, is_live_source, source_name = open_video_source(args.source, phone_server_url=args.phone_server)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

    print(f"[INFO] Source Dimensions: {frame_w}x{frame_h} | Source FPS: {source_fps:.1f}")

    # 4. Optional Video Writer
    writer = setup_video_writer(args.save, source_fps if source_fps > 0 else 30.0, frame_w, frame_h)

    # 5. Runtime Control State
    show_trails = args.show_trails
    show_hud = not args.no_hud
    is_paused = False
    frame_count = 0
    start_time = time.time()
    prev_frame_time = time.time()
    current_fps = 0.0

    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    window_title = "CodeAlpha AI - Object Detection & Tracking [YOLO + ByteTrack]"
    if not args.no_display:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, min(1280, frame_w), min(720, frame_h))

    print("\n[INFO] Starting detection and tracking loop...")
    print("[INFO] Press 'q' or 'ESC' in the video window to quit.")
    print("[INFO] Press 'p' or 'SPACE' to pause/resume.")
    print("[INFO] Press 's' to save a screenshot.")
    print("[INFO] Press 't' to toggle motion trails.\n")

    try:
        while cap.isOpened():
            if not is_paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    if is_live_source:
                        time.sleep(0.05)
                        continue
                    else:
                        print("\n[INFO] End of video stream reached.")
                        break

                frame_count += 1

                now = time.time()
                time_diff = now - prev_frame_time
                prev_frame_time = now
                instant_fps = 1.0 / time_diff if time_diff > 0 else 0.0
                current_fps = 0.85 * current_fps + 0.15 * instant_fps if current_fps > 0 else instant_fps

                tracked_objects = tracker.track_frame(frame)

                annotated_frame = tracker.draw_tracks(
                    frame=frame,
                    tracked_objects=tracked_objects,
                    show_trails=show_trails,
                    show_labels=True,
                )

                if show_hud:
                    annotated_frame = tracker.draw_hud(
                        frame=annotated_frame,
                        fps=current_fps,
                        model_name=os.path.basename(args.model),
                        source_name=source_name,
                        is_paused=is_paused,
                        show_trails=show_trails,
                    )

                if writer is not None:
                    h, w = annotated_frame.shape[:2]
                    if w != frame_w or h != frame_h:
                        frame_w, frame_h = w, h
                    writer.write(annotated_frame)

            if not args.no_display:
                cv2.imshow(window_title, annotated_frame)
                key = cv2.waitKey(1 if not is_paused else 30) & 0xFF

                if key in [ord("q"), ord("Q"), 27]:
                    print("\n[INFO] User requested termination (Quit key pressed).")
                    break
                elif key in [ord("p"), ord("P"), 32]:
                    is_paused = not is_paused
                    print(f"[STATUS] Playback {'PAUSED' if is_paused else 'RESUMED'}")
                elif key in [ord("t"), ord("T")]:
                    show_trails = not show_trails
                    print(f"[STATUS] Trajectory Trails: {'ON' if show_trails else 'OFF'}")
                elif key in [ord("h"), ord("H")]:
                    show_hud = not show_hud
                    print(f"[STATUS] Heads-Up Display: {'ON' if show_hud else 'OFF'}")
                elif key in [ord("s"), ord("S")]:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_file = os.path.join("screenshots", f"tracking_capture_{timestamp}.png")
                    cv2.imwrite(screenshot_file, annotated_frame)
                    print(f"[SUCCESS] Saved screenshot: '{screenshot_file}'")

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt detected. Exiting gracefully...")

    finally:
        elapsed_time = max(0.001, time.time() - start_time)
        avg_fps = frame_count / elapsed_time

        print("\n" + "=" * 70)
        print("                  SESSION SUMMARY & STATISTICS")
        print("=" * 70)
        print(f"Total Frames Processed : {frame_count}")
        print(f"Total Elapsed Time     : {elapsed_time:.2f} seconds")
        print(f"Average Processing FPS : {avg_fps:.2f} FPS")
        print(f"Total Unique Track IDs : {len(tracker.unique_track_ids)}")
        print("Class Breakdown:")
        for cls_name, count in tracker.class_counts.items():
            print(f"  - {cls_name}: {count} active")
        print("=" * 70)

        cap.release()
        if writer is not None:
            writer.release()
            print(f"[INFO] Video output safely written to '{args.save}'.")

        if not args.no_display:
            cv2.destroyAllWindows()
            for _ in range(3):
                cv2.waitKey(1)

        print("[INFO] Cleanup complete. Pipeline closed safely.\n")


if __name__ == "__main__":
    run_application()
