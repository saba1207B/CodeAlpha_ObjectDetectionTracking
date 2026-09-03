"""
backend/server.py
=================
Unified HTTP Server and Vision Pipeline for CodeAlpha Object Detection & Tracking.
Part of the CodeAlpha AI Internship - Task 4.

Supports:
  1. Android APK camera frame ingestion (POST /frame)
  2. Laptop Web Dashboard REST API and static file serving
  3. Real-time YOLO + ByteTrack / BoT-SORT multi-object tracking
  4. Snapshot capture and video recording
"""

import argparse
import datetime
import json
import mimetypes
import os
import socket
import sys
import threading
import time
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Optional, Tuple, Any, List, Dict

# Ensure backend package resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Graceful vision imports
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None
    np = None

try:
    from ultralytics import YOLO
    from src.detector import YOLODetector
    from src.tracker import ObjectTracker
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None
    YOLODetector = None
    ObjectTracker = None


def get_local_ip_addresses() -> List[str]:
    """Auto-detect the laptop's LAN IPv4 addresses."""
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        primary_ip = s.getsockname()[0]
        s.close()
        if primary_ip and not primary_ip.startswith("127."):
            ips.append(primary_ip)
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if ":" not in ip and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass

    return ips if ips else ["127.0.0.1"]


class FrameBuffer:
    """Thread-safe frame buffer holding incoming and processed frames."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_bgr: Optional[Any] = None
        self._processed_jpeg: Optional[bytes] = None
        self._processed_bgr: Optional[Any] = None
        self._frame_count: int = 0
        self._last_received_time: float = 0.0
        self._first_frame_logged: bool = False
        self._client_address: str = "None"
        self._fps_counter: int = 0
        self._fps_last_time: float = time.time()
        self._current_fps: float = 0.0
        self._frame_width: int = 0
        self._frame_height: int = 0
        self._new_frame_event = threading.Event()

    def update_raw(self, jpeg_bytes: bytes, client_ip: str) -> bool:
        if not jpeg_bytes or len(jpeg_bytes) < 10:
            return False

        decoded = None
        if OPENCV_AVAILABLE and np is not None:
            try:
                np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception:
                decoded = None

        now = time.time()
        with self._lock:
            self._latest_jpeg = jpeg_bytes
            self._latest_bgr = decoded
            self._frame_count += 1
            self._last_received_time = now
            self._client_address = client_ip

            if decoded is not None:
                self._frame_height, self._frame_width = decoded.shape[:2]

            self._fps_counter += 1
            if now - self._fps_last_time >= 1.0:
                self._current_fps = self._fps_counter / (now - self._fps_last_time)
                self._fps_counter = 0
                self._fps_last_time = now

            if not self._first_frame_logged:
                print(f"\n[+] FIRST FRAME RECEIVED from {client_ip}!")
                if self._frame_width > 0:
                    print(f"    Resolution: {self._frame_width}x{self._frame_height} px | Active stream.")
                self._first_frame_logged = True

        self._new_frame_event.set()
        return True

    def get_latest_bgr(self) -> Tuple[Optional[Any], float]:
        with self._lock:
            if self._latest_bgr is None:
                return None, 0.0
            return self._latest_bgr.copy(), self._last_received_time

    def get_latest_jpeg(self) -> Tuple[Optional[bytes], float]:
        with self._lock:
            return self._latest_jpeg, self._last_received_time

    def update_processed(self, bgr_frame: Any, jpeg_bytes: Optional[bytes] = None):
        with self._lock:
            self._processed_bgr = bgr_frame
            if jpeg_bytes is not None:
                self._processed_jpeg = jpeg_bytes
            elif OPENCV_AVAILABLE and bgr_frame is not None:
                _, enc = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                self._processed_jpeg = enc.tobytes()

    def get_processed_jpeg(self) -> Tuple[Optional[bytes], float]:
        with self._lock:
            if self._processed_jpeg is not None:
                return self._processed_jpeg, self._last_received_time
            # Fallback to raw if processed not yet ready
            return self._latest_jpeg, self._last_received_time

    def get_processed_bgr(self) -> Optional[Any]:
        with self._lock:
            if self._processed_bgr is None:
                return None
            return self._processed_bgr.copy()

    def wait_for_new_frame(self, timeout: float = 0.5) -> bool:
        signaled = self._new_frame_event.wait(timeout=timeout)
        self._new_frame_event.clear()
        return signaled

    def get_stats(self) -> dict:
        with self._lock:
            now = time.time()
            time_since = round(now - self._last_received_time, 2) if self._last_received_time > 0 else None
            return {
                "frame_count": self._frame_count,
                "current_fps": round(self._current_fps, 1),
                "resolution": f"{self._frame_width}x{self._frame_height}" if self._frame_width > 0 else "Pending",
                "client_address": self._client_address,
                "seconds_since_last_frame": time_since,
                "is_receiving": (time_since is not None and time_since < 3.5),
                "last_frame_timestamp": self._last_received_time,
            }


class DetectionManager:
    """Manages the background YOLO + ByteTrack/BoT-SORT execution and video recording."""

    def __init__(self, frame_buffer: FrameBuffer):
        self.buffer = frame_buffer
        self.lock = threading.Lock()
        self.is_active = True
        self.is_running = True
        self.worker_thread = None

        # Settings
        self.model_name = "yolo11n.pt"
        self.tracker_name = "bytetrack.yaml"
        self.conf_threshold = 0.35
        self.iou_threshold = 0.45
        self.filter_classes = None
        self.show_trails = True
        self.show_hud = True

        # Pipeline instances
        self.detector: Optional[Any] = None
        self.tracker: Optional[Any] = None

        # Analytics
        self.active_tracks: List[Dict[str, Any]] = []
        self.total_unique_tracks = 0
        self.active_count = 0
        self.last_inference_ms = 0.0

        # Video Recording
        self.is_recording = False
        self.video_writer: Optional[Any] = None
        self.current_record_path: Optional[str] = None

        # Start worker thread
        self.start_worker()

    def start_worker(self):
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _init_vision_pipeline(self):
        if not (OPENCV_AVAILABLE and YOLO_AVAILABLE):
            return False

        try:
            print(f"[ENGINE] Loading YOLO model: {self.model_name}...")
            self.detector = YOLODetector(
                model_name=self.model_name,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                classes=self.filter_classes,
            )
            print(f"[ENGINE] Initializing Tracker: {self.tracker_name}...")
            self.tracker = ObjectTracker(
                model=self.detector.model,
                tracker_type=self.tracker_name,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                classes=self.filter_classes,
                max_trail_length=30,
            )
            return True
        except Exception as e:
            print(f"[ENGINE ERROR] Failed to initialize vision pipeline: {e}")
            self.detector = None
            self.tracker = None
            return False

    def _worker_loop(self):
        last_processed_time = 0.0

        while self.is_running:
            self.buffer.wait_for_new_frame(timeout=0.1)

            if not self.is_active:
                time.sleep(0.05)
                continue

            frame, frame_time = self.buffer.get_latest_bgr()
            if frame is None or frame_time == last_processed_time:
                continue

            last_processed_time = frame_time

            # Lazily initialize or reload detector/tracker
            if self.tracker is None:
                success = self._init_vision_pipeline()
                if not success:
                    time.sleep(0.5)
                    continue

            t0 = time.time()
            try:
                tracked_objects = self.tracker.track_frame(frame)

                annotated = self.tracker.draw_tracks(
                    frame=frame,
                    tracked_objects=tracked_objects,
                    show_trails=self.show_trails,
                    show_labels=True,
                )

                if self.show_hud:
                    stats = self.buffer.get_stats()
                    annotated = self.tracker.draw_hud(
                        frame=annotated,
                        fps=stats["current_fps"],
                        model_name=os.path.basename(self.model_name),
                        source_name="Android APK (Phone)",
                        is_paused=not self.is_active,
                        show_trails=self.show_trails,
                    )

                self.buffer.update_processed(annotated)

                # Write to recording file if active
                if self.is_recording and self.video_writer is not None:
                    try:
                        self.video_writer.write(annotated)
                    except Exception as we:
                        print(f"[RECORDING ERROR] Failed to write frame: {we}")

                # Update Detections state
                inference_ms = (time.time() - t0) * 1000.0
                with self.lock:
                    now_str = datetime.datetime.now().strftime("%H:%M:%S")
                    formatted_tracks = []
                    for t in tracked_objects:
                        formatted_tracks.append({
                            "track_id": t["track_id"],
                            "class_name": t["class_name"],
                            "class_id": t["class_id"],
                            "conf": t["conf"],
                            "box": t["box"],
                            "centroid": t["centroid"],
                            "last_seen_time": now_str,
                            "trail_length": t.get("trail_length", 0)
                        })

                    self.active_tracks = formatted_tracks
                    self.active_count = len(formatted_tracks)
                    self.total_unique_tracks = len(self.tracker.unique_track_ids)
                    self.last_inference_ms = inference_ms

            except Exception as e:
                print(f"[ENGINE TRACKING ERROR] {e}")

    def update_settings(self, settings: dict):
        with self.lock:
            reinit_needed = False

            if "model" in settings and settings["model"] != self.model_name:
                self.model_name = settings["model"]
                reinit_needed = True

            if "tracker" in settings and settings["tracker"] != self.tracker_name:
                self.tracker_name = settings["tracker"]
                reinit_needed = True

            if "conf" in settings:
                self.conf_threshold = max(0.01, min(1.0, float(settings["conf"])))
                if self.tracker:
                    self.tracker.conf_threshold = self.conf_threshold

            if "iou" in settings:
                self.iou_threshold = max(0.01, min(1.0, float(settings["iou"])))
                if self.tracker:
                    self.tracker.iou_threshold = self.iou_threshold

            if "classes" in settings:
                self.filter_classes = settings["classes"]
                if self.tracker:
                    self.tracker.filter_classes = self.filter_classes

            if "show_trails" in settings:
                self.show_trails = bool(settings["show_trails"])

            if "show_hud" in settings:
                self.show_hud = bool(settings["show_hud"])

            if reinit_needed:
                self._init_vision_pipeline()

    def get_settings(self) -> dict:
        with self.lock:
            return {
                "model": self.model_name,
                "tracker": self.tracker_name,
                "conf": self.conf_threshold,
                "iou": self.iou_threshold,
                "classes": self.filter_classes,
                "show_trails": self.show_trails,
                "show_hud": self.show_hud,
                "recording": self.is_recording,
            }

    def clear_results(self):
        with self.lock:
            if self.tracker:
                self.tracker.reset()
            self.active_tracks = []
            self.total_unique_tracks = 0
            self.active_count = 0

    def start_recording(self, output_dir: str = "output") -> str:
        if not OPENCV_AVAILABLE:
            raise RuntimeError("OpenCV is not available for recording.")

        with self.lock:
            if self.is_recording:
                return self.current_record_path

            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"
            filepath = os.path.join(output_dir, filename)

            frame = self.buffer.get_processed_bgr()
            if frame is None:
                frame, _ = self.buffer.get_latest_bgr()

            h, w = (480, 640)
            if frame is not None:
                h, w = frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(filepath, fourcc, 20.0, (w, h))

            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                filepath = filepath.replace(".mp4", ".avi")
                writer = cv2.VideoWriter(filepath, fourcc, 20.0, (w, h))

            if not writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter for {filepath}")

            self.video_writer = writer
            self.current_record_path = filepath
            self.is_recording = True
            print(f"[RECORDING] Started: {filepath}")
            return filepath

    def stop_recording(self) -> Optional[str]:
        with self.lock:
            if not self.is_recording:
                return None

            path = self.current_record_path
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.current_record_path = None
            print(f"[RECORDING] Finalized: {path}")
            return path


# Global instances
FRAME_BUFFER = FrameBuffer()
DETECTION_MANAGER = DetectionManager(FRAME_BUFFER)


class UnifiedServerHandler(BaseHTTPRequestHandler):
    """Handles Android APK requests, dashboard API calls, and web asset serving."""

    server_version = "CodeAlphaVisionServer/2.0"

    def log_message(self, format, *args):
        # Suppress routine 200 logs for high-frequency video streams
        if len(args) >= 2 and any(p in str(args[0]) for p in ["/frame", "/latest_frame", "/processed_frame"]):
            return
        super().log_message(format, *args)

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        clean_path = self.path.split("?")[0].rstrip("/")
        if not clean_path:
            clean_path = "/"

        # 1. Health check & Root (Browser gets Dashboard, curl/APK get text)
        if clean_path == "/" or clean_path == "/health":
            accept_header = self.headers.get("Accept", "")
            if "text/html" in accept_header or "Mozilla" in self.headers.get("User-Agent", ""):
                self._serve_dashboard_file("index.html")
            else:
                body = (
                    "CodeAlpha Object Detection & Tracking Server\n"
                    "Status: Running\n"
                    "Endpoint: POST /frame\n"
                    "Dashboard: http://localhost:5000/\n"
                ).encode("utf-8")
                self.send_response(200)
                self._set_cors_headers()
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # 2. Latest Raw Phone Frame (CameraX JPEG)
        elif clean_path == "/latest_frame":
            jpeg_bytes, ts = FRAME_BUFFER.get_latest_jpeg()
            if jpeg_bytes is None:
                self._send_error_response(503, "No phone camera frame received yet.")
            else:
                self.send_response(200)
                self._set_cors_headers()
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg_bytes)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("X-Frame-Timestamp", str(ts))
                self.end_headers()
                self.wfile.write(jpeg_bytes)

        # 3. Latest Processed Detection/Tracking Frame
        elif clean_path == "/processed_frame":
            jpeg_bytes, ts = FRAME_BUFFER.get_processed_jpeg()
            if jpeg_bytes is None:
                self._send_error_response(503, "No processed detection frame available yet.")
            else:
                self.send_response(200)
                self._set_cors_headers()
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg_bytes)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("X-Frame-Timestamp", str(ts))
                self.end_headers()
                self.wfile.write(jpeg_bytes)

        # 4. Status & Diagnostic Metrics
        elif clean_path == "/status":
            stats = FRAME_BUFFER.get_stats()
            local_ips = get_local_ip_addresses()
            port = self.server.server_address[1]

            stats.update({
                "server_running": True,
                "opencv_loaded": OPENCV_AVAILABLE,
                "yolo_loaded": YOLO_AVAILABLE and (DETECTION_MANAGER.detector is not None),
                "tracker_loaded": YOLO_AVAILABLE and (DETECTION_MANAGER.tracker is not None),
                "detection_active": DETECTION_MANAGER.is_active,
                "recording": DETECTION_MANAGER.is_recording,
                "laptop_ips": local_ips,
                "server_port": port,
                "apk_url": f"http://{local_ips[0]}:{port}",
                "total_unique_tracks": DETECTION_MANAGER.total_unique_tracks,
                "active_tracks_count": DETECTION_MANAGER.active_count,
            })
            self._send_json_response(200, stats)

        # 5. Active Detections & Tracking Metadata
        elif clean_path == "/detections":
            with DETECTION_MANAGER.lock:
                payload = {
                    "status": "ok",
                    "timestamp": time.time(),
                    "is_detecting": DETECTION_MANAGER.is_active,
                    "active_tracks": DETECTION_MANAGER.active_tracks,
                    "active_count": DETECTION_MANAGER.active_count,
                    "total_unique_tracks": DETECTION_MANAGER.total_unique_tracks,
                    "inference_time_ms": DETECTION_MANAGER.last_inference_ms,
                    "recording": DETECTION_MANAGER.is_recording,
                }
            self._send_json_response(200, payload)

        # 6. Current Detection Settings
        elif clean_path == "/settings":
            settings = DETECTION_MANAGER.get_settings()
            self._send_json_response(200, settings)

        # 7. Dashboard Explicit Routes & Static Assets
        elif clean_path == "/dashboard" or clean_path == "/index.html":
            self._serve_dashboard_file("index.html")

        elif clean_path == "/style.css":
            self._serve_dashboard_file("style.css", content_type="text/css")

        elif clean_path == "/app.js":
            self._serve_dashboard_file("app.js", content_type="application/javascript")

        # 8. Screenshots Serving
        elif clean_path.startswith("/screenshots/"):
            fname = os.path.basename(clean_path)
            fpath = os.path.join("screenshots", fname)
            if os.path.exists(fpath):
                self._serve_binary_file(fpath, "image/jpeg")
            else:
                self._send_error_response(404, "Screenshot not found")

        else:
            self._send_error_response(404, f"Path not found: {clean_path}")

    def do_POST(self):
        clean_path = self.path.split("?")[0].rstrip("/")

        # 1. Android APK Camera Frame Intake
        if clean_path == "/frame":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length <= 0:
                    self._send_error_response(400, "Content-Length missing or zero")
                    return

                jpeg_bytes = self.rfile.read(content_length)
                client_ip = self.client_address[0]

                success = FRAME_BUFFER.update_raw(jpeg_bytes, client_ip)
                if not success:
                    self._send_error_response(400, "Failed to decode JPEG payload")
                    return

                # Strict compatibility with Android FrameSender.kt
                self._send_json_response(200, {"status": "ok"})
            except Exception as e:
                self._send_error_response(500, f"Server error: {str(e)}")

        # 2. Start Detection
        elif clean_path == "/start-detection":
            DETECTION_MANAGER.is_active = True
            self._send_json_response(200, {"status": "ok", "message": "Detection started", "active": True})

        # 3. Stop Detection
        elif clean_path == "/stop-detection":
            DETECTION_MANAGER.is_active = False
            self._send_json_response(200, {"status": "ok", "message": "Detection stopped", "active": False})

        # 4. Clear Results
        elif clean_path == "/clear-results":
            DETECTION_MANAGER.clear_results()
            self._send_json_response(200, {"status": "ok", "message": "Results cleared"})

        # 5. Update Settings
        elif clean_path == "/settings":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                settings_payload = json.loads(body)

                # Validation
                if "conf" in settings_payload:
                    conf = float(settings_payload["conf"])
                    if not (0.01 <= conf <= 1.0):
                        self._send_error_response(400, "conf must be between 0.01 and 1.0")
                        return

                if "iou" in settings_payload:
                    iou = float(settings_payload["iou"])
                    if not (0.01 <= iou <= 1.0):
                        self._send_error_response(400, "iou must be between 0.01 and 1.0")
                        return

                if "tracker" in settings_payload:
                    if settings_payload["tracker"] not in ["bytetrack.yaml", "botsort.yaml"]:
                        self._send_error_response(400, "tracker must be bytetrack.yaml or botsort.yaml")
                        return

                DETECTION_MANAGER.update_settings(settings_payload)
                self._send_json_response(200, {"status": "ok", "settings": DETECTION_MANAGER.get_settings()})
            except Exception as e:
                self._send_error_response(400, f"Invalid settings JSON: {e}")

        # 6. Screenshot Snapshot Capture
        elif clean_path == "/screenshot":
            os.makedirs("screenshots", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            filepath = os.path.join("screenshots", filename)

            # Try processed frame first, then raw frame
            frame = FRAME_BUFFER.get_processed_bgr()
            if frame is None:
                frame, _ = FRAME_BUFFER.get_latest_bgr()

            if frame is None or not OPENCV_AVAILABLE:
                # If no decoded frame, write raw JPEG bytes
                raw_jpeg, _ = FRAME_BUFFER.get_latest_jpeg()
                if raw_jpeg:
                    with open(filepath, "wb") as f:
                        f.write(raw_jpeg)
                else:
                    self._send_error_response(503, "No camera frame available to capture.")
                    return
            else:
                cv2.imwrite(filepath, frame)

            self._send_json_response(200, {
                "status": "ok",
                "file": filepath,
                "url": f"/screenshots/{filename}",
                "timestamp": timestamp,
            })

        # 7. Start Video Recording
        elif clean_path == "/recording/start":
            try:
                rec_path = DETECTION_MANAGER.start_recording()
                self._send_json_response(200, {"status": "ok", "file": rec_path, "recording": True})
            except Exception as e:
                self._send_error_response(500, f"Recording failed: {str(e)}")

        # 8. Stop Video Recording
        elif clean_path == "/recording/stop":
            try:
                saved_path = DETECTION_MANAGER.stop_recording()
                self._send_json_response(200, {"status": "ok", "file": saved_path, "recording": False})
            except Exception as e:
                self._send_error_response(500, f"Stop recording failed: {str(e)}")

        else:
            self._send_error_response(404, f"Endpoint not found: {clean_path}")

    def _serve_dashboard_file(self, filename: str, content_type: str = "text/html; charset=utf-8"):
        """Serve static files from laptop-dashboard directory."""
        candidates = [
            os.path.join(PROJECT_ROOT, "laptop-dashboard", filename),
            os.path.join(CURRENT_DIR, "laptop-dashboard", filename),
            os.path.join("laptop-dashboard", filename),
        ]
        target = None
        for c in candidates:
            if os.path.exists(c):
                target = c
                break

        if not target:
            self._send_error_response(404, f"Dashboard asset {filename} not found.")
            return

        with open(target, "rb") as f:
            data = f.read()

        self.send_response(200)
        self._set_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_binary_file(self, fpath: str, content_type: str):
        with open(fpath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self._set_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json_response(self, code: int, payload: dict):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(code)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_response(self, code: int, message: str):
        self._send_json_response(code, {"status": "error", "message": message})


def run_server(host: str = "0.0.0.0", port: int = 5000):
    server_address = (host, port)
    try:
        httpd = ThreadingHTTPServer(server_address, UnifiedServerHandler)
    except OSError as e:
        print("\n" + "=" * 65)
        print(f"[!] BIND ERROR: Unable to bind to {host}:{port}")
        print(f"    Details: {e}")
        if "Address already in use" in str(e) or 98 in getattr(e, "args", []):
            print(f"    Port {port} is occupied. Please specify another port: --port 5001")
        print("=" * 65 + "\n")
        sys.exit(1)

    local_ips = get_local_ip_addresses()
    primary_ip = local_ips[0]

    print("=" * 65)
    print("   CodeAlpha Object Detection & Tracking — Unified Server")
    print("=" * 65)
    print(" [1] Target URL for Android APK (CameraX):")
    print(f"     http://{primary_ip}:{port}")
    print()
    print(" [2] Laptop Web Dashboard URL (Open in browser):")
    print(f"     http://localhost:{port}/")
    print(f"     http://{primary_ip}:{port}/")
    print()
    print(" [3] Vision Stack Status:")
    print(f"     OpenCV Available   : {'YES' if OPENCV_AVAILABLE else 'NO (lightweight mode)'}")
    print(f"     YOLO Available     : {'YES' if YOLO_AVAILABLE else 'NO (install requirements.txt)'}")
    print("=" * 65 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[i] Stopping server gracefully...")
    finally:
        DETECTION_MANAGER.is_running = False
        DETECTION_MANAGER.stop_recording()
        httpd.server_close()
        print("[+] Server shutdown complete.\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="CodeAlpha Unified Vision Server & Web Dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (0.0.0.0 for LAN/Hotspot)")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_server(host=args.host, port=args.port)
