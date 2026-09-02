"""
CodeAlpha AI Internship - Task 4: Object Detection & Tracking
Phone Camera Receiver Server (Standard Library HTTP Server)

This server receives JPEG frames uploaded by the Android Camera Client via HTTP POST /frame,
stores the latest frame in a thread-safe buffer, and serves frames to YOLO/ByteTrack in app.py.

Endpoints:
    - GET  /             : Health check and connectivity verification
    - POST /frame        : Receives JPEG camera frames from Android phone
    - GET  /latest_frame : Serves latest frame to detection pipeline / testing tools
    - GET  /status       : Diagnostic metrics (frame count, FPS, client info)
"""

import argparse
import json
import sys
import time
import threading
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Optional, Tuple, Any

# Graceful OpenCV and NumPy import with clear user instructions
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None
    np = None


class PhoneFrameBuffer:
    """Thread-safe storage for the most recent camera frame received from the phone."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame: Optional[Any] = None
        self._frame_count: int = 0
        self._last_received_time: float = 0.0
        self._first_frame_logged: bool = False
        self._client_address: str = "None"
        self._fps_counter: int = 0
        self._fps_last_time: float = time.time()
        self._current_fps: float = 0.0
        self._frame_width: int = 0
        self._frame_height: int = 0

    def update(self, jpeg_bytes: bytes, client_ip: str) -> bool:
        """Store the latest JPEG and decode if OpenCV is available."""
        if not jpeg_bytes or len(jpeg_bytes) < 10:
            return False

        decoded = None
        if OPENCV_AVAILABLE and np is not None:
            np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if decoded is None or decoded.size == 0:
                return False

        with self._lock:
            self._latest_jpeg = jpeg_bytes
            self._latest_frame = decoded
            self._frame_count += 1
            now = time.time()
            self._last_received_time = now
            self._client_address = client_ip

            if decoded is not None:
                self._frame_height, self._frame_width = decoded.shape[:2]

            # Calculate intake FPS
            self._fps_counter += 1
            if now - self._fps_last_time >= 1.0:
                self._current_fps = self._fps_counter / (now - self._fps_last_time)
                self._fps_counter = 0
                self._fps_last_time = now

            if not self._first_frame_logged:
                print(f"\n[+] FIRST FRAME RECEIVED from {client_ip}!")
                if self._frame_width > 0:
                    print(f"    Resolution: {self._frame_width}x{self._frame_height} px | Active stream.")
                print("    Ready for: python app.py --source phone\n")
                self._first_frame_logged = True

        return True

    def get_latest_frame(self) -> Tuple[Optional[Any], float]:
        """Return a copy of the latest decoded frame and its timestamp."""
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._last_received_time

    def get_latest_jpeg(self) -> Tuple[Optional[bytes], float]:
        """Return raw JPEG bytes and timestamp."""
        with self._lock:
            return self._latest_jpeg, self._last_received_time

    def get_stats(self) -> dict:
        """Return operational statistics."""
        with self._lock:
            return {
                "frame_count": self._frame_count,
                "current_fps": round(self._current_fps, 1),
                "resolution": f"{self._frame_width}x{self._frame_height}" if self._frame_width > 0 else "Pending",
                "client_address": self._client_address,
                "opencv_loaded": OPENCV_AVAILABLE,
                "seconds_since_last_frame": round(time.time() - self._last_received_time, 2) if self._last_received_time > 0 else None,
                "is_receiving": (time.time() - self._last_received_time < 3.0) if self._last_received_time > 0 else False
            }


# Global shared buffer instance
FRAME_BUFFER = PhoneFrameBuffer()


class PhoneCameraHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for phone camera communication."""

    server_version = "CodeAlphaPhoneServer/1.0"

    def log_message(self, format, *args):
        # Suppress routine 200 access logs for POST /frame to avoid console flood
        if len(args) >= 2 and "POST /frame" in str(args[0]) and "200" in str(args[1]):
            return
        super().log_message(format, *args)

    def do_GET(self):
        """Handle GET requests for health check, status, and latest frame."""
        clean_path = self.path.split("?")[0]

        if clean_path == "/" or clean_path == "/health":
            body = (
                "CodeAlpha Object Detection & Tracking Server\n"
                "Status: Running\n"
                "Endpoint: POST /frame\n"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif clean_path == "/latest_frame":
            jpeg_bytes, timestamp = FRAME_BUFFER.get_latest_jpeg()
            if jpeg_bytes is None:
                err_msg = b"No phone frame received yet. Start camera in the Android app."
                self.send_response(503)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(err_msg)))
                self.end_headers()
                self.wfile.write(err_msg)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg_bytes)))
                self.send_header("X-Frame-Timestamp", str(timestamp))
                self.end_headers()
                self.wfile.write(jpeg_bytes)

        elif clean_path == "/status":
            stats = FRAME_BUFFER.get_stats()
            body = json.dumps(stats, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            msg = b"404 Not Found"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def do_POST(self):
        """Handle incoming JPEG frames from Android phone."""
        clean_path = self.path.split("?")[0]

        if clean_path == "/frame":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length <= 0:
                    self._send_error_response(400, "Content-Length header missing or zero")
                    return

                # Read JPEG body
                jpeg_bytes = self.rfile.read(content_length)
                client_ip = self.client_address[0]

                success = FRAME_BUFFER.update(jpeg_bytes, client_ip)
                if not success:
                    self._send_error_response(400, "Failed to decode JPEG payload")
                    return

                # Return success response
                resp = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            except Exception as e:
                self._send_error_response(500, f"Internal server error: {str(e)}")
        else:
            self._send_error_response(404, "Endpoint not found. Use POST /frame")

    def _send_error_response(self, code: int, message: str):
        body = json.dumps({"status": "error", "message": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str = "0.0.0.0", port: int = 5000) -> None:
    """Start the multi-threaded HTTP server and keep it running safely."""
    server_address = (host, port)

    try:
        httpd = ThreadingHTTPServer(server_address, PhoneCameraHTTPHandler)
    except OSError as e:
        print("\n" + "=" * 60)
        print(f"[!] SERVER BIND ERROR: Unable to bind to {host}:{port}")
        print(f"    Details: {e}")
        if "Address already in use" in str(e) or 98 in getattr(e, "args", []):
            print(f"    Port {port} is already occupied by another process.")
            print(f"    Please stop the existing process or use a different port:")
            print(f"    python phone_server.py --port 5001")
        print("=" * 60 + "\n")
        sys.exit(1)

    print("==================================================")
    print("CodeAlpha Object Detection & Tracking")
    print("Phone Camera Server")
    print("==================================================")
    print("Listening on:")
    print(f"http://{host}:{port}")
    print()
    print("Waiting for phone camera frames...")
    print("==================================================")
    if not OPENCV_AVAILABLE:
        print("[!] NOTICE: Running in lightweight storage mode (OpenCV not in active Python environment).")
        print("    For full YOLO detection and tracking in app.py, please activate your venv:")
        print("    pip install -r requirements.txt")
        print("==================================================")
    print(" [i] USAGE NOTES:")
    print("     - On Android phone, enter: http://<LAPTOP_IP>:5000")
    print("     - Find your laptop IP via 'ipconfig' (Windows) or 'ip a' (Linux)")
    print("     - Test server connectivity: curl http://127.0.0.1:5000")
    print("     - Press Ctrl+C at any time to shut down the server.")
    print("=" * 50 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[i] Shutting down phone camera server safely...")
    except Exception as e:
        print(f"\n[!] Unexpected server exception: {e}")
    finally:
        httpd.server_close()
        print("[+] Server stopped.\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="CodeAlpha Phone Camera Server for YOLO & ByteTrack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Network interface to bind to (0.0.0.0 enables LAN/Hotspot/USB access)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port number to listen on"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_server(host=args.host, port=args.port)
