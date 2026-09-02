#!/usr/bin/env python3
"""
phone_server.py
===============
Local HTTP camera receiver for the Android companion app.

The Android app sends JPEG frames to POST /frame. This server keeps only the
latest frame so the YOLO + ByteTrack pipeline can process it without building
an unbounded queue.
"""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np


class PhoneFrameStore:
    """Thread-safe latest-frame buffer shared by the HTTP server and AI loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._sequence = 0

    def set_jpeg(self, data: bytes) -> bool:
        array = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        with self._lock:
            self._frame = frame
            self._sequence += 1
        return True

    def get(self) -> tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self._frame is None:
                return None, self._sequence
            return self._frame.copy(), self._sequence


class PhoneRequestHandler(BaseHTTPRequestHandler):
    """Receive Android JPEG frames and expose a simple health endpoint."""

    server_version = "CodeAlphaPhoneCamera/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/health"):
            body = b"CodeAlpha phone camera receiver is running. POST JPEG frames to /frame."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/frame":
            self.send_error(404, "POST /frame expected")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 8 * 1024 * 1024:
                self.send_error(400, "Invalid frame size")
                return
            payload = self.rfile.read(length)
            if not self.server.frame_store.set_jpeg(payload):
                self.send_error(400, "Invalid JPEG frame")
                return

            body = b"OK"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionError, BrokenPipeError):
            pass
        except Exception as exc:
            self.send_error(500, f"Receiver error: {exc}")

    def log_message(self, format: str, *args) -> None:
        # Keep the terminal readable; the AI application prints its own status.
        return


class PhoneFrameServer:
    """Background HTTP server that provides the latest phone camera frame."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        self.host = host
        self.port = port
        self.frame_store = PhoneFrameStore()
        self._server = ThreadingHTTPServer((host, port), PhoneRequestHandler)
        self._server.frame_store = self.frame_store
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()
        print(f"[PHONE] Receiver listening on http://{self._local_ip()}:{self.port}")
        print(f"[PHONE] Android endpoint: http://<LAPTOP-IP>:{self.port}/frame")
        print("[PHONE] Waiting for the Android camera to send frames...\n")

    def read(self, last_sequence: int = -1) -> tuple[Optional[np.ndarray], int]:
        frame, sequence = self.frame_store.get()
        if sequence == last_sequence:
            return None, sequence
        return frame, sequence

    def is_ready(self) -> bool:
        frame, _ = self.frame_store.get()
        return frame is not None

    def release(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    @staticmethod
    def _local_ip() -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"
