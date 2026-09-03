"""Unified local-network server for the CodeAlpha object detection dashboard.

Android compatibility is intentionally preserved: POST /frame, GET /, GET
/latest_frame and GET /status remain available. The server binds to 0.0.0.0:5000
for a trusted local Wi-Fi network; do not expose this development server publicly.
"""
from __future__ import annotations
import datetime as dt
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path: sys.path.insert(0, BACKEND_DIR)
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None; np = None; OPENCV_AVAILABLE = False
try:
    from src.detector import YOLODetector
    from src.tracker import ObjectTracker
    YOLO_AVAILABLE = True
except ImportError:
    YOLODetector = None; ObjectTracker = None; YOLO_AVAILABLE = False

DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "laptop-dashboard")
SCREENSHOT_DIR = os.path.join(PROJECT_ROOT, "screenshots")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def get_local_ip_addresses() -> list[str]:
    ips: list[str] = []
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); sock.settimeout(0.2)
        sock.connect(("8.8.8.8", 80)); ip = sock.getsockname()[0]; sock.close()
        if ip and not ip.startswith("127."): ips.append(ip)
    except OSError: pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in ips: ips.append(ip)
    except OSError: pass
    return ips or ["127.0.0.1"]


class FrameBuffer:
    def __init__(self) -> None:
        self.lock = threading.Lock(); self.latest_jpeg: Optional[bytes] = None
        self.latest_bgr: Optional[Any] = None; self.processed_jpeg: Optional[bytes] = None
        self.processed_bgr: Optional[Any] = None; self.frame_count = 0
        self.last_received = 0.0; self.client_address = "None"; self.width = 0; self.height = 0
        self._fps_count = 0; self._fps_started = time.monotonic(); self.current_fps = 0.0
        self.new_frame = threading.Event()
    def update(self, jpeg: bytes, client_ip: str) -> bool:
        if not jpeg or len(jpeg) < 10: return False
        decoded = None
        if OPENCV_AVAILABLE:
            try: decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception: decoded = None
            if decoded is None: return False
        now = time.time()
        with self.lock:
            self.latest_jpeg = jpeg; self.latest_bgr = decoded; self.frame_count += 1
            self.last_received = now; self.client_address = client_ip
            if decoded is not None: self.height, self.width = decoded.shape[:2]
            self._fps_count += 1; elapsed = time.monotonic() - self._fps_started
            if elapsed >= 1.0:
                self.current_fps = self._fps_count / elapsed; self._fps_count = 0; self._fps_started = time.monotonic()
        self.new_frame.set(); return True
    def get_raw(self):
        with self.lock: return (self.latest_bgr.copy() if self.latest_bgr is not None else None, self.last_received)
    def get_raw_jpeg(self):
        with self.lock: return self.latest_jpeg, self.last_received
    def set_processed(self, frame: Any) -> None:
        if not OPENCV_AVAILABLE or frame is None: return
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            with self.lock: self.processed_bgr = frame.copy(); self.processed_jpeg = encoded.tobytes()
    def get_processed(self):
        with self.lock: return self.processed_jpeg, self.last_received
    def get_processed_bgr(self):
        with self.lock: return self.processed_bgr.copy() if self.processed_bgr is not None else None
    def wait(self, timeout: float = 0.2) -> None: self.new_frame.wait(timeout); self.new_frame.clear()
    def stats(self) -> dict:
        with self.lock:
            age = time.time() - self.last_received if self.last_received else None
            return {"frame_count": self.frame_count, "current_fps": round(self.current_fps,1),
                    "resolution": f"{self.width}x{self.height}" if self.width else "Pending",
                    "client_address": self.client_address, "seconds_since_last_frame": round(age,2) if age is not None else None,
                    "is_receiving": age is not None and age < 3.5, "last_frame_timestamp": self.last_received}


class DetectionManager:
    def __init__(self, buffer: FrameBuffer) -> None:
        self.buffer=buffer; self.lock=threading.RLock(); self.active=False; self.running=True
        self.model="yolo11n.pt"; self.tracker_name="bytetrack.yaml"; self.conf=0.35; self.iou=0.45; self.classes:Optional[list[int]]=None
        self.show_trails=True; self.show_hud=True; self.detector=None; self.tracker=None; self.engine_error:Optional[str]=None
        self.active_tracks:list[dict]=[]; self.total_unique_tracks=0; self.inference_ms=0.0
        self.recording=False; self.video_writer=None; self.record_path:Optional[str]=None
        self.worker=threading.Thread(target=self._worker,daemon=True,name="yolo-tracker"); self.worker.start()
    def _load_pipeline(self)->bool:
        if not (OPENCV_AVAILABLE and YOLO_AVAILABLE):
            self.engine_error="Missing Python packages. Install backend/requirements.txt (OpenCV, NumPy, PyTorch and Ultralytics)."; return False
        try:
            detector=YOLODetector(model_name=self.model,conf_threshold=self.conf,iou_threshold=self.iou,classes=self.classes)
            tracker=ObjectTracker(model=detector.model,tracker_type=self.tracker_name,conf_threshold=self.conf,iou_threshold=self.iou,classes=self.classes,max_trail_length=30)
            with self.lock: self.detector,self.tracker,self.engine_error=detector,tracker,None
            return True
        except Exception as exc:
            self.engine_error=f"YOLO/tracker initialization failed: {exc}"; self.detector=None; self.tracker=None; return False
    def _worker(self)->None:
        last_timestamp=0.0
        while self.running:
            self.buffer.wait()
            if not self.active: continue
            frame,timestamp=self.buffer.get_raw()
            if frame is None or timestamp==last_timestamp: continue
            last_timestamp=timestamp
            if self.tracker is None and not self._load_pipeline(): time.sleep(1.0); continue
            started=time.perf_counter()
            try:
                tracked=self.tracker.track_frame(frame)
                annotated=self.tracker.draw_tracks(frame=frame,tracked_objects=tracked,show_trails=self.show_trails,show_labels=True)
                if self.show_hud:
                    annotated=self.tracker.draw_hud(frame=annotated,fps=self.buffer.stats()["current_fps"],model_name=os.path.basename(self.model),source_name="Android APK (Phone)",is_paused=False,show_trails=self.show_trails)
                self.buffer.set_processed(annotated)
                with self.lock:
                    self.inference_ms=(time.perf_counter()-started)*1000.0
                    self.active_tracks=[{"track_id":x["track_id"],"class_name":x["class_name"],"class_id":x["class_id"],"conf":x["conf"],"box":x["box"],"centroid":x["centroid"],"last_seen_time":dt.datetime.now().strftime("%H:%M:%S"),"trail_length":x.get("trail_length",0)} for x in tracked]
                    self.total_unique_tracks=len(self.tracker.unique_track_ids); self.engine_error=None
                self._write_recording(annotated)
            except Exception as exc: self.engine_error=f"Detection engine error: {exc}"; time.sleep(0.2)
    def _write_recording(self,frame:Any)->None:
        with self.lock: writer=self.video_writer if self.recording else None
        if writer is not None:
            try: writer.write(frame)
            except Exception as exc: self.engine_error=f"Recording error: {exc}"
    def update(self,payload:dict)->None:
        allowed={"yolo11n.pt","yolo11s.pt","yolov8n.pt","yolov8s.pt"}; new_model=str(payload.get("model",self.model)); new_tracker=str(payload.get("tracker",self.tracker_name))
        if new_model not in allowed: raise ValueError("Unsupported YOLO model")
        if new_tracker not in {"bytetrack.yaml","botsort.yaml"}: raise ValueError("tracker must be bytetrack.yaml or botsort.yaml")
        new_conf=max(.01,min(1.,float(payload.get("conf",self.conf)))); new_iou=max(.01,min(1.,float(payload.get("iou",self.iou))))
        classes=payload.get("classes",self.classes); classes=None if classes is None else [int(c) for c in classes]
        reload_needed=new_model!=self.model or new_tracker!=self.tracker_name or classes!=self.classes
        with self.lock:
            self.model,self.tracker_name=new_model,new_tracker; self.conf,self.iou,self.classes=new_conf,new_iou,classes
            self.show_trails=bool(payload.get("show_trails",self.show_trails)); self.show_hud=bool(payload.get("show_hud",self.show_hud))
            if self.tracker is not None: self.tracker.conf_threshold=self.conf; self.tracker.iou_threshold=self.iou; self.tracker.filter_classes=self.classes
        if reload_needed: self.detector=None; self.tracker=None; self._load_pipeline()
    def settings(self)->dict:
        with self.lock: return {"model":self.model,"tracker":self.tracker_name,"conf":self.conf,"iou":self.iou,"classes":self.classes,"show_trails":self.show_trails,"show_hud":self.show_hud,"recording":self.recording}
    def clear(self)->None:
        with self.lock:
            if self.tracker: self.tracker.reset()
            self.active_tracks=[]; self.total_unique_tracks=0; self.inference_ms=0.0
    def start_recording(self)->str:
        if not OPENCV_AVAILABLE: raise RuntimeError("OpenCV is not installed; recording is unavailable.")
        os.makedirs(OUTPUT_DIR,exist_ok=True); frame=self.buffer.get_processed_bgr()
        if frame is None: frame,_=self.buffer.get_raw()
        h,w=frame.shape[:2] if frame is not None else (480,640); filename=f"recording_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"; path=os.path.join(OUTPUT_DIR,filename)
        writer=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*"mp4v"),20.0,(w,h))
        if not writer.isOpened(): raise RuntimeError("Could not open the MP4 video writer on this system.")
        with self.lock: self.video_writer,self.record_path,self.recording=writer,path,True
        return path
    def stop_recording(self)->Optional[str]:
        with self.lock:
            path=self.record_path; self.recording=False
            if self.video_writer is not None: self.video_writer.release()
            self.video_writer=None; self.record_path=None; return path

BUFFER=FrameBuffer(); ENGINE=DetectionManager(BUFFER)

def json_bytes(payload:dict)->bytes: return json.dumps(payload,separators=(",",":")).encode("utf-8")

class Handler(BaseHTTPRequestHandler):
    server_version="CodeAlphaVisionServer/3.0"
    def log_message(self,fmt,*args):
        if self.path.split("?")[0] not in {"/frame","/latest_frame","/processed_frame"}: super().log_message(fmt,*args)
    def _headers(self,content_type="application/json"):
        self.send_header("Access-Control-Allow-Origin","*"); self.send_header("Access-Control-Allow-Methods","GET, POST, OPTIONS"); self.send_header("Access-Control-Allow-Headers","Content-Type"); self.send_header("Content-Type",content_type)
    def _json(self,code,payload):
        body=json_bytes(payload); self.send_response(code); self._headers(); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
    def _file(self,path,content_type):
        if not os.path.isfile(path): self._json(404,{"status":"error","message":f"File not found: {os.path.basename(path)}"}); return
        with open(path,"rb") as h: body=h.read()
        self.send_response(200); self._headers(content_type); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_OPTIONS(self): self.send_response(204); self._headers(); self.end_headers()
    def do_GET(self):
        path=self.path.split("?")[0].rstrip("/") or "/"
        if path in {"/","/health"}:
            if "text/html" in self.headers.get("Accept","") or "Mozilla" in self.headers.get("User-Agent",""): self._file(os.path.join(DASHBOARD_DIR,"index.html"),"text/html; charset=utf-8")
            else: self._json(200,{"status":"ok","server":"CodeAlpha Object Detection & Tracking","endpoint":"POST /frame"})
        elif path=="/latest_frame":
            data,ts=BUFFER.get_raw_jpeg(); self._json(503,{"status":"error","message":"No phone camera frame received yet. Start the Android camera and check the Wi-Fi URL."}) if data is None else self._image(data,ts)
        elif path=="/processed_frame":
            data,ts=BUFFER.get_processed(); self._json(503,{"status":"error","message":"No processed frame is available. Start Detection after the phone sends frames."}) if data is None else self._image(data,ts)
        elif path=="/status":
            stats=BUFFER.stats(); ips=get_local_ip_addresses(); stats.update({"server_running":True,"server_host":"0.0.0.0","server_port":self.server.server_address[1],"laptop_ips":ips,"apk_url":f"http://{ips[0]}:{self.server.server_address[1]}","opencv_loaded":OPENCV_AVAILABLE,"yolo_loaded":YOLO_AVAILABLE and ENGINE.detector is not None,"tracker_loaded":YOLO_AVAILABLE and ENGINE.tracker is not None,"detection_active":ENGINE.active,"recording":ENGINE.recording,"total_unique_tracks":ENGINE.total_unique_tracks,"active_tracks_count":len(ENGINE.active_tracks),"inference_time_ms":round(ENGINE.inference_ms,1),"engine_error":ENGINE.engine_error}); self._json(200,stats)
        elif path=="/detections":
            with ENGINE.lock: payload={"status":"ok","timestamp":time.time(),"is_detecting":ENGINE.active,"active_tracks":ENGINE.active_tracks,"active_count":len(ENGINE.active_tracks),"total_unique_tracks":ENGINE.total_unique_tracks,"inference_time_ms":ENGINE.inference_ms,"recording":ENGINE.recording,"error":ENGINE.engine_error}
            self._json(200,payload)
        elif path=="/settings": self._json(200,ENGINE.settings())
        elif path=="/style.css": self._file(os.path.join(DASHBOARD_DIR,"style.css"),"text/css; charset=utf-8")
        elif path=="/app.js": self._file(os.path.join(DASHBOARD_DIR,"app.js"),"application/javascript; charset=utf-8")
        elif path in {"/dashboard","/index.html"}: self._file(os.path.join(DASHBOARD_DIR,"index.html"),"text/html; charset=utf-8")
        elif path.startswith("/screenshots/"): self._file(os.path.join(SCREENSHOT_DIR,os.path.basename(path)),"image/jpeg")
        else: self._json(404,{"status":"error","message":f"Path not found: {path}"})
    def _image(self,data,ts):
        self.send_response(200); self._headers("image/jpeg"); self.send_header("Cache-Control","no-store, no-cache, must-revalidate"); self.send_header("X-Frame-Timestamp",str(ts)); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
    def _body(self):
        length=int(self.headers.get("Content-Length","0")); return json.loads((self.rfile.read(length) if length else b"{}").decode("utf-8"))
    def do_POST(self):
        path=self.path.split("?")[0].rstrip("/")
        try:
            if path=="/frame":
                length=int(self.headers.get("Content-Length","0"))
                if length<=0: self._json(400,{"status":"error","message":"Empty JPEG request body."}); return
                if not BUFFER.update(self.rfile.read(length),self.client_address[0]): self._json(400,{"status":"error","message":"Invalid JPEG frame payload."}); return
                self._json(200,{"status":"ok"})
            elif path=="/start-detection": ENGINE.active=True; self._json(200,{"status":"ok","active":True,"message":"Detection started"})
            elif path=="/stop-detection": ENGINE.active=False; self._json(200,{"status":"ok","active":False,"message":"Detection stopped"})
            elif path=="/clear-results": ENGINE.clear(); self._json(200,{"status":"ok","message":"Tracking results cleared"})
            elif path=="/settings": ENGINE.update(self._body()); self._json(200,{"status":"ok","settings":ENGINE.settings()})
            elif path=="/screenshot":
                frame=BUFFER.get_processed_bgr(); raw,_=BUFFER.get_raw_jpeg()
                if frame is None and raw is None: self._json(503,{"status":"error","message":"No camera frame available for screenshot."}); return
                os.makedirs(SCREENSHOT_DIR,exist_ok=True); filename=f"screenshot_{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"; target=os.path.join(SCREENSHOT_DIR,filename)
                if frame is not None and OPENCV_AVAILABLE: cv2.imwrite(target,frame)
                else:
                    with open(target,"wb") as h: h.write(raw)
                self._json(200,{"status":"ok","file":target,"url":f"/screenshots/{filename}"})
            elif path=="/recording/start": self._json(200,{"status":"ok","file":ENGINE.start_recording(),"recording":True})
            elif path=="/recording/stop": self._json(200,{"status":"ok","file":ENGINE.stop_recording(),"recording":False})
            else: self._json(404,{"status":"error","message":f"Endpoint not found: {path}"})
        except json.JSONDecodeError as exc: self._json(400,{"status":"error","message":f"Invalid JSON: {exc}"})
        except Exception as exc: self._json(500,{"status":"error","message":str(exc)})


def run_server(host="0.0.0.0",port=5000):
    httpd=ThreadingHTTPServer((host,port),Handler); ips=get_local_ip_addresses()
    print("="*70); print("CodeAlpha Object Detection & Tracking — Unified Server"); print("="*70)
    print(f"Dashboard: http://localhost:{port}/"); print(f"Network dashboard: http://{ips[0]}:{port}/"); print(f"Android APK URL: http://{ips[0]}:{port}"); print("Listening on 0.0.0.0 for trusted local-network Wi-Fi clients."); print("Do not expose port 5000 publicly."); print("="*70)
    try: httpd.serve_forever()
    except KeyboardInterrupt: pass
    finally: ENGINE.running=False; ENGINE.stop_recording(); httpd.server_close()

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description="CodeAlpha Object Detection & Tracking server"); p.add_argument("--host",default="0.0.0.0"); p.add_argument("--port",type=int,default=5000); a=p.parse_args(); run_server(a.host,a.port)
