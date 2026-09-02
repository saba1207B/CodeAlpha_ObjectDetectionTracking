#!/usr/bin/env python3
"""CodeAlpha AI Internship - Task 4: Object Detection & Tracking.

Input modes:
    python app.py --source 0
    python app.py --source sample/video.mp4
    python app.py --source phone

Phone mode starts a local HTTP receiver. The Android companion app sends
JPEG camera frames over the same Wi-Fi network; this program feeds the
received frames into the existing YOLO + ByteTrack / BoT-SORT pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

import cv2

from phone_server import PhoneFrameServer
from src.detector import YOLODetector
from src.tracker import ObjectTracker


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CodeAlpha AI Internship - Object Detection and Tracking with YOLO & ByteTrack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=str, default="0", help="Webcam index, video path, or 'phone'.")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"])
    parser.add_argument("--classes", nargs="+", type=int, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--show-trails", action="store_true", default=True)
    parser.add_argument("--no-trails", dest="show_trails", action="store_false")
    parser.add_argument("--no-hud", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Phone receiver bind address.")
    parser.add_argument("--port", type=int, default=5000, help="Phone receiver port.")
    return parser.parse_args()


def setup_video_writer(save_path: Optional[str], fps: float, width: int, height: int) -> Optional[cv2.VideoWriter]:
    if not save_path:
        return None
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fps = fps if fps > 0 else 30.0
    for codec in ("mp4v", "avc1", "XVID"):
        extension = ".avi" if codec == "XVID" else ".mp4"
        target = save_path if os.path.splitext(save_path)[1] else save_path + extension
        writer = cv2.VideoWriter(target, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            print(f"[INFO] Saving processed output to '{target}'")
            return writer
    print("[WARNING] Could not initialize VideoWriter. Continuing without saving.")
    return None


def open_source(source_arg: str, host: str, port: int):
    if source_arg.lower() == "phone":
        server = PhoneFrameServer(host=host, port=port)
        server.start()
        return server, True, f"Android Phone (HTTP :{port})", 640, 480, 10.0

    is_webcam = source_arg.isdigit()
    if is_webcam:
        index = int(source_arg)
        print(f"[INFO] Connecting to Webcam (Device {index})...")
        cap = cv2.VideoCapture(index)
        source_name = f"Webcam (Device {index})"
    else:
        if not os.path.exists(source_arg):
            print(f"[ERROR] Video file not found: '{source_arg}'")
            sys.exit(1)
        print(f"[INFO] Opening video file: {source_arg}...")
        cap = cv2.VideoCapture(source_arg)
        source_name = f"File ({os.path.basename(source_arg)})"

    if not cap.isOpened():
        print(f"[ERROR] Failed to open video source: '{source_arg}'")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, is_webcam, source_name, width, height, fps


def print_banner() -> None:
    print("=" * 78)
    print("   CodeAlpha AI Internship - Task 4: Object Detection & Tracking System")
    print("=" * 78)
    print("  YOLO detection + ByteTrack / BoT-SORT tracking + OpenCV visualization")
    print("  Phone mode: Android camera -> Wi-Fi -> Python AI pipeline")
    print("  Processing remains local to the connected laptop.")
    print("=" * 78)
    print("  Controls: [Q/ESC] Quit | [P/SPACE] Pause | [T] Trails | [H] HUD | [S] Screenshot")
    print("=" * 78 + "\n")


def run_application() -> None:
    args = parse_arguments()
    print_banner()

    try:
        detector = YOLODetector(
            model_name=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            classes=args.classes,
        )
    except Exception as exc:
        print(f"[FATAL] Could not initialize YOLO model: {exc}")
        sys.exit(1)

    tracker = ObjectTracker(
        model=detector.model,
        tracker_type=args.tracker,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        classes=args.classes,
        max_trail_length=30,
    )

    source, is_live, source_name, frame_w, frame_h, source_fps = open_source(args.source, args.host, args.port)
    phone_mode = isinstance(source, PhoneFrameServer)

    print(f"[INFO] Source: {source_name} | Dimensions: {frame_w}x{frame_h}")
    if phone_mode:
        print("[INFO] Enter the laptop IP shown above in the Android app and tap Start Streaming.")

    writer = setup_video_writer(args.save, source_fps, frame_w, frame_h)
    show_trails = args.show_trails
    show_hud = not args.no_hud
    is_paused = False
    frame_count = 0
    start_time = time.time()
    prev_frame_time = time.time()
    current_fps = 0.0
    last_sequence = -1
    annotated_frame = None

    os.makedirs("screenshots", exist_ok=True)
    window_title = "CodeAlpha AI - Object Detection & Tracking [YOLO + ByteTrack]"
    if not args.no_display:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, min(1280, frame_w), min(720, frame_h))

    try:
        while True:
            if not is_paused:
                if phone_mode:
                    frame, sequence = source.read(last_sequence)
                    if frame is None:
                        time.sleep(0.02)
                        continue
                    last_sequence = sequence
                    frame_h, frame_w = frame.shape[:2]
                else:
                    ret, frame = source.read()
                    if not ret:
                        if is_live:
                            time.sleep(0.05)
                            continue
                        print("[INFO] End of video stream reached.")
                        break

                frame_count += 1
                now = time.time()
                delta = now - prev_frame_time
                prev_frame_time = now
                instant_fps = 1.0 / delta if delta > 0 else 0.0
                current_fps = 0.85 * current_fps + 0.15 * instant_fps if current_fps else instant_fps

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
                    writer.write(annotated_frame)

            if not args.no_display and annotated_frame is not None:
                cv2.imshow(window_title, annotated_frame)
                key = cv2.waitKey(1 if not is_paused else 30) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("p"), ord("P"), 32):
                    is_paused = not is_paused
                elif key in (ord("t"), ord("T")):
                    show_trails = not show_trails
                elif key in (ord("h"), ord("H")):
                    show_hud = not show_hud
                elif key in (ord("s"), ord("S")):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join("screenshots", f"tracking_capture_{timestamp}.png")
                    cv2.imwrite(path, annotated_frame)
                    print(f"[SUCCESS] Saved screenshot: '{path}'")

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt detected.")
    finally:
        elapsed = max(0.001, time.time() - start_time)
        print("\n" + "=" * 70)
        print("SESSION SUMMARY & STATISTICS")
        print("=" * 70)
        print(f"Total Frames Processed : {frame_count}")
        print(f"Total Elapsed Time     : {elapsed:.2f} seconds")
        print(f"Average Processing FPS : {frame_count / elapsed:.2f} FPS")
        print(f"Total Unique Track IDs : {len(tracker.unique_track_ids)}")
        for cls_name, count in tracker.class_counts.items():
            print(f"  - {cls_name}: {count} active")
        print("=" * 70)

        source.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run_application()
