#!/usr/bin/env python3
"""
app.py
======
Main Application Entry Point for CodeAlpha AI Internship Task 4:
Object Detection and Tracking System using Ultralytics YOLO & ByteTrack / BoT-SORT.

Usage Examples:
    # 1. Real-time Webcam Tracking:
    python app.py --source 0

    # 2. Tracking on a Video File:
    python app.py --source sample/video.mp4

    # 3. Custom Confidence Threshold & Model:
    python app.py --source sample/video.mp4 --conf 0.5 --model yolo11n.pt

    # 4. Save Processed Output Video:
    python app.py --source sample/video.mp4 --save output/tracked_output.mp4

    # 5. Filter Specific Classes (e.g., 0 for person, 2 for car):
    python app.py --source 0 --classes 0 2
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional, List, Union, Tuple
import cv2

# Import modular components from src package
from src.detector import YOLODetector
from src.tracker import ObjectTracker


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CodeAlpha AI Internship - Object Detection and Tracking with YOLO & ByteTrack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input video source: webcam index (e.g. '0', '1') or path to a video file ('sample/video.mp4').",
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


def open_video_source(source_arg: str) -> Tuple[cv2.VideoCapture, bool, str]:
    """
    Open the video capture source gracefully, supporting camera index or file paths.

    Returns:
        Tuple of (cv2.VideoCapture, is_webcam: bool, source_description: str)
    """
    is_webcam = False
    source_name = source_arg

    # Check if input is a digit index representing a webcam
    if source_arg.isdigit():
        camera_idx = int(source_arg)
        is_webcam = True
        source_name = f"Webcam (Device {camera_idx})"
        print(f"[INFO] Connecting to {source_name}...")
        
        # On Linux/macOS/Windows, standard VideoCapture
        cap = cv2.VideoCapture(camera_idx)
    else:
        # File path verification
        if not os.path.exists(source_arg):
            print(f"\n[ERROR] Video file not found: '{source_arg}'")
            print("[HELP] Please verify the path or provide a sample video.")
            print("[EXAMPLE] python app.py --source sample/video.mp4")
            sys.exit(1)

        source_name = f"File ({os.path.basename(source_arg)})"
        print(f"[INFO] Opening video file: {source_arg}...")
        cap = cv2.VideoCapture(source_arg)

    if not cap.isOpened():
        print(f"\n[ERROR] Failed to open video source: '{source_arg}'")
        if is_webcam:
            print("[TROUBLESHOOTING]")
            print("  1. Ensure your webcam is properly connected and not in use by another app.")
            print("  2. If using an external camera, try '--source 1' or '--source 2'.")
            print("  3. Check camera access permissions in your operating system settings.")
        sys.exit(1)

    return cap, is_webcam, source_name


def setup_video_writer(
    save_path: str,
    fps: float,
    width: int,
    height: int,
) -> Optional[cv2.VideoWriter]:
    """Initialize cv2.VideoWriter with fallback codecs."""
    if not save_path:
        return None

    # Ensure output directory exists
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
    print("  2. CAMERA & LIGHTING:")
    print("     - For webcam tracking, ensure adequate lighting and minimal background glare.")
    print("     - Ensure your webcam is not locked or used by another video application.")
    print("  3. PRIVACY & SECURITY DISCLAIMER:")
    print("     - All video processing runs 100% LOCALLY on your machine.")
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

    # Display opening disclaimer and pre-flight guide
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
        sys.exit(1)

    # 2. Initialize Object Tracker using the model instance
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
    cap, is_webcam, source_name = open_video_source(args.source)

    # Fetch Video Properties
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else None

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

    # Ensure screenshots folder exists
    os.makedirs("screenshots", exist_ok=True)

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
                if not ret:
                    if is_webcam:
                        print("[WARNING] Empty frame received from webcam stream. Retrying...")
                        time.sleep(0.05)
                        continue
                    else:
                        print("\n[INFO] End of video stream reached.")
                        break

                frame_count += 1

                # Calculate smooth rolling FPS
                now = time.time()
                time_diff = now - prev_frame_time
                prev_frame_time = now
                instant_fps = 1.0 / time_diff if time_diff > 0 else 0.0
                current_fps = 0.85 * current_fps + 0.15 * instant_fps if current_fps > 0 else instant_fps

                # Perform actual multi-object tracking
                tracked_objects = tracker.track_frame(frame)

                # Render tracking visualization (boxes, badges, trajectory lines)
                annotated_frame = tracker.draw_tracks(
                    frame=frame,
                    tracked_objects=tracked_objects,
                    show_trails=show_trails,
                    show_labels=True,
                )

                # Render HUD Overlay
                if show_hud:
                    annotated_frame = tracker.draw_hud(
                        frame=annotated_frame,
                        fps=current_fps,
                        model_name=os.path.basename(args.model),
                        source_name=source_name,
                        is_paused=is_paused,
                        show_trails=show_trails,
                    )

                # Save frame if writer configured
                if writer is not None:
                    writer.write(annotated_frame)

            # Display window
            if not args.no_display:
                cv2.imshow(window_title, annotated_frame)
                key = cv2.waitKey(1 if not is_paused else 30) & 0xFF

                # Key handling
                if key in [ord("q"), ord("Q"), 27]:  # 'q' or ESC to quit
                    print("\n[INFO] User requested termination (Quit key pressed).")
                    break
                elif key in [ord("p"), ord("P"), 32]:  # 'p' or SPACE to pause
                    is_paused = not is_paused
                    print(f"[STATUS] Playback {'PAUSED' if is_paused else 'RESUMED'}")
                elif key in [ord("t"), ord("T")]:  # 't' to toggle trails
                    show_trails = not show_trails
                    print(f"[STATUS] Trajectory Trails: {'ON' if show_trails else 'OFF'}")
                elif key in [ord("h"), ord("H")]:  # 'h' to toggle HUD
                    show_hud = not show_hud
                    print(f"[STATUS] Heads-Up Display: {'ON' if show_hud else 'OFF'}")
                elif key in [ord("s"), ord("S")]:  # 's' to capture screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_file = os.path.join("screenshots", f"tracking_capture_{timestamp}.png")
                    cv2.imwrite(screenshot_file, annotated_frame)
                    print(f"[SUCCESS] Saved screenshot: '{screenshot_file}'")

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt detected. Exiting gracefully...")

    finally:
        # Cleanup and release all hardware & memory resources
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

        # Release video capture & writer
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[INFO] Video output safely written to '{args.save}'.")

        # Destroy all OpenCV GUI windows safely
        if not args.no_display:
            cv2.destroyAllWindows()
            # On some platforms, multiple destroyAllWindows calls ensure prompt cleanup
            for _ in range(3):
                cv2.waitKey(1)

        print("[INFO] Cleanup complete. Project closed safely.\n")


if __name__ == "__main__":
    run_application()
