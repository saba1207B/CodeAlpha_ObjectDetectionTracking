"""
src/tracker.py
==============
Module handling Multi-Object Tracking (MOT) using YOLO + ByteTrack / BoT-SORT.
Maintains persistent track IDs, trajectory trails, and track analytics.
Part of the CodeAlpha AI Internship - Task 4 (Object Detection & Tracking).
"""

from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import cv2
from ultralytics import YOLO


class ObjectTracker:
    """
    Manages multi-object tracking across video frames using state-of-the-art
    computer vision algorithms (ByteTrack or BoT-SORT).
    """

    def __init__(
        self,
        model: YOLO,
        tracker_type: str = "bytetrack.yaml",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        classes: Optional[List[int]] = None,
        max_trail_length: int = 30,
    ):
        """
        Initialize the Object Tracker.

        Args:
            model: An instantiated Ultralytics YOLO model.
            tracker_type: Tracker configuration ('bytetrack.yaml' or 'botsort.yaml').
            conf_threshold: Minimum detection confidence threshold.
            iou_threshold: IOU threshold for association.
            device: Computation device ('cpu', 'cuda', 'mps', or None for auto).
            classes: Optional list of class IDs to track.
            max_trail_length: Maximum number of previous centroid positions to store per track.
        """
        self.model = model
        self.tracker_type = tracker_type
        self.conf_threshold = max(0.01, min(1.0, conf_threshold))
        self.iou_threshold = max(0.01, min(1.0, iou_threshold))
        self.device = device
        self.filter_classes = classes
        self.max_trail_length = max_trail_length

        # Class names dictionary from YOLO model
        self.class_names: Dict[int, str] = self.model.names

        # Tracking state and history
        self.track_history = defaultdict(lambda: deque(maxlen=self.max_trail_length))
        self.unique_track_ids: Set[int] = set()
        self.active_tracks_count: int = 0
        self.class_counts: Dict[str, int] = defaultdict(int)

        # Deterministic color mapping for unique track IDs
        np.random.seed(1337)
        self.id_colors = [
            tuple(int(c) for c in np.random.randint(60, 250, size=3))
            for _ in range(500)
        ]

    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Return a consistent color for a given track ID."""
        return self.id_colors[track_id % len(self.id_colors)]

    def track_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run tracking on the current frame.

        Args:
            frame: Input image/frame in BGR format.

        Returns:
            List of tracked object dictionaries:
                - 'track_id': int tracking ID (persistent across frames)
                - 'box': [x1, y1, x2, y2]
                - 'conf': float confidence score
                - 'class_id': int class ID
                - 'class_name': str class label
                - 'centroid': (cx, cy) integer coordinates
        """
        if frame is None or frame.size == 0:
            return []

        # Run YOLO with persistent tracking enabled
        results = self.model.track(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            tracker=self.tracker_type,
            persist=True,
            device=self.device,
            classes=self.filter_classes,
            verbose=False,
        )

        tracked_objects: List[Dict[str, Any]] = []
        self.class_counts.clear()

        if not results or len(results) == 0:
            self.active_tracks_count = 0
            return tracked_objects

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            self.active_tracks_count = 0
            return tracked_objects

        for box in boxes:
            # Extract tracking ID if available
            track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.class_names.get(cls_id, f"Class {cls_id}")

            # Compute centroid of bounding box for trajectory tracking
            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int((xyxy[1] + xyxy[3]) / 2)

            if track_id is not None:
                self.unique_track_ids.add(track_id)
                self.track_history[track_id].append((cx, cy))
                self.class_counts[cls_name] += 1

            tracked_objects.append({
                "track_id": track_id,
                "box": xyxy,
                "conf": conf,
                "class_id": cls_id,
                "class_name": cls_name,
                "centroid": (cx, cy),
            })

        self.active_tracks_count = len(tracked_objects)
        return tracked_objects

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracked_objects: List[Dict[str, Any]],
        show_trails: bool = True,
        show_labels: bool = True,
    ) -> np.ndarray:
        """
        Render tracking bounding boxes, ID badges, and movement trails onto the frame.

        Display Format:
            <class_name> | ID: <id> | <conf>   (e.g., "person | ID: 3 | 0.91")

        Args:
            frame: Base image frame in BGR format.
            tracked_objects: List of tracked objects from `track_frame`.
            show_trails: Whether to render motion trail curves.
            show_labels: Whether to render the formatted label badges.

        Returns:
            Annotated frame.
        """
        annotated = frame.copy()

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["box"]
            track_id = obj["track_id"]
            cls_name = obj["class_name"]
            conf = obj["conf"]

            # Choose color based on track ID (or default fallback)
            color = self.get_track_color(track_id) if track_id is not None else (0, 255, 0)

            # 1. Draw Bounding Box with subtle corner accents
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            self._draw_corner_accents(annotated, (x1, y1, x2, y2), color, length=12, thickness=3)

            # 2. Draw Trajectory Trails (fading motion line)
            if show_trails and track_id is not None:
                points = self.track_history[track_id]
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    # Dynamic thickness based on age in trail
                    thickness = int(np.sqrt(self.max_trail_length / float(i + 1)) * 1.8) + 1
                    cv2.line(annotated, points[i - 1], points[i], color, thickness, lineType=cv2.LINE_AA)
                # Centroid dot
                cx, cy = obj["centroid"]
                cv2.circle(annotated, (cx, cy), 4, color, -1, lineType=cv2.LINE_AA)

            # 3. Draw Formatted Label Badge: "person | ID: 3 | 0.91"
            if show_labels:
                if track_id is not None:
                    label_text = f"{cls_name} | ID: {track_id} | {conf:.2f}"
                else:
                    label_text = f"{cls_name} | {conf:.2f}"
                self._draw_badge(annotated, label_text, (x1, y1), color)

        return annotated

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        model_name: str,
        source_name: str,
        is_paused: bool = False,
        show_trails: bool = True,
    ) -> np.ndarray:
        """
        Draw an informational Heads-Up Display (HUD) overlay with statistics and controls.

        Args:
            frame: Annotated video frame.
            fps: Current frames-per-second rate.
            model_name: Name of the active YOLO model.
            source_name: Source descriptor (e.g. Webcam 0, or file path).
            is_paused: Video pause status.
            show_trails: Motion trails display status.

        Returns:
            Frame with HUD overlay.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Top Information Banner (Translucent Dark Gray)
        banner_h = 75
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Title and System Stats
        cv2.putText(
            frame,
            "CodeAlpha AI | Object Detection & Tracking",
            (15, 25),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (255, 215, 0),  # Gold accent
            1,
            cv2.LINE_AA,
        )

        stats_text = (
            f"FPS: {fps:5.1f} | Active Tracks: {self.active_tracks_count:2d} | "
            f"Total Unique: {len(self.unique_track_ids):2d} | Model: {model_name}"
        )
        cv2.putText(
            frame,
            stats_text,
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        # Status badge on top right
        status_text = "PAUSED" if is_paused else "TRACKING"
        status_color = (0, 165, 255) if is_paused else (50, 205, 50)
        cv2.putText(
            frame,
            f"[{status_text}]",
            (w - 130, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.60,
            status_color,
            1,
            cv2.LINE_AA,
        )

        # Bottom Shortcut Hints bar
        bottom_y = h - 12
        hints = "[Q] Quit  |  [P/SPACE] Pause  |  [T] Trails  |  [S] Screenshot  |  [H] Toggle HUD"
        cv2.putText(
            frame,
            hints,
            (15, bottom_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        return frame

    @staticmethod
    def _draw_corner_accents(
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        length: int = 10,
        thickness: int = 2,
    ) -> None:
        """Draw sleek decorative corner accents on bounding boxes."""
        x1, y1, x2, y2 = box
        # Top-left
        cv2.line(image, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(image, (x1, y1), (x1, y1 + length), color, thickness)
        # Top-right
        cv2.line(image, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(image, (x2, y1), (x2, y1 + length), color, thickness)
        # Bottom-left
        cv2.line(image, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(image, (x1, y2), (x1, y2 - length), color, thickness)
        # Bottom-right
        cv2.line(image, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(image, (x2, y2), (x2, y2 - length), color, thickness)

    @staticmethod
    def _draw_badge(
        image: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        bg_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Draw a high-contrast label badge coordinate."""
        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad = 4

        if y - text_h - baseline - (pad * 2) >= 0:
            box_coords = ((x, y - text_h - baseline - (pad * 2)), (x + text_w + (pad * 2), y))
            text_pos = (x + pad, y - pad - 2)
        else:
            box_coords = ((x, y), (x + text_w + (pad * 2), y + text_h + baseline + (pad * 2)))
            text_pos = (x + pad, y + text_h + pad)

        # Filled badge background
        cv2.rectangle(image, box_coords[0], box_coords[1], bg_color, -1)
        # Border
        cv2.rectangle(image, box_coords[0], box_coords[1], (0, 0, 0), 1)
        # Text label
        cv2.putText(image, text, text_pos, font, scale, text_color, thickness, cv2.LINE_AA)
