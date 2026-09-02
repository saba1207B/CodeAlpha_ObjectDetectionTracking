"""
src/detector.py
==============
Module handling YOLO model loading, inference, and object detection.
Part of the CodeAlpha AI Internship - Task 4 (Object Detection & Tracking).
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from ultralytics import YOLO


class YOLODetector:
    """
    Wraps the Ultralytics YOLO object detection model.
    Provides inference, class filtering, and bounding box formatting.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        classes: Optional[List[int]] = None,
    ):
        """
        Initialize the YOLO detector.

        Args:
            model_name: Name or path of pretrained model (e.g., 'yolo11n.pt', 'yolov8n.pt').
            conf_threshold: Minimum confidence score for valid detections (0.0 to 1.0).
            iou_threshold: Intersection over Union threshold for Non-Maximum Suppression.
            device: Computation device ('cpu', 'cuda', 'mps', or None for auto).
            classes: Optional list of class IDs to filter (e.g., [0] for persons only).
        """
        self.model_name = model_name
        self.conf_threshold = max(0.01, min(1.0, conf_threshold))
        self.iou_threshold = max(0.01, min(1.0, iou_threshold))
        self.device = device
        self.filter_classes = classes

        print(f"[INFO] Initializing YOLO Detector with model: {self.model_name}")
        try:
            self.model = YOLO(self.model_name)
            # Fetch model class names dictionary (id -> name)
            self.class_names: Dict[int, str] = self.model.names
            print(f"[INFO] Model loaded successfully. Total classes supported: {len(self.class_names)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model '{self.model_name}': {e}") from e

        # Generate a distinct deterministic color palette for up to 100 classes
        np.random.seed(42)
        self.color_palette = [
            tuple(int(c) for c in np.random.randint(50, 255, size=3))
            for _ in range(max(100, len(self.class_names)))
        ]

    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Return a consistent BGR color tuple for a given class ID."""
        return self.color_palette[class_id % len(self.color_palette)]

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform standard object detection on a single frame.

        Args:
            frame: Input image/frame in BGR format (numpy array).

        Returns:
            List of detection dictionaries containing:
                - 'box': [x1, y1, x2, y2]
                - 'conf': float confidence score
                - 'class_id': int class ID
                - 'class_name': str class label
        """
        if frame is None or frame.size == 0:
            return []

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            classes=self.filter_classes,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.class_names.get(cls_id, f"Class {cls_id}")

            detections.append({
                "box": xyxy,
                "conf": conf,
                "class_id": cls_id,
                "class_name": cls_name,
            })

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        draw_labels: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes and class labels for static detections on the frame.

        Args:
            frame: Image frame in BGR format.
            detections: List of detection dictionaries.
            draw_labels: Whether to render the label text badge.

        Returns:
            Annotated frame.
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_id = det["class_id"]
            cls_name = det["class_name"]
            conf = det["conf"]
            color = self.get_color(cls_id)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

            if draw_labels:
                label_text = f"{cls_name} | {conf:.2f}"
                self._draw_badge(annotated, label_text, (x1, y1), color)

        return annotated

    @staticmethod
    def _draw_badge(
        image: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        bg_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Draw a readable filled label badge above or below a bounding box coordinate."""
        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        
        # Position label above box if space permits, otherwise below
        if y - text_h - baseline - 8 >= 0:
            box_coords = ((x, y - text_h - baseline - 8), (x + text_w + 10, y))
            text_pos = (x + 5, y - 5)
        else:
            box_coords = ((x, y), (x + text_w + 10, y + text_h + baseline + 8))
            text_pos = (x + 5, y + text_h + 3)

        cv2.rectangle(image, box_coords[0], box_coords[1], bg_color, -1)
        cv2.putText(image, text, text_pos, font, scale, text_color, thickness, cv2.LINE_AA)
