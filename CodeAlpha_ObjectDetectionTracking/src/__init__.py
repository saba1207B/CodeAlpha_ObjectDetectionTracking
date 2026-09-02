"""
CodeAlpha_ObjectDetectionTracking
================================
A real-time AI computer vision application for Multi-Object Detection and
Deep Tracking using pretrained YOLO models and ByteTrack/BoT-SORT.

CodeAlpha Artificial Intelligence Internship - Task 4
"""

from .detector import YOLODetector
from .tracker import ObjectTracker

__version__ = "1.0.0"
__author__ = "CodeAlpha AI Intern"
__all__ = ["YOLODetector", "ObjectTracker"]
