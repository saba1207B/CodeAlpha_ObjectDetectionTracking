"""
backend/src
===========
Core vision modules: YOLO Detector and Object Tracker (ByteTrack / BoT-SORT).
Part of CodeAlpha AI Internship - Task 4: Object Detection and Tracking.
"""
from .detector import YOLODetector
from .tracker import ObjectTracker

__all__ = ["YOLODetector", "ObjectTracker"]
