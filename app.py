"""
Root entry point for CodeAlpha Object Detection & Tracking.
Delegates to CodeAlpha_ObjectDetectionTracking/app.py.
"""
import os
import sys

base_dir = os.path.join(os.path.dirname(__file__), "CodeAlpha_ObjectDetectionTracking")
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
os.chdir(base_dir)

from app import run_application

if __name__ == "__main__":
    run_application()
