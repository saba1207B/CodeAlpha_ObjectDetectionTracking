"""
Root entry point for CodeAlpha Object Detection & Tracking.
Delegates to backend/app.py.
"""
import os
import sys

base_dir = os.path.join(os.path.dirname(__file__), "backend")
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from app import run_application

if __name__ == "__main__":
    run_application()

