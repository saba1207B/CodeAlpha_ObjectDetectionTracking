"""
Root entry point for CodeAlpha Phone Camera Server.
Delegates to CodeAlpha_ObjectDetectionTracking/phone_server.py.
"""
import os
import sys

base_dir = os.path.join(os.path.dirname(__file__), "CodeAlpha_ObjectDetectionTracking")
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from phone_server import parse_arguments, run_server

if __name__ == "__main__":
    args = parse_arguments()
    run_server(host=args.host, port=args.port)
