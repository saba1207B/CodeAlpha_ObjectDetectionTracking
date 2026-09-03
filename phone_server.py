"""
Root entry point for CodeAlpha Phone Camera Server & Laptop Dashboard.
Delegates to backend/server.py.
"""
import os
import sys

base_dir = os.path.join(os.path.dirname(__file__), "backend")
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from server import parse_arguments, run_server

if __name__ == "__main__":
    args = parse_arguments()
    run_server(host=args.host, port=args.port)

