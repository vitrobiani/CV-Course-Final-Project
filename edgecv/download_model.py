#!/usr/bin/env python3
"""
Download a working YOLOv8n ONNX model for person detection.
Run this once on the Pi with network access:
    python3 download_model.py
"""

import urllib.request
import os

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.onnx")

def download():
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists: {MODEL_PATH}")
        return

    print(f"Downloading YOLOv8n ONNX model...")
    print(f"URL: {MODEL_URL}")
    print(f"Destination: {MODEL_PATH}")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"Done! Downloaded {size_mb:.1f} MB")
        print("Restart edgecv service: sudo systemctl restart edgecv")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Try manually:")
        print(f"  curl -L -o {MODEL_PATH} {MODEL_URL}")

if __name__ == "__main__":
    download()
