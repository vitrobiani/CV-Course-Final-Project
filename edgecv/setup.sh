#!/bin/bash
# Run this once on the Pi to install tflite-runtime
# Usage: sudo bash setup.sh

set -e

echo "Installing tflite-runtime for Raspberry Pi..."

# Try ai-edge-litert first (newer name), fall back to tflite-runtime
if pip3 install --break-system-packages ai-edge-litert 2>/dev/null; then
    echo "Installed ai-edge-litert"
elif pip3 install --break-system-packages tflite-runtime 2>/dev/null; then
    echo "Installed tflite-runtime"
else
    echo "Failed to install. Trying with --user flag..."
    pip3 install --user ai-edge-litert || pip3 install --user tflite-runtime
fi

echo "Restarting edgecv service..."
sudo systemctl restart edgecv.service

echo "Done! Check 'cat /home/alice/edgecv/detector.log' for status"
