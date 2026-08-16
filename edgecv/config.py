"""
config.py — all the knobs in one place. Edit this, not the app logic.

This is the file you'll touch most: touch calibration, model path, input
size, confidence, and labels all live here.
"""

# ---- Display (your ST7796S panel) ---------------------------------------
FB_DEVICE = "/dev/fb1"          # the SPI panel; HDMI is fb0

# ---- Touch (your FT6336U) -----------------------------------------------
TOUCH_DEVICE = "/dev/input/event0"

# The panel runs landscape over a portrait-native touch layer, so the raw
# axes are probably swapped/mirrored. Tap the on-screen Start button; if it
# doesn't respond where you actually touch, flip these one at a time until it
# tracks. Set DEBUG_TOUCH = True to print the mapped (x, y) while calibrating.
TOUCH_SWAP_XY  = False
TOUCH_INVERT_X = False
TOUCH_INVERT_Y = False
DEBUG_TOUCH    = False

# The FT6336U absinfo reports xmax=319, but hardware raw values reach ~479.
# Override here so the full screen width maps correctly instead of clipping.
TOUCH_X_MAX = 479
TOUCH_Y_MAX = 479

# ---- Camera -------------------------------------------------------------
# Capture/display resolution. Matches the panel; the camera ISP scales the
# full sensor down to this, so it's cheap.
CAM_SIZE = (480, 320)           # (width, height)

# ---- Model (the part that doesn't exist yet) ----------------------------
# Point this at your exported .tflite once training is done. Until the file
# exists on disk, the app runs the full camera + UI pipeline with detection
# simply disabled — nothing else changes.
import os as _os
_DIR = _os.path.dirname(_os.path.abspath(__file__))
MODEL_PATH_BASE = _os.path.join(_DIR, "best_float32_base.tflite")
MODEL_PATH_AUG  = _os.path.join(_DIR, "best_float32_aug.tflite")
MODEL_PATH = MODEL_PATH_BASE    # default / legacy
MODEL_INPUT_SIZE = (320, 320)   # actual model input size
CONF_THRESHOLD = 0.35
NMS_IOU = 0.45

# YOLOv8n-person model has only 1 class
LABELS = {0: "person"}

# ---- Cosmetic -----------------------------------------------------------
# If the camera colours look red/blue swapped, flip this.
SWAP_RB = False
SHOW_FPS = True
