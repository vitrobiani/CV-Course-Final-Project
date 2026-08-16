#!/usr/bin/env python3
"""
app.py — edge CV demo for Raspberry Pi Zero 2 W + ST7796S SPI panel + FT6336U touch.

Home screen with a Start button -> live camera stream with bounding boxes,
drawn straight to /dev/fb1 (no desktop), touch read from evdev.

Capture, inference, and display each run at their own pace so the preview stays
smooth even though detection on a Zero 2 W is only a couple of FPS.

Run:  python3 app.py        (Ctrl+C to quit)
"""

import os
import time
import queue
import signal
import fcntl
import threading

import numpy as np
import cv2

import config as cfg
from detector import Detector

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ===========================================================================
# Framebuffer  (RGB565, file-write path, with DRM pipeline unblank)
# ===========================================================================
class Framebuffer:
    def __init__(self, device, swap_rb=False):
        base = "/sys/class/graphics/" + os.path.basename(device)
        with open(base + "/virtual_size") as f:
            self.w, self.h = (int(v) for v in f.read().strip().split(","))
        with open(base + "/stride") as f:
            self.stride = int(f.read().strip())
        self.swap_rb = swap_rb
        self.row_bytes = self.w * 2
        self._f = open(device, "r+b")

        # panel-mipi-dbi is a DRM driver; /dev/fb1 is fbdev emulation. Writing
        # to it does NOT light the panel on its own -- explicitly unblank the
        # pipeline (this also switches the backlight on). Without it: black.
        FBIOBLANK, FB_BLANK_UNBLANK = 0x4611, 0
        try:
            fcntl.ioctl(self._f.fileno(), FBIOBLANK, FB_BLANK_UNBLANK)
        except OSError as e:
            print(f"[fb] unblank failed ({e})")

    def show(self, frame_bgr):
        if frame_bgr.shape[0] != self.h or frame_bgr.shape[1] != self.w:
            frame_bgr = cv2.resize(frame_bgr, (self.w, self.h))
        b = frame_bgr[:, :, 0].astype(np.uint16)
        g = frame_bgr[:, :, 1].astype(np.uint16)
        r = frame_bgr[:, :, 2].astype(np.uint16)
        if self.swap_rb:
            r, b = b, r
        rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
        data = rgb565.astype("<u2").tobytes()
        if self.stride != self.row_bytes:                 # pad rows to stride
            padded = bytearray(self.stride * self.h)
            for y in range(self.h):
                padded[y * self.stride:y * self.stride + self.row_bytes] = \
                    data[y * self.row_bytes:(y + 1) * self.row_bytes]
            data = bytes(padded)
        self._f.seek(0)
        self._f.write(data)
        self._f.flush()

    def clear(self):
        self._f.seek(0)
        self._f.write(b"\x00" * (self.stride * self.h))
        self._f.flush()

    def close(self):
        try:
            self.clear()
        except Exception:
            pass
        self._f.close()


# ===========================================================================
# Touch  (evdev; maps raw -> screen pixels with software calibration)
# ===========================================================================
class Touch(threading.Thread):
    def __init__(self, device, sw, sh):
        super().__init__(daemon=True)
        from evdev import InputDevice, ecodes
        self.ecodes = ecodes
        self.dev = InputDevice(device)
        self.sw, self.sh = sw, sh
        ax = self.dev.absinfo(ecodes.ABS_X)
        ay = self.dev.absinfo(ecodes.ABS_Y)
        self.xmin, self.xmax = ax.min, ax.max
        self.ymin, self.ymax = ay.min, ay.max
        self._rx = self._ry = 0
        self.taps = queue.Queue()
        print(f"[touch] X range: {self.xmin}-{self.xmax}, Y range: {self.ymin}-{self.ymax}, screen: {sw}x{sh}")

    def _map(self, rx, ry):
        xmax = getattr(cfg, 'TOUCH_X_MAX', self.xmax)
        ymax = getattr(cfg, 'TOUCH_Y_MAX', self.ymax)
        nx = (rx - self.xmin) / max(1, xmax - self.xmin)
        ny = (ry - self.ymin) / max(1, ymax - self.ymin)
        if cfg.TOUCH_SWAP_XY:
            nx, ny = ny, nx
        if cfg.TOUCH_INVERT_X:
            nx = 1.0 - nx
        if cfg.TOUCH_INVERT_Y:
            ny = 1.0 - ny
        return (int(np.clip(nx, 0, 1) * (self.sw - 1)),
                int(np.clip(ny, 0, 1) * (self.sh - 1)))

    def run(self):
        ec = self.ecodes
        for ev in self.dev.read_loop():
            if ev.type == ec.EV_ABS:
                if ev.code == ec.ABS_X:
                    self._rx = ev.value
                elif ev.code == ec.ABS_Y:
                    self._ry = ev.value
            elif ev.type == ec.EV_KEY and ev.code == ec.BTN_TOUCH and ev.value == 1:
                sx, sy = self._map(self._rx, self._ry)
                if cfg.DEBUG_TOUCH:
                    print(f"[touch] raw=({self._rx},{self._ry}) -> screen=({sx},{sy})")
                    # Also log to file
                    try:
                        with open("/home/alice/edgecv/touch.log", "a") as f:
                            f.write(f"raw=({self._rx},{self._ry}) -> screen=({sx},{sy})\n")
                    except:
                        pass
                self.taps.put((sx, sy))


# ===========================================================================
# Camera  (Picamera2, capture thread keeps only the latest frame)
# ===========================================================================
class Camera(threading.Thread):
    def __init__(self, size):
        super().__init__(daemon=True)
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        c = self.picam2.create_preview_configuration(
            main={"size": size, "format": "RGB888"})     # array comes out BGR-ordered
        self.picam2.configure(c)
        self._lock = threading.Lock()
        self._frame = None
        self.running = False

    def begin(self):
        if not self.running:
            self.picam2.start()
            self.running = True
            self.start()

    def run(self):
        while self.running:
            arr = self.picam2.capture_array("main")
            with self._lock:
                self._frame = arr

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self.running = False
        time.sleep(0.1)
        try:
            self.picam2.stop()
        except Exception:
            pass


# ===========================================================================
# Inference thread  (latest frame -> latest boxes, at its own slow pace)
# ===========================================================================
class Inference(threading.Thread):
    def __init__(self, camera, detector):
        super().__init__(daemon=True)
        self.camera, self.detector = camera, detector
        self._lock = threading.Lock()
        self._boxes = []
        self.running = True

    def run(self):
        while self.running:
            if not self.detector.ok:
                time.sleep(0.2)
                continue
            frame = self.camera.latest()
            if frame is None:
                time.sleep(0.03)
                continue
            dets = self.detector.detect(frame)
            with self._lock:
                self._boxes = dets

    def latest(self):
        with self._lock:
            return list(self._boxes)


# ===========================================================================
# UI
# ===========================================================================
HOME, STREAM = 0, 1

# button rects in screen coords (x, y, w, h)
# NOTE: top ~106px is a touch dead zone (raw_y < 160 never fires on this panel).
# All interactive elements must sit below screen_y ≈ 110.
_W, _H = cfg.CAM_SIZE
START_BTN = (_W // 2 - 120, 150, 240, 90)       # centred, in the reachable zone
STOP_BTN  = (_W // 2 - 100, _H - 58, 200, 50)  # bottom-centre, always reachable


def _in(rect, x, y):
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def draw_button(canvas, rect, text, bg, fg, scale=0.7, thick=2):
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), fg, 1)
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(canvas, text, (x + (w - tw) // 2, y + (h + th) // 2),
                FONT, scale, fg, thick, cv2.LINE_AA)


def render_home(w, h):
    c = np.zeros((h, w, 3), np.uint8)
    c[:] = (40, 30, 25)
    cv2.putText(c, "RPI CV", (w // 2 - 60, 70), FONT, 1.1, (230, 230, 230), 2, cv2.LINE_AA)
    draw_button(c, START_BTN, "Start", (60, 140, 60), (240, 240, 240), 0.75, 2)
    return c


def draw_detections(frame, dets):
    for d in dets:
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (80, 220, 120), 2)
        tag = f"{d.label} {d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.5, 1)
        cv2.rectangle(frame, (d.x1, d.y1 - th - 6), (d.x1 + tw + 4, d.y1), (80, 220, 120), -1)
        cv2.putText(frame, tag, (d.x1 + 2, d.y1 - 4), FONT, 0.5, (20, 20, 20), 1, cv2.LINE_AA)


# ===========================================================================
# Main
# ===========================================================================
def main():
    fb = Framebuffer(cfg.FB_DEVICE, swap_rb=cfg.SWAP_RB)
    detector = Detector(cfg, model_path=cfg.MODEL_PATH_BASE)
    camera = Camera(cfg.CAM_SIZE)
    infer = Inference(camera, detector)
    infer.start()

    try:
        touch = Touch(cfg.TOUCH_DEVICE, fb.w, fb.h)
        touch.start()
    except Exception as e:
        print(f"[touch] unavailable ({e}); running without touch.")
        touch = None

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    state = HOME
    home_canvas = render_home(fb.w, fb.h)
    t_prev, fps = time.time(), 0.0

    while not stop["flag"]:
        # ---- handle taps ----
        if touch:
            while not touch.taps.empty():
                tx, ty = touch.taps.get()
                if state == HOME and _in(START_BTN, tx, ty):
                    camera.begin()
                    state = STREAM
                elif state == STREAM and _in(STOP_BTN, tx, ty):
                    state = HOME

        # ---- render: HOME ----
        if state == HOME:
            fb.show(home_canvas)
            time.sleep(0.05)
            continue

        # ---- render: STREAM ----
        frame = camera.latest()
        if frame is None:
            wait = home_canvas.copy()
            cv2.putText(wait, "starting camera...", (fb.w // 2 - 110, fb.h - 30),
                        FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            fb.show(wait)
            time.sleep(0.05)
            continue

        draw_detections(frame, infer.latest())
        draw_button(frame, STOP_BTN, "STOP", (60, 60, 200), (240, 240, 240), 0.65, 2)

        if cfg.SHOW_FPS:
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, now - t_prev))
            t_prev = now
            det = "on" if infer.detector.ok else "off"
            cv2.putText(frame, f"{fps:4.1f} fps  det:{det}",
                        (fb.w - 150, fb.h - 10),
                        FONT, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        fb.show(frame)

    # ---- shutdown ----
    print("\n[app] shutting down...")
    infer.running = False
    camera.stop()
    fb.close()


if __name__ == "__main__":
    main()
