#!/usr/bin/env python3
"""
calibrate.py — interactive touch calibration for FT6336U + ST7796S landscape panel.

Run:  python3 calibrate.py
      (stop the edgecv service first: sudo systemctl stop edgecv)

Shows crosshair targets at three screen corners one at a time.
Tap each target as accurately as you can.
The script prints the correct SWAP_XY / INVERT_X / INVERT_Y values and
writes them to calibration.log.
"""

import fcntl, time, os
import numpy as np
import cv2
from evdev import InputDevice, ecodes

import config as cfg

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.log")
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Framebuffer (same as app.py)
# ---------------------------------------------------------------------------
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
        FBIOBLANK, FB_BLANK_UNBLANK = 0x4611, 0
        try:
            fcntl.ioctl(self._f.fileno(), FBIOBLANK, FB_BLANK_UNBLANK)
        except OSError:
            pass

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
        if self.stride != self.row_bytes:
            padded = bytearray(self.stride * self.h)
            for y in range(self.h):
                padded[y * self.stride:y * self.stride + self.row_bytes] = \
                    data[y * self.row_bytes:(y + 1) * self.row_bytes]
            data = bytes(padded)
        self._f.seek(0)
        self._f.write(data)
        self._f.flush()

    def close(self):
        self._f.seek(0)
        self._f.write(b"\x00" * (self.stride * self.h))
        self._f.flush()
        self._f.close()


# ---------------------------------------------------------------------------
# Touch (raw, no calibration — read directly)
# ---------------------------------------------------------------------------
def read_one_tap(dev):
    """Block until BTN_TOUCH press; return (raw_x, raw_y). Ignores (0,0)."""
    rx = ry = 0
    for ev in dev.read_loop():
        if ev.type == ecodes.EV_ABS:
            if ev.code == ecodes.ABS_X:
                rx = ev.value
            elif ev.code == ecodes.ABS_Y:
                ry = ev.value
        elif (ev.type == ecodes.EV_KEY
              and ev.code == ecodes.BTN_TOUCH
              and ev.value == 1):
            if rx == 0 and ry == 0:
                continue           # skip spurious (0,0) events
            return rx, ry


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_crosshair(canvas, x, y, color=(0, 255, 0), size=24, thick=2):
    cv2.circle(canvas, (x, y), size, color, thick)
    cv2.circle(canvas, (x, y), 3, color, -1)
    cv2.line(canvas, (x - size - 8, y), (x + size + 8, y), color, thick)
    cv2.line(canvas, (x, y - size - 8), (x, y + size + 8), color, thick)


def show_target(fb, step, total, sx, sy, label):
    c = np.zeros((fb.h, fb.w, 3), np.uint8)
    c[:] = (20, 20, 35)
    cv2.putText(c, "Touch Calibration", (fb.w // 2 - 115, 38),
                FONT, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(c, f"Step {step}/{total}: tap the {label} target",
                (fb.w // 2 - 170, fb.h - 15),
                FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    draw_crosshair(c, sx, sy)
    fb.show(c)


def show_result(fb, lines):
    c = np.zeros((fb.h, fb.w, 3), np.uint8)
    c[:] = (20, 35, 20)
    y = 40
    for line in lines:
        cv2.putText(c, line, (20, y), FONT, 0.55, (100, 255, 100), 1, cv2.LINE_AA)
        y += 32
    fb.show(c)


# ---------------------------------------------------------------------------
# Calibration logic
# ---------------------------------------------------------------------------
def compute_config(targets_screen, taps_raw, ax, ay):
    """
    Given 3 (screen_x, screen_y) targets and their (raw_x, raw_y) taps,
    derive SWAP_XY, INVERT_X, INVERT_Y.

    Strategy: use TL→TR (horizontal motion) and TL→BL (vertical motion)
    to see which raw axis changes and in which direction.
    """
    (sx_tl, sy_tl), (sx_tr, sy_tr), (sx_bl, sy_bl) = targets_screen
    (rx_tl, ry_tl), (rx_tr, ry_tr), (rx_bl, ry_bl) = taps_raw

    # Horizontal screen motion (TL→TR): which raw axis moves more?
    horiz_dx_raw_x = abs(rx_tr - rx_tl)
    horiz_dx_raw_y = abs(ry_tr - ry_tl)
    swap = horiz_dx_raw_y > horiz_dx_raw_x

    if not swap:
        # raw X → screen X, raw Y → screen Y
        invert_x = (rx_tr - rx_tl) < 0   # moving right decreases raw X → invert
        invert_y = (ry_bl - ry_tl) < 0   # moving down decreases raw Y → invert
    else:
        # raw Y → screen X, raw X → screen Y
        invert_x = (ry_tr - ry_tl) < 0   # moving right decreases raw Y → invert X
        invert_y = (rx_bl - rx_tl) < 0   # moving down decreases raw X → invert Y

    return swap, invert_x, invert_y


def main():
    fb = Framebuffer(cfg.FB_DEVICE, swap_rb=cfg.SWAP_RB)
    dev = InputDevice(cfg.TOUCH_DEVICE)

    ax = dev.absinfo(ecodes.ABS_X)
    ay = dev.absinfo(ecodes.ABS_Y)
    info = f"Touch absinfo — X: {ax.min}..{ax.max}  Y: {ay.min}..{ay.max}  Screen: {fb.w}x{fb.h}"
    print(info)

    M = 18   # margin from corner so the crosshair is visible
    targets = [
        ("top-left",     (M,          M)),
        ("top-right",    (fb.w - M,   M)),
        ("bottom-left",  (M,          fb.h - M)),
    ]

    taps = []
    for i, (label, (sx, sy)) in enumerate(targets, 1):
        show_target(fb, i, len(targets), sx, sy, label)
        print(f"  Tap the {label} crosshair …", flush=True)
        time.sleep(0.4)          # brief pause so the finger-down from "ready" is ignored
        rx, ry = read_one_tap(dev)
        taps.append((rx, ry))
        print(f"    raw = ({rx}, {ry})")
        time.sleep(0.3)

    swap, inv_x, inv_y = compute_config(
        [pos for _, pos in targets],
        taps,
        ax, ay,
    )

    lines = [
        "=== Calibration Result ===",
        f"",
        f"TOUCH_SWAP_XY  = {swap}",
        f"TOUCH_INVERT_X = {inv_x}",
        f"TOUCH_INVERT_Y = {inv_y}",
        f"",
        f"absinfo X: {ax.min}..{ax.max}",
        f"absinfo Y: {ay.min}..{ay.max}",
        f"",
        "Tap taps:",
    ] + [f"  {label}: raw={t}" for (label, _), t in zip(targets, taps)] + [
        "",
        "Copy the 3 lines above into config.py",
    ]

    result_text = "\n".join(lines)
    print(result_text)
    with open(LOG, "a") as f:
        f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(result_text + "\n")

    show_result(fb, lines)
    time.sleep(8)
    fb.close()
    print(f"\nResult also saved to {LOG}")


if __name__ == "__main__":
    main()
