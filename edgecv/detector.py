"""
detector.py — the model connector.

Supports ONNX via onnxruntime and TFLite via tflite-runtime.
For YOLO models, expects output shape (1, 4+nc, N) with xywh + class scores.

detect(frame_bgr) -> list[Detection], in ORIGINAL FRAME pixel coordinates.
"""

import os
import numpy as np
import cv2

# Log to file since we're running as a systemd service
_LOGFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector.log")
def _log(msg):
    print(msg)
    try:
        with open(_LOGFILE, "a") as f:
            f.write(msg + "\n")
    except:
        pass


class Detection:
    __slots__ = ("x1", "y1", "x2", "y2", "label", "score")

    def __init__(self, x1, y1, x2, y2, label, score):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.label, self.score = label, score


class Detector:
    def __init__(self, cfg, model_path=None):
        self.cfg = cfg
        self.ok = False
        self.net = None

        _log(f"[detector] OpenCV version: {cv2.__version__}")

        tflite_path = model_path or cfg.MODEL_PATH

        if not os.path.exists(tflite_path):
            _log(f"[detector] Model not found: {tflite_path}")
            return

        candidates = [tflite_path]
        _log(f"[detector] Found TFLite model: {tflite_path}")

        import traceback
        for model_path in candidates:
            try:
                self._load(model_path)
                self.ok = True
                _log(f"[detector] Loaded {model_path}  input={self.in_w}x{self.in_h}")
                break
            except Exception as e:
                _log(f"[detector] Failed to load {model_path}: {e}")
                _log(traceback.format_exc())
        else:
            _log("[detector] All model candidates failed. Detection DISABLED.")

    def _download_model(self, dest_path):
        """Download YOLOv8n ONNX model."""
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx"
        try:
            import urllib.request
            _log(f"[detector] Downloading from {url}")
            urllib.request.urlretrieve(url, dest_path)
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            _log(f"[detector] Downloaded {size_mb:.1f} MB")
            return True
        except Exception as e:
            _log(f"[detector] Download failed: {e}")
            return False

    def _load(self, path):
        if not path.endswith(".tflite"):
            raise ValueError(f"Unsupported model format: {path}")

        # Support both tflite_runtime (legacy) and ai_edge_litert (Python 3.13+)
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from ai_edge_litert.interpreter import Interpreter

        _log("[detector] Loading TFLite model...")
        self._interp = Interpreter(model_path=path)
        self._interp.allocate_tensors()
        self._inp_idx = self._interp.get_input_details()[0]["index"]
        self._out_idx = self._interp.get_output_details()[0]["index"]
        self.in_w, self.in_h = self.cfg.MODEL_INPUT_SIZE
        _log("[detector] TFLite model ready")

    def detect(self, frame_bgr):
        if not self.ok:
            return []

        h0, w0 = frame_bgr.shape[:2]
        img, scale, padx, pady = self._letterbox(frame_bgr, self.in_w, self.in_h)

        # TFLite expects NHWC (1, H, W, 3), RGB, float32 normalized to [0,1]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = rgb[np.newaxis]
        self._interp.set_tensor(self._inp_idx, inp)
        self._interp.invoke()
        out = self._interp.get_tensor(self._out_idx)

        try:
            return self._postprocess(out, w0, h0, scale, padx, pady)
        except Exception as e:
            import traceback
            _log(f"[detector] postprocess error (output shape {out.shape}): {e}")
            _log(traceback.format_exc())
            return []

    def _letterbox(self, img, tw, th):
        h, w = img.shape[:2]
        s = min(tw / w, th / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((th, tw, 3), 114, np.uint8)
        padx, pady = (tw - nw) // 2, (th - nh) // 2
        canvas[pady:pady + nh, padx:padx + nw] = resized
        return canvas, s, padx, pady

    def _postprocess(self, out, w0, h0, scale, padx, pady):
        arr = out[0]

        if not getattr(self, '_shape_logged', False):
            _log(f"[detector] output tensor shape: {out.shape}  dtype={out.dtype}")
            self._shape_logged = True

        # Detect output format by shape.
        #
        # SSD / TF Detection API:  (N, 6)  →  [y1, x1, y2, x2, score, cls] normalized
        # YOLOv8 standard:         (N, 4+nc) →  [cx, cy, w, h, cls0, …] in model-px
        # YOLOv8 transposed:       (4+nc, N) →  same, needs .T first
        rows, cols = arr.shape

        if cols == 6:
            return self._decode_ssd(arr, w0, h0, scale, padx, pady)

        # YOLOv8: orient so rows=proposals, cols=features
        if rows < cols:
            arr = arr.T          # (4+nc, N) → (N, 4+nc)

        return self._decode_yolo(arr, w0, h0, scale, padx, pady)

    def _decode_ssd(self, arr, w0, h0, scale, padx, pady):
        """SSD / TF Detection API: (N, 6) [y1, x1, y2, x2, score, cls] normalized."""
        scores = arr[:, 4]
        arr = arr[scores >= self.cfg.CONF_THRESHOLD]
        if len(arr) == 0:
            return []
        dets = []
        for y1, x1, y2, x2, score, cls_id in arr:
            px1 = np.clip(x1 * self.in_w, 0, self.in_w)
            py1 = np.clip(y1 * self.in_h, 0, self.in_h)
            px2 = np.clip(x2 * self.in_w, 0, self.in_w)
            py2 = np.clip(y2 * self.in_h, 0, self.in_h)
            ox1 = int(np.clip((px1 - padx) / scale, 0, w0 - 1))
            oy1 = int(np.clip((py1 - pady) / scale, 0, h0 - 1))
            ox2 = int(np.clip((px2 - padx) / scale, 0, w0 - 1))
            oy2 = int(np.clip((py2 - pady) / scale, 0, h0 - 1))
            dets.append(Detection(ox1, oy1, ox2, oy2,
                                  self.cfg.LABELS.get(int(cls_id), str(int(cls_id))),
                                  float(score)))
        if dets:
            _log(f"[detector] {len(dets)} detections")
        return dets

    def _decode_yolo(self, arr, w0, h0, scale, padx, pady):
        """YOLOv8: (N, 4+nc) [cx, cy, w, h, cls0, …] in model-input pixels."""
        nc = arr.shape[1] - 4
        box_xy = arr[:, :4]          # cx, cy, w, h
        cls_scores = arr[:, 4:]      # (N, nc)

        if nc == 1:
            scores = cls_scores[:, 0]
            cls_ids = np.zeros(len(arr), dtype=int)
        else:
            cls_ids = np.argmax(cls_scores, axis=1)
            scores = cls_scores[np.arange(len(arr)), cls_ids]

        keep = scores >= self.cfg.CONF_THRESHOLD
        if not keep.any():
            return []

        cx, cy, w, h = box_xy[keep, 0], box_xy[keep, 1], box_xy[keep, 2], box_xy[keep, 3]
        scores = scores[keep]
        cls_ids = cls_ids[keep]

        # cx/cy/w/h are in model-input pixel space; map back to original frame
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        dets = []
        for i in range(len(x1)):
            ox1 = int(np.clip((x1[i] - padx) / scale, 0, w0 - 1))
            oy1 = int(np.clip((y1[i] - pady) / scale, 0, h0 - 1))
            ox2 = int(np.clip((x2[i] - padx) / scale, 0, w0 - 1))
            oy2 = int(np.clip((y2[i] - pady) / scale, 0, h0 - 1))
            dets.append(Detection(ox1, oy1, ox2, oy2,
                                  self.cfg.LABELS.get(int(cls_ids[i]), str(int(cls_ids[i]))),
                                  float(scores[i])))
        if dets:
            _log(f"[detector] {len(dets)} detections")
        return dets
