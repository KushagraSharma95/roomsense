"""Person detectors.

YoloxDetector runs a COCO-trained YOLOX model with ONNX Runtime on the CPU and keeps only the
"person" class. MotionDetector needs no model: it finds moving blobs with background subtraction.
It is here as a baseline, because it fails the same way a PIR sensor does (still people disappear).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

PERSON_CLASS = 0  # COCO index for "person"


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in image pixels
    score: float

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre of the box: the point where the person meets the floor."""
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y2)


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression. boxes: (N, 4) as x1, y1, x2, y2."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def letterbox(frame_bgr: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Resize keeping aspect ratio, pad with grey (114) to `size` (h, w), return CHW float32 and scale.

    This matches the preprocessing the official YOLOX ONNX models expect (BGR, 0-255, no normalisation).
    """
    h, w = size
    r = min(h / frame_bgr.shape[0], w / frame_bgr.shape[1])
    nh, nw = int(frame_bgr.shape[0] * r), int(frame_bgr.shape[1] * r)
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((h, w, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), r


def make_grids(size: tuple[int, int], strides: tuple[int, ...] = (8, 16, 32)) -> tuple[np.ndarray, np.ndarray]:
    grids, expanded = [], []
    for s in strides:
        hs, ws = size[0] // s, size[1] // s
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
        grids.append(grid)
        expanded.append(np.full((grid.shape[0], 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(expanded).astype(np.float32)


def decode_yolox(
    raw: np.ndarray,
    ratio: float,
    grids: np.ndarray,
    strides: np.ndarray,
    score_threshold: float,
    nms_threshold: float,
    image_shape: tuple[int, int] | None = None,
    class_id: int = PERSON_CLASS,
) -> list[Detection]:
    """Turn raw YOLOX head output (N, 5 + classes) into person detections in original image pixels."""
    scores = raw[:, 4] * raw[:, 5 + class_id]  # objectness and class score are already sigmoid-ed
    keep = scores >= score_threshold
    if not np.any(keep):
        return []
    xy = (raw[keep, :2] + grids[keep]) * strides[keep]
    wh = np.exp(raw[keep, 2:4]) * strides[keep]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1) / ratio
    if image_shape is not None:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_shape[1] - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_shape[0] - 1)
    scores = scores[keep]
    idx = nms(boxes, scores, nms_threshold)
    return [Detection(tuple(float(v) for v in boxes[i]), float(scores[i])) for i in idx]


class YoloxDetector:
    def __init__(
        self,
        model_path: str | Path,
        score_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        threads: int = 2,
    ) -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"model not found: {path}. Download it with: roomsense models download nano")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = (int(inp.shape[2]), int(inp.shape[3]))
        self.grids, self.strides = make_grids(self.input_size)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        tensor, ratio = letterbox(frame_bgr, self.input_size)
        raw = self.session.run(None, {self.input_name: tensor[None]})[0][0]
        return decode_yolox(
            raw, ratio, self.grids, self.strides, self.score_threshold, self.nms_threshold, frame_bgr.shape[:2]
        )


class MotionDetector:
    """Moving-blob detector (MOG2 background subtraction). Cheap, no model, but blind to people sitting still."""

    def __init__(self, min_area: int = 1500, history: int = 300) -> None:
        self.bg = cv2.createBackgroundSubtractorMOG2(history=history, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.min_area = min_area

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        mask = self.bg.apply(frame_bgr)
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]  # drop shadows (127)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            out.append(Detection((float(x), float(y), float(x + w), float(y + h)), 1.0))
        return out


def frame_quality(frame_bgr: np.ndarray, min_brightness: float = 20.0, min_detail: float = 15.0) -> tuple[bool, str]:
    """Is the picture usable? Catches a covered lens, lights off, or a dead feed."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (160, 120))
    brightness = float(small.mean())
    detail = float(cv2.Laplacian(small, cv2.CV_64F).var())
    if brightness < min_brightness:
        return False, f"too dark (mean {brightness:.0f})"
    if detail < min_detail:
        return False, f"no detail (lens covered or out of focus, laplacian var {detail:.0f})"
    return True, "ok"
