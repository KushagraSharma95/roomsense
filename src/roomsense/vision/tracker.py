"""A small multi-person tracker: Hungarian matching on IoU, a second pass on centre distance for fast
movers, and a constant-velocity guess to bridge short occlusions.

Good enough to keep IDs in a room. It is not a re-identification model: someone who leaves the frame
and comes back gets a new ID.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detection


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N, 4) and b (M, 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / (area_a + area_b - inter + 1e-9)


def centre_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance between box centres, in units of the height of the boxes in `a`."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ca = np.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2], axis=1)
    cb = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
    height = (a[:, 3] - a[:, 1]).clip(1.0)[:, None]
    return np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2) / height


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    score: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    hits: int = 1
    missed: int = 0
    confirmed: bool = False

    @property
    def foot(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.box
        return (float(x1 + x2) / 2.0, float(y2))


class IouTracker:
    def __init__(
        self,
        iou_threshold: float = 0.25,
        min_hits: int = 3,
        max_missed: int = 15,
        report_missed: int | None = None,
        max_centre_distance: float = 0.6,
        tentative_max_missed: int = 2,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_missed = max_missed  # frames a lost track is kept alive, so it can be picked up again
        # frames a lost track still counts as a person; shorter than max_missed so people who walked
        # out of view stop being counted quickly
        self.report_missed = max(1, max_missed // 3) if report_missed is None else report_missed
        self.max_centre_distance = max_centre_distance
        # A new track survives a couple of missed frames while it collects its first hits, so one missed
        # detection doesn't restart confirmation. A one-frame false alarm still never reaches min_hits.
        self.tentative_max_missed = tentative_max_missed
        self.tracks: list[Track] = []
        self._next_id = 1

    def _match(self, predicted: np.ndarray, dets: np.ndarray) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        if not len(predicted) or not len(dets):
            return pairs
        iou = iou_matrix(predicted, dets)
        rows, cols = linear_sum_assignment(-iou)
        pairs = [(r, c) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= self.iou_threshold]
        # Second pass for whatever is left: fast movers at low frame rates overlap too little for IoU.
        free_t = [i for i in range(len(predicted)) if i not in {r for r, _ in pairs}]
        free_d = [j for j in range(len(dets)) if j not in {c for _, c in pairs}]
        if free_t and free_d:
            dist = centre_distance(predicted[free_t], dets[free_d])
            rows, cols = linear_sum_assignment(dist)
            pairs += [
                (free_t[r], free_d[c])
                for r, c in zip(rows, cols, strict=True)
                if dist[r, c] <= self.max_centre_distance
            ]
        return pairs

    def update(self, detections: list[Detection]) -> list[Track]:
        """Feed one frame of detections; returns the tracks that currently count as people."""
        dets = np.array([d.box for d in detections], dtype=float).reshape(-1, 4)
        predicted = np.array([t.box + t.velocity for t in self.tracks], dtype=float).reshape(-1, 4)

        pairs = self._match(predicted, dets)
        matched_t = {r for r, _ in pairs}
        matched_d = {c for _, c in pairs}
        for r, c in pairs:
            t = self.tracks[r]
            t.velocity = 0.6 * t.velocity + 0.4 * (dets[c] - t.box)
            t.box = dets[c].copy()
            t.score = detections[c].score
            t.hits += 1
            t.missed = 0
            t.confirmed = t.confirmed or t.hits >= self.min_hits

        survivors = []
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.missed += 1
                t.box = predicted[i]
                t.velocity *= 0.5  # coast, but slow down so a lost track doesn't drift across the room
            if t.missed <= (self.max_missed if t.confirmed else self.tentative_max_missed):
                survivors.append(t)
        self.tracks = survivors

        for j, d in enumerate(detections):
            if j not in matched_d:
                self.tracks.append(Track(self._next_id, dets[j].copy(), d.score, confirmed=self.min_hits <= 1))
                self._next_id += 1

        visible = [t for t in self.tracks if t.confirmed and t.missed == 0]
        coasting = [t for t in self.tracks if t.confirmed and 0 < t.missed <= self.report_missed]
        if visible and coasting:
            # a coasting track sitting on top of a visible one is usually the same person after an ID switch
            overlap = iou_matrix(np.array([t.box for t in coasting]), np.array([t.box for t in visible]))
            coasting = [t for t, row in zip(coasting, overlap, strict=True) if row.max() < 0.3]
        return visible + coasting
