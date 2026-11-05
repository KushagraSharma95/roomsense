"""CPU latency of the person detector, the number that decides what hardware a room needs."""

from __future__ import annotations

import time

import numpy as np

from .vision.detector import YoloxDetector


def benchmark(detector: YoloxDetector, frames: list[np.ndarray], runs: int = 100, warmup: int = 10) -> dict:
    for i in range(warmup):
        detector.detect(frames[i % len(frames)])
    times = []
    people = []
    for i in range(runs):
        frame = frames[i % len(frames)]
        start = time.perf_counter()
        dets = detector.detect(frame)
        times.append((time.perf_counter() - start) * 1000)
        people.append(len(dets))
    ms = np.array(times)
    return {
        "runs": runs,
        "input": f"{detector.input_size[1]}x{detector.input_size[0]}",
        "frame": f"{frames[0].shape[1]}x{frames[0].shape[0]}",
        "mean_ms": round(float(ms.mean()), 1),
        "p50_ms": round(float(np.percentile(ms, 50)), 1),
        "p95_ms": round(float(np.percentile(ms, 95)), 1),
        "fps": round(1000.0 / float(ms.mean()), 1),
        "avg_people": round(float(np.mean(people)), 2),
    }
