"""One loop step: detections -> tracks -> floor positions and zones -> room state -> device decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .adapters.sinks import Sink
from .audio import SpeechActivity
from .config import RoomConfig
from .control import ControlPolicy, Decision
from .fusion import RoomState, RoomStateEstimator, StateChange
from .spatial import FloorMapper, ZoneMap
from .vision.detector import Detection, Detector, frame_quality
from .vision.tracker import IouTracker, Track


@dataclass
class StepResult:
    t: float
    tracks: list[Track]
    positions: dict[int, tuple[float, float]]  # floor coordinates of people inside the room
    zones: dict[str, list[int]]
    state: RoomState
    people: int
    speech_ratio: float | None
    vision_ok: bool
    vision_note: str
    change: StateChange | None = None
    decisions: list[Decision] = field(default_factory=list)


class RoomPipeline:
    def __init__(
        self,
        cfg: RoomConfig,
        detector: Detector | None = None,
        sinks: list[Sink] | None = None,
        speech: SpeechActivity | None = None,
        status_every_s: float = 1.0,
    ) -> None:
        self.cfg = cfg
        self.detector = detector
        self.sinks = sinks or []
        self.speech = speech
        self.mapper = FloorMapper(cfg.camera)
        self.zone_map = ZoneMap(cfg.zones)
        self.tracker = IouTracker(cfg.tracker.iou_threshold, cfg.tracker.min_hits, cfg.tracker.max_missed)
        self.estimator = RoomStateEstimator(cfg.fusion, use_audio=cfg.audio.enabled)
        self.policy = ControlPolicy(cfg.room_id, cfg.policy.actions, cfg.policy.min_hold_s)
        self.status_every_s = status_every_s
        self._last_status = float("-inf")
        self._frame_idx = 0
        self._tracks: list[Track] = []

    def process_frame(self, t: float, frame_bgr: np.ndarray) -> StepResult:
        """Run detection on this frame (or reuse the last tracks on skipped frames) and update the room."""
        ok, note = frame_quality(frame_bgr)
        run_detector = self._frame_idx % self.cfg.detector.frame_stride == 0
        self._frame_idx += 1
        if not run_detector:
            return self._update(t, self._tracks, ok, note)
        detections = self.detector.detect(frame_bgr) if (ok and self.detector) else []
        return self.process_detections(t, detections, ok, note)

    def process_detections(
        self, t: float, detections: list[Detection], vision_ok: bool = True, vision_note: str = "ok"
    ) -> StepResult:
        self._tracks = self.tracker.update(detections if vision_ok else [])
        return self._update(t, self._tracks, vision_ok, vision_note)

    def _update(self, t: float, tracks: list[Track], vision_ok: bool, vision_note: str) -> StepResult:
        floor = self.mapper.to_floor([tr.foot for tr in tracks])
        all_pos = {tr.track_id: (float(x), float(y)) for tr, (x, y) in zip(tracks, floor, strict=True)}
        in_room = self.zone_map.people_in_room(all_pos)
        positions = {tid: all_pos[tid] for tid in in_room}
        zones = self.zone_map.assign(positions)
        zone_counts = {name: len(ids) for name, ids in zones.items() if name != ZoneMap.ROOM}
        speech = self.speech.ratio(t) if self.speech is not None else None

        change = self.estimator.update(t, len(positions), zone_counts, speech, vision_ok)
        decisions = self.policy.on_state_change(change) if change else []
        decisions += self.policy.tick(t)
        for d in decisions:
            for s in self.sinks:
                s.publish_decision(d)

        ev = self.estimator.last_evidence
        result = StepResult(t, tracks, positions, zones, self.estimator.state, ev.people if ev else 0,
                            speech, vision_ok, vision_note, change, decisions)
        if t - self._last_status >= self.status_every_s:
            self._last_status = t
            status = self.status_dict(result)
            for s in self.sinks:
                s.publish_status(status)
        return result

    def status_dict(self, r: StepResult) -> dict:
        return {
            "room": self.cfg.room_id,
            "t": round(r.t, 3),
            "state": r.state.value,
            "people": r.people,
            "zones": {k: len(v) for k, v in r.zones.items() if k != ZoneMap.ROOM},
            "speech_ratio": None if r.speech_ratio is None else round(r.speech_ratio, 3),
            "vision_ok": r.vision_ok,
            "vision_note": r.vision_note,
            "devices": dict(self.policy.current),
        }

    def close(self) -> None:
        for s in self.sinks:
            s.close()


STATE_COLOURS = {  # BGR
    RoomState.VACANT: (120, 120, 120),
    RoomState.OCCUPIED: (60, 180, 75),
    RoomState.CONVERSATION: (200, 130, 0),
}


def draw_overlay(frame_bgr: np.ndarray, r: StepResult, pipeline: RoomPipeline, blur: bool = True) -> np.ndarray:
    """Annotated copy of the frame. People are blurred by default so previews and recordings stay anonymous."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    if blur:
        for tr in r.tracks:
            x1, y1, x2, y2 = (int(v) for v in tr.box)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 > 4 and y2 - y1 > 4:
                roi = out[y1:y2, x1:x2]
                k = max(3, (min(roi.shape[:2]) // 3) | 1)
                out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

    for z in pipeline.zone_map.zones:
        pts = pipeline.mapper.to_image(z.polygon).astype(np.int32)
        colour = (255, 255, 255) if z.name == ZoneMap.ROOM else (0, 200, 255)
        cv2.polylines(out, [pts], True, colour, 1, cv2.LINE_AA)
        if z.name != ZoneMap.ROOM:
            n = len(r.zones.get(z.name, []))
            x, y = pts.mean(axis=0).astype(int)
            cv2.putText(out, f"{z.name}: {n}", (int(x) - 30, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colour, 1, cv2.LINE_AA)

    colour = STATE_COLOURS[r.state]
    for tr in r.tracks:
        x1, y1, x2, y2 = (int(v) for v in tr.box)
        inside = tr.track_id in r.positions
        c = colour if inside else (150, 150, 150)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        cv2.putText(out, f"#{tr.track_id}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1,
                    cv2.LINE_AA)
        fx, fy = (int(v) for v in tr.foot)
        cv2.circle(out, (fx, fy), 3, c, -1)

    speech = "n/a" if r.speech_ratio is None else f"{r.speech_ratio:.0%}"
    lines = [f"{r.state.value.upper()}", f"people: {r.people}", f"speech: {speech}",
             f"camera: {r.vision_note}"]
    cv2.rectangle(out, (8, 8), (260, 18 + 20 * len(lines)), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (16, 28 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    colour if i == 0 else (255, 255, 255), 1, cv2.LINE_AA)
    return out
