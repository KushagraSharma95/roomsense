"""Room state from vision and audio together.

The camera answers "how many people are here", the microphone answers "is anyone talking". Either
one alone fails in a familiar way: a camera misses someone behind a pillar or in the dark, and a
motion sensor (or a camera-only motion check) misses someone reading quietly. Here a person seen
by the camera OR sustained speech keeps the room occupied, and it only goes vacant after neither
has been observed for `vacant_after_s`. Speech alone can't switch an empty room on while the camera
can see. If the camera is blocked or too dark, the room falls back to audio with a longer timeout.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum

from .config import FusionConfig


class RoomState(str, Enum):
    VACANT = "vacant"
    OCCUPIED = "occupied"
    CONVERSATION = "conversation"


@dataclass
class Evidence:
    t: float
    people: int  # median over the smoothing window
    people_raw: int  # this frame
    zones: dict[str, int] = field(default_factory=dict)
    speech_ratio: float | None = None
    vision_ok: bool = True

    def as_dict(self) -> dict:
        d = asdict(self)
        if self.speech_ratio is not None:
            d["speech_ratio"] = round(self.speech_ratio, 3)
        d["t"] = round(self.t, 3)
        return d


@dataclass
class StateChange:
    t: float
    old: RoomState
    new: RoomState
    reason: str
    evidence: Evidence


class RoomStateEstimator:
    def __init__(self, cfg: FusionConfig | None = None, use_audio: bool = True) -> None:
        self.cfg = cfg or FusionConfig()
        self.use_audio = use_audio
        self.state = RoomState.VACANT
        self._counts: deque[tuple[float, int]] = deque()
        self._presence_since: float | None = None
        self._last_presence: float | None = None
        self._conv_since: float | None = None
        self._conv_last: float | None = None
        self.last_evidence: Evidence | None = None

    def _smoothed(self, t: float, count: int) -> int:
        self._counts.append((t, count))
        while self._counts and self._counts[0][0] < t - self.cfg.count_window_s:
            self._counts.popleft()
        return int(round(statistics.median(c for _, c in self._counts)))

    def update(
        self,
        t: float,
        people_raw: int,
        zones: dict[str, int] | None = None,
        speech_ratio: float | None = None,
        vision_ok: bool = True,
    ) -> StateChange | None:
        cfg = self.cfg
        speech = speech_ratio if self.use_audio else None
        people = self._smoothed(t, people_raw if vision_ok else 0)
        ev = Evidence(t, people, people_raw, dict(zones or {}), speech, vision_ok)
        self.last_evidence = ev

        seen = vision_ok and people > 0
        heard = speech is not None and speech >= cfg.speech_presence_ratio
        # Speech keeps an occupied room on (someone the camera can't see is still talking). It does not
        # switch an empty room on while the camera has a clear view and sees nobody: that is more likely
        # a TV, a speakerphone or voices in the corridor. If the camera is blocked or dark, speech counts.
        if self.state == RoomState.VACANT and vision_ok:
            heard = False
        if seen or heard:
            self._last_presence = t
            if self._presence_since is None:
                self._presence_since = t
        else:
            self._presence_since = None

        talking = (
            vision_ok
            and people >= cfg.conversation_min_people
            and speech is not None
            and speech >= cfg.conversation_speech_ratio
        )
        if talking:
            self._conv_last = t
            if self._conv_since is None:
                self._conv_since = t
        else:
            self._conv_since = None
            # Can't see, but people are still talking (say, lights down for a presentation):
            # keep an ongoing conversation going rather than guessing it ended.
            if self.state == RoomState.CONVERSATION and not vision_ok and heard:
                self._conv_last = t

        def why_present() -> str:
            parts = []
            if seen:
                parts.append(f"{people} {'person' if people == 1 else 'people'} in view")
            if heard:
                parts.append(f"speech in {speech:.0%} of the last window")
            return " and ".join(parts)

        new, reason = self.state, ""
        if self.state == RoomState.VACANT:
            if self._presence_since is not None and t - self._presence_since >= cfg.occupied_after_s:
                new, reason = RoomState.OCCUPIED, why_present()
        else:
            timeout = cfg.vacant_after_s if vision_ok else cfg.degraded_vacant_after_s
            idle = t - self._last_presence if self._last_presence is not None else float("inf")
            if idle >= timeout:
                cam = "" if vision_ok else " (camera unusable, audio-only timeout)"
                new, reason = RoomState.VACANT, f"nobody seen or heard for {idle:.0f} s{cam}"
            elif self.state == RoomState.OCCUPIED:
                if self._conv_since is not None and t - self._conv_since >= cfg.conversation_after_s:
                    new = RoomState.CONVERSATION
                    reason = f"{people} people in view and speech in {speech:.0%} of the last window"
            elif self.state == RoomState.CONVERSATION:
                quiet = t - self._conv_last if self._conv_last is not None else float("inf")
                if quiet >= cfg.conversation_end_s:
                    new, reason = RoomState.OCCUPIED, f"no conversation for {quiet:.0f} s"

        if new == self.state:
            return None
        change = StateChange(t, self.state, new, reason, ev)
        self.state = new
        return change
