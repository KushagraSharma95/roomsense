"""Turn room-state changes into device commands, each with a record of why it happened."""

from __future__ import annotations

from dataclasses import dataclass, field

from .fusion import StateChange


@dataclass(frozen=True)
class Action:
    target: str  # e.g. "lights", "av", "hvac"
    value: str  # e.g. "meeting", "standby", "setback"

    @property
    def key(self) -> str:
        return f"{self.target}.{self.value}"


@dataclass
class Decision:
    """One command plus the evidence behind it, so any automatic change can be explained later."""

    t: float
    room_id: str
    action: Action
    state: str
    reason: str
    evidence: dict = field(default_factory=dict)
    deferred_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "t": round(self.t, 3),
            "room": self.room_id,
            "target": self.action.target,
            "value": self.action.value,
            "state": self.state,
            "reason": self.reason,
            "deferred_s": round(self.deferred_s, 1),
            "evidence": self.evidence,
        }


class ControlPolicy:
    """Maps each room state to device settings.

    A device is only commanded when its setting actually changes, and never twice within
    `min_hold_s`. A change that arrives too soon is held and sent once the hold expires, unless
    the room has moved on again by then (then the newer setting wins).
    """

    def __init__(self, room_id: str, actions: dict[str, dict[str, str]], min_hold_s: float = 30.0) -> None:
        self.room_id = room_id
        self.actions = actions
        self.min_hold_s = min_hold_s
        self.current: dict[str, str] = {}
        self.last_sent: dict[str, float] = {}
        self.pending: dict[str, Decision] = {}

    def on_state_change(self, change: StateChange) -> list[Decision]:
        out = []
        for target, value in self.actions.get(change.new.value, {}).items():
            d = Decision(change.t, self.room_id, Action(target, value), change.new.value, change.reason,
                         change.evidence.as_dict())
            if self.current.get(target) == value:
                self.pending.pop(target, None)  # already there; drop any queued change
                continue
            last = self.last_sent.get(target)
            if last is not None and change.t - last < self.min_hold_s:
                self.pending[target] = d
                continue
            self.pending.pop(target, None)
            out.append(self._send(d))
        return out

    def tick(self, t: float) -> list[Decision]:
        """Release held changes whose hold time has passed. Call this regularly."""
        out = []
        for target, d in list(self.pending.items()):
            if t - self.last_sent.get(target, float("-inf")) >= self.min_hold_s:
                del self.pending[target]
                d.deferred_s = t - d.t
                d.t = t
                out.append(self._send(d))
        return out

    def _send(self, d: Decision) -> Decision:
        self.current[d.action.target] = d.action.value
        self.last_sent[d.action.target] = d.t
        return d
