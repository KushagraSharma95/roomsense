"""Where status updates and decisions go. Every sink gets the same two calls."""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Protocol

from ..control import Decision


class Sink(Protocol):
    def publish_status(self, status: dict) -> None: ...
    def publish_decision(self, decision: Decision) -> None: ...
    def close(self) -> None: ...


class ConsoleSink:
    def __init__(self, print_fn=print) -> None:
        self.print = print_fn

    def publish_status(self, status: dict) -> None:
        pass  # status is too chatty for a terminal; decisions are printed

    def publish_decision(self, d: Decision) -> None:
        held = f" (held {d.deferred_s:.0f} s)" if d.deferred_s else ""
        self.print(f"[{d.t:9.1f}s] {d.room_id}: {d.action.target} -> {d.action.value}{held}  because {d.reason}")

    def close(self) -> None:
        pass


class JsonlSink:
    """Appends every decision as one JSON line: an audit log of what the room did and why."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", encoding="utf-8")

    def publish_status(self, status: dict) -> None:
        pass

    def publish_decision(self, d: Decision) -> None:
        self._f.write(json.dumps(d.as_dict()) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class SharedState:
    """Latest status, recent decisions and the latest preview frame, shared with the HTTP API."""

    def __init__(self, max_decisions: int = 200) -> None:
        self._lock = threading.Lock()
        self.status: dict = {}
        self.decisions: deque[dict] = deque(maxlen=max_decisions)
        self.preview_jpeg: bytes | None = None
        self.version = 0

    def publish_status(self, status: dict) -> None:
        with self._lock:
            self.status = status
            self.version += 1

    def publish_decision(self, d: Decision) -> None:
        with self._lock:
            self.decisions.append(d.as_dict())

    def set_preview(self, jpeg: bytes) -> None:
        with self._lock:
            self.preview_jpeg = jpeg

    def snapshot(self) -> tuple[int, dict, list[dict]]:
        with self._lock:
            return self.version, dict(self.status), list(self.decisions)

    def close(self) -> None:
        pass
