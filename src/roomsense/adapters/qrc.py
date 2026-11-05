"""Q-SYS Remote Control (QRC) output.

QRC is the public JSON-RPC 2.0 protocol a Q-SYS Core accepts on TCP port 1710: each message is
JSON followed by a null byte, `Logon` authenticates, `NoOp` keeps the socket alive, and
`Control.Set` / `Component.Set` change named controls. This client covers that subset. It is
tested against the local mock in `qrc_mock.py`, not against a real Core.

QrcSink never blocks the video loop: decisions go onto a queue and one worker thread owns the
connection. The worker remembers the latest wanted setting per device and keeps retrying while the
Core is unreachable, so a room that went vacant during an outage still gets its lights switched off
once the Core is back.
"""

from __future__ import annotations

import itertools
import json
import logging
import queue
import socket
import threading
import time
from typing import Any

from ..config import QrcConfig
from ..control import Decision

log = logging.getLogger(__name__)

LOGON_REQUIRED = 10


class QrcError(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"QRC error {code}: {message}")
        self.code = code


class QrcClient:
    """Synchronous QRC client. Reconnects on demand and logs on again after every new connection."""

    def __init__(self, host: str, port: int = 1710, user: str = "", password: str = "", timeout: float = 5.0) -> None:
        self.host, self.port, self.timeout = host, port, timeout
        self.user, self.password = user, password
        self.sock: socket.socket | None = None
        self._buf = b""
        self._ids = itertools.count(1)
        self._lock = threading.RLock()
        self.notifications: list[dict] = []  # pushes such as EngineStatus

    @property
    def connected(self) -> bool:
        return self.sock is not None

    def connect(self) -> None:
        with self._lock:
            self.close()
            self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self._buf = b""
            if self.user:
                self._call_locked("Logon", {"User": self.user, "Password": self.password})

    def close(self) -> None:
        with self._lock:
            if self.sock is not None:
                try:
                    self.sock.close()
                finally:
                    self.sock = None

    def _send(self, msg: dict) -> None:
        assert self.sock is not None, "not connected"
        self.sock.sendall(json.dumps(msg).encode("utf-8") + b"\0")

    def _recv(self) -> dict:
        assert self.sock is not None, "not connected"
        while b"\0" not in self._buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("QRC connection closed by the Core")
            self._buf += chunk
        raw, self._buf = self._buf.split(b"\0", 1)
        return json.loads(raw.decode("utf-8"))

    def _call_locked(self, method: str, params: Any) -> Any:
        msg_id = next(self._ids)
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        while True:
            reply = self._recv()
            if reply.get("id") != msg_id:
                self.notifications.append(reply)  # unsolicited push, keep reading
                continue
            if "error" in reply:
                err = reply["error"]
                raise QrcError(int(err.get("code", -1)), str(err.get("message", "")))
            return reply.get("result")

    def call(self, method: str, params: Any = None) -> Any:
        with self._lock:
            if self.sock is None:
                self.connect()
            try:
                return self._call_locked(method, {} if params is None else params)
            except OSError:
                self.close()  # a broken socket must not be reused
                raise

    def logon(self, user: str, password: str) -> Any:
        self.user, self.password = user, password
        return self.call("Logon", {"User": user, "Password": password})

    def noop(self) -> Any:
        return self.call("NoOp", {})

    def status(self) -> dict:
        return self.call("StatusGet", 0)

    def control_set(self, name: str, value: Any, ramp: float | None = None) -> Any:
        params: dict[str, Any] = {"Name": name, "Value": value}
        if ramp is not None:
            params["Ramp"] = ramp
        return self.call("Control.Set", params)

    def control_get(self, names: list[str]) -> list[dict]:
        return self.call("Control.Get", names)

    def component_set(self, component: str, controls: list[dict]) -> Any:
        return self.call("Component.Set", {"Name": component, "Controls": controls})

    def component_get(self, component: str, controls: list[str]) -> dict:
        return self.call("Component.Get", {"Name": component, "Controls": [{"Name": c} for c in controls]})


class QrcSink:
    """Sends decisions to the Core using the action-to-control map from the config."""

    def __init__(
        self, cfg: QrcConfig, client: QrcClient | None = None, keepalive_s: float = 30.0, retry_s: float = 5.0
    ) -> None:
        self.cfg = cfg
        self.client = client or QrcClient(cfg.host, cfg.port, cfg.user, cfg.password)
        self.keepalive_s, self.retry_s = keepalive_s, retry_s
        self._queue: queue.Queue[tuple[str, str, dict]] = queue.Queue()
        self.desired: dict[str, tuple[str, dict]] = {}  # device -> (action key, control mapping)
        self.unsent: set[str] = set()  # devices whose desired setting hasn't reached the Core yet
        self._outstanding: dict[str, int] = {}  # decisions per device not yet delivered or rejected
        self._cond = threading.Condition()
        self._queued = 0
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="qrc-sink", daemon=True)
        self._worker.start()

    def publish_status(self, status: dict) -> None:
        pass

    def publish_decision(self, d: Decision) -> None:
        mapping = self.cfg.controls.get(d.action.key)
        if mapping is None:
            log.debug("no QRC mapping for %s", d.action.key)
            return
        with self._cond:
            self._queued += 1
        self._queue.put((d.action.target, d.action.key, mapping))

    def _apply(self, mapping: dict) -> None:
        if "component" in mapping:
            ctl = {"Name": mapping["control"], "Value": mapping["value"]}
            if "ramp" in mapping:
                ctl["Ramp"] = mapping["ramp"]
            self.client.component_set(mapping["component"], [ctl])
        else:
            self.client.control_set(mapping["control"], mapping["value"], mapping.get("ramp"))

    def _done(self, target: str) -> None:
        with self._cond:
            self._queued -= self._outstanding.pop(target, 0)
            self._cond.notify_all()

    def _run(self) -> None:
        last_io = time.monotonic()
        while not self._stop.is_set():
            try:
                # with work pending, only drain what is already queued; otherwise wait for work
                if self.unsent:
                    target, key, mapping = self._queue.get_nowait()
                else:
                    target, key, mapping = self._queue.get(timeout=min(self.keepalive_s, self.retry_s))
                self.desired[target] = (key, mapping)  # the newest wanted setting replaces older ones
                self.unsent.add(target)
                self._outstanding[target] = self._outstanding.get(target, 0) + 1
                continue
            except queue.Empty:
                pass
            if self.unsent:
                try:
                    for target in sorted(self.unsent):
                        key, mapping = self.desired[target]
                        try:
                            self._apply(mapping)
                        except QrcError as e:
                            if e.code == LOGON_REQUIRED:
                                raise
                            log.warning("QRC rejected %s (%s); check the control mapping", key, e)
                        self.unsent.discard(target)
                        self._done(target)
                    last_io = time.monotonic()
                except (OSError, QrcError, ValueError) as e:
                    log.warning("QRC Core unreachable or refused logon (%s); retrying in %.0f s", e, self.retry_s)
                    self.client.close()
                    self._stop.wait(self.retry_s)
            elif self.client.connected and time.monotonic() - last_io >= self.keepalive_s:
                try:
                    self.client.noop()
                    last_io = time.monotonic()
                except (OSError, QrcError, ValueError) as e:
                    log.warning("QRC keepalive failed: %s", e)
                    self.client.close()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every published decision has reached the Core or been rejected. For tests and shutdown."""
        with self._cond:
            return self._cond.wait_for(lambda: self._queued == 0, timeout)

    def close(self) -> None:
        self.flush(timeout=2.0)
        self._stop.set()
        self._worker.join(timeout=2.0)
        self.client.close()
