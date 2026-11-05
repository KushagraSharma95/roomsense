"""Q-SYS Remote Control (QRC) output.

QRC is the public JSON-RPC 2.0 protocol a Q-SYS Core accepts on TCP port 1710: each message is
JSON followed by a null byte, `Logon` authenticates, `NoOp` keeps the socket alive, and
`Control.Set` / `Component.Set` change named controls. This client covers that subset. It is
tested against the local mock in `qrc_mock.py`, not against a real Core.
"""

from __future__ import annotations

import itertools
import json
import logging
import socket
import threading
from typing import Any

from ..config import QrcConfig
from ..control import Decision

log = logging.getLogger(__name__)


class QrcError(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"QRC error {code}: {message}")
        self.code = code


class QrcClient:
    def __init__(self, host: str, port: int = 1710, timeout: float = 5.0) -> None:
        self.host, self.port, self.timeout = host, port, timeout
        self.sock: socket.socket | None = None
        self._buf = b""
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.notifications: list[dict] = []  # pushes such as EngineStatus

    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._buf = b""

    def close(self) -> None:
        if self.sock:
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

    def call(self, method: str, params: Any = None) -> Any:
        with self._lock:
            if self.sock is None:
                self.connect()
            msg_id = next(self._ids)
            self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": {} if params is None else params})
            while True:
                reply = self._recv()
                if reply.get("id") != msg_id:
                    self.notifications.append(reply)  # unsolicited push, keep reading
                    continue
                if "error" in reply:
                    err = reply["error"]
                    raise QrcError(int(err.get("code", -1)), str(err.get("message", "")))
                return reply.get("result")

    def logon(self, user: str, password: str) -> Any:
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
    """Sends each decision to the Core using the action-to-control map from the config."""

    def __init__(self, cfg: QrcConfig, client: QrcClient | None = None, keepalive_s: float = 30.0) -> None:
        self.cfg = cfg
        self.client = client or QrcClient(cfg.host, cfg.port)
        self.keepalive_s = keepalive_s
        self._stop = threading.Event()
        self._logged_on = False
        self._keepalive = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._keepalive.start()

    def _ensure_session(self) -> None:
        if self.client.sock is None:
            self._logged_on = False  # call() reconnects on its own; a new socket needs a new Logon
        if not self._logged_on and self.cfg.user:
            self.client.logon(self.cfg.user, self.cfg.password)
        self._logged_on = True

    def _keepalive_loop(self) -> None:
        while not self._stop.wait(self.keepalive_s):
            try:
                if self.client.sock is not None:
                    self.client.noop()
            except (OSError, QrcError) as e:
                log.warning("QRC keepalive failed: %s", e)
                self.client.close()

    def publish_status(self, status: dict) -> None:
        pass

    def publish_decision(self, d: Decision) -> None:
        mapping = self.cfg.controls.get(d.action.key)
        if mapping is None:
            log.debug("no QRC mapping for %s", d.action.key)
            return
        try:
            self._ensure_session()
            if "component" in mapping:
                ctl = {"Name": mapping["control"], "Value": mapping["value"]}
                if "ramp" in mapping:
                    ctl["Ramp"] = mapping["ramp"]
                self.client.component_set(mapping["component"], [ctl])
            else:
                self.client.control_set(mapping["control"], mapping["value"], mapping.get("ramp"))
        except (OSError, QrcError) as e:
            log.warning("QRC command %s failed: %s", d.action.key, e)
            self.client.close()

    def close(self) -> None:
        self._stop.set()
        self.client.close()
