"""A tiny stand-in for a Q-SYS Core that speaks the QRC subset roomsense uses. For tests and demos."""

from __future__ import annotations

import json
import socketserver
import threading
from typing import Any


class MockCore:
    def __init__(self, user: str = "", password: str = "") -> None:
        self.user, self.password = user, password
        self.controls: dict[str, Any] = {}
        self.components: dict[str, dict[str, Any]] = {}
        self.log: list[dict] = []  # every request received, for assertions
        self._server: socketserver.ThreadingTCPServer | None = None

    def start(self, host: str = "127.0.0.1", port: int = 0) -> tuple[str, int]:
        core = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                logged_on = not core.user
                self._push({"jsonrpc": "2.0", "method": "EngineStatus",
                            "params": {"State": "Active", "DesignName": "roomsense-mock", "IsEmulator": True}})
                buf = b""
                while True:
                    chunk = self.request.recv(65536)
                    if not chunk:
                        return
                    buf += chunk
                    while b"\0" in buf:
                        raw, buf = buf.split(b"\0", 1)
                        try:
                            msg = json.loads(raw.decode("utf-8"))
                        except ValueError:
                            parse_error = {"code": -32700, "message": "Parse error"}
                            self._push({"jsonrpc": "2.0", "id": None, "error": parse_error})
                            continue
                        core.log.append(msg)
                        logged_on, reply = core.handle(msg, logged_on)
                        if reply is not None:
                            self._push(reply)

            def _push(self, obj: dict) -> None:
                self.request.sendall(json.dumps(obj).encode("utf-8") + b"\0")

        socketserver.ThreadingTCPServer.allow_reuse_address = True
        self._server = socketserver.ThreadingTCPServer((host, port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self._server.server_address[0], self._server.server_address[1]

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    def handle(self, msg: dict, logged_on: bool) -> tuple[bool, dict | None]:
        method, params, msg_id = msg.get("method"), msg.get("params"), msg.get("id")

        def ok(result: Any = True) -> dict | None:
            return None if msg_id is None else {"jsonrpc": "2.0", "id": msg_id, "result": result}

        def err(code: int, message: str) -> dict:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

        if method == "Logon":
            if params.get("User") == self.user and params.get("Password") == self.password:
                return True, ok()
            return logged_on, err(10, "Logon required")
        if method == "NoOp":
            return logged_on, ok()
        if not logged_on:
            return logged_on, err(10, "Logon required")
        if method == "StatusGet":
            return logged_on, ok({"Platform": "mock", "State": "Active", "DesignName": "roomsense-mock",
                                  "IsEmulator": True, "Status": {"Code": 0, "String": "OK"}})
        if method == "Control.Set":
            self.controls[params["Name"]] = params["Value"]
            return logged_on, ok()
        if method == "Control.Get":
            missing = [n for n in params if n not in self.controls]
            if missing:
                return logged_on, err(8, f"Unknown control: {missing[0]}")
            return logged_on, ok([{"Name": n, "Value": self.controls[n]} for n in params])
        if method == "Component.Set":
            comp = self.components.setdefault(params["Name"], {})
            for c in params.get("Controls", []):
                comp[c["Name"]] = c["Value"]
            return logged_on, ok()
        if method == "Component.Get":
            comp = self.components.get(params["Name"])
            if comp is None:
                return logged_on, err(7, f"Unknown component name: {params['Name']}")
            names = [c["Name"] for c in params.get("Controls", [])]
            return logged_on, ok({"Name": params["Name"],
                                  "Controls": [{"Name": n, "Value": comp.get(n)} for n in names]})
        return logged_on, err(-32601, "Method not found")
