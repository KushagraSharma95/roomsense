"""Small HTTP API: current status, recent decisions, a live WebSocket feed and a privacy-blurred preview.

Binds to 127.0.0.1 by default. Only expose it on the network if you mean to.
(No `from __future__ import annotations` here: FastAPI needs the real WebSocket type on the endpoint.)
"""

import asyncio
import threading

from .adapters.sinks import SharedState


def create_app(state: SharedState):
    from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect

    app = FastAPI(title="roomsense", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/status")
    def status() -> dict:
        _, st, _ = state.snapshot()
        return st

    @app.get("/decisions")
    def decisions(limit: int = 50) -> list[dict]:
        _, _, ds = state.snapshot()
        return ds[-max(1, min(limit, 200)):]

    @app.get("/preview.jpg")
    def preview() -> Response:
        jpeg = state.preview_jpeg
        if jpeg is None:
            return Response(status_code=204)
        return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        seen = -1
        try:
            while True:
                version, st, _ = state.snapshot()
                if version != seen:
                    await websocket.send_json(st)
                    seen = version
                await asyncio.sleep(0.25)
        except WebSocketDisconnect:
            return

    return app


def serve_in_background(state: SharedState, host: str = "127.0.0.1", port: int = 8080) -> threading.Thread:
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(create_app(state), host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return thread
