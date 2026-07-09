"""Starlette WebSocket server route injector for low-lag live camera streaming.

Locates the running Streamlit server in memory via garbage collection and
injects a custom WebSocket endpoint (`/livecamws`) directly into Starlette's
routing table. This allows streaming frames on the same port (8501/443),
bypassing firewalls and Mixed Content blocks, while eliminating continuous reruns.
"""
from __future__ import annotations

import asyncio
import base64
import gc
import json
import logging
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from starlette.websockets import WebSocket

LOGGER = logging.getLogger(__name__)

# Thread-safe global registry mapping session_id to (AttendanceSession, RealtimeAttendanceEngine)
_SESSION_REGISTRY: Dict[str, Tuple[object, object]] = {}
_REGISTRY_LOCK = threading.Lock()


def register_session(session_id: str, session: object, engine: object) -> None:
    """Register an active attendance session and engine."""
    with _REGISTRY_LOCK:
        _SESSION_REGISTRY[session_id] = (session, engine)
        LOGGER.debug("Registered session ID: %s", session_id)


def get_session(session_id: str) -> Optional[Tuple[object, object]]:
    """Retrieve the registered session and engine."""
    with _REGISTRY_LOCK:
        return _SESSION_REGISTRY.get(session_id)


async def livecam_websocket_endpoint(websocket: WebSocket) -> None:
    """Starlette WebSocket connection handler."""
    await websocket.accept()
    session_id = "default"
    last_sync_time = 0.0
    LOGGER.info("LiveCam Starlette WebSocket connection accepted.")

    try:
        while True:
            # Receive frame or handshake
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
            except ValueError:
                continue

            msg_type = data.get("type")

            if msg_type == "handshake":
                session_id = data.get("session_id", "default")
                LOGGER.info("WebSocket handshake for session: %s", session_id)
                await websocket.send_text(json.dumps({"status": "connected", "session_id": session_id}))
                continue

            elif msg_type == "frame":
                reg = get_session(session_id)
                if reg is None:
                    await websocket.send_text(json.dumps({
                        "status": "error",
                        "message": f"Session {session_id} not registered."
                    }))
                    continue

                session, engine = reg

                img_b64 = data.get("image")
                if not img_b64:
                    continue

                try:
                    img_bytes = base64.b64decode(img_b64)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception as exc:
                    LOGGER.warning("Failed to decode base64 image: %s", exc)
                    continue

                if frame is None or frame.size == 0:
                    continue

                loop = asyncio.get_running_loop()
                prev_records_len = len(session.records)

                try:
                    # Run CPU-bound ML in executor
                    annotated_frame, predictions, annotations = await loop.run_in_executor(
                        None,
                        engine.process_photo,
                        frame
                    )
                except Exception as exc:
                    LOGGER.exception("Error processing frame in ML engine:")
                    continue

                has_new_record = len(session.records) > prev_records_len

                # Throttle UI reruns: only sync metrics every 1.5 seconds, or immediately on check-in
                now = time.time()
                sync_ui = False
                if has_new_record:
                    sync_ui = True
                    last_sync_time = now
                elif now - last_sync_time >= 1.5:
                    sync_ui = True
                    last_sync_time = now

                # Serialize annotations
                serialized_annotations = []
                for face in annotations:
                    serialized_annotations.append({
                        "bbox": [int(v) for v in face.bbox],
                        "confidence": float(face.confidence),
                        "label": str(face.label),
                        "status": str(face.status),
                        "color": [int(c) for c in face.color]
                    })

                response = {
                    "status": "success",
                    "annotations": serialized_annotations,
                    "sync_ui": sync_ui,
                    "event": "play_beep" if has_new_record else None
                }
                await websocket.send_text(json.dumps(response))

    except Exception as exc:
        LOGGER.info("LiveCam WebSocket connection closed/terminated.")


def register_starlette_route() -> None:
    """Finds the active Starlette Server in memory and registers the /livecamws WebSocket route."""
    from streamlit.web.server.server import Server
    
    server = None
    # Traverse active objects in garbage collector to retrieve the running server instance
    for obj in gc.get_objects():
        if isinstance(obj, Server):
            server = obj
            break
            
    if server is None:
        LOGGER.warning("Streamlit Server instance not found in memory.")
        return
        
    if not hasattr(server, "_starlette_server") or server._starlette_server is None:
        LOGGER.warning("Starlette server wrapper is not initialized on Server.")
        return
        
    starlette_server = server._starlette_server
    if not hasattr(starlette_server, "_server") or starlette_server._server is None:
        LOGGER.warning("Uvicorn server is not started on UvicornServer.")
        return
        
    # Retrieve the Starlette Application instance
    app = starlette_server._server.config.app
    
    # Register the route if not already in routing table
    has_route = any(getattr(route, "path", None) == "/livecamws" for route in app.routes)
    if not has_route:
        from starlette.routing import WebSocketRoute
        # Insert at index 0 to override catch-all routes
        app.routes.insert(0, WebSocketRoute("/livecamws", livecam_websocket_endpoint))
        LOGGER.info("Successfully registered custom Starlette WebSocket route /livecamws!")
