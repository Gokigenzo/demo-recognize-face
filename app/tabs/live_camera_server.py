"""WebSocket server for low-lag browser-based live camera streaming.

Handles incoming base64 JPEG frames, decodes them, runs them through the
attendance engine in a background thread executor, and returns bounding box
annotations and detection events.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import socket
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import websockets

LOGGER = logging.getLogger(__name__)

# Thread-safe global registry mapping session_id to (AttendanceSession, RealtimeAttendanceEngine)
_SESSION_REGISTRY: Dict[str, Tuple[object, object]] = {}
_REGISTRY_LOCK = threading.Lock()

# Global server instance tracking to prevent multiple server instances on rerun
_SERVER_STATE = {
    "started": False,
    "port": None,
    "thread": None,
    "loop": None,
    "exception": None
}
_STATE_LOCK = threading.Lock()


def register_session(session_id: str, session: object, engine: object) -> None:
    """Register an active attendance session and engine."""
    with _REGISTRY_LOCK:
        _SESSION_REGISTRY[session_id] = (session, engine)
        LOGGER.debug("Registered session ID: %s", session_id)


def get_session(session_id: str) -> Optional[Tuple[object, object]]:
    """Retrieve the registered session and engine."""
    with _REGISTRY_LOCK:
        return _SESSION_REGISTRY.get(session_id)


def find_free_port(start_port: int = 8504, max_attempts: int = 20) -> int:
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise OSError(f"Could not find a free port in range {start_port} to {start_port + max_attempts - 1}")


class LiveCameraWebSocketServer:
    """WebSocket server running in a daemon thread to receive camera frames."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.server = None

    async def handler(self, websocket: websockets.WebSocketServerProtocol, path: str = "") -> None:
        """Handle incoming WebSocket connection."""
        session_id = "default"
        last_sync_time = 0.0
        LOGGER.info("New live-camera websocket connection established.")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                except ValueError:
                    continue

                msg_type = data.get("type")

                if msg_type == "handshake":
                    session_id = data.get("session_id", "default")
                    LOGGER.info("Websocket handshake received for session: %s", session_id)
                    await websocket.send(json.dumps({"status": "connected", "session_id": session_id}))
                    continue

                elif msg_type == "frame":
                    # Retrieve the active session & engine
                    reg = get_session(session_id)
                    if reg is None:
                        await websocket.send(json.dumps({
                            "status": "error",
                            "message": f"Session {session_id} not registered."
                        }))
                        continue

                    session, engine = reg

                    # Decode base64 frame
                    img_b64 = data.get("image")
                    if not img_b64:
                        continue

                    try:
                        img_bytes = base64.b64decode(img_b64)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    except Exception as exc:
                        LOGGER.warning("Failed to decode frame from base64: %s", exc)
                        continue

                    if frame is None or frame.size == 0:
                        continue

                    # Run ML inference in thread pool executor to avoid blocking the event loop
                    loop = asyncio.get_running_loop()
                    
                    prev_records_len = len(session.records)
                    
                    try:
                        # process_photo runs detection, tracking, classifier prediction, & updates session state
                        annotated_frame, predictions, annotations = await loop.run_in_executor(
                            None,
                            engine.process_photo,
                            frame
                        )
                    except Exception as exc:
                        LOGGER.exception("Error running face recognition inside executor:")
                        continue

                    # Detect if a student was newly confirmed
                    has_new_record = len(session.records) > prev_records_len
                    
                    # Throttle UI synchronizations to prevent Streamlit from running page updates too fast
                    now = time.time()
                    sync_ui = False
                    if has_new_record:
                        sync_ui = True
                        last_sync_time = now
                    elif now - last_sync_time >= 1.5:
                        sync_ui = True
                        last_sync_time = now

                    # Format annotations for client drawing
                    serialized_annotations = []
                    for face in annotations:
                        serialized_annotations.append({
                            "bbox": [int(v) for v in face.bbox],
                            "confidence": float(face.confidence),
                            "label": str(face.label),
                            "status": str(face.status),
                            "color": [int(c) for c in face.color]
                        })

                    # Prepare and send response
                    response = {
                        "status": "success",
                        "annotations": serialized_annotations,
                        "sync_ui": sync_ui,
                        "event": "play_beep" if has_new_record else None
                    }
                    await websocket.send(json.dumps(response))

        except websockets.exceptions.ConnectionClosed:
            LOGGER.info("Live-camera websocket connection closed by client.")
        except Exception as exc:
            LOGGER.exception("Error in websocket handler:")
        finally:
            LOGGER.info("Websocket handler task terminated.")

    async def start(self) -> None:
        """Start the websockets server."""
        # Clean up any old server running on this port in this process loop
        try:
            self.server = await websockets.serve(
                self.handler,
                "0.0.0.0",
                self.port,
                ping_interval=10,
                ping_timeout=10,
                max_size=10 * 1024 * 1024 # max size 10MB to avoid frame cap limits
            )
            LOGGER.info("WebSocket server started on port %d", self.port)
        except Exception as exc:
            LOGGER.exception("Failed to start websockets server on port %d:", self.port)
            raise

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            LOGGER.info("WebSocket server stopped.")


def _run_server_thread(port: int, loop: asyncio.AbstractEventLoop) -> None:
    """Run the asyncio event loop inside the background thread."""
    asyncio.set_event_loop(loop)
    server = LiveCameraWebSocketServer(port)
    
    try:
        loop.run_until_complete(server.start())
        with _STATE_LOCK:
            _SERVER_STATE["started"] = True
            _SERVER_STATE["port"] = port
            _SERVER_STATE["loop"] = loop
            _SERVER_STATE["exception"] = None
            
        loop.run_forever()
    except BaseException as exc:
        LOGGER.exception("Error in background WebSocket server thread:")
        with _STATE_LOCK:
            _SERVER_STATE["exception"] = exc
    finally:
        # Cleanup state on exit
        with _STATE_LOCK:
            _SERVER_STATE["started"] = False
            _SERVER_STATE["port"] = None
            _SERVER_STATE["thread"] = None
            _SERVER_STATE["loop"] = None
        LOGGER.info("Background WebSocket server thread finished.")


def start_server_background(default_port: int = 8504) -> int:
    """Starts the WebSocket server in a background thread if it is not already running.

    Returns the port number on which the server is listening.
    """
    with _STATE_LOCK:
        if _SERVER_STATE["started"]:
            LOGGER.info("WebSocket server already running on port %d", _SERVER_STATE["port"])
            return _SERVER_STATE["port"]

        # Probe for a free port
        port = find_free_port(default_port)
        
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=_run_server_thread,
            args=(port, loop),
            name="LiveCameraWebSocketServerThread",
            daemon=True
        )
        _SERVER_STATE["thread"] = thread
        thread.start()
        
        # Wait a moment for server to initialize
        retries = 10
        while retries > 0:
            if _SERVER_STATE["started"]:
                break
            time.sleep(0.05)
            retries -= 1

        return port
