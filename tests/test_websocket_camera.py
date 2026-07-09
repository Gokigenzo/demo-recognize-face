"""Unit tests for the Live Camera WebSocket Server and Registry."""
from __future__ import annotations

import asyncio
import socket
import time
import pytest
import websockets

from app.tabs.live_camera_server import (
    register_session,
    get_session,
    find_free_port,
    start_server_background,
    _SERVER_STATE
)
from ml.attendance_session import AttendanceSession
from ml.realtime_engine import RealtimeAttendanceEngine


def test_session_registry() -> None:
    """Test registering and retrieving sessions."""
    users = {"user_1": {"name": "Test User"}}
    session = AttendanceSession(users=users)
    engine = RealtimeAttendanceEngine(session=session)
    
    register_session("test_session_123", session, engine)
    
    reg = get_session("test_session_123")
    assert reg is not None
    assert reg[0] is session
    assert reg[1] is engine
    
    # Non-existent session
    assert get_session("non_existent_session") is None


def test_find_free_port() -> None:
    """Test finding a free port."""
    port = find_free_port(start_port=9900, max_attempts=5)
    assert port >= 9900
    
    # Verify we can bind to it
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))
        assert True


def test_start_server_background() -> None:
    """Test starting the WebSocket server in a background thread."""
    # Ensure server is not started under previous tests
    port = start_server_background(default_port=9950)
    assert port >= 9950

    # Wait for thread to start and set started to True
    for _ in range(40):
        if _SERVER_STATE["started"]:
            break
        time.sleep(0.05)

    print("TEST SERVER STATE:", _SERVER_STATE)
    if _SERVER_STATE.get("exception") is not None:
        raise _SERVER_STATE["exception"]
    assert _SERVER_STATE["started"] is True
    assert _SERVER_STATE["port"] == port
    assert _SERVER_STATE["thread"] is not None
    assert _SERVER_STATE["thread"].is_alive()

    # Re-calling should return the same port
    port2 = start_server_background(default_port=9950)
    assert port2 == port
