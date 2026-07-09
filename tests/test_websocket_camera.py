"""Unit tests for the Live Camera Starlette Registry and Route Injection."""
from __future__ import annotations

import pytest

from app.tabs.live_camera_server import (
    register_session,
    get_session,
    register_starlette_route,
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


def test_register_starlette_route_no_server() -> None:
    """Test that register_starlette_route executes safely when Server is not in memory."""
    # This should log a warning but return safely without raising any exceptions
    try:
        register_starlette_route()
        assert True
    except Exception as exc:
        pytest.fail(f"register_starlette_route raised an unexpected exception: {exc}")
