import pytest
from datetime import datetime, timedelta, UTC

from src.services.session_manager import SessionManager, Session
from src.core.exceptions import SessionError


def test_create_session(session_manager):
    session = session_manager.create_session()
    
    assert session.session_id is not None
    assert isinstance(session.created_at, datetime)
    assert isinstance(session.last_active, datetime)


def test_get_session(session_manager):
    session = session_manager.create_session()
    
    retrieved = session_manager.get_session(session.session_id)
    
    assert retrieved is not None
    assert retrieved.session_id == session.session_id


def test_get_nonexistent_session(session_manager):
    result = session_manager.get_session("nonexistent-id")
    assert result is None


def test_get_or_create_session_creates_new(session_manager):
    session = session_manager.get_or_create_session()
    
    assert session is not None
    assert session.session_id is not None


def test_get_or_create_session_gets_existing(session_manager):
    session1 = session_manager.create_session()
    session2 = session_manager.get_or_create_session(session_id=session1.session_id)
    
    assert session1.session_id == session2.session_id


def test_get_or_create_session_raises_on_invalid(session_manager):
    with pytest.raises(SessionError):
        session_manager.get_or_create_session(session_id="invalid-id")


def test_delete_session(session_manager):
    session = session_manager.create_session()
    
    result = session_manager.delete_session(session.session_id)
    
    assert result is True
    assert session_manager.get_session(session.session_id) is None


def test_delete_nonexistent_session(session_manager):
    result = session_manager.delete_session("nonexistent-id")
    assert result is False


def test_session_is_expired():
    session = Session(session_id="test-id")
    
    assert not session.is_expired(timeout_seconds=3600)
    
    session.last_active = datetime.now(UTC) - timedelta(seconds=7200)
    
    assert session.is_expired(timeout_seconds=3600)


def test_cleanup_expired_sessions():
    manager = SessionManager(session_timeout=10)
    
    session1 = manager.create_session()
    session2 = manager.create_session()
    
    session1.last_active = datetime.now(UTC) - timedelta(seconds=20)
    
    count = manager.cleanup_expired_sessions()
    
    assert count == 1
    assert manager.get_session(session1.session_id) is None
    assert manager.get_session(session2.session_id) is not None
