import uuid
from datetime import datetime, timedelta, UTC
from typing import Dict, Optional
from dataclasses import dataclass, field

from src.core.logger import logger
from src.core.exceptions import SessionError


@dataclass
class Session:
    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_active: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict = field(default_factory=dict)
    
    def touch(self) -> None:
        self.last_active = datetime.now(UTC)
    
    def is_expired(self, timeout_seconds: int = 3600) -> bool:
        return (datetime.now(UTC) - self.last_active).total_seconds() > timeout_seconds


class SessionManager:
    def __init__(self, session_timeout: int = 3600):
        self._sessions: Dict[str, Session] = {}
        self._session_timeout = session_timeout
    
    def create_session(self, metadata: Optional[Dict] = None) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            metadata=metadata or {}
        )
        self._sessions[session_id] = session
        logger.info(f"Created session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        session = self._sessions.get(session_id)
        if session:
            session.touch()
        return session
    
    def get_or_create_session(self, session_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Session:
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
            raise SessionError(f"Session not found: {session_id}")
        
        return self.create_session(metadata)
    
    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    def cleanup_expired_sessions(self) -> int:
        expired_ids = [
            sid for sid, session in self._sessions.items()
            if session.is_expired(self._session_timeout)
        ]
        
        for sid in expired_ids:
            self.delete_session(sid)
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired sessions")
        
        return len(expired_ids)
    
    def get_active_session_count(self) -> int:
        return len(self._sessions)
