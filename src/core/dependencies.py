from typing import Optional

from src.agent.graph import create_conversation_graph
from src.models.voice.factory import VoiceProviderFactory
from src.services.session_manager import SessionManager
from src.core.logger import logger


class DependencyContainer:
    _instance: Optional["DependencyContainer"] = None
    
    def __init__(self):
        self._session_manager: Optional[SessionManager] = None
        self._conversation_graph = None
    
    @classmethod
    def get_instance(cls) -> "DependencyContainer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_session_manager(self) -> SessionManager:
        if self._session_manager is None:
            self._session_manager = SessionManager()
            logger.debug("SessionManager initialized")
        return self._session_manager
    
    def get_conversation_graph(self):
        if self._conversation_graph is None:
            self._conversation_graph = create_conversation_graph()
            logger.debug("Conversation graph initialized")
        return self._conversation_graph
    
    def get_voice_provider(self, provider_name: Optional[str] = None):
        return VoiceProviderFactory.create_provider(provider_name)
    
    def reset(self) -> None:
        self._session_manager = None
        self._conversation_graph = None
        logger.debug("Dependency container reset")


def get_container() -> DependencyContainer:
    return DependencyContainer.get_instance()
