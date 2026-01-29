import pytest
from unittest.mock import patch

from src.agent.llm_factory import LLMFactory


def test_llm_factory_caching():
    with patch("src.agent.llm_factory.settings") as mock_settings:
        mock_settings.llm.LLM_PROVIDER = "nvidia"
        mock_settings.llm.NVIDIA_MODEL = "test-model"
        mock_settings.llm.NVIDIA_API_KEY = "test-key"
        mock_settings.llm.LLM_TEMPERATURE = 0.7
        mock_settings.llm.LLM_MAX_TOKENS = 1024
        
        LLMFactory.reset_cache()
        
        llm1 = LLMFactory.create_llm(use_cache=True)
        llm2 = LLMFactory.create_llm(use_cache=True)
        
        assert llm1 is llm2


def test_llm_factory_no_caching():
    with patch("src.agent.llm_factory.settings") as mock_settings:
        mock_settings.llm.LLM_PROVIDER = "nvidia"
        mock_settings.llm.NVIDIA_MODEL = "test-model"
        mock_settings.llm.NVIDIA_API_KEY = "test-key"
        mock_settings.llm.LLM_TEMPERATURE = 0.7
        mock_settings.llm.LLM_MAX_TOKENS = 1024
        
        LLMFactory.reset_cache()
        
        llm1 = LLMFactory.create_llm(use_cache=False)
        llm2 = LLMFactory.create_llm(use_cache=False)
        
        assert llm1 is not llm2


def test_llm_factory_invalid_provider():
    with patch("src.agent.llm_factory.settings") as mock_settings:
        mock_settings.llm.LLM_PROVIDER = "invalid_provider"
        
        LLMFactory.reset_cache()
        
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMFactory.create_llm(use_cache=False)
