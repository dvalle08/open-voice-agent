from __future__ import annotations

from langchain_core.messages import AIMessageChunk
from livekit.plugins.langchain import langgraph as lk_langgraph

from src.agent._langchain_usage_patch import (
    _PATCH_FLAG,
    _ORIGINAL_TO_CHAT_CHUNK,
    _completion_usage_from_message_chunk,
    apply_langchain_usage_patch,
)


def test_completion_usage_is_extracted_from_usage_metadata() -> None:
    chunk = AIMessageChunk(
        content="hello",
        usage_metadata={
            "input_tokens": 12,
            "output_tokens": 24,
            "total_tokens": 36,
            "input_token_details": {"cache_read": 7},
        },
    )

    usage = _completion_usage_from_message_chunk(chunk)
    assert usage is not None
    assert usage.prompt_tokens == 12
    assert usage.completion_tokens == 24
    assert usage.total_tokens == 36
    assert usage.prompt_cached_tokens == 7


def test_completion_usage_falls_back_to_response_metadata_token_usage() -> None:
    chunk = AIMessageChunk(
        content="hello",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 2},
            }
        },
    )

    usage = _completion_usage_from_message_chunk(chunk)
    assert usage is not None
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15
    assert usage.prompt_cached_tokens == 2


def test_patch_preserves_usage_only_chunks(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    original = lk_langgraph._to_chat_chunk
    monkeypatch.setattr(lk_langgraph, _PATCH_FLAG, False, raising=False)
    monkeypatch.delattr(lk_langgraph, _ORIGINAL_TO_CHAT_CHUNK, raising=False)
    monkeypatch.setattr(lk_langgraph, "_to_chat_chunk", original)

    assert apply_langchain_usage_patch() is True
    assert apply_langchain_usage_patch() is False

    usage_only_chunk = AIMessageChunk(
        content="",
        usage_metadata={
            "input_tokens": 8,
            "output_tokens": 3,
            "total_tokens": 11,
        },
    )
    chat_chunk = lk_langgraph._to_chat_chunk(usage_only_chunk)
    assert chat_chunk is not None
    assert chat_chunk.delta is None
    assert chat_chunk.usage is not None
    assert chat_chunk.usage.prompt_tokens == 8
    assert chat_chunk.usage.completion_tokens == 3

    # Restore to avoid bleeding monkeypatches between tests.
    monkeypatch.setattr(lk_langgraph, "_to_chat_chunk", original)
    monkeypatch.setattr(lk_langgraph, _PATCH_FLAG, False, raising=False)
    monkeypatch.delattr(lk_langgraph, _ORIGINAL_TO_CHAT_CHUNK, raising=False)
