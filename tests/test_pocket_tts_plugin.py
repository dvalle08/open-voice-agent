from __future__ import annotations

import asyncio
import importlib
import sys
import threading
import time
import types
from collections.abc import Generator
from typing import Any

import numpy as np
import pytest

from livekit.agents import APIConnectionError, APITimeoutError
from livekit.agents.types import APIConnectOptions


@pytest.fixture
def pocket_plugin(monkeypatch: pytest.MonkeyPatch) -> Any:
    calls: dict[str, Any] = {
        "num_chunks": 2,
        "chunk_samples": 9600,
        "per_chunk_sleep": 0.0,
        "pause_after_first_chunk": False,
        "allow_generation_finish": None,
        "raise_on_generate": None,
        "active_generations": 0,
        "max_active_generations": 0,
        "texts": [],
    }

    class _FakeModel:
        def get_state_for_audio_prompt(self, voice: str, truncate: bool = True) -> dict[str, str]:
            calls["voice"] = voice
            if voice == "missing":
                raise FileNotFoundError("voice prompt not found")
            if voice == "bad-voice":
                raise RuntimeError("bad voice")
            return {"voice": voice}

        def generate_audio_stream(
            self,
            state: dict[str, str],
            text: str,
            copy_state: bool = True,
        ) -> Generator[np.ndarray[Any, np.dtype[np.float32]], None, None]:
            calls["state"] = state
            calls["text"] = text
            calls["texts"].append(text)
            calls["copy_state"] = copy_state
            calls["active_generations"] += 1
            calls["max_active_generations"] = max(
                calls["max_active_generations"], calls["active_generations"]
            )
            try:
                if calls["raise_on_generate"] is not None:
                    raise calls["raise_on_generate"]

                for i in range(calls["num_chunks"]):
                    if calls["per_chunk_sleep"] > 0:
                        time.sleep(calls["per_chunk_sleep"])

                    yield np.linspace(
                        -0.25 + i * 0.05,
                        0.25 - i * 0.05,
                        calls["chunk_samples"],
                        dtype=np.float32,
                    )

                    if i == 0 and calls["pause_after_first_chunk"]:
                        gate = calls["allow_generation_finish"]
                        if isinstance(gate, threading.Event):
                            gate.wait(timeout=2.0)
            finally:
                calls["active_generations"] -= 1

    class _FakeTTSModel:
        @staticmethod
        def load_model(*, temp: float, lsd_decode_steps: int) -> _FakeModel:
            calls["temperature"] = temp
            calls["lsd_decode_steps"] = lsd_decode_steps
            return _FakeModel()

    monkeypatch.setitem(sys.modules, "pocket_tts", types.SimpleNamespace(TTSModel=_FakeTTSModel))

    for name in list(sys.modules):
        if name == "src.plugins.pocket_tts" or name.startswith("src.plugins.pocket_tts."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    module = importlib.import_module("src.plugins.pocket_tts.tts")
    calls["module"] = module
    return calls


def test_fallback_voice_and_missing_voice_error(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]

    tts_fallback = module.PocketTTS(voice="bad-voice")
    assert tts_fallback._voice == "alba"

    with pytest.raises(ValueError, match="Failed to load voice"):
        module.PocketTTS(voice="missing")


def test_tensor_to_pcm_bytes_handles_channel_first_and_last(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]

    channel_first = np.array(
        [[1.0, -1.0, 0.5, -0.5], [-1.0, 1.0, -0.5, 0.5]],
        dtype=np.float32,
    )
    channel_last = np.array(
        [[1.0, -1.0], [-1.0, 1.0], [0.5, -0.5], [-0.5, 0.5]],
        dtype=np.float32,
    )

    pcm_first = module._tensor_to_pcm_bytes(
        audio_chunk=channel_first,
        output_sample_rate=24000,
        native_sample_rate=24000,
    )
    pcm_last = module._tensor_to_pcm_bytes(
        audio_chunk=channel_last,
        output_sample_rate=24000,
        native_sample_rate=24000,
    )

    assert len(pcm_first) == 8
    assert len(pcm_last) == 8


def test_tensor_to_pcm_bytes_rejects_unsupported_shape(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    invalid = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="unsupported audio tensor shape"):
        module._tensor_to_pcm_bytes(
            audio_chunk=invalid,
            output_sample_rate=24000,
            native_sample_rate=24000,
        )


def test_tensor_to_pcm_bytes_resamples_to_configured_rate(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    src = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)

    pcm = module._tensor_to_pcm_bytes(
        audio_chunk=src,
        output_sample_rate=48000,
        native_sample_rate=24000,
    )

    # 5 input samples at 24kHz should become 10 output samples at 48kHz.
    assert len(pcm) == 20


def test_stream_and_chunked_emit_audio_without_synthesize_attribute_error(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    tts_v = module.PocketTTS(voice="alba", sample_rate=48000)

    async def _run() -> None:
        async with tts_v.stream() as synth_stream:
            synth_stream.push_text("hola")
            synth_stream.end_input()
            streamed_events = await asyncio.wait_for(_collect_events(synth_stream), timeout=3.0)

        assert streamed_events
        assert streamed_events[-1].is_final
        assert all(event.frame.sample_rate == 48000 for event in streamed_events)
        assert all(event.frame.num_channels == 1 for event in streamed_events)
        assert streamed_events[0].segment_id.startswith("SEG_")

        chunked_events = await asyncio.wait_for(_collect_events(tts_v.synthesize("hola")), timeout=3.0)
        assert chunked_events
        assert chunked_events[-1].is_final

    asyncio.run(_run())


def test_stream_emits_before_generation_completes(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    pocket_plugin["pause_after_first_chunk"] = True
    pocket_plugin["allow_generation_finish"] = threading.Event()
    pocket_plugin["num_chunks"] = 2
    pocket_plugin["chunk_samples"] = 9600

    tts_v = module.PocketTTS(voice="alba")

    async def _run() -> None:
        async with tts_v.stream() as synth_stream:
            synth_stream.push_text("hola")
            synth_stream.end_input()

            first_event = await asyncio.wait_for(synth_stream.__anext__(), timeout=1.0)
            assert not first_event.is_final

            pocket_plugin["allow_generation_finish"].set()
            remaining_events = [event async for event in synth_stream]

        all_events = [first_event, *remaining_events]
        assert all_events[-1].is_final

    asyncio.run(_run())


def test_stream_uses_single_segment_for_one_flush(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    tts_v = module.PocketTTS(voice="alba")
    long_text = (
        "First sentence with enough words to trigger internal chunking. " * 6
        + "Second sentence also long enough to split. " * 6
    )

    async def _run() -> None:
        async with tts_v.stream() as synth_stream:
            synth_stream.push_text(long_text)
            synth_stream.end_input()
            events = await asyncio.wait_for(_collect_events(synth_stream), timeout=3.0)

        segment_ids = {
            event.segment_id
            for event in events
            if not event.is_final and isinstance(event.segment_id, str) and event.segment_id
        }
        assert len(segment_ids) == 1

    asyncio.run(_run())


def test_chunked_generation_serializes_concurrent_requests(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    pocket_plugin["per_chunk_sleep"] = 0.03
    pocket_plugin["num_chunks"] = 3

    tts_v = module.PocketTTS(voice="alba")

    async def _run() -> None:
        await asyncio.gather(
            _collect_events(tts_v.synthesize("uno")),
            _collect_events(tts_v.synthesize("dos")),
        )

    asyncio.run(_run())
    assert pocket_plugin["max_active_generations"] == 1


def test_generation_errors_are_mapped_to_api_errors(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    pocket_plugin["raise_on_generate"] = RuntimeError("boom")

    tts_v = module.PocketTTS(voice="alba")

    async def _run() -> None:
        with pytest.raises(APIConnectionError):
            await _collect_events(tts_v.synthesize("hola"))

    asyncio.run(_run())


def test_generation_timeout_is_mapped_to_api_timeout_error(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    gate = threading.Event()
    pocket_plugin["pause_after_first_chunk"] = True
    pocket_plugin["allow_generation_finish"] = gate
    pocket_plugin["num_chunks"] = 2

    tts_v = module.PocketTTS(voice="alba")
    conn_options = APIConnectOptions(max_retry=0, timeout=0.1)

    async def _run() -> None:
        with pytest.raises(APITimeoutError):
            await _collect_events(tts_v.synthesize("hola", conn_options=conn_options))

    try:
        asyncio.run(_run())
    finally:
        gate.set()


def test_sanitize_tts_text_removes_markdown_noise(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    raw_text = """
    ## Title
    - **Bold** item with [link text](https://example.com)
    1. `code` item
    """

    sanitized = module._sanitize_tts_text(raw_text)
    assert "##" not in sanitized
    assert "**" not in sanitized
    assert "`" not in sanitized
    assert "[link text]" not in sanitized
    assert "(https://example.com)" not in sanitized
    assert "link text" in sanitized
    assert "Bold item with" in sanitized


def test_chunk_tts_text_respects_length_limit(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    text = " ".join(["word"] * 80)

    chunks = module._chunk_tts_text(text, max_chars=40)
    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)
    assert " ".join(chunks).replace("  ", " ").strip() == text


def test_chunked_synthesize_sanitizes_and_splits_long_text(pocket_plugin: Any) -> None:
    module = pocket_plugin["module"]
    tts_v = module.PocketTTS(voice="alba")
    text = (
        "## Header\n"
        "- **First** item with [a link](https://example.com).\n"
        + "Second sentence keeps going with enough words to exceed the segment limit. " * 5
        + "Third sentence keeps going with enough words to exceed the segment limit. " * 5
    )

    async def _run() -> None:
        await _collect_events(tts_v.synthesize(text))

    asyncio.run(_run())

    generated_texts = pocket_plugin["texts"]
    assert len(generated_texts) >= 2
    assert all(len(part) <= module.MAX_TTS_SEGMENT_CHARS for part in generated_texts)
    assert all("**" not in part and "`" not in part for part in generated_texts)
    assert all("https://example.com" not in part for part in generated_texts)


async def _collect_events(stream: Any) -> list[Any]:
    return [event async for event in stream]
