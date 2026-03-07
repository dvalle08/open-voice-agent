"""Assistant prompt definitions."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from src.core.settings import Settings, mask_sensitive_data, settings as global_settings


def _format_human_date(value: date) -> str:
    return f"{value.strftime('%B')} {value.day}, {value.year}"


def _format_agent_config_summary(current_settings: Settings) -> str:
    masked = mask_sensitive_data(current_settings.model_dump())

    voice = masked.get("voice", {})
    stt = masked.get("stt", {})
    llm = masked.get("llm", {})
    livekit = masked.get("livekit", {})

    stt_provider = str(stt.get("STT_PROVIDER", "")).lower()
    if stt_provider == "nvidia":
        stt_model = stt.get("NVIDIA_STT_MODEL")
        stt_language = stt.get("NVIDIA_STT_LANGUAGE_CODE")
    else:
        stt_model = stt.get("MOONSHINE_MODEL_ID")
        stt_language = stt.get("MOONSHINE_LANGUAGE")

    llm_provider = str(llm.get("LLM_PROVIDER", "")).lower()
    if llm_provider == "nvidia":
        llm_model = llm.get("NVIDIA_MODEL")
        llm_mode = None
    else:
        llm_model = llm.get("OLLAMA_MODEL")
        llm_mode = llm.get("OLLAMA_CLOUD_MODE")

    mcp_extra_servers = llm.get("MCP_EXTRA_SERVER_URLS") or "<none>"

    sections = [
        (
            "Voice stack: "
            f"STT provider={stt_provider}, model={stt_model}, language={stt_language}; "
            f"LLM provider={llm_provider}, model={llm_model}, "
            f"{f'cloud_mode={llm_mode}, ' if llm_mode is not None else ''}"
            f"temperature={llm.get('LLM_TEMPERATURE')}, max_tokens={llm.get('LLM_MAX_TOKENS')}, "
            f"timeout_sec={llm.get('LLM_CONN_TIMEOUT_SEC')}, max_retry={llm.get('LLM_CONN_MAX_RETRY')}, "
            f"retry_interval_sec={llm.get('LLM_CONN_RETRY_INTERVAL_SEC')}, "
            f"startup_greeting_timeout_sec={llm.get('MCP_STARTUP_GREETING_TIMEOUT_SEC')}; "
            f"TTS provider=pocket-tts, voice={voice.get('POCKET_TTS_VOICE')}, "
            f"temperature={voice.get('POCKET_TTS_TEMPERATURE')}, "
            f"lsd_decode_steps={voice.get('POCKET_TTS_LSD_DECODE_STEPS')}, "
            f"timeout_sec={voice.get('POCKET_TTS_CONN_TIMEOUT_SEC')}."
        ),
        (
            "Turn handling: "
            f"VAD min_speech_duration={voice.get('VAD_MIN_SPEECH_DURATION')} s, "
            f"min_silence_duration={voice.get('VAD_MIN_SILENCE_DURATION')} s, "
            f"threshold={voice.get('VAD_THRESHOLD')}; "
            f"endpointing min_delay={voice.get('MIN_ENDPOINTING_DELAY')} s, "
            f"max_delay={voice.get('MAX_ENDPOINTING_DELAY')} s, "
            f"preemptive_generation={voice.get('PREEMPTIVE_GENERATION')}."
        ),
        (
            "LiveKit: "
            f"runtime agent_name={livekit.get('LIVEKIT_AGENT_NAME')}, "
            f"num_idle_processes={livekit.get('LIVEKIT_NUM_IDLE_PROCESSES')}, "
            f"job_memory_warn_mb={livekit.get('LIVEKIT_JOB_MEMORY_WARN_MB')}; "
            f"audio sample_rate={voice.get('LIVEKIT_SAMPLE_RATE')} Hz, "
            f"channels={voice.get('LIVEKIT_NUM_CHANNELS')}, "
            f"frame_size_ms={voice.get('LIVEKIT_FRAME_SIZE_MS')}, "
            f"pre_connect_audio={voice.get('LIVEKIT_PRE_CONNECT_AUDIO')}, "
            f"pre_connect_timeout={voice.get('LIVEKIT_PRE_CONNECT_TIMEOUT')}."
        ),
        (
            "MCP runtime: "
            f"enabled={llm.get('MCP_ENABLED')}, "
            f"primary_server={llm.get('MCP_SERVER_URL')}, "
            f"extra_servers={mcp_extra_servers}."
        ),
        (
            "Credential state (redacted): "
            f"NVIDIA_API_KEY={llm.get('NVIDIA_API_KEY')}, "
            f"NVIDIA_STT_API_KEY={stt.get('NVIDIA_STT_API_KEY')}, "
            f"OLLAMA_API_KEY={llm.get('OLLAMA_API_KEY')}, "
            f"LIVEKIT_API_KEY={livekit.get('LIVEKIT_API_KEY')}, "
            f"LIVEKIT_API_SECRET={livekit.get('LIVEKIT_API_SECRET')}."
        ),
    ]
    return "\n".join(sections)


def build_assistant_instructions(
    *,
    current_date: Optional[date] = None,
    current_settings: Optional[Settings] = None,
) -> str:
    today = current_date or datetime.now().astimezone().date()
    settings_obj = current_settings or global_settings
    config_summary = _format_agent_config_summary(settings_obj)

    return f"""You are Open Voice Agent, a real-time voice pipeline.
Current date: {_format_human_date(today)}.
Priority 1. Identity. You are a pipeline, not a single model. For self-description, describe the pipeline briefly and mention VAD, turn detection, STT, LLM, and TTS only when relevant.
Priority 2. Response style. Answer as quickly as possible. Give the shortest complete answer. Default to one short sentence. If a few words are enough, use a few words. Expand only when the user asks or accuracy requires it. The user only speaks to you, not types, so do not ask them to write, type, paste, or use chat.
Priority 3. Tools. Answer from your own knowledge first, then from the setup summary. Only call tools that are available in the current session. Do not claim you can generate or edit images. Use Hugging Face tools only when needed and only if they are available in the current session. Use LiveKit tools only for LiveKit-specific questions not covered by your own knowledge or the setup summary. Before any tool call, say one short lead-in sentence. Never invent tools.
Priority 4. Safety and setup. For setup questions, answer from the setup summary. Never reveal raw keys, tokens, passwords, or secrets.
Use plain voice-friendly text only; no markdown, emojis, bullets, or decorative punctuation.
Setup summary:
{config_summary}"""


ASSISTANT_INSTRUCTIONS = build_assistant_instructions()
