---
title: Open Voice Agent
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---

# 🎤 Open Voice Agent

> A real-time voice AI agent powered by open-source components.

**[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/dvalle08/open-voice-agent)**

<!-- Replace with your actual GIF after recording -->
![Open Voice Agent Demo](assets/demo2.gif)

---

## Description
A real-time conversational voice agent built with open-source components, deployable on consumer hardware (8GB VRAM).

The core value is in the **custom LiveKit plugins** for STT/TTS and a clean runtime layer for MCP-enabled LLM providers.

## Architecture

```
User Audio → Moonshine STT → Ollama / NVIDIA LLM (+ optional MCP tools) → Pocket TTS → Audio Response
                          ↕ LiveKit (WebRTC) ↕
```

| Component | Model | Why this one |
|-----------|-------|-------------|
| **STT** | [Moonshine](https://huggingface.co/usefulsensors/moonshine-streaming-medium) (61M params) | Edge-optimized, streaming, open-source |
| **LLM** | [Ollama](https://ollama.com/) or [NVIDIA NIM](https://build.nvidia.com/) | Local or hosted OpenAI-compatible backends |
| **TTS** | [Pocket TTS](https://huggingface.co/kyutai/pocket-tts) (Kyutai) | Local inference, streaming, no API needed |
| **VAD** | Silero VAD | Industry standard voice activity detection |
| **Transport** | [LiveKit](https://livekit.io) | WebRTC, open-source, low-latency |
| **UI** | Streamlit | Simple, functional browser interface |

## Custom LiveKit Plugins

**This is the most reusable part of the project.** The `src/plugins/` directory contains custom LiveKit agent plugins for the realtime voice pipeline:

```
src/plugins/
├── moonshine_stt/     # Moonshine streaming STT as a LiveKit plugin
│   └── stt.py         # MoonshineSTT class + MoonshineSTTStream
└── pocket_tts/        # Pocket TTS as a LiveKit plugin
    └── tts.py         # PocketTTS class + PocketSynthesizeStream
```

Each plugin follows LiveKit's official plugin pattern (extending `stt.STT`, `tts.TTS`), so you can drop them into any LiveKit agent project.

### [Moonshine STT Plugin](src/plugins/moonshine_stt/stt.py)

Built on [usefulsensors/moonshine-streaming-medium](https://huggingface.co/usefulsensors/moonshine-streaming-medium).

- Extends `stt.STT` with both batch (`_recognize_impl`) and streaming (`stream()`) support
- Automatic silence detection and segment management
- Audio resampling to 16kHz using polyphase filtering
- Proper metrics emission for LiveKit's monitoring

### [Pocket TTS Plugin](src/plugins/pocket_tts/tts.py)

Built on [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts).

- Extends `tts.TTS` with both `stream()` and `synthesize()` support (explicit `ChunkedStream`)
- Chunk-progressive synthesis: audio is pushed while generation is still running
- Timeout-aware generation pipeline with API error mapping (`APITimeoutError`, `APIConnectionError`)
- Native 24kHz output path for stable realtime browser playback
- Multiple voice support with fallback handling (`voice -> alba`)
- Generation runs in background thread(s) to keep the async event loop responsive

## Quick Start

### Prerequisites

- Python 3.10+
- [UV](https://github.com/astral-sh/uv) (package manager)
- [LiveKit server](https://docs.livekit.io) (cloud or self-hosted)

### Install & Run

```bash
git clone https://github.com/dvalle08/open-voice-agent.git
cd open-voice-agent

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your LiveKit credentials plus LLM/TTS provider choices

# Terminal 1: Start the LiveKit agent
uv run src/agent/agent.py start

# Terminal 2: Start the Streamlit UI
uv run streamlit run src/streamlit_app.py
```

### Docker

```bash
docker build -t open-voice-agent .
docker run -p 8501:8501 --env-file .env open-voice-agent
```

## Configuration

### LLM Provider (choose one)

**Local (Ollama):**
```bash
LLM_PROVIDER=ollama
OLLAMA_CLOUD_MODE=false
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_API_KEY=ollama
```

**Ollama Cloud (OpenAI-compatible):**
```bash
LLM_PROVIDER=ollama
OLLAMA_CLOUD_MODE=true
OLLAMA_MODEL=qwen3-next:80b
OLLAMA_API_KEY=your_ollama_api_key_here
```

- `OLLAMA_CLOUD_MODE=true` derives `https://ollama.com/v1`; `false` derives `http://localhost:11434/v1`.
- For `https://ollama.com/v1`, `OLLAMA_MODEL` must be an exact ID returned by `GET /v1/models`.
- `OLLAMA_API_KEY` is required when `OLLAMA_CLOUD_MODE=true`.
- Do not use `:cloud` aliases (for example `qwen3.5:cloud`) with the OpenAI-compatible endpoint.
- Quick check:
  ```bash
  curl -sS https://ollama.com/v1/models \
    -H "Authorization: Bearer $OLLAMA_API_KEY"
  ```

**API (NVIDIA NIM):**
```bash
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-xxx
NVIDIA_MODEL=meta/llama-3.1-8b-instruct
```

### STT Provider (choose one)

**Local (Moonshine):**
```bash
STT_PROVIDER=moonshine
MOONSHINE_MODEL_ID=usefulsensors/moonshine-streaming-medium
MOONSHINE_LANGUAGE=en
```

**API (Deepgram):**
```bash
STT_PROVIDER=deepgram
DEEPGRAM_STT_MODEL=nova-3
DEEPGRAM_STT_LANGUAGE=en-US
DEEPGRAM_API_KEY=your_deepgram_api_key_here
```

**API (NVIDIA Riva):**
```bash
STT_PROVIDER=nvidia
NVIDIA_STT_MODEL=parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer
NVIDIA_STT_LANGUAGE_CODE=en-US
# Optional override; otherwise falls back to NVIDIA_API_KEY
NVIDIA_STT_API_KEY=
```

### TTS Provider (choose one)

**Local (PocketTTS):**
```bash
TTS_PROVIDER=pocket
POCKET_TTS_VOICE=alba
```

**API (Deepgram):**
```bash
TTS_PROVIDER=deepgram
DEEPGRAM_API_KEY=your_deepgram_api_key_here
```

**API or self-hosted (NVIDIA Riva):**
```bash
TTS_PROVIDER=nvidia
NVIDIA_TTS_VOICE=Magpie-Multilingual.EN-US.Leo
NVIDIA_TTS_LANGUAGE_CODE=en-US
NVIDIA_TTS_SERVER=grpc.nvcf.nvidia.com:443
NVIDIA_TTS_FUNCTION_ID=877104f7-e885-42b9-8de8-f6e4c6303969
NVIDIA_TTS_USE_SSL=true
# Optional override; otherwise falls back to NVIDIA_API_KEY
NVIDIA_TTS_API_KEY=
```

- `NVIDIA_TTS_API_KEY` overrides the shared `NVIDIA_API_KEY` when set.
- For self-hosted Riva, point `NVIDIA_TTS_SERVER` at your server and set `NVIDIA_TTS_USE_SSL=false` if TLS is disabled.
- `DEEPGRAM_API_KEY` is shared across Deepgram STT and TTS in this app.

See `.env.example` for all available options.

### MCP runtime (default on)

```bash
MCP_ENABLED=true
MCP_SERVER_URL=https://huggingface.co/mcp
MCP_EXTRA_SERVER_URLS=https://docs.livekit.io/mcp
# Works with either provider:
LLM_PROVIDER=ollama
# or LLM_PROVIDER=nvidia (+ NVIDIA_API_KEY)
```

- Primary MCP endpoint defaults to `https://huggingface.co/mcp` (no auth configured).
- Extra MCP endpoints can be configured with `MCP_EXTRA_SERVER_URLS` (comma-separated), defaulting to LiveKit Docs.
- MCP tools are available when `MCP_ENABLED=true` and `LLM_PROVIDER` is `nvidia` or `ollama`.
- In MCP mode, startup greeting is sent with `session.say(...)` and manual `session.generate_reply(...)` calls are disabled by policy.
- There is no legacy LangGraph fallback path.

### Langfuse tracing (one trace per turn)

```bash
LANGFUSE_ENABLED=true
LANGFUSE_HOST=https://cloud.langfuse.com
# LANGFUSE_BASE_URL=https://us.cloud.langfuse.com  # optional alternative
LANGFUSE_PROJECT_ID=clkpwwm0m000gmm094odg11gi
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_TRACES=false
```

Each finalized user transcript creates a new trace with spans `stt`, `llm`, and `tts`.
The Streamlit client generates a new `session_id` on each Connect click and sends it to the agent.
The header includes a live traces dropdown that updates as finalized turns arrive.
Each entry shows the full `trace_id`, local created-at time, and an `Open Trace` link to:
`https://cloud.langfuse.com/project/<project_id>/traces/<trace_id>`.

Notes:

- `LANGFUSE_PROJECT_ID` is required to build trace deep links in the UI.
- Session pages are intentionally not linked from the frontend because they are project-member scoped in Langfuse Cloud.

### LLM/TTS runtime resilience

Tune request behavior with:

```bash
LLM_CONN_TIMEOUT_SEC=12.0
LLM_CONN_MAX_RETRY=1
LLM_CONN_RETRY_INTERVAL_SEC=1.0
TURN_LLM_STALL_TIMEOUT_SEC=8.0
MCP_STARTUP_GREETING_TIMEOUT_SEC=0.0
POCKET_TTS_CONN_TIMEOUT_SEC=45.0
```

- `LLM_CONN_*` controls timeout/retry behavior for LLM requests.
- `POCKET_TTS_CONN_TIMEOUT_SEC` controls the timeout for one TTS synthesis attempt across PocketTTS, Deepgram, and NVIDIA Riva.
- `MCP_STARTUP_GREETING_TIMEOUT_SEC=0.0` disables forced interruption of the startup greeting; set a positive value to restore a cutoff.
- `TURN_LLM_STALL_TIMEOUT_SEC` emits a backend warning if a finalized user turn never reaches the LLM stage.

### LiveKit worker startup

Tune idle worker warm-up with:

```bash
LIVEKIT_NUM_IDLE_PROCESSES=1
LIVEKIT_INITIALIZE_PROCESS_TIMEOUT_SEC=20.0
```

- `LIVEKIT_INITIALIZE_PROCESS_TIMEOUT_SEC` maps to LiveKit's idle worker bootstrap timeout.
- `LIVEKIT_NUM_IDLE_PROCESSES=1` is the intended local baseline to reduce idle worker memory and CPU pressure.
- `20.0` seconds is the intended baseline when keeping the current worker behavior and avoiding idle worker init timeouts.

### Troubleshooting: STT works but no voice reply

If the UI only shows silence/STT activity and never reaches LLM/TTS:

- Check backend logs for `Turn stalled before LLM stage`.
- Check backend logs for `Agent session pipeline error` with `source=...` and `error_type=...`.
- Verify your selected LLM provider config:
  - `nvidia`: `NVIDIA_API_KEY` and `NVIDIA_MODEL`
  - `ollama` local: `OLLAMA_CLOUD_MODE=false`, local server reachable at `http://localhost:11434/v1`, and model pulled in Ollama
  - `ollama` cloud: `OLLAMA_CLOUD_MODE=true`, valid `OLLAMA_API_KEY`, and model ID from `/v1/models`
- Verify Deepgram credentials when using `STT_PROVIDER=deepgram` or `TTS_PROVIDER=deepgram`.
- Verify NVIDIA STT credentials when using `STT_PROVIDER=nvidia`; the agent logs which STT key source is used.
- If local memory warnings are noisy, raise `LIVEKIT_JOB_MEMORY_WARN_MB` (for local setups, `6144` is a practical baseline).

## Project Structure

```
open-voice-agent/
├── src/
│   ├── agent/
│   │   ├── agent.py              # LiveKit CLI entrypoint
│   │   ├── runtime/              # Session wiring + Assistant + lifecycle tasks
│   │   ├── models/               # LLM/STT runtime providers and factory
│   │   ├── traces/               # Metrics collector + turn tracing + Langfuse setup
│   │   ├── tools/                # Tool feedback controller + pre-tool speech injection
│   │   └── prompts/              # Assistant/system and runtime prompt text
│   ├── plugins/                  # ← Custom LiveKit plugins
│   │   ├── moonshine_stt/        # Moonshine streaming STT
│   │   └── pocket_tts/           # Pocket TTS streaming synthesis
│   ├── api/
│   │   └── livekit_tokens.py     # Token generation for WebRTC
│   ├── core/
│   │   ├── settings.py           # Configuration management
│   │   └── logger.py             # Logging setup
│   └── streamlit_app.py          # Browser UI
├── Dockerfile                    # For HF Spaces / local Docker
├── pyproject.toml                # UV dependencies
├── uv.lock
├── .env.example                  # Template with all env vars
└── start.sh                      # Container entrypoint
```
## License
Apache 2.0
