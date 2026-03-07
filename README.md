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
![Open Voice Agent Demo](assets/demo.gif)

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
# Edit .env with your LiveKit credentials and LLM provider choice

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
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_API_KEY=ollama
```

**Ollama Cloud (OpenAI-compatible):**
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=https://ollama.com/v1
OLLAMA_MODEL=qwen3-next:80b
OLLAMA_API_KEY=your_ollama_api_key_here
```

- For `https://ollama.com/v1`, `OLLAMA_MODEL` must be an exact ID returned by `GET /v1/models`.
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
```

- `LLM_CONN_*` controls timeout/retry behavior for both LLM and PocketTTS requests.
- `TURN_LLM_STALL_TIMEOUT_SEC` emits a backend warning if a finalized user turn never reaches the LLM stage.

### Troubleshooting: STT works but no voice reply

If the UI only shows silence/STT activity and never reaches LLM/TTS:

- Check backend logs for `Turn stalled before LLM stage`.
- Check backend logs for `Agent session pipeline error` with `source=...` and `error_type=...`.
- Verify your selected LLM provider config:
  - `nvidia`: `NVIDIA_API_KEY` and `NVIDIA_MODEL`
  - `ollama` local: local server reachable at `OLLAMA_BASE_URL` and model pulled in Ollama
  - `ollama` cloud: `OLLAMA_BASE_URL=https://ollama.com/v1` and model ID from `/v1/models`
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
