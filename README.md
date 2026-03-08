---

## title: Open Voice Agent
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false

# 🎤 Open Voice Agent

> A real-time voice AI agent powered by open-source components.

**[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/dvalle08/open-voice-agent)**



Open Voice Agent Demo

---

## Description

A real-time conversational voice agent built with open-source components, deployable on consumer hardware (8GB VRAM).

The core value is in the **custom LiveKit plugins** for STT/TTS and a clean runtime layer for MCP-enabled LLM providers.

## Architecture

Open Voice Agent is built as a local, open-source-first voice pipeline, with hosted providers available when you want to swap a layer.

```text
Recommended stack:
User Audio -> Silero VAD + turn detection -> Moonshine STT -> Ollama -> Pocket TTS -> Audio Response
                            ^ LiveKit realtime transport + Streamlit UI ^
```


| Layer         | Recommended path                                                             | Also supported                                                                      | Notes                                                             |
| ------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **STT**       | [Moonshine](https://huggingface.co/usefulsensors/moonshine-streaming-medium) | Deepgram `nova-3`, NVIDIA `parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer` | Moonshine is the local open-source path                           |
| **LLM**       | [Ollama](https://ollama.com/)                                                | NVIDIA NIM                                                                          | Ollama can run locally or via Ollama Cloud                        |
| **TTS**       | [Pocket TTS](https://huggingface.co/kyutai/pocket-tts)                       | Deepgram `aura-2-thalia-en`, NVIDIA Riva with `Magpie-Multilingual.EN-US.Leo`       | Pocket TTS is the local open-source path                          |
| **VAD**       | Silero VAD                                                                   | -                                                                                   | Handles speech start and end detection inside the LiveKit session |
| **Transport** | [LiveKit](https://livekit.io)                                                | -                                                                                   | Realtime WebRTC transport and agent runtime                       |
| **UI**        | Streamlit                                                                    | -                                                                                   | Lightweight browser UI for local and Spaces deployments           |


## Open-Source Core

The reusable core lives in `src/plugins/`. It provides custom LiveKit-compatible components for the local speech stack:

```text
/plugins/
├── moonshine_stt/   # Streaming STT plugin built on Moonshinesrc
└── pocket_tts/      # Streaming TTS plugin built on Pocket TTS
```

These plugins implement LiveKit's `stt.STT` and `tts.TTS` interfaces, so the local stack behaves like any other provider-backed LiveKit pipeline while staying easy to reuse in other agent projects.

- `moonshine_stt` wraps `usefulsensors/moonshine-streaming-medium` for streaming speech-to-text.
- `pocket_tts` wraps `kyutai/pocket-tts` for streaming text-to-speech with native 24 kHz output.
- Deepgram and NVIDIA can replace individual layers without changing the rest of the agent flow.

## Quick Start

- Python 3.10+
- [UV](https://github.com/astral-sh/uv)
- [LiveKit server](https://docs.livekit.io) (cloud or self-hosted)

Recommended setup: Moonshine for STT, Ollama for LLM, and Pocket TTS for speech output.

```bash
git clone https://github.com/dvalle08/open-voice-agent.git
cd open-voice-agent

uv sync
cp .env.example .env
# Edit .env with your LiveKit credentials and provider choices

uv run src/agent/agent.py start
uv run streamlit run src/streamlit_app.py
```

If you prefer hosted components, switch the STT, LLM, or TTS provider in `.env` to Deepgram or NVIDIA for the layers that need it.

Docker remains available for local or Spaces-style deployment:

```bash
docker build -t open-voice-agent .
docker run -p 8501:8501 --env-file .env open-voice-agent
```

## Configuration


| Layer          | Supported providers               | Recommended choice |
| -------------- | --------------------------------- | ------------------ |
| `STT_PROVIDER` | `moonshine`, `deepgram`, `nvidia` | `moonshine`        |
| `LLM_PROVIDER` | `ollama`, `nvidia`                | `ollama`           |
| `TTS_PROVIDER` | `pocket`, `deepgram`, `nvidia`    | `pocket`           |


Recommended `.env` for the local open-source stack:

```bash
STT_PROVIDER=moonshine
MOONSHINE_MODEL_ID=usefulsensors/moonshine-streaming-medium
MOONSHINE_LANGUAGE=en

LLM_PROVIDER=ollama
OLLAMA_CLOUD_MODE=false
OLLAMA_MODEL=your-local-ollama-model
OLLAMA_API_KEY=ollama

TTS_PROVIDER=pocket
POCKET_TTS_VOICE=alba
```

Hosted alternatives by layer:

```bash
STT_PROVIDER=deepgram
DEEPGRAM_STT_MODEL=nova-3
DEEPGRAM_STT_LANGUAGE=en-US
DEEPGRAM_API_KEY=your_deepgram_api_key_here

# or NVIDIA STT
STT_PROVIDER=nvidia
NVIDIA_STT_MODEL=parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer
NVIDIA_STT_LANGUAGE_CODE=en-US
NVIDIA_STT_API_KEY=

# NVIDIA LLM
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_MODEL=meta/llama-3.1-8b-instruct

# Deepgram TTS
TTS_PROVIDER=deepgram
DEEPGRAM_API_KEY=your_deepgram_api_key_here

# or NVIDIA TTS / self-hosted Riva
TTS_PROVIDER=nvidia
NVIDIA_TTS_VOICE=Magpie-Multilingual.EN-US.Leo
NVIDIA_TTS_LANGUAGE_CODE=en-US
NVIDIA_TTS_SERVER=grpc.nvcf.nvidia.com:443
NVIDIA_TTS_FUNCTION_ID=877104f7-e885-42b9-8de8-f6e4c6303969
NVIDIA_TTS_USE_SSL=true
NVIDIA_TTS_API_KEY=
```

## MCP Support

```bash
MCP_ENABLED=true
MCP_SERVER_URL=https://huggingface.co/mcp
MCP_EXTRA_SERVER_URLS=https://docs.livekit.io/mcp
```

## Langfuse Tracing

Langfuse tracing is optional and supported out of the box.

```bash
LANGFUSE_ENABLED=true
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PROJECT_ID=your_project_id
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_TRACES=false
```

- The app creates one Langfuse trace per finalized user turn.
- Traces cover the core voice pipeline stages `stt`, `llm`, and `tts`, plus tool activity when tools are used.
- The Streamlit UI shows a live trace dropdown with `Open Trace` links for the current session.
- `LANGFUSE_PROJECT_ID` is required for those trace links.

## Operational Notes

```bash
LLM_CONN_TIMEOUT_SEC=20.0
LLM_CONN_MAX_RETRY=1
LLM_CONN_RETRY_INTERVAL_SEC=1.0
TURN_LLM_STALL_TIMEOUT_SEC=12.0
MCP_STARTUP_GREETING_TIMEOUT_SEC=0.0
POCKET_TTS_CONN_TIMEOUT_SEC=45.0
LIVEKIT_NUM_IDLE_PROCESSES=1
LIVEKIT_INITIALIZE_PROCESS_TIMEOUT_SEC=20.0
```

- `LLM_CONN_*` controls LLM timeout and retry behavior.
- `POCKET_TTS_CONN_TIMEOUT_SEC` is the per-attempt TTS timeout used by Pocket TTS, Deepgram TTS, and NVIDIA TTS in this app.
- `MCP_STARTUP_GREETING_TIMEOUT_SEC=0.0` disables forced interruption of the startup greeting.
- `LIVEKIT_NUM_IDLE_PROCESSES` and `LIVEKIT_INITIALIZE_PROCESS_TIMEOUT_SEC` control idle worker warm-up.
- If STT is working but no reply is produced, start with the backend log line `Turn stalled before LLM stage`.

## Project Structure

```text
open-voice-agent/
├── src/
│   ├── agent/            # LiveKit agent runtime, prompts, tools, tracing
│   ├── plugins/          # Custom Moonshine STT and Pocket TTS plugins
│   ├── api/              # Session bootstrap and token helpers
│   ├── core/             # Settings and logging
│   ├── ui/               # Browser client assets
│   └── streamlit_app.py  # Streamlit entrypoint
├── .env.example          # Full provider and runtime configuration reference
├── Dockerfile            # Container build for local use or Spaces
├── pyproject.toml        # Dependencies
└── start.sh              # Container entrypoint
```

## License

Apache 2.0