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
![Open Voice Agent Demo](demo.gif)

---

## Description
A real-time conversational voice agent built with open-source components, deployable on consumer hardware (8GB VRAM)

The core value is in the **custom LiveKit plugins** — reusable integrations that let you plug any HuggingFace model into LiveKit's agent framework.

## Architecture

```
User Audio → Moonshine STT → Qwen2.5-3B / NVIDIA LLM → Pocket TTS → Audio Response
                          ↕ LiveKit (WebRTC) ↕
```

| Component | Model | Why this one |
|-----------|-------|-------------|
| **STT** | [Moonshine](https://huggingface.co/usefulsensors/moonshine-streaming-medium) (61M params) | Edge-optimized, streaming, open-source |
| **LLM** | [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) or [NVIDIA NIM](https://build.nvidia.com/) | Fits 8GB VRAM / flexible fallback |
| **TTS** | [Pocket TTS](https://huggingface.co/kyutai/pocket-tts) (Kyutai) | Local inference, streaming, no API needed |
| **VAD** | Silero VAD | Industry standard voice activity detection |
| **Transport** | [LiveKit](https://livekit.io) | WebRTC, open-source, low-latency |
| **UI** | Streamlit | Simple, functional browser interface |

## Custom LiveKit Plugins

**This is the most reusable part of the project.** The `src/plugins/` directory contains custom LiveKit agent plugins that integrate HuggingFace models into LiveKit's streaming pipeline:

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

- Extends `tts.TTS` with streaming synthesis via `AudioEmitter` API
- Sentence-level streaming: audio starts playing before full generation completes
- Automatic resampling from 24kHz native to configurable output rate
- Multiple voice support with fallback handling
- Runs generation in background thread to keep the async loop responsive

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

**Local (HuggingFace):**
```bash
LLM_PROVIDER=huggingface
HUGGINGFACE_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
HUGGINGFACE_DEVICE=cuda
HF_TOKEN=hf_xxx  # optional, for gated models
```

**API (NVIDIA NIM):**
```bash
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-xxx
NVIDIA_MODEL=meta/llama-3.1-8b-instruct
```

See `.env.example` for all available options.

## Project Structure

```
open-voice-agent/
├── src/
│   ├── agent/
│   │   ├── agent.py              # LiveKit agent entry point
│   │   └── graph.py              # LangGraph conversation graph
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

## Deployment

This project runs as a HuggingFace Docker Space:
**[dvalle08/open-voice-agent](https://huggingface.co/spaces/dvalle08/open-voice-agent)**

The `Dockerfile` handles both the LiveKit agent process and the Streamlit UI, managed by `start.sh`.

## License
Apache 2.0