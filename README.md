---
title: Open Voice Agent
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---

# Open Voice Agent

Real-time AI voice conversation application powered by LiveKit Agents, Moonshine STT, and Pocket TTS.

## Features

- **Streaming Speech-to-Text**: Moonshine (HuggingFace transformers)
- **LLM Integration**: HuggingFace models or NVIDIA API via LangGraph
- **Text-to-Speech**: Pocket TTS (Kyutai) with local inference
- **Voice Activity Detection**: Silero VAD
- **Web Interface**: Streamlit-based UI

## Setup

### Local Development

1. Install dependencies:
```bash
uv sync
source .venv/bin/activate
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the application:
```bash
# Terminal 1: Start LiveKit agent
uv run src/agent/agent.py start

# Terminal 2: Start Streamlit UI
streamlit run src/streamlit_app.py
```

### Docker

```bash
docker build -t open-voice-agent .
docker run -p 8501:8501 --env-file .env open-voice-agent
```

## Environment Variables

### Required

- `LIVEKIT_URL`: WebSocket URL for LiveKit server (wss://...)
- `LIVEKIT_API_KEY`: LiveKit API key
- `LIVEKIT_API_SECRET`: LiveKit API secret

### LLM Provider (choose one)

**HuggingFace** (local inference):
```bash
LLM_PROVIDER=huggingface
HUGGINGFACE_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
HUGGINGFACE_DEVICE=cuda  # or 'cpu' or leave empty for auto
HF_TOKEN=hf_xxx  # optional, for private models
```

**NVIDIA** (API-based):
```bash
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-xxx
NVIDIA_MODEL=meta/llama-3.1-8b-instruct
```

### Optional

See `.env.example` for all available configuration options.

## Requirements

- Python >= 3.12, < 3.13
- LiveKit server (cloud or self-hosted)
- NVIDIA API key OR sufficient compute for local LLM inference

## License

Apache 2.0
