# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open Voice Agent is a real-time AI voice conversation application built on LiveKit Agents. It integrates Moonshine STT (streaming speech-to-text), NVIDIA LLM (via LangGraph), and Pocket TTS (local text-to-speech) for low-latency voice conversations.

## Core Commands

### Development Setup
```bash
# Install dependencies using uv (REQUIRED - never use pip directly)
uv sync

# Activate virtual environment
source .venv/bin/activate

# Copy environment template and configure
cp .env.example .env
# Edit .env to add your API keys
```

### Running the Application
```bash
# Run the LiveKit voice agent
python src/agent/agent.py start
```

### Testing
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run LiveKit integration tests
python dev/test/livekit_langgraph_moonshine.py
```

## Architecture

### High-Level Structure

The codebase follows a modular architecture with clear separation of concerns:

**src/agent/** - LangGraph-based conversation orchestration and LiveKit agent entry point
- `agent.py`: LiveKit `AgentServer` and `AgentSession` setup; wires together STT, LLM, TTS, VAD, and turn detection
- `graph.py`: Single-node `StateGraph` using `ChatNVIDIA` as the LLM backend

**src/plugins/** - Provider-specific LiveKit plugin implementations
- `moonshine_stt/`: Streaming STT using `MoonshineStreamingForConditionalGeneration` (HuggingFace transformers)
- `pocket_tts/`: Local TTS using Kyutai's `pocket_tts` library

**src/core/** - Application configuration and utilities
- `settings.py`: Pydantic-based settings with `VoiceSettings` and `LLMSettings`
- `logger.py`: Centralized logging configuration

### Key Architectural Patterns

**LangGraph StateGraph**: The LLM workflow is a simple single-node graph:
- Uses `MessagesState` for conversation history
- Wraps `ChatNVIDIA` and is passed to LiveKit via `langchain.LLMAdapter`

**LiveKit AgentSession**: All voice components are composed via `AgentSession`:
- `MoonshineSTT` for streaming speech recognition
- `langchain.LLMAdapter(create_graph())` for LLM responses
- `PocketTTS` for local speech synthesis
- `silero.VAD` for voice activity detection
- `MultilingualModel` for turn detection

**Settings Management**: Pydantic `BaseSettings` with two nested groups (`voice`, `llm`). Environment variables are loaded from `.env` with automatic masking of sensitive data in logs.

**Plugin System**: Plugins implement LiveKit's `stt.STT` and `tts.TTS` abstract classes, allowing easy swapping of STT/TTS backends.

### Provider Details

**LLM**: NVIDIA via `langchain_nvidia_ai_endpoints.ChatNVIDIA`

**STT**: Moonshine (`usefulsensors/moonshine-streaming-*`) using HuggingFace transformers with automatic CUDA/CPU detection

**TTS**: Pocket TTS (Kyutai) — local inference with configurable voice, temperature, and LSD decode steps. Native sample rate 24000 Hz, resampled to configured output rate.

## Coding Standards

### Type Hints (REQUIRED)
All function parameters and return types MUST have type hints. This is enforced by project standards.

### Documentation
Code should be self-documenting. Only add comments that explain **why**, never **what**. Do not create markdown documentation files unless explicitly requested by the user.

### Dependencies
Always use `uv` for package management. Never use `pip` directly. The project requires Python >=3.10,<3.11 (see pyproject.toml).

### Modern Python
Use modern Python features appropriate for Python 3.10+.

## Environment Configuration

Required environment variables (see `.env.example`):
- `NVIDIA_API_KEY`: For NVIDIA LLM via `langchain_nvidia_ai_endpoints`

Optional configuration:
- `NVIDIA_MODEL`: NVIDIA model ID (default: `meta/llama-3.1-8b-instruct`)
- `LLM_TEMPERATURE`: LLM temperature (0.0-2.0, default 0.7)
- `LLM_MAX_TOKENS`: Maximum tokens in LLM response (default 1024)
- `MOONSHINE_MODEL_ID`: Moonshine model ID (default: `usefulsensors/moonshine-streaming-medium`)
- `POCKET_TTS_VOICE`: Voice name or path to audio file (default: `alba`)
- `POCKET_TTS_TEMPERATURE`: TTS sampling temperature (default 0.7)
- `POCKET_TTS_LSD_DECODE_STEPS`: LSD decoding steps; higher = better quality, slower (default 1)
- `SAMPLE_RATE_OUTPUT`: Output audio sample rate in Hz (default 48000)

## Common Patterns

### Running the Agent
```python
# src/agent/agent.py is the entry point
# The agent uses AgentServer and is started via the LiveKit CLI:
# python src/agent/agent.py start
```

### Creating the LangGraph Workflow
```python
from src.agent.graph import create_graph

graph = create_graph()
# Returns a compiled StateGraph with a single ChatNVIDIA node
```

### Using Settings
```python
from src.core.settings import settings

model_id = settings.voice.MOONSHINE_MODEL_ID
api_key = settings.llm.NVIDIA_API_KEY
```

## Important Notes

- Python version constraint: Requires Python >=3.10,<3.11 (see pyproject.toml)
- GPU acceleration: Moonshine STT automatically uses CUDA with fp16 when available, falls back to CPU with fp32
- The `dev/` directory contains development scripts and integration tests (not production code)
- LiveKit environment variables (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`) are required at runtime for the agent to connect to a LiveKit server
