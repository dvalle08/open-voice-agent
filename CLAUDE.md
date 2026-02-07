# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open Voice Agent is a real-time AI voice conversation application that integrates speech-to-text (STT), large language models (LLMs), and text-to-speech (TTS) capabilities. The project supports multiple providers including NVIDIA, HuggingFace, and custom implementations.

## Core Commands

### Development Setup
```bash
# Install dependencies using uv (REQUIRED - never use pip directly)
uv sync

# Activate virtual environment
source .venv/bin/activate

# Copy environment template and configure
cp .env.example .env
# Edit .env to add your API keys (NVIDIA_API_KEY, HF_TOKEN, etc.)
```

### Running the Application
```bash
# Run both FastAPI server and Streamlit UI (default)
python main.py

# Run only the FastAPI server
python main.py api

# Run only the Streamlit UI
python main.py streamlit
```

The application will start:
- FastAPI server on http://localhost:8000
- Streamlit UI on http://localhost:8501

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_models.py

# Run with verbose output
pytest -v

# Run LiveKit integration tests
python test/livekit_langgraph_moonshine.py
```

### NVIDIA TTS Docker Setup
```bash
# Login to NVIDIA Container Registry
source .env
echo "$NVIDIA_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

# Run Riva TTS NIM container
docker run -it --rm --name=magpie-tts-multilingual \
  --runtime=nvidia --gpus '"device=0"' --shm-size=8GB \
  -e NGC_API_KEY=$NGC_API_KEY \
  -p 9000:9000 -p 50051:50051 \
  nvcr.io/nim/nvidia/magpie-tts-multilingual:latest
```

## Architecture

### High-Level Structure

The codebase follows a modular architecture with clear separation of concerns:

**src/agent/** - LangGraph-based conversation orchestration
- `graph.py`: StateGraph implementation with nodes for processing user input and generating responses
- `state.py`: ConversationState TypedDict defines the state structure (messages, current_transcript, context, turn_active)
- `llm_factory.py`: Factory pattern for creating LLM, STT, and TTS providers
- `prompts.py`: System prompts for the conversation agent

**src/models/voice/** - Voice provider abstractions
- `base.py`: BaseVoiceProvider abstract class defining the interface for STT/TTS providers
- `types.py`: Shared types (TranscriptionResult, VADInfo, etc.)

**src/plugins/** - Provider-specific implementations
- `moonshine_stt/`: ONNX-based Moonshine STT plugin for LiveKit agents

**src/core/** - Application configuration and utilities
- `settings.py`: Pydantic-based settings with VoiceSettings, LLMSettings, and APISettings
- `logger.py`: Centralized logging configuration

**src/api/** - FastAPI REST API implementation

### Key Architectural Patterns

**Provider Factory Pattern**: LLMFactory centralizes creation of all AI providers (NVIDIA, HuggingFace, Moonshine, Kokoro). Each provider has specific initialization parameters and supports both cloud and local execution modes.

**LangGraph StateGraph**: The conversation flow is managed by a StateGraph with conditional edges:
- `process_input` node: Adds user transcripts to message history
- `should_respond` conditional: Decides whether to generate response or wait based on turn_active and transcript presence
- `generate_response` node: Invokes LLM with conversation history

**Settings Management**: Pydantic BaseSettings with nested configuration groups (voice, llm, api). Environment variables are loaded from .env file with automatic masking of sensitive data in logs.

**Plugin System**: Voice providers inherit from BaseVoiceProvider abstract class, implementing connect/disconnect, text_to_speech, speech_to_text, and get_vad_info methods. This allows easy swapping of STT/TTS backends.

### Multi-Provider Support

**LLM Providers**:
- NVIDIA: ChatNVIDIA via `langchain_nvidia_ai_endpoints`
- HuggingFace: ChatHuggingFace with HuggingFaceEndpoint (cloud) or HuggingFacePipeline (local)

**STT Providers**:
- Moonshine: ONNX-based streaming STT (configurable: tiny, base, small, medium models)
- HuggingFace: InferenceClient for cloud STT or pipeline for local

**TTS Providers**:
- NVIDIA: Magpie-TTS-Multilingual via Riva endpoint
- HuggingFace: InferenceClient for cloud TTS or pipeline for local
- Kokoro: Local TTS using KPipeline (hexgrad/Kokoro-82M)

### LiveKit Integration

The project uses LiveKit agents for real-time voice conversations:
- `AgentSession` manages STT, LLM, TTS, VAD, and turn detection
- Supports noise cancellation via BVC/BVCTelephony
- Integrates LangGraph workflows through `langchain.LLMAdapter`

## Coding Standards

### Type Hints (REQUIRED)
All function parameters and return types MUST have type hints. This is enforced by project standards.

### Documentation
Code should be self-documenting. Only add comments that explain **why**, never **what**. Do not create markdown documentation files unless explicitly requested by the user.

### Dependencies
Always use `uv` for package management. Never use `pip` directly. The project requires Python 3.10 (specified in pyproject.toml).

### Modern Python
Use modern Python features appropriate for Python 3.10+.

## Environment Configuration

Required environment variables (see .env.example):
- `NVIDIA_API_KEY`: For NVIDIA LLM and TTS services
- `HF_TOKEN`: For HuggingFace models and inference
- `NVIDIA_TTS_ENDPOINT`: NVIDIA TTS service endpoint (from build.nvidia.com)

Optional configuration:
- `VOICE_PROVIDER`: Default voice provider (nvidia, huggingface, etc.)
- `STT_PROVIDER`: Speech-to-text provider (moonshine, assemblyai, etc.)
- `MOONSHINE_MODEL_SIZE`: Moonshine model size (tiny, base, small, medium)
- `LLM_TEMPERATURE`: LLM temperature (0.0-2.0, default 0.7)
- `LLM_MAX_TOKENS`: Maximum tokens in LLM response (default 1024)

## Common Patterns

### Creating LLM Instances
```python
from src.agent.llm_factory import LLMFactory

# NVIDIA LLM
llm = LLMFactory.create_nvidia_llm(
    model="meta/llama-3.1-8b-instruct",
    temperature=0.7,
    max_tokens=1024
)

# HuggingFace LLM (cloud)
llm = LLMFactory.create_huggingface_llm(
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    provider="auto"
)

# HuggingFace LLM (local)
llm = LLMFactory.create_huggingface_llm(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    run_local=True
)
```

### Creating STT/TTS Instances
```python
# Moonshine STT (ONNX-based, efficient)
stt = LLMFactory.create_moonshine_stt(model_size="base")

# Kokoro TTS (local)
tts = LLMFactory.create_kokoro_tts(lang_code="a")

# HuggingFace STT (cloud)
stt = LLMFactory.create_huggingface_stt(model_id="openai/whisper-large-v3")
```

### Using LangGraph Conversation Graph
```python
from src.agent.graph import create_conversation_graph

graph = create_conversation_graph()
result = graph.invoke({
    "messages": [],
    "current_transcript": "Hello, how are you?",
    "context": {},
    "turn_active": False
})
```

## Important Notes

- Python version constraint: Requires Python >=3.10,<3.11 (see pyproject.toml)
- GPU acceleration: Models automatically detect CUDA availability and use fp16 on GPU, fp32 on CPU
- LiveKit agent scripts are in test/ directory (not production code)
- The test/ directory contains integration test scripts, not unit tests with assertions
