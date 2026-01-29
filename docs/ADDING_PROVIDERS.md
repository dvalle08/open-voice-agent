# Adding New Providers

## Voice Providers

### Step 1: Implement Base Interface

Create a new file in `src/models/voice/`:

```python
from typing import AsyncIterator, Optional

from src.models.voice.base import BaseVoiceProvider, VoiceProviderConfig
from src.models.voice.types import TranscriptionResult, VADInfo


class MyProviderConfig(VoiceProviderConfig):
    provider_name: str = "myprovider"
    api_key: str
    # Add provider-specific settings


class MyProvider(BaseVoiceProvider):
    def __init__(self, config: MyProviderConfig):
        super().__init__(config)
        self.config: MyProviderConfig = config
        self.client = None
    
    async def connect(self) -> None:
        # Initialize provider client
        self.client = MyProviderClient(api_key=self.config.api_key)
        self._connected = True
    
    async def disconnect(self) -> None:
        # Cleanup resources
        if self.client:
            await self.client.close()
        self._connected = False
    
    async def text_to_speech(
        self, text: str, stream: bool = True
    ) -> AsyncIterator[bytes]:
        # Generate speech audio
        async for chunk in self.client.tts(text):
            yield chunk
    
    async def speech_to_text(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptionResult]:
        # Transcribe audio
        async for result in self.client.stt(audio_stream):
            yield TranscriptionResult(
                text=result.text,
                start_s=result.start_time,
                is_final=result.is_final,
            )
    
    async def get_vad_info(self) -> Optional[VADInfo]:
        # Return VAD info if available
        return None
```

### Step 2: Register Provider

In `src/models/voice/factory.py`:

```python
from src.models.voice.myprovider import MyProvider

VoiceProviderFactory.register("myprovider", MyProvider)
```

### Step 3: Add Configuration

In `src/core/settings.py`:

```python
class VoiceSettings(CoreSettings):
    # Existing settings...
    
    # MyProvider settings
    MY_PROVIDER_API_KEY: str = Field(default="")
    MY_PROVIDER_SETTING: str = Field(default="default")
```

### Step 4: Update Factory Logic

In `src/models/voice/factory.py`, add configuration:

```python
@classmethod
def create_provider(cls, provider_name: str = None) -> BaseVoiceProvider:
    provider_name = provider_name or settings.voice.VOICE_PROVIDER
    provider_name = provider_name.lower()
    
    if provider_name not in cls._registry:
        raise ValueError(f"Unknown voice provider: {provider_name}")
    
    if provider_name == "myprovider":
        config = MyProviderConfig(
            api_key=settings.voice.MY_PROVIDER_API_KEY,
            # Map other settings...
        )
        return cls._registry[provider_name](config)
    
    # Existing providers...
```

### Step 5: Add Tests

Create `tests/unit/test_myprovider.py`:

```python
import pytest
from unittest.mock import AsyncMock

from src.models.voice.myprovider import MyProvider, MyProviderConfig


@pytest.fixture
def my_provider():
    config = MyProviderConfig(api_key="test-key")
    return MyProvider(config)


@pytest.mark.asyncio
async def test_connect(my_provider):
    await my_provider.connect()
    assert my_provider.is_connected


@pytest.mark.asyncio
async def test_tts(my_provider):
    await my_provider.connect()
    
    chunks = []
    async for chunk in my_provider.text_to_speech("Hello"):
        chunks.append(chunk)
    
    assert len(chunks) > 0
```

### Step 6: Update Documentation

Add to `.env.example`:

```env
# MyProvider
VOICE_PROVIDER=myprovider
MY_PROVIDER_API_KEY=your_api_key_here
```

Update `README.md`:

```markdown
### MyProvider

To use MyProvider:

1. Get API key from https://myprovider.com
2. Set environment variables:
   ```env
   VOICE_PROVIDER=myprovider
   MY_PROVIDER_API_KEY=your_key
   ```
3. Run the application
```

## LLM Providers

### Step 1: Update LLM Factory

In `src/agent/llm_factory.py`:

```python
from langchain_myprovider import MyProviderLLM

class LLMFactory:
    @classmethod
    def create_llm(cls, use_cache: bool = True) -> BaseLanguageModel:
        # Existing code...
        
        if provider == "myprovider":
            logger.info(f"Initializing MyProvider LLM: {settings.llm.MY_PROVIDER_MODEL}")
            llm = MyProviderLLM(
                model=settings.llm.MY_PROVIDER_MODEL,
                api_key=settings.llm.MY_PROVIDER_API_KEY,
                temperature=settings.llm.LLM_TEMPERATURE,
                max_tokens=settings.llm.LLM_MAX_TOKENS,
            )
        # ...
```

### Step 2: Add Configuration

In `src/core/settings.py`:

```python
class LLMSettings(CoreSettings):
    # Existing settings...
    
    # MyProvider LLM settings
    MY_PROVIDER_API_KEY: Optional[str] = Field(default=None)
    MY_PROVIDER_MODEL: Optional[str] = Field(default="default-model")
    
    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid = ["nvidia", "huggingface", "myprovider"]
        if v.lower() not in valid:
            raise ValueError(f"LLM_PROVIDER must be one of: {valid}")
        return v.lower()
```

### Step 3: Add Dependencies

In `pyproject.toml`:

```toml
dependencies = [
    # ...
    "langchain-myprovider>=1.0.0",
]
```

### Step 4: Test Integration

```python
def test_myprovider_llm():
    with patch("src.agent.llm_factory.settings") as mock_settings:
        mock_settings.llm.LLM_PROVIDER = "myprovider"
        mock_settings.llm.MY_PROVIDER_MODEL = "test-model"
        mock_settings.llm.MY_PROVIDER_API_KEY = "test-key"
        
        llm = LLMFactory.create_llm(use_cache=False)
        assert llm is not None
```

## Best Practices

### Error Handling

Always wrap provider calls in try-except:

```python
try:
    result = await provider.text_to_speech(text)
except ProviderException as e:
    logger.error(f"Provider error: {e}")
    raise TTSError(f"Failed to generate speech: {e}") from e
```

### Logging

Log important events:

```python
logger.info(f"Connecting to MyProvider...")
logger.debug(f"Sending request with params: {params}")
logger.error(f"Failed to connect: {error}")
```

### Configuration Validation

Validate settings on startup:

```python
@field_validator("MY_PROVIDER_API_KEY")
@classmethod
def validate_api_key(cls, v: str) -> str:
    if not v:
        raise ValueError("MY_PROVIDER_API_KEY is required")
    if not v.startswith("mp_"):
        raise ValueError("Invalid API key format")
    return v
```

### Resource Cleanup

Always cleanup in disconnect:

```python
async def disconnect(self) -> None:
    try:
        if self._stream:
            await self._stream.close()
        if self.client:
            await self.client.close()
    finally:
        self._connected = False
```

### Testing

Test all required methods:

- Connection/disconnection
- Text-to-speech
- Speech-to-text
- VAD info (if supported)
- Error conditions

## Example: OpenAI Provider

```python
class OpenAIVoiceConfig(VoiceProviderConfig):
    provider_name: str = "openai"
    api_key: str
    model: str = "whisper-1"
    voice: str = "alloy"
    sample_rate_input: int = 16000
    sample_rate_output: int = 24000


class OpenAIVoiceProvider(BaseVoiceProvider):
    def __init__(self, config: OpenAIVoiceConfig):
        super().__init__(config)
        self.config: OpenAIVoiceConfig = config
        self.client = None
    
    async def connect(self) -> None:
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=self.config.api_key)
        self._connected = True
    
    async def disconnect(self) -> None:
        if self.client:
            await self.client.close()
        self._connected = False
    
    async def text_to_speech(
        self, text: str, stream: bool = True
    ) -> AsyncIterator[bytes]:
        response = await self.client.audio.speech.create(
            model="tts-1",
            voice=self.config.voice,
            input=text,
        )
        
        async for chunk in response.iter_bytes():
            yield chunk
    
    async def speech_to_text(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptionResult]:
        # Collect audio chunks
        audio_data = b""
        async for chunk in audio_stream:
            audio_data += chunk
        
        # Transcribe
        response = await self.client.audio.transcriptions.create(
            model=self.config.model,
            file=audio_data,
        )
        
        yield TranscriptionResult(
            text=response.text,
            start_s=0.0,
            is_final=True,
        )
    
    async def get_vad_info(self) -> Optional[VADInfo]:
        return None  # OpenAI doesn't provide VAD
```

## Common Pitfalls

### Forgetting Async/Await

```python
# Wrong
def text_to_speech(self, text):
    return self.client.generate(text)

# Correct
async def text_to_speech(self, text):
    return await self.client.generate(text)
```

### Not Yielding in Generators

```python
# Wrong
async def text_to_speech(self, text):
    for chunk in await self.client.generate(text):
        return chunk  # Wrong!

# Correct
async def text_to_speech(self, text):
    async for chunk in self.client.generate(text):
        yield chunk
```

### Hardcoding Configuration

```python
# Wrong
def __init__(self):
    self.api_key = "hardcoded_key"

# Correct
def __init__(self, config: MyConfig):
    self.config = config
    self.api_key = config.api_key
```

### Missing Error Handling

Always handle provider errors gracefully and convert to appropriate custom exceptions.
