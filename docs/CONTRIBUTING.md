# Contributing Guide

## Development Setup

### Prerequisites

- Python 3.13+
- uv (recommended) or pip
- Git

### Clone and Install

```bash
git clone <repository-url>
cd open-voice-agent
uv sync --all-extras
```

### Environment Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

## Code Style

### General Principles

- Code is self-documenting - avoid "what" comments
- Comments explain "why" when non-obvious
- No unnecessary docstrings
- Clean, readable code over clever code

### Python Style

- Type hints for all function parameters and returns
- Descriptive variable names
- Maximum line length: 100 characters
- Use f-strings for string formatting

### Import Order

1. Standard library
2. Third-party packages
3. Local modules

Example:
```python
import os
from typing import Optional

from fastapi import WebSocket

from src.core.logger import logger
```

## Architecture Guidelines

### Service Layer

Services should:
- Have single responsibility
- Accept dependencies via constructor
- Return domain objects, not primitives when appropriate
- Raise custom exceptions, not generic ones

### Error Handling

- Use custom exceptions from `src.core.exceptions`
- Log errors before raising
- Provide context in error messages
- Don't catch exceptions you can't handle

### Testing

- Write tests for new features
- Mock external dependencies
- Test edge cases and error paths
- Keep tests fast

## Making Changes

### Branch Strategy

- `main`: Stable production code
- Feature branches: `feature/description`
- Bug fixes: `fix/description`

### Commit Messages

Write clear, concise commit messages:

```
Add session management with UUID generation

- Implement SessionManager class
- Add session lifecycle methods
- Add tests for session expiration
```

### Pull Request Process

1. Create feature branch
2. Make changes with tests
3. Ensure tests pass: `pytest`
4. Check linting (if configured)
5. Open pull request
6. Address review feedback
7. Squash and merge

## Testing

### Run All Tests

```bash
pytest
```

### Run Specific Tests

```bash
pytest tests/unit/test_prompts.py
pytest tests/unit/ -v
pytest -k "test_session"
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html
```

### Writing Tests

**Unit Test Example:**

```python
def test_audio_service_add_chunk():
    service = AudioService()
    service.add_chunk(b"data")
    assert service.get_buffer_size() == 1
```

**Async Test Example:**

```python
@pytest.mark.asyncio
async def test_transcription(mock_provider):
    service = TranscriptionService(mock_provider)
    result = await service.get_full_transcript([b"audio"])
    assert isinstance(result, str)
```

## Adding Features

### New Voice Provider

1. Implement `BaseVoiceProvider`:
```python
class MyProvider(BaseVoiceProvider):
    async def connect(self): ...
    async def disconnect(self): ...
    async def text_to_speech(self, text): ...
    async def speech_to_text(self, audio): ...
    async def get_vad_info(self): ...
```

2. Add config class:
```python
class MyProviderConfig(VoiceProviderConfig):
    provider_name: str = "myprovider"
    api_key: str
```

3. Register in factory:
```python
VoiceProviderFactory.register("myprovider", MyProvider)
```

4. Add settings:
```python
class VoiceSettings(CoreSettings):
    MY_PROVIDER_API_KEY: str = Field(default="")
```

5. Add tests

### New Service

1. Create service class:
```python
class MyService:
    def __init__(self, dependency: SomeDependency):
        self._dependency = dependency
    
    async def do_something(self, param: str) -> Result:
        # Implementation
```

2. Add to service layer
3. Write unit tests
4. Update WebSocket handler if needed

## Debugging

### Enable Debug Logging

```bash
export LOG_LEVEL=DEBUG
python main.py both
```

### Common Issues

**Import errors**: Run `uv sync`
**API key errors**: Check `.env` file
**Test failures**: Check for stale mocks

## Code Review Guidelines

### As Author

- Keep PRs focused and small
- Write clear descriptions
- Add tests
- Update documentation

### As Reviewer

- Check for edge cases
- Verify tests are meaningful
- Ensure error handling
- Look for performance issues

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release branch
4. Test thoroughly
5. Tag release
6. Deploy

## Questions?

- Check existing issues
- Review documentation
- Ask in discussions
