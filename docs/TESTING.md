# Testing Guide

## Overview

The project uses pytest for testing with async support, mocking, and coverage reporting.

## Test Structure

```
tests/
├── conftest.py          # Shared fixtures
├── unit/                # Unit tests
│   ├── test_prompts.py
│   ├── test_llm_factory.py
│   ├── test_session_manager.py
│   └── test_services.py
├── integration/         # Integration tests
└── fixtures/           # Test data and mocks
```

## Running Tests

### All Tests

```bash
pytest
```

### Specific Test File

```bash
pytest tests/unit/test_prompts.py
```

### Specific Test Function

```bash
pytest tests/unit/test_prompts.py::test_get_system_prompt
```

### By Marker

```bash
pytest -m unit
pytest -m integration
pytest -m "not slow"
```

### With Coverage

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Verbose Output

```bash
pytest -v
pytest -vv  # Extra verbose
```

### Show Print Statements

```bash
pytest -s
```

## Writing Tests

### Unit Tests

Test individual components in isolation:

```python
def test_prompt_template_render():
    template = PromptTemplate("Hello {name}")
    result = template.render(name="Alice")
    assert result == "Hello Alice"
```

### Async Tests

Use `@pytest.mark.asyncio` decorator:

```python
@pytest.mark.asyncio
async def test_transcription_service(mock_voice_provider):
    service = TranscriptionService(mock_voice_provider)
    result = await service.get_full_transcript([b"audio"])
    assert isinstance(result, str)
```

### Using Fixtures

Fixtures are defined in `conftest.py`:

```python
def test_session_creation(session_manager):
    session = session_manager.create_session()
    assert session.session_id is not None
```

### Mocking

Use `unittest.mock` for mocking:

```python
from unittest.mock import Mock, AsyncMock, patch

def test_with_mock():
    mock_provider = Mock()
    mock_provider.is_connected = True
    
    # Use mock
```

### Async Mocks

```python
@pytest.mark.asyncio
async def test_async_function():
    mock = AsyncMock(return_value="result")
    result = await mock()
    assert result == "result"
```

### Patching

```python
@patch("src.module.function")
def test_with_patch(mock_function):
    mock_function.return_value = "mocked"
    # Test code
```

## Available Fixtures

### `mock_voice_provider`

Mocked voice provider with async methods:

```python
def test_something(mock_voice_provider):
    assert mock_voice_provider.is_connected is True
```

### `mock_transcription_result`

Sample transcription result:

```python
def test_transcription(mock_transcription_result):
    assert mock_transcription_result.text == "Hello world"
```

### `mock_vad_info`

Sample VAD information:

```python
def test_vad(mock_vad_info):
    assert mock_vad_info.inactivity_prob == 0.8
```

### `session_manager`

Real session manager instance:

```python
def test_sessions(session_manager):
    session = session_manager.create_session()
    # Test session operations
```

### `conversation_graph`

Real LangGraph conversation graph:

```python
def test_graph(conversation_graph):
    result = conversation_graph.invoke(state)
    # Test graph execution
```

### `sample_audio_bytes`

Sample audio data:

```python
def test_audio(sample_audio_bytes):
    assert len(sample_audio_bytes) > 0
```

## Test Markers

### Built-in Markers

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Slow-running tests

### Skip/Xfail

```python
@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    pass

@pytest.mark.xfail(reason="Known issue")
def test_broken_feature():
    assert False
```

### Parametrize

```python
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
])
def test_upper(input, expected):
    assert input.upper() == expected
```

## Testing Best Practices

### Arrange-Act-Assert

```python
def test_something():
    # Arrange
    service = MyService()
    
    # Act
    result = service.do_something()
    
    # Assert
    assert result == expected
```

### One Assert Per Test

Prefer focused tests:

```python
# Good
def test_session_has_id(session_manager):
    session = session_manager.create_session()
    assert session.session_id is not None

def test_session_has_timestamp(session_manager):
    session = session_manager.create_session()
    assert session.created_at is not None

# Avoid
def test_session_creation(session_manager):
    session = session_manager.create_session()
    assert session.session_id is not None
    assert session.created_at is not None
    assert session.metadata == {}
```

### Test Edge Cases

```python
def test_empty_input():
    service = AudioService()
    assert service.get_buffer_size() == 0

def test_invalid_input():
    with pytest.raises(ValueError):
        service.process(None)

def test_large_input():
    data = b"x" * 1_000_000
    service.process(data)
    assert service.get_buffer_size() == 1
```

### Mock External Dependencies

```python
def test_transcription_service(mock_voice_provider):
    # Don't call real Gradium API
    service = TranscriptionService(mock_voice_provider)
    # Test service logic
```

### Test Error Paths

```python
@pytest.mark.asyncio
async def test_transcription_empty_audio(mock_voice_provider):
    service = TranscriptionService(mock_voice_provider)
    
    with pytest.raises(TranscriptionError):
        async for _ in service.transcribe_audio([]):
            pass
```

## Coverage Goals

- **Minimum**: 70% overall coverage
- **Target**: 80% overall coverage
- **Critical paths**: 90%+ coverage

### Check Coverage

```bash
pytest --cov=src --cov-report=term-missing
```

### Coverage Report

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## Integration Tests

Integration tests use real components:

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_conversation_flow():
    # Test with real graph, real services
    # May use mock providers to avoid external calls
```

## Debugging Tests

### Run Single Test with Print

```bash
pytest -s tests/unit/test_prompts.py::test_get_system_prompt
```

### Use PDB

```python
def test_something():
    import pdb; pdb.set_trace()
    # Test code
```

### PyCharm/VS Code Debugging

Set breakpoints and run tests in debug mode.

## CI/CD Integration

Tests run automatically on:
- Pull requests
- Commits to main
- Scheduled runs

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -e .[dev]
      - run: pytest --cov
```

## Common Issues

### Async Test Not Running

Add `@pytest.mark.asyncio` decorator

### Fixture Not Found

Check `conftest.py` is in correct location

### Import Errors

Run `uv sync --all-extras` to install test dependencies

### Slow Tests

Use `-m "not slow"` to skip slow tests during development
