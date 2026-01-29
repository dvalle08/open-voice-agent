import pytest

from src.agent.prompts import (
    PromptTemplate,
    PromptVersion,
    get_system_prompt,
    get_custom_prompt,
)


def test_prompt_template_render():
    template = PromptTemplate("Hello {name}, you are {age} years old.")
    result = template.render(name="Alice", age=30)
    assert result == "Hello Alice, you are 30 years old."


def test_get_system_prompt():
    prompt = get_system_prompt()
    assert "helpful AI voice assistant" in prompt
    assert "conversational dialogue" in prompt


def test_get_system_prompt_with_version():
    prompt = get_system_prompt(PromptVersion.V1)
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_get_custom_prompt():
    result = get_custom_prompt("User {name} asked: {question}", name="Bob", question="What time is it?")
    assert result == "User Bob asked: What time is it?"


def test_prompt_version_enum():
    assert PromptVersion.DEFAULT == PromptVersion.V1
    assert PromptVersion.V1.value == "v1"
