from typing import Any, Optional
from enum import Enum


class PromptVersion(str, Enum):
    V1 = "v1"
    DEFAULT = "v1"


class PromptTemplate:
    def __init__(self, template: str, version: PromptVersion = PromptVersion.DEFAULT):
        self.template = template
        self.version = version
    
    def render(self, **kwargs: Any) -> str:
        return self.template.format(**kwargs)


SYSTEM_PROMPT_V1 = """You are a helpful AI voice assistant. You engage in natural, conversational dialogue with users.

Guidelines:
- Keep responses concise and natural for voice interaction
- Be friendly and engaging
- Ask clarifying questions when needed
- Acknowledge what the user says before responding
- Keep your responses focused and to the point (2-3 sentences typically)
"""

SYSTEM_PROMPTS = {
    PromptVersion.V1: PromptTemplate(SYSTEM_PROMPT_V1, PromptVersion.V1),
}


def get_system_prompt(version: Optional[PromptVersion] = None) -> str:
    version = version or PromptVersion.DEFAULT
    return SYSTEM_PROMPTS[version].render()


def get_custom_prompt(template: str, **context: Any) -> str:
    prompt = PromptTemplate(template)
    return prompt.render(**context)
