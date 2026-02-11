import torch
from typing import Any, Optional, Iterator
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

from src.core.logger import logger


class HuggingFaceLLM(BaseChatModel):
    """Custom LangChain LLM using HuggingFace transformers with streaming support."""

    model_id: str
    device: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.95
    repetition_penalty: float = 1.0

    _tokenizer: Any = None
    _model: Any = None
    _device: torch.device = None
    _dtype: torch.dtype = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Load the model and tokenizer with appropriate device/dtype."""
        logger.info(f"Loading HuggingFace model: {self.model_id}")

        if self.device:
            self._device = torch.device(self.device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._dtype = torch.float16 if self._device.type == "cuda" else torch.float32

        logger.info(f"Using device: {self._device}, dtype: {self._dtype}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=self._dtype,
        )

        self._model = self._model.to(self._device)

        self._model.eval()
        logger.info(f"Model loaded successfully on {self._device}")

    @property
    def _llm_type(self) -> str:
        return "huggingface_transformers"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response without streaming."""
        formatted_messages = [
            {"role": self._get_role(msg), "content": msg.content}
            for msg in messages
        ]

        inputs = self._tokenizer.apply_chat_template(
            formatted_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_length:]
        response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream response token by token."""
        formatted_messages = [
            {"role": self._get_role(msg), "content": msg.content}
            for msg in messages
        ]

        inputs = self._tokenizer.apply_chat_template(
            formatted_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": True,
            "pad_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk

        thread.join()

    @staticmethod
    def _get_role(message: BaseMessage) -> str:
        """Convert LangChain message type to chat role."""
        if isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, AIMessage):
            return "assistant"
        elif isinstance(message, SystemMessage):
            return "system"
        else:
            return "user"
