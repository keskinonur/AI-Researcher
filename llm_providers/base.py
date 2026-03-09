# llm_providers/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Iterator, Union, TypedDict

class Message(TypedDict, total=False):
    role: str       # "system" | "user" | "assistant"
    content: str

class LLMClient(ABC):
    """
    Tüm sağlayıcı istemcileri için ortak arayüz.
    """
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        stream: bool = False,
    ) -> Union[Iterator[str], str]:
        """
        messages: [{"role": "...", "content": "..."}]
        stream=False -> tek string döner
        stream=True  -> iterator döner (parça parça text)
        """
        raise NotImplementedError
