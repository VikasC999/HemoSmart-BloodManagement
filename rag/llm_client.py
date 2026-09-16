"""
HemoSmart - LLM Client Wrapper (Person B)
------------------------------------------------
A single, swappable interface for calling either Ollama (local, free,
offline) or Groq (cloud, free tier, faster/more accurate). Everything
downstream (RAG explanation, PDF extraction) calls `.generate(prompt)`
and never needs to know which provider is behind it.

Usage:
    llm = get_llm_client(provider="ollama")     # or provider="groq"
    response = llm.generate("Explain this in one sentence: ...")
"""

from abc import ABC, abstractmethod
import os


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OllamaClient(BaseLLMClient):
    """Local, offline LLM via Ollama. Requires `ollama pull llama3.2` first."""

    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

    def generate(self, prompt: str) -> str:
        import ollama
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]


class GroqClient(BaseLLMClient):
    """Cloud LLM via Groq's free tier. Requires GROQ_API_KEY env variable."""

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at console.groq.com "
                "and set it with: set GROQ_API_KEY=your_key (Windows) "
                "or export GROQ_API_KEY=your_key (Mac/Linux)"
            )
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


def get_llm_client(provider: str = "ollama", **kwargs) -> BaseLLMClient:
    """
    Factory function -- this is the ONE place that decides which LLM
    provider is active. Swapping providers project-wide means changing
    one line here, not hunting through every file that calls an LLM.
    """
    provider = provider.lower()
    if provider == "ollama":
        return OllamaClient(**kwargs)
    elif provider == "groq":
        return GroqClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Use 'ollama' or 'groq'.")