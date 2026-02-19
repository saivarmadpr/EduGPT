import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.9) -> ChatOpenAI:
    """Create a ChatOpenAI instance using env-based configuration.

    Supports both direct OpenAI and OpenRouter (OpenAI-compatible) backends
    via OPENAI_API_BASE / OPENAI_MODEL environment variables.
    """
    kwargs: dict = {"temperature": temperature}

    base_url = os.environ.get("OPENAI_API_BASE")
    if base_url:
        kwargs["base_url"] = base_url

    model = os.environ.get("OPENAI_MODEL")
    if model:
        kwargs["model"] = model

    return ChatOpenAI(**kwargs)
