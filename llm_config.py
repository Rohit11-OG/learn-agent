"""Shared LLM config — swap the model or provider here, in ONE place.

Every agent file imports get_llm() from here, so changing the model is a
one-line edit instead of touching every file.
"""

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()  # loads NVIDIA_API_KEY from .env

# NVIDIA NIM — Nemotron Super (agentic flagship, strong tool calling)
MODEL = "nvidia/nemotron-3-super-120b-a12b"


def get_llm(temperature: float = 0):
    """Return the project's chat model. Change MODEL above to swap everywhere."""
    return ChatNVIDIA(model=MODEL, temperature=temperature)
