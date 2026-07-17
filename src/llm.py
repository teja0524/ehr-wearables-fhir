"""
Provider-agnostic LLM completion.

The pipeline's code mapper sends one prompt and expects one text response, so a
single `complete()` is all the rest of the system needs. Routing:

  * Anthropic (claude-*)  → native `anthropic` SDK (the verified default path).
  * Everything else        → LiteLLM, which unifies OpenAI, Google Gemini,
                             Meta Llama (via Groq/Together/Ollama), etc. behind
                             one call and reads each provider's key from the
                             environment.

Required environment variables (only for the providers you actually use):
  ANTHROPIC_API_KEY                       claude-*
  OPENAI_API_KEY                          gpt-*, o*
  GEMINI_API_KEY (or GOOGLE_API_KEY)      gemini/*
  GROQ_API_KEY / TOGETHER_API_KEY / …     llama / groq/* / together_ai/*

Model ids follow LiteLLM's convention for non-Anthropic providers, e.g.
"gpt-4o", "gemini/gemini-1.5-pro", "groq/llama-3.3-70b-versatile".
"""

from __future__ import annotations

import os


def provider_for(model: str) -> str:
    """Best-effort provider label from a model id (for routing + error messages)."""
    m = (model or "").lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4") or m.startswith("openai/"):
        return "openai"
    if m.startswith("gemini") or m.startswith("google/"):
        return "gemini"
    if "llama" in m or m.startswith("groq/") or m.startswith("together"):
        return "meta"
    return "other"


def complete(prompt: str, model: str, max_tokens: int = 400,
             anthropic_api_key: str | None = None) -> str:
    """Return the model's text response to a single user prompt."""
    if provider_for(model) == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", ""))
        resp = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    # All non-Anthropic providers go through LiteLLM.
    try:
        import litellm
    except ImportError as exc:
        raise RuntimeError(
            f"Model '{model}' requires LiteLLM for multi-provider support. "
            f"Install it with: pip install litellm"
        ) from exc

    # LiteLLM maps GEMINI_API_KEY/GOOGLE_API_KEY itself; nothing else to do here.
    resp = litellm.completion(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp["choices"][0]["message"]["content"]
