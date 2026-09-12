"""
llm_provider.py
---------------
Provider-agnostic LLM interface supporting Groq and Google Gemini.

Design notes:
- API keys are only ever held in memory (Streamlit session_state) for the
  duration of the session. They are never written to disk or logged.
- Model lists are fetched live from each provider's model-listing endpoint
  where possible, with a small, clearly documented fallback list of
  currently-supported free-tier models in case the listing call fails
  (e.g. due to network restrictions or an invalid key at listing time).
- All calls are wrapped so that invalid keys, unsupported models, rate
  limits and quota errors surface as friendly, actionable messages
  instead of raw stack traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import requests


class LLMError(Exception):
    """Friendly, user-facing LLM error."""
    pass


@dataclass
class ModelInfo:
    id: str
    label: str
    recommended: bool = False


# --------------------------------------------------------------------------- #
# Fallback model lists (used only if live listing fails).
# Keep this short and periodically reviewed — these are free-tier-suitable,
# lightweight models as of the last manual review of this file.
# --------------------------------------------------------------------------- #
GROQ_FALLBACK_MODELS = [
    ModelInfo(id="llama-3.1-8b-instant", label="Llama 3.1 8B Instant (fast, free-tier)", recommended=True),
    ModelInfo(id="llama-3.3-70b-versatile", label="Llama 3.3 70B Versatile (higher quality)"),
    ModelInfo(id="gemma2-9b-it", label="Gemma2 9B IT"),
]

GEMINI_FALLBACK_MODELS = [
    ModelInfo(id="gemini-2.5-flash", label="Gemini 2.5 Flash (fast, free-tier)", recommended=True),
    ModelInfo(id="gemini-2.5-flash-lite", label="Gemini 2.5 Flash-Lite (lightweight, highest free rate limit)"),
    ModelInfo(id="gemini-3-flash-preview", label="Gemini 3 Flash Preview (newer, may have tighter free quota)"),
]
# NOTE (last manually reviewed 2026-09): Google moved Pro-tier models behind a
# paid plan as of April 2026 — only Flash / Flash-Lite models remain free-tier
# eligible via Google AI Studio keys. Gemini 1.x models have been phased out.
# This fallback list is only used if the live /v1beta/models listing call
# fails; the app otherwise always prefers the live list.


# --------------------------------------------------------------------------- #
# Model discovery
# --------------------------------------------------------------------------- #
def list_groq_models(api_key: str) -> List[ModelInfo]:
    """Attempt to list available Groq models via their /models endpoint."""
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code != 200:
            return GROQ_FALLBACK_MODELS
        data = resp.json().get("data", [])
        candidate_ids = [m["id"] for m in data if "id" in m]
        # Prefer known lightweight instruct/chat models; filter out
        # whisper/audio/guard/vision models not suited for this text task.
        blocked_substrings = ["whisper", "guard", "vision", "tts", "embedding"]
        filtered = [
            mid for mid in candidate_ids
            if not any(b in mid.lower() for b in blocked_substrings)
        ]
        if not filtered:
            return GROQ_FALLBACK_MODELS
        models = []
        for mid in filtered:
            recommended = "8b-instant" in mid or "llama-3.1-8b" in mid
            models.append(ModelInfo(id=mid, label=mid, recommended=recommended))
        if not any(m.recommended for m in models):
            models[0].recommended = True
        return sorted(models, key=lambda m: not m.recommended)
    except Exception:
        return GROQ_FALLBACK_MODELS


def list_gemini_models(api_key: str) -> List[ModelInfo]:
    """Attempt to list available Gemini models via the generativelanguage API."""
    try:
        resp = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=10,
        )
        if resp.status_code != 200:
            return GEMINI_FALLBACK_MODELS
        data = resp.json().get("models", [])
        models = []
        for m in data:
            name = m.get("name", "").replace("models/", "")
            methods = m.get("supportedGenerationMethods", [])
            if "generateContent" not in methods:
                continue
            if "flash" not in name and "pro" not in name:
                continue
            # Prefer a stable (non-preview), non-lite Flash model as the default
            # recommendation, since Pro models are generally paid-only and
            # preview models can have tighter/less predictable free quotas.
            recommended = "flash" in name and "lite" not in name and "preview" not in name and "pro" not in name
            models.append(ModelInfo(id=name, label=name, recommended=recommended))
        if not models:
            return GEMINI_FALLBACK_MODELS
        if not any(m.recommended for m in models):
            models[0].recommended = True
        return sorted(models, key=lambda m: not m.recommended)
    except Exception:
        return GEMINI_FALLBACK_MODELS


def list_models_for_provider(provider: str, api_key: str) -> List[ModelInfo]:
    if provider == "Groq":
        return list_groq_models(api_key) if api_key else GROQ_FALLBACK_MODELS
    elif provider == "Gemini":
        return list_gemini_models(api_key) if api_key else GEMINI_FALLBACK_MODELS
    return []


# --------------------------------------------------------------------------- #
# Chat completion calls
# --------------------------------------------------------------------------- #
def _call_groq(api_key: str, model: str, system_prompt: str, user_prompt: str,
                temperature: float = 0.2, max_tokens: int = 2000) -> str:
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
    except requests.exceptions.RequestException as e:
        raise LLMError(f"Network error while contacting Groq: {e}")

    if resp.status_code == 401:
        raise LLMError("Invalid Groq API key. Please check and re-enter your key.")
    if resp.status_code == 404:
        raise LLMError(f"The selected model '{model}' is not available on Groq. Please pick another model.")
    if resp.status_code == 429:
        raise LLMError("Groq rate limit or quota exceeded. Please wait a moment and try again, or switch models.")
    if resp.status_code >= 400:
        raise LLMError(f"Groq API error ({resp.status_code}): {resp.text[:300]}")

    try:
        return resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError):
        raise LLMError("Unexpected response format from Groq API.")


def _call_gemini(api_key: str, model: str, system_prompt: str, user_prompt: str,
                  temperature: float = 0.2, max_tokens: int = 2000) -> str:
    model_path = model if model.startswith("models/") else f"models/{model}"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent"
        f"?key={api_key}"
    )
    try:
        resp = requests.post(
            url,
            json={
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
            timeout=60,
        )
    except requests.exceptions.RequestException as e:
        raise LLMError(f"Network error while contacting Gemini: {e}")

    if resp.status_code == 400:
        raise LLMError("Invalid Gemini request or API key. Please check your key and model selection.")
    if resp.status_code == 403:
        raise LLMError("Invalid or unauthorized Gemini API key.")
    if resp.status_code == 404:
        raise LLMError(f"The selected model '{model}' is not available on Gemini. Please pick another model.")
    if resp.status_code == 429:
        raise LLMError("Gemini rate limit or quota exceeded. Please wait a moment and try again, or switch models.")
    if resp.status_code >= 400:
        raise LLMError(f"Gemini API error ({resp.status_code}): {resp.text[:300]}")

    try:
        candidates = resp.json()["candidates"]
        parts = candidates[0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, ValueError):
        raise LLMError("Unexpected response format from Gemini API. The prompt may have been blocked by safety filters.")


def generate(provider: str, api_key: str, model: str, system_prompt: str, user_prompt: str,
             temperature: float = 0.2, max_tokens: int = 2000) -> str:
    """Unified entry point for chat completion across providers."""
    if not api_key:
        raise LLMError("Please enter an API key before running the audit.")
    if provider == "Groq":
        return _call_groq(api_key, model, system_prompt, user_prompt, temperature, max_tokens)
    elif provider == "Gemini":
        return _call_gemini(api_key, model, system_prompt, user_prompt, temperature, max_tokens)
    else:
        raise LLMError(f"Unknown provider: {provider}")
