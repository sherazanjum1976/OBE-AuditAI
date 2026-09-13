"""
llm_provider.py
---------------
Provider-agnostic LLM interface supporting Groq and Google Gemini.

Design notes:
- API keys are only ever held in memory (Streamlit session_state) for the
  duration of the session. They are never written to disk or logged.
- Each provider uses exactly ONE fixed, curated text model (see
  GROQ_MODEL / GEMINI_MODEL below). There is no model-picker dropdown:
  this app fires several sequential LLM calls per audit run, and letting
  the user pick an arbitrary model from a live catalog previously caused
  two real problems — (1) some Groq-hosted "models" are actually
  non-chat audio/TTS models that reject text requests outright, and
  (2) many chat models have much tighter free-tier rate/quota limits,
  causing repeated 429 errors mid-audit. Pinning one well-tested,
  generous-free-tier model per provider avoids both issues.
- All calls are wrapped so that invalid keys, unsupported models, rate
  limits and quota errors surface as friendly, actionable messages
  instead of raw stack traces.
"""

from __future__ import annotations

import re
import time
import requests


class LLMError(Exception):
    """Friendly, user-facing LLM error."""
    pass


# --------------------------------------------------------------------------- #
# Fixed model configuration — one text model per provider.
# Last manually reviewed 2026-09.
# --------------------------------------------------------------------------- #
GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3.5-flash-lite"


def get_model_for_provider(provider: str) -> str:
    """Return the single fixed model id used for a given provider."""
    if provider == "Groq":
        return GROQ_MODEL
    elif provider == "Gemini":
        return GEMINI_MODEL
    raise LLMError(f"Unknown provider: {provider}")


# --------------------------------------------------------------------------- #
# Chat completion calls
# --------------------------------------------------------------------------- #
def _parse_retry_after_seconds(resp: "requests.Response", attempt: int) -> float:
    """Figure out how long to wait before retrying a Groq 429.

    Prefers the standard Retry-After header. Groq's free tier doesn't
    always send one, so we fall back to parsing the "try again in Xs"
    wording Groq puts in the error message body, and finally to a plain
    exponential backoff if neither is present.
    """
    header_val = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if header_val:
        try:
            return max(float(header_val), 0.5)
        except ValueError:
            pass
    try:
        msg = resp.json().get("error", {}).get("message", "")
        match = re.search(r"try again in ([\d.]+)s", msg, re.IGNORECASE)
        if match:
            return max(float(match.group(1)), 0.5)
    except Exception:
        pass
    return min(2.0 * (2 ** attempt), 20.0)  # 2s, 4s, 8s, ... capped at 20s


def _call_groq(api_key: str, model: str, system_prompt: str, user_prompt: str,
                temperature: float = 0.2, max_tokens: int = 2000,
                max_retries: int = 4) -> str:
    last_wait = 0.0
    # openai/gpt-oss-120b is a reasoning model. Groq docs confirm it doesn't
    # support `reasoning_format` (so we can't ask it to hide its
    # chain-of-thought), and by default its internal reasoning both (a) eats
    # into the max_tokens budget, sometimes leaving too little room for the
    # actual JSON answer, and (b) can leak into the `content` field alongside
    # the answer. `reasoning_effort: low` is the supported lever to reduce
    # both effects for a structured-output task like this one, where we
    # don't need deep reasoning — just correct extraction/classification.
    is_gpt_oss = model.startswith("openai/gpt-oss")

    for attempt in range(max_retries + 1):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if is_gpt_oss:
            payload["reasoning_effort"] = "low"

        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
        except requests.exceptions.RequestException as e:
            raise LLMError(f"Network error while contacting Groq: {e}")

        if resp.status_code == 401:
            raise LLMError("Invalid Groq API key. Please check and re-enter your key.")
        if resp.status_code == 404:
            raise LLMError(f"The selected model '{model}' is not available on Groq. Please pick another model.")

        if resp.status_code == 429:
            # Distinguish a per-minute rate limit (worth retrying — it will
            # clear itself in seconds) from a hard daily/monthly quota
            # exhaustion (retrying won't help, so fail fast with a clear
            # message instead of stalling the whole audit for ~40s).
            body_text = resp.text[:500].lower()
            is_hard_quota = "per day" in body_text or "monthly" in body_text or "insufficient_quota" in body_text
            if is_hard_quota or attempt == max_retries:
                waited_note = f" (already waited ~{last_wait:.0f}s across retries)" if last_wait else ""
                raise LLMError(
                    "Groq rate limit or quota exceeded. Please wait a moment and try again, "
                    f"or switch models.{waited_note}"
                )
            wait_s = _parse_retry_after_seconds(resp, attempt)
            last_wait += wait_s
            time.sleep(wait_s)
            continue

        if resp.status_code >= 400:
            raise LLMError(f"Groq API error ({resp.status_code}): {resp.text[:300]}")

        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError):
            raise LLMError("Unexpected response format from Groq API.")

    raise LLMError("Groq rate limit or quota exceeded. Please wait a moment and try again, or switch models.")


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
    """Unified entry point for chat completion across providers.

    `model` is expected to be the fixed model returned by
    get_model_for_provider(provider) — the app never lets the user choose
    a different one — but we still validate it here as a safety net in
    case a caller passes something unexpected.
    """
    if not api_key:
        raise LLMError("Please enter an API key before running the audit.")
    expected_model = get_model_for_provider(provider)
    if model != expected_model:
        raise LLMError(
            f"'{model}' is not the supported model for {provider} "
            f"(expected '{expected_model}')."
        )
    if provider == "Groq":
        return _call_groq(api_key, model, system_prompt, user_prompt, temperature, max_tokens)
    elif provider == "Gemini":
        return _call_gemini(api_key, model, system_prompt, user_prompt, temperature, max_tokens)
    else:
        raise LLMError(f"Unknown provider: {provider}")
