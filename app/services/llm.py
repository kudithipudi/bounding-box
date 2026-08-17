"""Vision-LLM bounding-box detection via any OpenAI-compatible endpoint.

The app POSTs the normalized image (as a base64 data URL in an OpenAI-style
image_url content part) plus the user's description to {base}/chat/completions
and asks the model to return the detected object's bounding box as normalized
[0,1] coordinates (same coordinate space as the displayed image).
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504, 522, 524}
_RETRYABLE_EXC = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.NetworkError)


class LlmError(Exception):
    pass


class BadBoxError(LlmError):
    pass


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float
    model: str
    raw: str


SYSTEM_PROMPT = (
    "You are an object-detection assistant. The user supplies one image and a "
    "text description of an object they want located. Find the object in the "
    "image and return a single JSON object, with no extra text or markdown, "
    "shaped exactly like: "
    '{"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "label": "short label", "confidence": 0.95} '
    "where x1/y1/x2/y2 are the normalized bounding-box corners (all values in "
    "[0,1] relative to the image's pixel dimensions, x1 < x2, y1 < y2) tightly "
    "around the described object. The label should be a short phrase describing "
    "what was detected; confidence is your certainty in [0,1]."
)


def _user_prompt(description: str) -> str:
    return (
        f"Find and bound the following object in the supplied image:\n\n"
        f"{description}\n\n"
        "Return only the JSON described in the system message."
    )


def _extract_json(content: str) -> dict:
    """Parse JSON out of the model's reply, tolerating stray prose/markdown."""
    content = content.strip()
    # Strip a ```json ... ``` fence if the model wrapped the answer.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        content = fenced.group(1)
    else:
        brace = re.search(r"\{.*\}", content, re.DOTALL)
        if brace:
            content = brace.group(0)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise BadBoxError(f"Model returned unparseable JSON: {content[:300]!r}") from exc
    if not isinstance(data, dict):
        raise BadBoxError("Model response was not a JSON object.")
    return data


def _parse_box(data: dict) -> tuple[float, float, float, float]:
    try:
        x1 = float(data["x1"])
        y1 = float(data["y1"])
        x2 = float(data["x2"])
        y2 = float(data["y2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BadBoxError(f"Model response missing box coordinates: {data!r}") from exc

    # Clamp to the unit square and fix degenerate boxes so the overlay is always sane.
    x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
    y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
        raise BadBoxError(f"Detected box is degenerate: {data!r}")
    return x1, y1, x2, y2


def _call_chat(image_data_url: str, description: str) -> tuple[str, str]:
    """POST to the OpenAI-compatible endpoint with retry on transient failures.

    Returns (content, model).
    """
    settings = get_settings()
    if not settings.llm_api_key:
        raise LlmError("No LLM_API_KEY configured.")
    base = settings.llm_base_url.rstrip("/")
    url = f"{base}/chat/completions"
    body = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _user_prompt(description)},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(settings.llm_timeout_seconds)

    backoffs = [0.0] + [0.8 * (2 ** i) for i in range(settings.llm_max_retries)]
    last_exc: Exception | None = None
    started = time.perf_counter()
    for attempt, wait in enumerate(backoffs):
        if wait:
            log.warning("llm retry %d after %.1fs: %s", attempt, wait, last_exc)
            time.sleep(wait)
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, json=body, headers=headers)
        except _RETRYABLE_EXC as e:
            last_exc = e
            continue
        if 400 <= r.status_code < 500 and r.status_code not in _RETRYABLE_STATUS:
            raise LlmError(
                f"{r.status_code} from {settings.llm_base_url} "
                f"(model={settings.llm_model}): {(r.text or '')[:600].strip()}"
            )
        if r.status_code in _RETRYABLE_STATUS and attempt < len(backoffs) - 1:
            last_exc = httpx.HTTPStatusError(
                f"{r.status_code} transient", request=r.request, response=r
            )
            continue
        break
    else:
        raise LlmError(f"LLM call failed after retries: {last_exc}")

    if r.status_code >= 400:
        raise LlmError(
            f"{r.status_code} from {settings.llm_base_url} after "
            f"{attempt + 1} attempt(s): {(r.text or '')[:600].strip()}"
        )
    data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
        model = data.get("model", "") or settings.llm_model
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmError(f"Unexpected response shape: {(r.text or '')[:600].strip()}") from exc
    log.info(
        "llm call ok model=%s latency=%dms", model, int((time.perf_counter() - started) * 1000)
    )
    return str(content), model


def detect(image_data_url: str, description: str) -> Detection:
    """Detect the object described by `description` in the given image.

    Raises LlmError / BadBoxError on failure. The returned box is normalized
    to [0,1] relative to the image the caller supplied.
    """
    content, model = _call_chat(image_data_url, description)
    data = _extract_json(content)
    x1, y1, x2, y2 = _parse_box(data)
    label = str(data.get("label", "")).strip()
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    return Detection(
        x1=x1, y1=y1, x2=x2, y2=y2, label=label, confidence=confidence,
        model=model, raw=content,
    )