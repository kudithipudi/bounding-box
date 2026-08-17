"""Vision-LLM bounding-box detection via any OpenAI-compatible endpoint.

Two calls per run:

1. ``interpret()`` — a cheap text-only call that restates the user's free-text
   request as a crisp detection target (e.g. "find all circles" -> "all
   circles"). The app shows this to the user for confirmation *before* the
   (more expensive) vision call.
2. ``detect()`` — the vision call. It POSTs the normalized image (base64 data
   URL in an OpenAI-style image_url content part) plus the confirmed target and
   asks the model to return one or more bounding boxes as normalized [0,1]
   coordinates (same coordinate space as the displayed image).

Both calls tolerate empty/"None" content and unparseable JSON by retrying once
with a stricter "return only JSON" instruction before giving up.
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

_MAX_JSON_ATTEMPTS = 2  # 1 normal + 1 strict retry when parsing fails


class LlmError(Exception):
    pass


class BadBoxError(LlmError):
    pass


@dataclass
class DetectionBox:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float


@dataclass
class Detection:
    boxes: list[DetectionBox]
    model: str
    raw: str

    @property
    def primary(self) -> DetectionBox | None:
        return self.boxes[0] if self.boxes else None


INTERPRET_PROMPT = (
    "You turn a user's request for finding objects in an image into a short, "
    "concrete detection target. The user may describe objects in natural "
    "language (e.g. 'find all circles', 'banana', 'the red chair next to the "
    "window', 'the first three signatures'). Restate it as exactly what the "
    "model should look for. Keep quantifiers that control how many boxes are "
    "wanted: plural words like 'all'/'every'/'each' (one box per instance), "
    "and count/ordinal phrases like 'first three'/'the top two' (exactly that "
    "many, in order). A bare singular noun such as 'banana' still means every "
    "matching instance, so restate it as 'all bananas' unless the user limits "
    "the count. Return a single JSON object, with no extra text or markdown, "
    'shaped exactly like: {"target": "all bananas"}.'
)


DETECT_PROMPT = (
    "You are an object-detection assistant. The user supplies one image and a "
    "detection target. Find the described object(s) in the image and return a "
    "single JSON object, with no extra text or markdown, shaped exactly like: "
    '{"boxes": [{"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "label": "short label", "confidence": 0.95}]} '
    "where x1/y1/x2/y2 are normalized bounding-box corners (all values in "
    "[0,1] relative to the image's pixel dimensions, x1 < x2, y1 < y2) tightly "
    "around each found object. "
    "Unless the target limits the count, return one box per instance you can "
    "find: 'all circles' means every circle in the image, and a bare noun like "
    "'banana' also means every banana in the image. "
    "If the target names a count or an ordinal range (e.g. 'first three "
    "signatures', 'the top two rows', 'all 4 tires'), return exactly that many "
    "boxes, stopping once you have that many. 'First N'/'top N' means the N "
    "instances from the top of the image downward unless the user says "
    "otherwise, so order the returned boxes from top to bottom (smallest y1 "
    "first) to match. "
    "The label is a short phrase describing that box; confidence is your "
    "certainty in [0,1]. If nothing matches, return an empty list: {\"boxes\": []}."
)


def _user_prompt(description: str) -> str:
    return (
        f"Find and bound the following object(s) in the supplied image:\n\n"
        f"{description}\n\n"
        "Return only the JSON described in the system message."
    )


def _extract_json(content: str | None) -> dict:
    """Parse JSON out of the model's reply, tolerating stray prose/markdown.

    Raises BadBoxError on any failure, including an empty/None reply (some
    vision models return a literal 'None' or an empty string when their output
    lands in a reasoning field instead of content).
    """
    if not content or not content.strip():
        raise BadBoxError("Model returned an empty response.")
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


def _parse_boxes(data: dict) -> list[DetectionBox]:
    """Parse the model's reply into one or more boxes.

    Accepts both the canonical ``{"boxes": [...]}`` shape and a legacy single
    ``{"x1": ..., "y1": ..., ...}`` shape (in case a provider/model returns the
    old format).
    """
    raw_boxes = data.get("boxes")
    if raw_boxes is None:
        raw_boxes = [data]  # legacy single-box reply
    if not isinstance(raw_boxes, list):
        raise BadBoxError(f"Model response 'boxes' is not a list: {data!r}")

    boxes: list[DetectionBox] = []
    for entry in raw_boxes:
        if not isinstance(entry, dict):
            continue
        try:
            x1, y1, x2, y2 = _parse_box(entry)
        except BadBoxError:
            continue  # skip malformed entries rather than fail the whole run
        label = str(entry.get("label", "")).strip()
        try:
            confidence = float(entry.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        boxes.append(DetectionBox(
            x1=x1, y1=y1, x2=x2, y2=y2, label=label,
            confidence=min(max(confidence, 0.0), 1.0),
        ))
    return boxes


def _post(messages: list[dict], max_tokens: int = 1024) -> tuple[str | None, str]:
    """POST to the OpenAI-compatible endpoint with retry on transient failures.

    Returns (content, model). content may be None for empty model replies.
    """
    settings = get_settings()
    if not settings.llm_api_key:
        raise LlmError("No LLM_API_KEY configured.")
    base = settings.llm_base_url.rstrip("/")
    url = f"{base}/chat/completions"
    body = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": max_tokens,
        "messages": messages,
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
        message = data["choices"][0]["message"]
        # Some reasoning models leave `content` empty and put their answer in a
        # `reasoning` / `reasoning_content` field instead — fall back to those
        # so their output is still parsed (and can be re-sent by the caller).
        content = (
            message.get("content")
            or message.get("reasoning")
            or message.get("reasoning_content")
        )
        model = data.get("model", "") or settings.llm_model
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmError(f"Unexpected response shape: {(r.text or '')[:600].strip()}") from exc
    log.info(
        "llm call ok model=%s latency=%dms", model, int((time.perf_counter() - started) * 1000)
    )
    return content, model


def interpret(description: str) -> str:
    """Restate the user's request as a concrete detection target.

    Falls back to the raw description if the model can't be bothered.
    """
    messages = [
        {"role": "system", "content": INTERPRET_PROMPT},
        {"role": "user", "content": description},
    ]
    for attempt in range(_MAX_JSON_ATTEMPTS):
        content, _ = _post(messages, max_tokens=256)
        try:
            data = _extract_json(content)
            target = str(data.get("target", "")).strip()
        except BadBoxError:
            log.warning("interpret bad JSON attempt %d: %s", attempt + 1, content)
            if attempt == _MAX_JSON_ATTEMPTS - 1:
                break
            messages = messages + [
                {"role": "user", "content": "Output ONLY a single JSON object with no other text."}
            ]
            continue
        if target:
            return target
        break
    return description.strip()


def detect(image_data_url: str, description: str) -> Detection:
    """Detect the object(s) described by `description` in the given image.

    Raises LlmError / BadBoxError on failure (after one strict-format retry).
    Returns boxes normalized to [0,1] relative to the image the caller supplied.
    """
    messages = [
        {"role": "system", "content": DETECT_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _user_prompt(description)},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]
    last_exc: BadBoxError | None = None
    for attempt in range(_MAX_JSON_ATTEMPTS):
        content, model = _post(messages)
        try:
            boxes = _parse_boxes(_extract_json(content))
            return Detection(boxes=boxes, model=model, raw=content or "")
        except BadBoxError as exc:
            last_exc = exc
            log.warning("llm bad JSON attempt %d: %s", attempt + 1, exc)
            if attempt == _MAX_JSON_ATTEMPTS - 1:
                break
            # Retry with a strict instruction appended — models often comply
            # when told plainly to output nothing but JSON.
            messages = messages + [
                {"role": "user", "content": "Output ONLY a single JSON object with no other text."}
            ]
    assert last_exc is not None
    raise last_exc