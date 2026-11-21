"""Classify feedback text into triage categories using a configurable LLM."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

CATEGORIES = [
    "Bug Report",
    "Feature Request",
    "Accessibility Concern",
    "Question",
    "Other",
]

SYSTEM_PROMPT = """You are a product feedback triage assistant.
Classify each feedback item into exactly one category:
- Bug Report: broken behavior, crashes, errors, regressions, or incorrect output.
- Feature Request: new capability, enhancement, or improvement to existing behavior.
- Accessibility Concern: barriers for users with disabilities (screen readers, contrast, keyboard nav, etc.).
- Question: how-to, clarification, documentation, or support-style inquiry.
- Other: praise, off-topic content, spam, or feedback that does not fit the categories above.

Respond with valid JSON only, using this schema:
{"category": "<one of the five categories>", "rationale": "<one concise sentence explaining why>"}
"""

USER_PROMPT_TEMPLATE = """Classify the following feedback:

\"\"\"
{text}
\"\"\"
"""


def _normalize_category(value: str) -> str:
    cleaned = value.strip()
    for category in CATEGORIES:
        if cleaned.lower() == category.lower():
            return category
    raise ValueError(f"Unknown category returned by model: {value!r}")


def _parse_model_response(raw: str) -> dict[str, str]:
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        object_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not object_match:
            raise ValueError(f"Model did not return JSON: {raw!r}") from None
        payload = json.loads(object_match.group(0))

    category = _normalize_category(str(payload.get("category", "")))
    rationale = " ".join(str(payload.get("rationale", "")).split())
    if not rationale:
        raise ValueError("Model response missing rationale.")
    return {"category": category, "rationale": rationale}


def _call_openai(text: str, api_key: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
        ],
    )
    return response.choices[0].message.content or ""


def _call_anthropic(text: str, api_key: str, model: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)}],
    )
    return response.content[0].text


def classify_text(
    text: str,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, str]:
    """Send feedback text to the configured LLM and return category + rationale."""
    chosen_provider = (provider or os.environ.get("LLM_PROVIDER", "openai")).lower()
    if chosen_provider == "openai":
        resolved_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
        resolved_model = (
            model
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("LLM_MODEL")
            or "gpt-4o-mini"
        )
        if not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY or LLM_API_KEY is required for provider 'openai'."
            )
        raw = _call_openai(text, resolved_key, resolved_model)
    elif chosen_provider == "anthropic":
        resolved_key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
        resolved_model = (
            model
            or os.environ.get("ANTHROPIC_MODEL")
            or os.environ.get("LLM_MODEL")
            or "claude-haiku-4-5-20251001"
        )
        if not resolved_key:
            raise ValueError(
                "ANTHROPIC_API_KEY or LLM_API_KEY is required for provider 'anthropic'."
            )
        raw = _call_anthropic(text, resolved_key, resolved_model)
    else:
        raise ValueError(
            f"Unsupported LLM provider: {chosen_provider}. Use 'openai' or 'anthropic'."
        )

    return _parse_model_response(raw)


def classify_items(
    items: list[dict[str, Any]],
    *,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Classify a list of feedback items in place, adding category and rationale."""
    labeled: list[dict[str, Any]] = []
    for item in items:
        result = classify_text(
            item["text"],
            provider=provider,
            api_key=api_key,
            model=model,
        )
        labeled.append({**item, **result})
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one feedback string.")
    parser.add_argument("text", help="Feedback text to classify.")
    parser.add_argument("--provider", help="LLM provider override: openai or anthropic.")
    parser.add_argument("--model", help="Model name override.")
    args = parser.parse_args()

    result = classify_text(args.text, provider=args.provider, model=args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
