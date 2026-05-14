"""Shared helper for routing local handlers through a vLLM OpenAI-compatible
text-completions server instead of loading the model in-process.

Local handlers render prompts via Jinja templates (raw strings), so we use the
``/v1/completions`` endpoint (not chat completions). Sampling parameters match
those used by the direct vLLM ``SamplingParams`` path exactly; non-standard
OpenAI fields (``top_k``, ``min_p``) are forwarded via ``extra_body``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"

_EXTRA_BODY_KEYS = {"top_k", "min_p", "repetition_penalty"}


def make_client(base_url: str = DEFAULT_BASE_URL, api_key: str = DEFAULT_API_KEY):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


def _split_params(sampling: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    extra_body: dict[str, Any] = {}
    standard: dict[str, Any] = {}
    for k, v in sampling.items():
        if k in _EXTRA_BODY_KEYS:
            extra_body[k] = v
        else:
            standard[k] = v
    return standard, extra_body


def run_completions(
    client,
    model: str,
    prompts: list[str],
    sampling: dict[str, Any],
    extra_body_extra: dict[str, Any] | None = None,
) -> list[str]:
    standard, extra_body = _split_params(sampling)
    if extra_body_extra:
        extra_body = {**extra_body, **extra_body_extra}

    def infer_one(prompt: str) -> str:
        try:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                extra_body=extra_body or None,
                **standard,
            )
            text = response.choices[0].text or ""
            return text.strip()
        except Exception as exc:
            return f"[ERROR]: {exc}"

    results: list[str] = [""] * len(prompts)
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = {pool.submit(infer_one, p): i for i, p in enumerate(prompts)}
        with tqdm(total=len(prompts), desc="API inference") as pbar:
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
                pbar.update(1)
    return results
