"""Local LLM client. Two backends, same return shape (Ollama's):

    {"message": {"content": str}, "logprobs": [{"top_logprobs": [{"token", "logprob"}, ...]}], ...}

- "llama" (default): llama-server's OpenAI-compatible API on DECISOR_LLAMA_URL. It serves several
  requests at once (continuous batching), so generation and labeling do not queue behind each other.
- "ollama": Ollama's /api/chat.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

BACKEND = os.environ.get("DECISOR_LLM_BACKEND", "llama")
LLAMA = os.environ.get("DECISOR_LLAMA_URL", "http://127.0.0.1:8089").rstrip("/")
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA


def _post(url: str, body: dict, timeout: int, retries: int) -> dict:
    err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post(url, json=body, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM request failed: {err}")


def chat(model: str, messages: list[dict[str, str]], *, fmt: Any = None, temperature: float = 0.8,
         num_predict: int = 2048, logprobs: int = 0, timeout: int = 900, retries: int = 3,
         num_ctx: int = 8192) -> dict[str, Any]:
    if BACKEND == "llama":
        body: dict[str, Any] = {
            "model": model, "messages": messages, "temperature": temperature, "max_tokens": num_predict,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if temperature > 0:
            body.update({"top_p": 0.95, "top_k": 20, "presence_penalty": 1.0})
        if fmt is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": fmt}}
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = logprobs
        out = _post(f"{LLAMA}/v1/chat/completions", body, timeout, retries)
        ch = out["choices"][0]
        lp = ((ch.get("logprobs") or {}).get("content")) or []
        return {"message": {"content": ch["message"].get("content") or ""},
                "logprobs": [{"top_logprobs": t.get("top_logprobs") or []} for t in lp],
                "eval_count": (out.get("usage") or {}).get("completion_tokens", 0)}
    body = {
        "model": model, "messages": messages, "stream": False, "think": False,
        "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx},
        "keep_alive": "30m",
    }
    if fmt is not None:
        body["format"] = fmt
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = logprobs
    return _post(f"{OLLAMA}/api/chat", body, timeout, retries)


def chat_json(model: str, messages: list[dict[str, str]], schema: dict[str, Any], **kw) -> Any:
    out = chat(model, messages, fmt=schema, **kw)
    return json.loads(out["message"]["content"])
