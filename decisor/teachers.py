"""The teacher: Jev (TypeSafe API). Returns, per question, a probability vector aligned with
`Question.options`.

A local open-weights teacher (Qwen3.5-9B) was tried and dropped: 0.51 / 0.60 / 0.45 accuracy against
typed-decisions gold (choice / noul / score) versus Jev's 0.72 / 0.80 / 0.73. Qwen only generates
synthetic cases now (scripts/generate.py).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import requests

from decisor.schema import normalize_probs, parse_questions



def load_env(path: str = ".env") -> None:
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


class BudgetExceeded(RuntimeError):
    pass


class Jev:
    """TypeSafe /v1/systemone client with a persistent spend cap."""

    # Published input price is $0.042 / 1M tokens; output price is not published, so it is
    # counted conservatively at 5x.
    PRICE_IN = 0.042 / 1e6
    PRICE_OUT = 0.21 / 1e6

    def __init__(self, usage_file: str = "data/labels/jev_usage.json") -> None:
        load_env()
        self.key = os.environ["TYPESAFE_API_KEY"]
        self.base = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
        self.budget = float(os.environ.get("JEV_BUDGET_USD", "10"))
        self.usage_path = Path(usage_file)
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.usage = json.loads(self.usage_path.read_text()) if self.usage_path.exists() else \
            {"input_tokens": 0, "output_tokens": 0, "requests": 0}

    @property
    def spent(self) -> float:
        return self.usage["input_tokens"] * self.PRICE_IN + self.usage["output_tokens"] * self.PRICE_OUT

    def _record(self, usage: dict[str, int]) -> None:
        with self.lock:
            self.usage["input_tokens"] += int(usage.get("input_tokens", 0))
            self.usage["output_tokens"] += int(usage.get("output_tokens", 0))
            self.usage["requests"] += 1
            self.usage_path.write_text(json.dumps(self.usage))

    def ask(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        if self.spent >= self.budget:
            raise BudgetExceeded(f"Jev budget reached: ${self.spent:.2f} of ${self.budget:.2f}")
        body = {"model": "jev-latest", "state": state, "questions": questions}
        for attempt in range(6):
            try:
                r = requests.post(f"{self.base}/v1/systemone", json=body, timeout=120,
                                  headers={"Authorization": f"Bearer {self.key}"})
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(60, 2 ** attempt * 2))
                    continue
                r.raise_for_status()
                out = r.json()
                self._record(out.get("usage") or {})
                return out
            except requests.RequestException:
                time.sleep(min(60, 2 ** attempt * 2))
        raise RuntimeError("Jev request failed after retries")

    def distributions(self, state: Any, questions: dict[str, Any]) -> dict[str, list[float]]:
        out = self.ask(state, questions)
        answers = out.get("answers") or {}
        res = {}
        for q in parse_questions(questions):
            a = answers.get(q.qid)
            if not a:
                continue
            if q.kind == "noul":
                p = float(a.get("noul", 0.5))
                res[q.qid] = [1 - p, p]
            else:
                res[q.qid] = normalize_probs(q, a.get("probabilities") or {})
        return {"model": out.get("model"), "dists": res}
