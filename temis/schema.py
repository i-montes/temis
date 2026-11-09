"""Typed decision schemas, compatible with the TypeSafe / System One request shape.

A request has a `state` (string or JSON object) and `questions`: {qid: question}.

    choice: {"type": "choice", "instructions": str, "criteria": {label: description, ...}}
    score:  {"type": "score",  "instructions": str, "criteria": [level_0, level_1, ...]}
    noul:   {"type": "noul",   "instructions": str, "criteria": {"false": ..., "true": ...}?}

Internally every question becomes a list of options (key + description) so the same scorer
handles all three types. `noul` options are always ("false", "true").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Kind = Literal["choice", "score", "noul"]
KINDS: tuple[Kind, ...] = ("choice", "score", "noul")
NOUL_DEFAULT = {"false": "No, the statement is false.", "true": "Yes, the statement is true."}


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Question:
    qid: str
    kind: Kind
    instructions: str
    options: tuple[str, ...]
    descriptions: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.options)


@dataclass
class Answer:
    kind: Kind
    label: str
    probabilities: dict[str, float]
    confidence: float
    expected: float | None = None  # score questions: E[level]
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_typesafe(self) -> dict[str, Any]:
        """Same shape as the TypeSafe /v1/systemone answers."""
        if self.kind == "noul":
            return {"type": "noul", "noul": round(self.probabilities.get("true", 0.0), 4),
                    "probabilities": _round(self.probabilities)}
        out: dict[str, Any] = {"type": self.kind, "confidence": round(self.confidence, 4),
                               "probabilities": _round(self.probabilities)}
        if self.kind == "choice":
            out["choice"] = self.label
        else:
            out["score"] = round(self.expected or 0.0, 4)
        if self.evidence:
            out["evidence"] = self.evidence
        return out


def _round(p: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 4) for k, v in p.items()}


def parse_question(qid: str, q: dict[str, Any]) -> Question:
    if not isinstance(q, dict):
        raise SchemaError(f"question {qid!r} must be an object")
    kind = q.get("type")
    if kind not in KINDS:
        raise SchemaError(f"question {qid!r}: unknown type {kind!r} (expected one of {KINDS})")
    instructions = str(q.get("instructions") or "").strip()
    if not instructions:
        raise SchemaError(f"question {qid!r}: missing instructions")
    crit = q.get("criteria")
    if kind == "noul":
        c = crit if isinstance(crit, dict) else {}
        c = {str(k).lower(): str(v) for k, v in c.items()}
        opts = ("false", "true")
        desc = tuple(c.get(o) or NOUL_DEFAULT[o] for o in opts)
    elif kind == "score":
        if isinstance(crit, list) and len(crit) >= 2:
            opts = tuple(str(i) for i in range(len(crit)))
            desc = tuple(str(c) for c in crit)
        elif isinstance(crit, dict) and len(crit) >= 2 and all(str(k).lstrip("-").isdigit() for k in crit):
            keys = sorted(crit, key=lambda k: int(k))
            opts = tuple(str(i) for i in range(len(keys)))
            desc = tuple(str(crit[k]) for k in keys)
        else:
            raise SchemaError(f"question {qid!r}: score criteria must be a list of >= 2 levels")
    else:
        if not isinstance(crit, dict) or len(crit) < 2:
            raise SchemaError(f"question {qid!r}: choice criteria must be an object with >= 2 options")
        opts = tuple(str(k) for k in crit)
        desc = tuple(str(v) for v in crit.values())
    if len(opts) > 12:
        raise SchemaError(f"question {qid!r}: at most 12 options")
    return Question(qid, kind, instructions, opts, desc)


def parse_questions(questions: Any) -> list[Question]:
    if isinstance(questions, str):
        questions = json.loads(questions)
    if not isinstance(questions, dict) or not questions:
        raise SchemaError("questions must be a non-empty object")
    return [parse_question(qid, q) for qid, q in questions.items()]


def render_state(state: Any, max_chars: int = 6000) -> str:
    """JSON states become readable `key: value` lines; strings pass through."""
    if isinstance(state, str):
        s = state.strip()
        if s[:1] in "{[":
            try:
                state = json.loads(s)
            except json.JSONDecodeError:
                return s[:max_chars]
        else:
            return s[:max_chars]
    lines: list[str] = []

    def walk(prefix: str, v: Any) -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                walk(f"{prefix} · {k}" if prefix else str(k), x)
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            for i, x in enumerate(v):
                walk(f"{prefix} [{i + 1}]", x)
        elif isinstance(v, list):
            lines.append(f"{prefix.replace('_', ' ')}: " + "; ".join(map(str, v)))
        else:
            lines.append(f"{prefix.replace('_', ' ')}: {v}")

    walk("", state)
    return "\n".join(lines)[:max_chars]


def normalize_probs(q: Question, probs: dict[str, float]) -> list[float]:
    """Map an answer's probabilities onto q.options (noul accepts yes/no), renormalized."""
    alias = {"yes": "true", "no": "false"} if q.kind == "noul" else {}
    p = {alias.get(str(k).lower() if q.kind == "noul" else str(k), str(k).lower() if q.kind == "noul" else str(k)): float(v)
         for k, v in probs.items()}
    vec = [max(p.get(o, 0.0), 0.0) for o in q.options]
    s = sum(vec)
    return [x / s for x in vec] if s > 0 else [1.0 / q.n] * q.n
