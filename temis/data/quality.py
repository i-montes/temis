"""Quality filter for generated cases.

A `noul` question must be a statement that is true or false about the state. The generator often
writes commands instead ("Determine if X", "Evalúa el peso emocional"). Conditional ones are rewritten
into the statement they contain ("Determine if X" -> "X"); vague ones, or either/or ones, are dropped.
States that are almost empty ("Layer 30 failed.") are dropped too, unless the case is marked `short`:
what a person types to an agent is often shorter than that, and those batches are written to be short
(docs/prompt_casos_cortos.md).
"""

from __future__ import annotations

import json
import re
from typing import Any

VERBS = (r"determine|determina|determinar|verify|verifica|verifique|verificar|check|confirm|confirma|confirme|"
         r"assess|evaluate|eval[uú]a|eval[uú]e|evaluar|indica|indique|analiza|analice|analyze|identify|"
         r"identifica|identifique|comprueba|compruebe|revisa|revise|establish|establece|decide|decida|examine|"
         r"examina|review|determine")
CONDITIONAL = re.compile(rf"^\s*(?:{VERBS})\s+(?:if|whether|si)\s+(.+)$", re.IGNORECASE | re.DOTALL)
IMPERATIVE = re.compile(rf"^\s*(?:{VERBS}|busque|busca|elige|elija|selecciona|seleccione|select|choose|rate|"
                        r"califica|clasifica|clasifique|describe|explain|explica)\b", re.IGNORECASE)
NOMINAL = re.compile(r"^\s*(evaluaci[oó]n|verificaci[oó]n|confirmaci[oó]n|an[aá]lisis|assessment|evaluation|"
                     r"verification|check)\b", re.IGNORECASE)
WH = re.compile(r"^\s*¿?\s*(qu[eé]|cu[aá]l|cu[aá]les|c[oó]mo|cu[aá]nto|cu[aá]nta|cu[aá]ndo|d[oó]nde|qui[eé]n|"
                r"por qu[eé]|what|which|how|when|where|who|why)\b", re.IGNORECASE)
EITHER_OR = re.compile(r"\s(o si|or whether|or if)\s", re.IGNORECASE)
MIN_STATE_CHARS = 120
MIN_SHORT_TEXT_CHARS = 2  # a case marked short still has to say something


def normalize_noul(instructions: str) -> str | None:
    """The statement to keep, or None if the question cannot be a clean yes/no statement."""
    s = " ".join(str(instructions).split())
    if len(s) < 12:
        return None
    m = CONDITIONAL.match(s)
    if m:
        s = m.group(1).strip()
        if EITHER_OR.search(s):
            return None
        s = s[0].upper() + s[1:]
        if not s.endswith((".", "?", "!")):
            s += "."
        return s
    if s.startswith("Si ") or s.endswith(":") or NOMINAL.match(s) or IMPERATIVE.match(s) or WH.match(s):
        return None
    if s.endswith("?") and EITHER_OR.search(s):
        return None
    return s


def _words(state: Any) -> str:
    """The text a state carries, without its keys."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return " ".join(_words(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return " ".join(_words(v) for v in state)
    return ""


def state_ok(state: Any, short: bool = False) -> bool:
    if short:
        return len(_words(state).strip()) >= MIN_SHORT_TEXT_CHARS
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return len(text) >= MIN_STATE_CHARS


def clean_questions(questions: dict[str, dict]) -> tuple[dict[str, dict], int, int]:
    """Returns (cleaned questions, rewritten, dropped). Non-noul questions pass unchanged."""
    out, rewritten, dropped = {}, 0, 0
    for qid, q in questions.items():
        if q.get("type") != "noul":
            out[qid] = q
            continue
        new = normalize_noul(q.get("instructions", ""))
        if new is None:
            dropped += 1
            continue
        if new != " ".join(str(q.get("instructions", "")).split()):
            rewritten += 1
            q = {**q, "instructions": new}
        out[qid] = q
    return out, rewritten, dropped
