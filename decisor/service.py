"""The decisor as a service: model + precedents + optional RAG + decision memory."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from decisor.retrieval import ClioMemoryRetriever, Passage, format_context
from decisor.runtime import Decider, make_answer, uncertainty
from decisor.schema import parse_questions, render_state
from decisor.store import Store

MODELS = Path("runs")
ACTIVE = Path("models/active.json")


def active_model_dir() -> Path | None:
    if ACTIVE.exists():
        d = Path(json.loads(ACTIVE.read_text())["dir"])
        if (d / "config.json").exists():
            return d
    cands = sorted((p for p in MODELS.glob("*/onnx/config.json") if "smoke" not in str(p)),
                   key=lambda p: p.stat().st_mtime)
    return cands[-1].parent if cands else None


def set_active(model_dir: Path) -> None:
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE.write_text(json.dumps({"dir": str(model_dir)}))


class Service:
    def __init__(self, store: Store | None = None) -> None:
        self.store = store or Store()
        self.lock = threading.Lock()
        self.decider: Decider | None = None
        self.reload()
        self._retriever: ClioMemoryRetriever | None = None

    def reload(self) -> None:
        d = active_model_dir()
        with self.lock:
            self.decider = Decider(d) if d else None
            self.loaded_dir = d

    def maybe_reload(self) -> None:
        """Pick up a version promoted by another process (nightly loop, retrain)."""
        if active_model_dir() != getattr(self, "loaded_dir", None):
            self.reload()

    @property
    def version(self) -> str:
        return self.decider.version if self.decider else "none"

    @property
    def display(self) -> str:
        return self.decider.display if self.decider else "sin modelo"

    def retriever(self) -> ClioMemoryRetriever | None:
        if self._retriever is None:
            try:
                self._retriever = ClioMemoryRetriever()
            except Exception:  # noqa: BLE001
                return None
        return self._retriever

    def decide(self, state: Any, questions: dict, *, context: Any = None, workflow: str = "",
               origin: str = "api", log: bool = True, use_precedents: bool = True) -> dict:
        self.maybe_reload()
        if self.decider is None:
            raise RuntimeError("no trained model is active yet")
        qs = parse_questions(questions)
        st = render_state(state)
        passages: list = []
        if isinstance(context, dict):  # {"k": 5, "scope": "clio"} -> Clio Memory
            r = self.retriever()
            if r is not None:
                r.scope = context.get("scope", r.scope)
                query = context.get("query") or st
                passages = r.search(query, int(context.get("k", 5)))
        elif context:
            passages = context if isinstance(context, list) else [str(context)]
        ctx = format_context(passages)
        with self.lock:
            logits = self.decider.logits(qs, st, ctx)
            answers = {}
            for q, lg in zip(qs, logits):
                p = self.decider.probabilities(q, lg)
                ev: list[dict] = []
                if use_precedents:
                    prec = self.store.precedents(q, st)
                    if prec:
                        lam = min(0.6, 0.15 * len(prec))
                        pp = np.zeros(q.n)
                        for r in prec:
                            v = np.full(q.n, 0.1 / max(q.n - 1, 1))
                            v[q.options.index(r["label"])] = 0.9
                            pp += v
                        pp /= pp.sum()
                        p = (1 - lam) * p + lam * pp
                        ev += [{"type": "precedent", "decision": r["decision_id"], "label": r["label"]} for r in prec]
                a = make_answer(q, p)
                a.evidence = ev + [{"type": "passage", "id": x.id, "source": x.source, "text": x.text[:300]}
                                   if isinstance(x, Passage) else {"type": "passage", "text": str(x)[:300]}
                                   for x in passages]
                answers[q.qid] = a
        u = uncertainty(answers)
        out = {qid: a.to_typesafe() for qid, a in answers.items()}
        did = None
        if log:
            did = self.store.add_decision(origin=origin, state=state, questions=questions, answers=out,
                                          model_version=self.version, uncertainty=u, context=ctx, workflow=workflow)
        return {"model": self.decider.model_id, "run": self.version, "decision_id": did, "uncertainty": u,
                "answers": out}
