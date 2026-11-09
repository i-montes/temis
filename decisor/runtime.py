"""Torch-free runtime: ONNX Runtime + tokenizers + numpy. Runs on CPU, CoreML (Mac) or CUDA.

    from decisor.runtime import Decider
    d = Decider("runs/v1/onnx")
    answers = d.decide(state, questions)              # {qid: Answer}
    d.decide(state, questions, context="...")        # with retrieved evidence
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from decisor.encoding import Encoder
from decisor.schema import Answer, Question, parse_questions, render_state


def _providers(prefer: str | None) -> list[str]:
    import onnxruntime as ort
    avail = ort.get_available_providers()
    order = [prefer] if prefer else []
    order += ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in dict.fromkeys(order) if p and p in avail]


class Decider:
    def __init__(self, model_dir: str | Path, int8: bool = True, provider: str | None = None,
                 threads: int = 4, calibration: str = "accuracy") -> None:
        """calibration: "accuracy" (a 70 % answer is right ~70 % of the time; default) or "soft"
        (matches the teachers' distributions, which is what the KL benchmarks measure)."""
        import onnxruntime as ort
        d = Path(model_dir)
        self.config = json.loads((d / "config.json").read_text())
        self.version = self.config.get("version", d.parent.name)  # run id, e.g. v1108-1022
        from decisor.naming import display, model_id
        self.display = display(self.version)   # "Temis 1.0.5"
        self.model_id = model_id(self.version)  # "temis-1.0.5"
        f = d / ("decisor.int8.onnx" if int8 and (d / "decisor.int8.onnx").exists() else "decisor.onnx")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(f), so, providers=_providers(provider))
        self.encoder = Encoder(str(d / "tokenizer.json"), self.config.get("max_len", 1024))
        self.temps_soft = self.config.get("temperatures", {})
        acc = self.config.get("temperatures_accuracy")
        self.calibration = calibration if (calibration == "soft" or acc) else "soft"
        self.temps = acc if self.calibration == "accuracy" else self.temps_soft

    def logits(self, qs: list[Question], state_text: str, context: str | None = None) -> list[np.ndarray]:
        encs = [self.encoder.encode(q, state_text, context, max_len=2048 if context else None) for q in qs]
        self.last_input_tokens = sum(len(e.ids) for e in encs)  # read under the caller's lock (API usage)
        B, T, K = len(encs), max(len(e.ids) for e in encs), max(len(e.opt_pos) for e in encs)
        ids = np.zeros((B, T), np.int64)
        am = np.zeros((B, T), np.int64)
        pos = np.zeros((B, K), np.int64)
        om = np.zeros((B, K), np.int64)
        for i, e in enumerate(encs):
            ids[i, :len(e.ids)] = e.ids
            am[i, :len(e.ids)] = 1
            pos[i, :len(e.opt_pos)] = e.opt_pos
            om[i, :len(e.opt_pos)] = 1
        out = self.session.run(None, {"input_ids": ids, "attention_mask": am, "opt_pos": pos, "opt_mask": om})[0]
        res = []
        for i, (q, e) in enumerate(zip(qs, encs)):
            lg = np.empty(q.n, np.float32)
            lg[e.order] = out[i, :q.n]
            res.append(lg)
        return res

    def probabilities(self, q: Question, logits: np.ndarray) -> np.ndarray:
        z = logits / float(self.temps.get(q.kind, 1.0))
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    def decide(self, state: Any, questions: dict[str, Any] | str, context: str | None = None) -> dict[str, Answer]:
        qs = parse_questions(questions)
        st = render_state(state)
        out = {}
        for q, lg in zip(qs, self.logits(qs, st, context)):
            out[q.qid] = make_answer(q, self.probabilities(q, lg))
        return out


def make_answer(q: Question, p: np.ndarray) -> Answer:
    p = np.asarray(p, dtype=np.float64)
    j = int(p.argmax())
    probs = {o: float(x) for o, x in zip(q.options, p)}
    expected = float(sum(int(o) * float(x) for o, x in zip(q.options, p))) if q.kind == "score" else None
    return Answer(q.kind, q.options[j], probs, float(p[j]), expected)


def uncertainty(answers: dict) -> float:
    """0 = sure of everything, 1 = a coin flip on at least one question."""
    u = 0.0
    for a in answers.values():
        n = len(a.probabilities)
        u = max(u, (1 - a.confidence) / (1 - 1 / n) if n > 1 else 0.0)
    return round(min(u, 1.0), 4)
