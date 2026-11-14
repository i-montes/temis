"""Torch-free runtime: ONNX Runtime + tokenizers + numpy. Runs on CPU, CoreML (Mac) or CUDA.

    from temis.runtime import Decider
    d = Decider("runs/v1/onnx")
    answers = d.decide(state, questions)              # {qid: Answer}
    d.decide(state, questions, context="...")        # with retrieved evidence

Evaluations can run the same model on PyTorch instead (`backend="torch"` or TEMIS_BACKEND=torch): the
fp32 weights in runs/<name>/model.pt, on the GPU when there is one. That is for quick exploration only:
what ships, and what the promotion gate compares, is the ONNX int8 model on CPU, so those results go
to their own files (eval_path).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from temis.encoding import Encoder
from temis.schema import Answer, Question, parse_questions, render_state


def _providers(prefer: str | None) -> list[str]:
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if prefer == "CPUExecutionProvider":
        return [prefer]  # CPU only: no GPU provider loaded on the side (CPU benchmarks)
    order = [prefer] if prefer else []
    order += ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in dict.fromkeys(order) if p and p in avail]


def backend_from_env() -> str:
    return os.environ.get("TEMIS_BACKEND", "onnx").strip().lower() or "onnx"


def eval_path(model_dir: str | Path, stem: str) -> Path:
    """Where an evaluation writes its result: next to the run, with `.torch` in the name when it did
    not come from the ONNX model (so it never stands in for the numbers the promotion gate reads)."""
    suffix = "" if backend_from_env() == "onnx" else f".{backend_from_env()}"
    return Path(model_dir).parent / f"{stem}{suffix}.json"


class TorchSession:
    """The ONNX session's `run` on the PyTorch model (fp32), for evaluations on the GPU."""

    def __init__(self, model_pt: Path, base: str, device: str | None = None) -> None:
        import torch
        from temis.model import Decisor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Decisor(base, attn="sdpa" if self.device == "cuda" else "eager")
        self.model.load_state_dict(torch.load(model_pt, map_location="cpu"))
        self.model.to(self.device).eval()

    def run(self, _outputs, feed: dict) -> list:
        t = {k: self.torch.from_numpy(v).to(self.device) for k, v in feed.items()}
        with self.torch.inference_mode():
            out = self.model(t["input_ids"], t["attention_mask"], t["opt_pos"], t["opt_mask"].bool())
        return [out.float().cpu().numpy()]


class Decider:
    def __init__(self, model_dir: str | Path, int8: bool = True, provider: str | None = None,
                 threads: int = 4, calibration: str = "accuracy", backend: str | None = None,
                 per_question: bool | None = None) -> None:
        """calibration: "accuracy" (a 70 % answer is right ~70 % of the time; default), "soft"
        (matches the teachers' distributions, which is what the KL benchmarks measure) or "jevbench"
        (by type and number of options, decisive on yes/no; see temis/calibration.py).
        backend: "onnx" (default) or "torch" (fp32 weights in ../model.pt, GPU if any); None reads
        TEMIS_BACKEND.
        per_question: one forward pass per question, at its own length. Default on the CPU: there a
        batch costs the sum of its rows, so padding every question to the longest one is pure waste
        (2.4.0 int8, 4 threads: p50 579 -> 522 ms, typed-decisions 1001 -> 742 ms), and with dynamic
        int8 the padding rows even move the other questions' quantization scale, so an answer depended
        on which questions came with it. GPUs keep batching."""
        d = Path(model_dir)
        self.config = json.loads((d / "config.json").read_text())
        self.version = self.config.get("version", d.parent.name)  # run id, e.g. v1108-1022
        from temis.naming import display, model_id
        self.display = display(self.version)   # "Temis 1.0.5"
        self.model_id = model_id(self.version)  # "temis-1.0.5"
        self.backend = backend or backend_from_env()
        if self.backend == "torch":
            self.session = TorchSession(d.parent / "model.pt", self.config["base"])
        elif self.backend == "onnx":
            import onnxruntime as ort
            f = d / ("decisor.int8.onnx" if int8 and (d / "decisor.int8.onnx").exists() else "decisor.onnx")
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            self.session = ort.InferenceSession(str(f), so, providers=_providers(provider))
        else:
            raise ValueError(f"unknown backend {self.backend!r} (onnx or torch)")
        if per_question is None:
            per_question = self.backend == "onnx" and self.session.get_providers()[0] == "CPUExecutionProvider"
        self.per_question = per_question
        # long states are read up to long_max_len tokens (short ones cost the same as before)
        self.long_max_len = int(self.config.get("long_max_len", 1024))
        self.encoder = Encoder(str(d / "tokenizer.json"), max(int(self.config.get("max_len", 1024)), self.long_max_len))
        self.temps_soft = self.config.get("temperatures", {})
        acc = self.config.get("temperatures_accuracy")
        jev = self.config.get("temperatures_jevbench")
        if calibration == "jevbench" and not jev:
            calibration = "accuracy"  # not fitted yet: fall back
        self.calibration = calibration if (calibration == "soft" or acc) else "soft"
        self.temps = {"soft": self.temps_soft, "accuracy": acc, "jevbench": jev}[self.calibration] or {}

    def logits(self, qs: list[Question], state_text: str, context: str | None = None) -> list[np.ndarray]:
        encs = [self.encoder.encode(q, state_text, context, max_len=2048 if context else self.long_max_len) for q in qs]
        self.last_input_tokens = sum(len(e.ids) for e in encs)  # read under the caller's lock (API usage)
        groups = [[i] for i in range(len(qs))] if self.per_question else [list(range(len(qs)))]
        self.last_padded_tokens = 0  # what the model actually computes (a batch is padded to its longest)
        res: list[np.ndarray] = [None] * len(qs)  # type: ignore[list-item]
        for g in groups:
            for i, lg in zip(g, self._run([qs[i] for i in g], [encs[i] for i in g])):
                res[i] = lg
        return res

    def _run(self, qs: list[Question], encs: list) -> list[np.ndarray]:
        B, T, K = len(encs), max(len(e.ids) for e in encs), max(len(e.opt_pos) for e in encs)
        self.last_padded_tokens += B * T
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
        from temis.calibration import pick_temperature
        z = logits / pick_temperature(self.temps, q.kind, q.n)
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
