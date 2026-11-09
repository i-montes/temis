"""Evaluate an exported model on the typed-decisions TEST split with the official metrics
(accuracy vs gold argmax, KL from gold, Brier, ECE), using the torch-free runtime.

  python -m scripts.evaluate --model runs/v1/onnx [--fp32]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from temis.data.sources import load_td
from temis.jevbench import ece
from temis.runtime import Decider


def score_one(probs: dict[str, float], gold: dict) -> tuple[bool, float, float, float]:
    opts = list(gold["probabilities"].keys())
    g = [float(gold["probabilities"][o]) for o in opts]
    p = [float(probs.get(o, 0.0)) for o in opts]
    s = sum(p)
    p = [v / s for v in p] if s > 0 else [1 / len(opts)] * len(opts)
    ok = p.index(max(p)) == g.index(max(g))
    kl = sum(gi * math.log(gi / max(pi, 1e-9)) for gi, pi in zip(g, p) if gi > 0)
    brier = sum((pi - gi) ** 2 for pi, gi in zip(p, g))
    return ok, kl, brier, max(p)


def evaluate(model_dir: str, int8: bool = True, limit: int | None = None) -> dict:
    d = Decider(model_dir, int8=int8, calibration="soft")  # soft keeps KL comparable with the leaderboard
    cases = load_td("test")[:limit]
    accs, kls, brs, confs, kinds, wfs = [], [], [], [], [], []
    t0 = time.time()
    for c in cases:
        ans = d.decide(c["state"], c["questions"])
        for qid, g in c["gold"].items():
            a = ans.get(qid)
            if a is None:
                continue
            probs = a.probabilities
            ok, kl, br, conf = score_one(probs, g)
            accs.append(ok); kls.append(kl); brs.append(br); confs.append(conf)
            kinds.append(g["type"]); wfs.append(c["workflow"])
    n = len(accs)
    res = {"model": model_dir, "int8": int8, "n_cases": len(cases), "n_decisions": n,
           "accuracy": sum(accs) / n, "kl_from_gold": sum(kls) / n, "brier": sum(brs) / n,
           "ece": ece(accs, confs), "ms_per_case": (time.time() - t0) / len(cases) * 1000}
    for key, vals in (("by_type", kinds), ("by_workflow", wfs)):
        grp = defaultdict(list)
        for v, ok in zip(vals, accs):
            grp[v].append(ok)
        res[key] = {k: round(sum(v) / len(v), 4) for k, v in sorted(grp.items())}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    res = evaluate(a.model, int8=not a.fp32, limit=a.limit)
    print(json.dumps(res, indent=2))
    out = Path(a.model).parent / f"eval_td_test{'_fp32' if a.fp32 else ''}.json"
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
