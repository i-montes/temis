"""Evaluate a model on Iván's personal test set: the 1 in 5 labeled cases that are never trained on.

  python -m scripts.eval_user --model runs/v1/onnx

Accuracy = how often the model's top answer is what Iván chose. Versions trained before the holdout
existed (2026-11-08 ~13:30) saw these cases, so their score is optimistic.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from decisor.runtime import Decider
from decisor.schema import parse_questions, render_state
from decisor.store import Store
from decisor.userlabels import is_user_test


def evaluate(model_dir: str) -> dict:
    d = Decider(model_dir)
    res: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for case in Store().user_labeled():
        if not is_user_test(case["id"]):
            continue
        qs = [q for q in parse_questions(case["questions"]) if q.qid in case["labels"]]
        if not qs:
            continue
        for q, lg in zip(qs, d.logits(qs, render_state(case["state"]))):
            ok = q.options[int(lg.argmax())] == case["labels"][q.qid]
            for key in ("all", f"family:{case.get('workflow') or '-'}"):
                res[key][0] += ok
                res[key][1] += 1
    out = {"n": res["all"][1], "accuracy": round(res["all"][0] / res["all"][1], 4) if res["all"][1] else None,
           "by_family": {k.split(":", 1)[1]: {"n": n, "accuracy": round(h / n, 4)} for k, (h, n) in res.items() if k != "all"}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    r = evaluate(a.model)
    print(json.dumps(r, ensure_ascii=False))
    (Path(a.model).parent / "eval_user.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
