"""Evaluate an exported model on the internal hard-like set (data/hardlike.jsonl).

  python -m scripts.eval_hardlike --model runs/<run>/onnx

Same metrics as scripts/evaluate.py (accuracy vs gold argmax, KL, Brier, ECE), plus accuracy by task
and by level. Writes runs/<run>/eval_hardlike.json. This set is part of the promotion gate.
"""

from __future__ import annotations

import argparse
import json

from scripts.build_hardlike import HARDLIKE
from scripts.evaluate import evaluate_cases
from temis.data.sources import read_jsonl
from temis.runtime import Decider, eval_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cases = list(read_jsonl(HARDLIKE))[: a.limit]
    if not cases:
        raise SystemExit(f"{HARDLIKE} missing or empty: python -m scripts.build_hardlike")
    d = Decider(a.model, calibration="soft")
    res = evaluate_cases(d, cases, group_keys=("workflow", "level"))
    res["model"] = a.model
    print(json.dumps({k: v for k, v in res.items() if k != "by_level"}, indent=2))
    print("by level:", res["by_level"])
    out = eval_path(a.model, "eval_hardlike")
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
