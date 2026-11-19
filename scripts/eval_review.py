"""Evaluate an exported model on Clio's `needs_review` test (data/review_test.jsonl, never trained on):
30 turns as Clio sends them, 15 that deserve an independent review and 15 that do not.

  python -m scripts.eval_review --model runs/<run>/onnx

Clio starts a review from 0.8 and reads anything under 0.5 as a clear no. The test passes when at
most 2 yes cases score under 0.8, at most 1 no case scores 0.5 or more, and no no case reaches 0.8.
Writes runs/<run>/eval_review.json with the score of every case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_review import QID, QUESTION, TEST
from temis.data.sources import read_jsonl

YES_AT, NO_UNDER = 0.8, 0.5


def verdict(rows: list[dict]) -> dict:
    """rows: {id, expect: "yes"|"no", p}."""
    yes_low = [r["id"] for r in rows if r["expect"] == "yes" and r["p"] < YES_AT]
    no_high = [r["id"] for r in rows if r["expect"] == "no" and r["p"] >= NO_UNDER]
    no_fired = [r["id"] for r in rows if r["expect"] == "no" and r["p"] >= YES_AT]
    return {"passed": len(yes_low) <= 2 and len(no_high) <= 1 and not no_fired,
            "yes_under_0.8": yes_low, "no_from_0.5": no_high, "no_from_0.8": no_fired}


def main() -> None:
    from temis.runtime import Decider, eval_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(TEST))
    ap.add_argument("--name", default="eval_review", help="result file stem (another one for another data set)")
    a = ap.parse_args()
    d = Decider(a.model)
    question = {QID: {"type": "noul", "instructions": QUESTION}}
    rows = [{"id": c["id"], "expect": c["expect"],
             "p": round(d.decide(c["state"], question)[QID].probabilities["true"], 4)}
            for c in read_jsonl(Path(a.data))]
    res = {**verdict(rows), "model": a.model, "rows": rows}
    eval_path(a.model, a.name).write_text(json.dumps(res, indent=2))
    for r in rows:
        print(f"{r['id']}  {r['p']:.4f}")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))


if __name__ == "__main__":
    main()
