"""Evaluate an exported model on the held-out short set (data/cliolike.jsonl, scripts/build_cliolike.py):
messages and memories shaped like Clio's production calls, far shorter than the generated cases.

  python -m scripts.eval_cliolike --model runs/<run>/onnx

Decisions are scored the way Clio uses them: a yes/no counts as yes from 0.5, a choice by its most
likely option. Reports accuracy by family, the yes/no split (a model that never says yes gets most
tool questions right) and the share of cases with every question right. Writes
runs/<run>/eval_cliolike.json with one row per question, for paired comparisons between runs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from scripts.build_cliolike import CLIOLIKE
from temis.data.sources import read_jsonl


def right(answer: dict | None, gold: str) -> bool:
    if not answer:
        return False
    if answer.get("type") == "noul":
        return (float(answer.get("noul", 0.0)) >= 0.5) == (gold == "true")
    return answer.get("choice") == gold


def _rate(hits: list[bool]) -> dict:
    return {"n": len(hits), "accuracy": round(sum(hits) / len(hits), 4)}


def score(cases: list[dict], decide: Callable[[dict], dict]) -> dict:
    """`decide(case)` -> {qid: answer in TypeSafe shape}."""
    rows = []
    fam: dict[str, list[bool]] = defaultdict(list)
    whole: dict[str, list[bool]] = defaultdict(list)
    yes_no: dict[str, list[bool]] = {"yes": [], "no": []}
    for c in cases:
        answers = decide(c)
        hits = []
        for qid, gold in c["gold"].items():
            a = answers.get(qid)
            ok = right(a, gold)
            hits.append(ok)
            fam[c["family"]].append(ok)
            if gold in ("true", "false"):
                yes_no["yes" if gold == "true" else "no"].append(ok)
            rows.append({"id": c["id"], "qid": qid, "family": c["family"], "gold": gold, "right": ok,
                         "p": (a or {}).get("noul") if gold in ("true", "false") else (a or {}).get("choice")})
        whole[c["family"]].append(all(hits))
    return {"n": len(rows), "accuracy": round(sum(r["right"] for r in rows) / len(rows), 4),
            "by_family": {k: _rate(v) for k, v in sorted(fam.items())},
            "yes_no": {k: _rate(v) for k, v in yes_no.items() if v},
            "whole_case": {k: _rate(v) for k, v in sorted(whole.items())}, "rows": rows}


def main() -> None:
    from temis.runtime import Decider

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    cases = list(read_jsonl(CLIOLIKE))
    if not cases:
        raise SystemExit(f"{CLIOLIKE} missing or empty: python -m scripts.build_cliolike")
    d = Decider(a.model)
    res = score(cases, lambda c: {k: v.to_typesafe() for k, v in d.decide(c["state"], c["questions"]).items()})
    res["model"] = a.model
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))
    (Path(a.model).parent / "eval_cliolike.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
