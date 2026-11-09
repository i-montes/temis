"""Evaluate an exported model on Clio's decision benchmark (external/clio/testing/decisiones):
72 Spanish cases about memory operations, needing memory, forgetting, and tools.
Evaluation only: these cases are never used for training.

  python -m scripts.eval_clio --model runs/v1/onnx
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from decisor.runtime import Decider

CASES = Path("external/clio/testing/decisiones/cases.py")
# Jev's published result on this set (2026-11-03): op 93 %, mem 100 %, del 100 %, tool 88 %.


def load_cases():
    spec = importlib.util.spec_from_file_location("clio_cases", CASES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate(model_dir: str) -> dict:
    C = load_cases()
    d = Decider(model_dir)
    res: dict[str, list[int]] = {}
    for c in C.build():
        qs = C.questions(c)
        ans = {k: v.to_typesafe() for k, v in d.decide(c["state"], qs).items()}
        for q, g in c["gold"].items():
            a = ans.get(q)
            if a is None:
                good = False
            elif a["type"] == "noul":
                good = (a["noul"] >= 0.5) == bool(g)
            else:
                good = a["choice"] in (g if isinstance(g, list) else [g])
            r = res.setdefault(c["task"], [0, 0])
            r[0] += good
            r[1] += 1
    out = {t: round(h / n, 4) for t, (h, n) in res.items()}
    out["mean"] = round(sum(out.values()) / len(out), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    r = evaluate(a.model)
    print(json.dumps(r))
    (Path(a.model).parent / "eval_clio.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
