"""Build the internal hard-like evaluation set: data/hardlike.jsonl (never trained on).

  python -m scripts.build_hardlike [--n 2000]

2,000 cases from the `test` split of tasksource/procedural-typed-decisions: half from the two tasks
held out of training entirely, half from levels 3–4 of the rest. Used by scripts/eval_hardlike.py
and by the promotion gate in scripts/retrain.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from temis.data.procedural import load_hardlike

HARDLIKE = Path("data/hardlike.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    cases = load_hardlike(a.n)
    HARDLIKE.parent.mkdir(parents=True, exist_ok=True)
    with HARDLIKE.open("w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[hardlike] wrote {len(cases)} cases to {HARDLIKE}")


if __name__ == "__main__":
    main()
