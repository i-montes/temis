"""Where cases come from. A case: {id, workflow, state, questions, source, gold?}.

typed-decisions (LocalLLaMA/typed-decisions, Apache-2.0): `train` is used for training and for
calibrating the teachers against its gold; `test` is NEVER used for training — only for the final
evaluation.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

SYNTH = Path("data/synth/cases.jsonl")
EXTERNAL = Path("data/synth/external.jsonl")  # cases written by other models (scripts/import_cases.py)
REJECTED = Path("data/synth/rejected.txt")  # broken cases (one id per line, # comments): never trained on


def rejected_ids() -> set[str]:
    if not REJECTED.exists():
        return set()
    return {ln.split("#")[0].strip() for ln in REJECTED.read_text().splitlines() if ln.split("#")[0].strip()}


def load_td(split: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split=split)
    out = []
    for r in ds:
        out.append({"id": f"td-{r['id']}", "workflow": r["workflow"], "state": json.loads(r["state"]),
                    "questions": json.loads(r["questions"]), "gold": json.loads(r["gold"]),
                    "source": f"td-{split}", "language": "en"})
    return out


def load_synth(clean: bool = True) -> list[dict]:
    """Synthetic cases. With clean=True (default) the quality filter runs: yes/no questions that are
    commands become statements or are dropped, and near-empty states are dropped."""
    from temis.data.quality import clean_questions, state_ok

    lines = []
    for path in (SYNTH, EXTERNAL):
        if path.exists():
            lines += path.open().readlines()
    bad = rejected_ids() if clean else set()
    out = []
    for line in lines:
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            continue
        if clean:
            if c.get("id") in bad:
                continue
            if not state_ok(c.get("state")):
                continue
            c["questions"], _, _ = clean_questions(c.get("questions") or {})
            if not c["questions"]:
                continue
        out.append(c)
    return out


def read_jsonl(path: Path) -> Iterator[dict]:
    if path.exists():
        for line in path.open():
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
