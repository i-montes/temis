"""How many times each synthetic target goes into training (0 = left out).

The memory_op cases that Qwen generated lean heavily on "corrects a memory": Jev picks a correction
option in 51 % of them and "already said" in only 4 %. Temis learned that a close memory means
"correct it", and on Clio's benchmark it calls every already-stored proposal a correction (0 of 6).

Two fixes, both deterministic by case id:
- Qwen memory_op questions whose Jev answer is a correction are kept only CORRECTION_KEEP of the time;
- memory_op cases written by other models (ext-*, balanced and clearer) count EXTERNAL_REPEAT times.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CORRECTION_PREFIXES = ("corrige_", "updates_", "update_", "corrects_")
CORRECTION_KEEP = 0.3
EXTERNAL_REPEAT = 2


def _unit(key: str) -> float:
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def copies(case: dict, options: list[str], dist: list[float]) -> int:
    if case.get("workflow") != "memory_op":
        return 1
    if str(case.get("id", "")).startswith("ext-"):
        return EXTERNAL_REPEAT
    top = options[max(range(len(dist)), key=dist.__getitem__)]
    if top.startswith(CORRECTION_PREFIXES):
        return 1 if _unit(case["id"]) < CORRECTION_KEEP else 0
    return 1


def cap_share(rows: list[dict], source: str, max_share: float, seed: int = 0) -> list[dict]:
    """At most `max_share` of the rows may come from `source`; the excess is dropped deterministically
    (by position, seeded) and the order of everything else is kept."""
    import random

    idx = [i for i, r in enumerate(rows) if r.get("source") == source]
    others = len(rows) - len(idx)
    allowed = int(max_share * others / (1 - max_share)) if max_share < 1 else len(idx)
    if len(idx) <= allowed:
        return list(rows)
    keep = set(random.Random(seed).sample(idx, allowed))
    return [r for i, r in enumerate(rows) if r.get("source") != source or i in keep]


NO_OPINION = object()  # nobody has given a second opinion on this question yet
HELD_BACK = Path("data/held_back.jsonl")  # hard questions where the teachers disagree: queue, not training


def agree(jev_dist: list[float], options: list[str], label) -> bool:
    """A hard question enters training only if the second opinion (another model,
    docs/prompt_segunda_opinion.md) picks the same option as Jev's argmax. NO_OPINION (nobody asked):
    it enters on Jev alone. None (the reviewer said no option is justified): held back."""
    if label is NO_OPINION:
        return True
    if label is None:
        return False
    top = options[max(range(len(jev_dist)), key=jev_dist.__getitem__)]
    return str(label) == top


def write_held_back(pairs: list[tuple[str, str]], path: Path = HELD_BACK) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for cid, qid in pairs:
            f.write(json.dumps({"id": cid, "qid": qid}) + "\n")


def read_held_back(path: Path = HELD_BACK) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.add((r["id"], r["qid"]))
    return out
