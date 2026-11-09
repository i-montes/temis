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
