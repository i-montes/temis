"""Temis: the model's name, and its versions.

Temis (Themis), the Titaness of judgment and good order, beside Clio. Versions are MAJOR.MINOR.PATCH:
  MAJOR  the base model or the architecture changes (mmBERT-small = 0, mmBERT-base = 1, MrBERT-es = 2)
  MINOR  a training from the base model (a fresh run, e.g. the nightly one)
  PATCH  a short fine-tune of a previous Temis version
Each training run keeps its id (`v1108-1022`); runs/registry.json maps run id -> version.
"""

from __future__ import annotations

import json
from pathlib import Path

NAME = "Temis"
REGISTRY = Path("runs/registry.json")
MAJOR_OF_BASE = {"jhu-clsp/mmBERT-small": 0, "jhu-clsp/mmBERT-base": 1, "BSC-LT/MrBERT-es": 2}

# Runs trained before versions existed (2026-11-08), numbered in the order they were trained.
BACKFILL = {"v0": "0.1.0", "v1b": "0.2.0", "v1c": "1.0.0", "v2": "1.0.1", "v1108-0552": "1.0.2",
            "v1108-0717": "1.0.3", "v1108-0830": "1.0.4", "v1108-1022": "1.0.5", "v1108-1157": "1.0.6"}


def _load() -> dict[str, str]:
    reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    for k, v in BACKFILL.items():
        reg.setdefault(k, v)
    return reg


def _key(v: str) -> tuple[int, int, int]:
    a, b, c = (int(x) for x in v.split("."))
    return a, b, c


def version_of(run: str) -> str | None:
    return _load().get(run)


def assign(run: str, base: str, fine_tune: bool) -> str:
    """Give `run` the next version and store it. Idempotent: a run keeps the version it got."""
    reg = _load()
    if run in reg:
        return reg[run]
    major = MAJOR_OF_BASE.get(base, max(MAJOR_OF_BASE.values()) + 1)
    same = sorted((_key(v) for v in reg.values() if _key(v)[0] == major), reverse=True)
    if not same:
        v = (major, 0, 0)
    elif fine_tune:
        v = (major, same[0][1], max(p for (M, m, p) in same if m == same[0][1]) + 1)
    else:
        v = (major, same[0][1] + 1, 0)
    reg[run] = ".".join(map(str, v))
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(dict(sorted(reg.items(), key=lambda kv: _key(kv[1]))), indent=2))
    return reg[run]


def display(run: str) -> str:
    v = version_of(run)
    return f"{NAME} {v}" if v else run


def model_id(run: str) -> str:
    v = version_of(run)
    return f"{NAME.lower()}-{v}" if v else run
