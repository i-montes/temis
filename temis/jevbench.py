"""JevBench's public items, scored locally.

JevBench (https://github.com/fstandhartinger/jevbench) publishes 231 of its items (easy 48, standard
72 in `original.jsonl`, hard 111). This module turns an item into a Temis request, maps the answer
back to the item's labels and scores it two ways:

- v1.2: argmax accuracy (for `score`, argmax == expected level), top-label ECE;
- v1.5 (local estimate, NOT the official score): `noul` counts as wrong when P(yes) is strictly
  inside (0.20, 0.80); `score` is judged by the expected level's normalized MAE against uniform
  chance; chance-corrected scores per tier and type, weighted easy 10 / standard 20 / hard 40
  (the judge tier is not public) and choice 50 / noul 25 / score 25.

These items are for evaluation only: they never enter `data/` or any training target.
"""

from __future__ import annotations

import json
import math
import statistics
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from temis.schema import Answer

PUBLIC_FILES = ("easy", "original", "hard")
RAW = "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/datasets/public/{name}.jsonl"
PUBLIC_DIR = Path("external/jevbench/public")
TIER_OF_FILE = {"easy": "easy", "original": "standard", "hard": "hard"}
TIER_WEIGHT = {"easy": 10.0, "standard": 20.0, "hard": 40.0}
TYPE_WEIGHT = {"choice": 50.0, "noul": 25.0, "score": 25.0}
NOUL_BAND = (0.20, 0.80)


def fetch_public(dir: Path = PUBLIC_DIR) -> list[Path]:
    dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name in PUBLIC_FILES:
        p = dir / f"{name}.jsonl"
        if not p.exists():
            urllib.request.urlretrieve(RAW.format(name=name), p)  # noqa: S310 (fixed https URL)
        out.append(p)
    return out


def load_public(dir: Path = PUBLIC_DIR) -> list[dict]:
    items = []
    for name in PUBLIC_FILES:
        p = dir / f"{name}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run `python -m scripts.eval_jevbench --fetch`")
        for line in p.read_text().splitlines():
            if line.strip():
                it = json.loads(line)
                it["tier"] = TIER_OF_FILE[name]
                items.append(it)
    return items


def to_request(item: dict) -> tuple[Any, dict]:
    return item["state"], {"decision": item["question"]}


def probs_for_labels(item: dict, answer: Answer) -> dict[str, float]:
    """The answer's probabilities on the item's labels, renormalized."""
    kind = item["question"]["type"]
    if kind == "noul":
        p_yes = float(answer.probabilities.get("true", 0.0))
        raw = {"yes": p_yes, "no": float(answer.probabilities.get("false", 1.0 - p_yes))}
    else:
        raw = {str(k): float(v) for k, v in answer.probabilities.items()}
    vec = {lab: max(raw.get(lab, 0.0), 0.0) for lab in item["labels"]}
    s = sum(vec.values())
    return {k: v / s for k, v in vec.items()} if s > 0 else {k: 1.0 / len(vec) for k in vec}


def _argmax(probs: dict[str, float]) -> str:
    """Ties go to the lexicographically smallest label, as JevBench does."""
    return min(probs, key=lambda k: (-probs[k], k))


def score_item(item: dict, probs: dict[str, float]) -> dict:
    kind = item["question"]["type"]
    labels = list(item["labels"])
    expected = str(item["expected"])
    pred = _argmax(probs)
    out: dict[str, Any] = {"correct": pred == expected, "predicted": pred, "conf": max(probs.values()),
                           "chance": 1.0 / len(labels), "type": kind}
    if kind == "noul":
        lo, hi = NOUL_BAND
        decided = probs.get("yes", 0.0) <= lo or probs.get("yes", 0.0) >= hi
        out["v15_correct"] = out["correct"] and decided
        out["in_band"] = not decided
    elif kind == "score":
        k = len(labels)
        ev = sum(int(lab) * p for lab, p in probs.items())
        gold = int(item["expected"])
        out["v15_correct"] = out["correct"]
        out["nmae"] = abs(ev - gold) / max(k - 1, 1)
        out["nmae_chance"] = abs((k - 1) / 2 - gold) / max(k - 1, 1)
    else:
        out["v15_correct"] = out["correct"]
    gold_probs = (item.get("provenance") or {}).get("gold_probs")
    if gold_probs:
        out["tvd"] = 0.5 * sum(abs(probs.get(k, 0.0) - float(v)) for k, v in gold_probs.items())
    return out


def ece(accs: list[bool], confs: list[float], n_bins: int = 10) -> float:
    """Top-label expected calibration error with equal-width bins."""
    if not accs:
        return 0.0
    bt, bh, bc = [0] * n_bins, [0] * n_bins, [0.0] * n_bins
    for ok, c in zip(accs, confs):
        b = min(int(c * n_bins), n_bins - 1)
        bt[b] += 1
        bh[b] += ok
        bc[b] += c
    return sum(bt[b] / len(accs) * abs(bh[b] / bt[b] - bc[b] / bt[b]) for b in range(n_bins) if bt[b])


def _acc(rows: list[dict], key: str = "correct") -> dict:
    n = len(rows)
    c = sum(bool(r[key]) for r in rows)
    return {"n": n, "correct": c, "accuracy": c / n if n else None}


def _cc(rows: list[dict], key: str) -> float | None:
    """Chance-corrected accuracy, 0..100, clipped at 0."""
    if not rows:
        return None
    acc = sum(bool(r[key]) for r in rows) / len(rows)
    chance = sum(r["chance"] for r in rows) / len(rows)
    return max(0.0, 100.0 * (acc - chance) / (1.0 - chance)) if chance < 1 else 0.0


def _cc_score(rows: list[dict]) -> float | None:
    if not rows:
        return None
    nm = sum(r["nmae"] for r in rows) / len(rows)
    nc = sum(r["nmae_chance"] for r in rows) / len(rows)
    return max(0.0, 100.0 * (1.0 - nm / nc)) if nc > 0 else (100.0 if nm == 0 else 0.0)


def _percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def summarize(rows: list[dict]) -> dict:
    """rows: score_item(...) plus `tier`, `family`, `type`, `id`, `ms` (optionally `pred_rot1`, `pred_blank`)."""
    by_tier: dict[str, list[dict]] = defaultdict(list)
    by_family: dict[str, list[dict]] = defaultdict(list)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)
        by_family[f"{r['tier']}/{r['family']}"].append(r)
        by_type[r["type"]].append(r)
    cc_by_tier: dict[str, dict] = {}
    for tier, trs in by_tier.items():
        parts = {"choice": _cc([r for r in trs if r["type"] == "choice"], "correct"),
                 "noul": _cc([r for r in trs if r["type"] == "noul"], "v15_correct"),
                 "score": _cc_score([r for r in trs if r["type"] == "score"])}
        present = {k: v for k, v in parts.items() if v is not None}
        wsum = sum(TYPE_WEIGHT[k] for k in present)
        parts["tier"] = sum(TYPE_WEIGHT[k] * v for k, v in present.items()) / wsum if wsum else None
        cc_by_tier[tier] = parts
    tiers = {t: v["tier"] for t, v in cc_by_tier.items() if v["tier"] is not None and t in TIER_WEIGHT}
    wsum = sum(TIER_WEIGHT[t] for t in tiers)
    intelligence = sum(TIER_WEIGHT[t] * v for t, v in tiers.items()) / wsum if wsum else None
    nouls = [r for r in rows if r["type"] == "noul"]
    ms = [float(r["ms"]) for r in rows if r.get("ms") is not None]
    out = {
        "n": len(rows),
        "accuracy": _acc(rows)["accuracy"],
        "by_tier": {t: _acc(v) for t, v in sorted(by_tier.items())},
        "by_family": {f: _acc(v) for f, v in sorted(by_family.items())},
        "by_type": {t: _acc(v) for t, v in sorted(by_type.items())},
        "ece": ece([bool(r["correct"]) for r in rows], [float(r["conf"]) for r in rows]),
        "latency_ms": {"p50": _percentile(ms, 0.5), "p95": _percentile(ms, 0.95)},
        "v15": {"accuracy": _acc(rows, "v15_correct")["accuracy"],
                "noul_in_band": sum(bool(r.get("in_band")) for r in nouls),
                "noul_n": len(nouls),
                "cc_by_tier": cc_by_tier,
                "intelligence": intelligence},
    }
    tvd = [r["tvd"] for r in rows if "tvd" in r]
    if tvd:
        out["v15"]["probability_tvd"] = statistics.fmean(tvd)
    return out


__all__ = ["PUBLIC_DIR", "PUBLIC_FILES", "fetch_public", "load_public", "to_request", "probs_for_labels",
           "score_item", "ece", "summarize", "math"]
