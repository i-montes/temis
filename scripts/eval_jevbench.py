"""Run JevBench's public items against an exported model, without a server.

  python -m scripts.eval_jevbench --model runs/<run>/onnx [--fetch] [--limit N] [--calibration accuracy]

Scores every item as JevBench v1.2 does (argmax, ECE) and with a local estimate of the v1.5 rules
(see temis/jevbench.py). Writes runs/<run>/eval_jevbench.json. These numbers are reported, never
used to pick a version, and the items are never trained on.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from temis.jevbench import (PUBLIC_DIR, fetch_public, load_public, options_only_request, probs_for_labels,
                            rotate_question, score_item, summarize, to_request)
from temis.runtime import Decider


def run(d: Decider, items: list[dict], probes: bool = False) -> list[dict]:
    """With probes, every choice item with >= 3 options is also decided with its options rotated by one
    and with an empty state (3x the cost on those items)."""
    rows = []
    for it in items:
        state, qs = to_request(it)
        t0 = time.perf_counter()
        ans = d.decide(state, qs)["decision"]
        ms = (time.perf_counter() - t0) * 1000
        r = score_item(it, probs_for_labels(it, ans))
        if probes and it["question"]["type"] == "choice" and len(it["labels"]) >= 3:
            rot = d.decide(state, {"decision": rotate_question(it["question"], 1)})["decision"]
            r["pred_rot1"] = score_item(it, probs_for_labels(it, rot))["predicted"]
            blank = d.decide(*options_only_request(it))["decision"]
            r["pred_blank"] = score_item(it, probs_for_labels(it, blank))["predicted"]
        rows.append({**r, "id": it["id"], "tier": it["tier"], "family": it["family"], "ms": ms})
    return rows


def print_report(s: dict) -> None:
    print(f"\n{'tier/family':34s} {'acc':>7s} {'n':>4s}")
    for f, v in s["by_family"].items():
        print(f"{f:34s} {v['accuracy']:7.1%} {v['n']:4d}")
    for t, v in s["by_tier"].items():
        print(f"{t:34s} {v['accuracy']:7.1%} {v['n']:4d}")
    cc = {t: round(v["tier"], 1) for t, v in s["v15"]["cc_by_tier"].items() if v["tier"] is not None}
    print(f"\naccuracy (v1.2) {s['accuracy']:.1%} · ECE {s['ece']:.3f} · latency p50 {s['latency_ms']['p50']:.0f} ms "
          f"p95 {s['latency_ms']['p95']:.0f} ms")
    print(f"v1.5 estimate: accuracy {s['v15']['accuracy']:.1%} · noul in band {s['v15']['noul_in_band']}/{s['v15']['noul_n']} "
          f"· CC by tier {cc} · intelligence {s['v15']['intelligence']:.1f}")
    if "probes" in s:
        p = s["probes"]
        print(f"probes: rotation flip rate {p['rotation_flip_rate']:.1%} · options-only accuracy "
              f"{p['options_only_accuracy']:.1%} · first option picked {p['first_option_rate']['pred']:.1%} "
              f"(gold {p['first_option_rate']['gold']:.1%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fetch", action="store_true", help="download the public items into external/jevbench/public")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--calibration", default="accuracy", choices=["soft", "accuracy", "jevbench"])
    ap.add_argument("--probes", action="store_true", help="bias probes: rotated options and empty state (slower)")
    a = ap.parse_args()
    if a.fetch:
        fetch_public()
    if not (PUBLIC_DIR / "hard.jsonl").exists():
        raise SystemExit("public items missing: python -m scripts.eval_jevbench --model ... --fetch")
    items = load_public()[: a.limit]
    d = Decider(a.model, calibration=a.calibration)
    rows = run(d, items, probes=a.probes)
    s = summarize(rows)
    s["calibration"] = d.calibration
    s["model"] = a.model
    s["rows"] = [{k: r.get(k) for k in ("id", "tier", "family", "type", "predicted", "correct", "v15_correct", "conf", "ms")}
                 for r in rows]
    print_report(s)
    out = Path(a.model).parent / "eval_jevbench.json"
    out.write_text(json.dumps(s, indent=2))
    print(f"[eval_jevbench] wrote {out}")


if __name__ == "__main__":
    main()
