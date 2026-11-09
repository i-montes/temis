"""Fit the student's temperatures on the validation split and store them next to the model.

  python -m scripts.calibrate --name v2

Three families go into runs/<name>/onnx/config.json (and meta.json); see temis/calibration.py:
  temperatures           soft: match the teachers' distributions (fitted at training time)
  temperatures_accuracy  accuracy: NLL of the target's top option, one per type (the API/UI default)
  temperatures_jevbench  by type and number of options, aimed at JevBench v1.5 (noul outside its
                         abstention band, score by expected level)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.train import load_examples
from temis.calibration import fit_jevbench, fit_nll, pick_temperature, softmax
from temis.jevbench import NOUL_BAND, ece
from temis.runtime import Decider


def stats(rows: list, t: float, kind: str) -> dict:
    ps = [softmax(lg, t) for lg, _ in rows]
    conf = np.array([p.max() for p in ps])
    ok = np.array([int(p.argmax() == y) for p, (_, y) in zip(ps, rows)])
    out = {"t": round(float(t), 2), "conf": round(float(conf.mean()), 4), "ece": round(ece(list(ok.astype(bool)), list(conf)), 4)}
    if kind == "noul":
        lo, hi = NOUL_BAND
        decided = np.array([not (lo < p[1] < hi) for p in ps])
        out["decided_right"] = round(float((ok.astype(bool) & decided).mean()), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    cfg_path = Path("runs") / a.name / "onnx" / "config.json"
    d = Decider(cfg_path.parent, calibration="soft")
    _, val = load_examples([])
    by_kind: dict[str, list] = defaultdict(list)
    by_kind_n: dict[tuple[str, int], list] = defaultdict(list)
    for ex in val:
        row = (d.logits([ex["q"]], ex["state"])[0], int(np.argmax(ex["target"])))
        by_kind[ex["kind"]].append(row)
        by_kind_n[(ex["kind"], ex["q"].n)].append(row)

    temps_acc = {kind: round(fit_nll(rows), 2) for kind, rows in by_kind.items()}
    temps_jev = fit_jevbench(by_kind_n)
    report = {}
    for kind, rows in by_kind.items():
        soft_t = float(d.temps_soft.get(kind, 1.0))
        report[kind] = {"n": len(rows), "soft": stats(rows, soft_t, kind), "accuracy": stats(rows, temps_acc[kind], kind),
                        "jevbench": {str(n): stats(rs, pick_temperature(temps_jev, kind, n), kind)
                                     for (k, n), rs in sorted(by_kind_n.items()) if k == kind}}
        print(f"[calibrate] {kind}: n={len(rows)} soft T={soft_t} ECE={report[kind]['soft']['ece']:.3f} -> "
              f"accuracy T={temps_acc[kind]} ECE={report[kind]['accuracy']['ece']:.3f} · jevbench {temps_jev.get(kind)}", flush=True)
        if kind == "noul":
            print(f"[calibrate]   noul decided & right: soft {report[kind]['soft']['decided_right']:.3f} -> "
                  f"jevbench {report[kind]['jevbench']['2']['decided_right']:.3f}", flush=True)

    cfg = json.loads(cfg_path.read_text())
    cfg["temperatures_accuracy"] = temps_acc
    cfg["temperatures_jevbench"] = temps_jev
    cfg["calibration_report"] = report
    cfg_path.write_text(json.dumps(cfg, indent=2))
    meta_p = Path("runs") / a.name / "meta.json"
    meta = json.loads(meta_p.read_text())
    meta["temperatures_accuracy"] = temps_acc
    meta["temperatures_jevbench"] = temps_jev
    meta_p.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
