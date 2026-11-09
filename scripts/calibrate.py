"""Fit accuracy calibration for an exported model and store it next to the soft one.

  python -m scripts.calibrate --name v2

Two temperatures per question type live in runs/<name>/onnx/config.json:
  temperatures           soft: match the teachers' distributions (best KL; used by the benchmarks)
  temperatures_accuracy  accuracy: a 70 % answer is right ~70 % of the time (used by the API/UI)
Training targets are soft averages of hesitant teachers, so the soft temperature leaves the model
underconfident; this fits a temperature by NLL of the target's top option on the validation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from decisor.runtime import Decider
from scripts.train import load_examples


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def ece(conf: np.ndarray, ok: np.ndarray, n_bins: int = 10) -> float:
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(ok[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(n_bins) if (idx == b).any()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    cfg_path = Path("runs") / a.name / "onnx" / "config.json"
    d = Decider(cfg_path.parent, calibration="soft")
    _, val = load_examples([])
    by: dict[str, list] = {}
    for ex in val:
        by.setdefault(ex["kind"], []).append((d.logits([ex["q"]], ex["state"])[0], int(np.argmax(ex["target"]))))
    temps, report = {}, {}
    for kind, rows in by.items():
        def nll(t: float) -> float:
            return -float(np.mean([np.log(max(softmax(lg / t)[y], 1e-9)) for lg, y in rows]))
        grid = np.arange(0.2, 2.51, 0.05)
        t_best = float(min(grid, key=nll))
        temps[kind] = round(t_best, 2)

        def stats(t: float) -> tuple[float, float]:
            ps = [softmax(lg / t) for lg, _ in rows]
            conf = np.array([p.max() for p in ps])
            ok = np.array([int(p.argmax() == y) for p, (_, y) in zip(ps, rows)])
            return float(conf.mean()), ece(conf, ok)
        soft_t = float(d.temps_soft.get(kind, 1.0))
        report[kind] = {"n": len(rows), "soft": {"t": soft_t, "conf": stats(soft_t)[0], "ece": stats(soft_t)[1]},
                        "accuracy": {"t": temps[kind], "conf": stats(t_best)[0], "ece": stats(t_best)[1]}}
        print(f"[calibrate] {kind}: n={len(rows)} soft T={soft_t} ECE={report[kind]['soft']['ece']:.3f} -> "
              f"accuracy T={temps[kind]} ECE={report[kind]['accuracy']['ece']:.3f}", flush=True)
    cfg = json.loads(cfg_path.read_text())
    cfg["temperatures_accuracy"] = temps
    cfg["calibration_report"] = report
    cfg_path.write_text(json.dumps(cfg, indent=2))
    meta_p = Path("runs") / a.name / "meta.json"
    meta = json.loads(meta_p.read_text())
    meta["temperatures_accuracy"] = temps
    meta_p.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
