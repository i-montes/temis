"""Free disk space: old versions keep only what is needed to use them (ONNX int8, config, tokenizer,
metrics); their PyTorch weights (model.pt) and ONNX fp32 are deleted.

Never touched: the active version, the `--keep-best` best versions by the promotion score, the best on
typed-decisions alone, any run that
is still training or not yet evaluated, and anything outside runs/. Test runs (`smoke`) and aborted
runs (weights but no meta.json, not training) are deleted whole.

  python -m scripts.prune_runs            # shows what it would delete
  python -m scripts.prune_runs --apply    # deletes it
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from temis.service import active_model_dir

RUNS = Path("runs")


def training_now() -> set[str]:
    out = subprocess.run(["pgrep", "-fl", r"python.* -m scripts\.train"], capture_output=True, text=True).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if "--name" in parts:
            names.add(parts[parts.index("--name") + 1])
    return names


def combined(run: Path) -> float | None:
    ev, cl = run / "eval_td_test.json", run / "eval_clio.json"
    if not (ev.exists() and cl.exists()):
        return None
    return (json.loads(ev.read_text())["accuracy"] + json.loads(cl.read_text())["mean"]) / 2


def plan(keep_best: int = 2) -> tuple[list[Path], list[Path], list[str]]:
    act = active_model_dir()
    active = act.parent.name if act else None
    busy = training_now()
    runs = [p for p in RUNS.iterdir() if p.is_dir() and p.name != "logs"]
    scored = sorted(((combined(p), p.name) for p in runs if combined(p) is not None), reverse=True)
    best = {name for _, name in scored[:keep_best]}
    td = [(json.loads((p / "eval_td_test.json").read_text())["accuracy"], p.name) for p in runs
          if (p / "eval_td_test.json").exists()]
    if td:
        best.add(max(td)[1])  # also the best on typed-decisions alone
    files, whole, kept = [], [], []
    for p in runs:
        if p.name in busy:
            kept.append(f"{p.name} (entrenando)")
            continue
        if p.name == "smoke" or (not (p / "meta.json").exists() and (p / "model.pt").exists()):
            whole.append(p)
            continue
        if p.name == active or p.name in best:
            kept.append(f"{p.name} ({'activa' if p.name == active else 'de las mejores'})")
            continue
        if combined(p) is None:
            kept.append(f"{p.name} (sin evaluar)")
            continue
        if not (p / "onnx" / "decisor.int8.onnx").exists():
            kept.append(f"{p.name} (sin int8)")
            continue
        files += [f for f in (p / "model.pt", p / "onnx" / "decisor.onnx") if f.exists()]
    return files, whole, kept


def size(p: Path) -> int:
    return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--keep-best", type=int, default=2)
    a = ap.parse_args()
    files, whole, kept = plan(a.keep_best)
    total = sum(size(p) for p in files + whole)
    print(f"[prune] se conservan completas: {', '.join(kept) or '-'}")
    for p in whole:
        print(f"[prune] carpeta entera: {p} ({size(p) / 1e9:.1f} GB)")
    for f in files:
        print(f"[prune] archivo: {f} ({size(f) / 1e9:.1f} GB)")
    print(f"[prune] total: {total / 1e9:.1f} GB {'liberados' if a.apply else '(simulación; usa --apply)'}")
    if a.apply:
        for f in files:
            f.unlink()
        for p in whole:
            shutil.rmtree(p)


if __name__ == "__main__":
    main()
