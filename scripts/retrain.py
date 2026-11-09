"""Retrain with the user's labels + replay of all previous targets; promote only if not worse.

  python -m scripts.retrain

0. Jev labels any synthetic or imported case it has not seen yet; targets are rebuilt
1. user labels (data/decisor.db) -> data/user_targets.jsonl (gold, x3 weight by repetition);
   reviewer answers -> data/reviewer_targets.jsonl (soft, weight 1)
2. fine-tune a new version starting from the active model, replaying data/targets.jsonl
3. export ONNX, evaluate on typed-decisions test, Clio, the personal test, the internal hard-like set
   and JevBench's public items (reported only)
4. activate it only if validation KL, test accuracy, Clio, the hard-like set and the option-rotation
   flip rate are not worse than the active version's
5. refresh the review queue and free disk space from old versions

This is how the model is trained now: the automatic nightly loop is off (runs/nightly.stop),
so run this (or press "Reentrenar" in the web page) after adding cases or labels.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from temis.service import active_model_dir, set_active
from temis.userlabels import REVIEWER_TARGETS, USER_TARGETS, export_reviewer_targets, export_user_targets


def promotion(new: dict, old: dict) -> tuple[bool, str]:
    """new/old: {kl, test_acc, clio, hardlike, flip} of the candidate and the active run (None = unknown).
    A candidate is promoted when it is not worse on any gate; a missing measurement on either side
    skips that gate. JevBench accuracy is reported, never a gate; its option-rotation flip rate only
    blocks a candidate that got worse than the active run by more than 5 points."""
    def delta(k: str) -> float | None:
        return None if new.get(k) is None or old.get(k) is None else new[k] - old[k]
    d = {k: delta(k) for k in ("kl", "test_acc", "clio", "hardlike", "flip")}
    ok = ((d["kl"] is None or d["kl"] <= 0.01) and (d["test_acc"] is None or d["test_acc"] >= -0.005)
          and (d["clio"] is None or d["clio"] >= -0.01) and (d["hardlike"] is None or d["hardlike"] >= -0.01)
          and (d["flip"] is None or d["flip"] <= 0.05))
    fmt = lambda v: "n/a" if v is None else f"{v:+.4f}"  # noqa: E731
    why = ", ".join(f"{k} {fmt(v)}" for k, v in d.items())
    return ok, why


def run(*args: str) -> None:
    print(f"[retrain] $ {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)


def main() -> None:
    run("scripts.label", "--sources", "synth", "--workers", "4")  # new cases only; already labeled ones are skipped
    run("scripts.build_targets")
    n = export_user_targets()
    out = USER_TARGETS
    print(f"[retrain] {n} user-labeled questions", flush=True)
    nr = export_reviewer_targets()
    print(f"[retrain] {nr} reviewer answers (second opinion, weight of one Jev target)", flush=True)

    act = active_model_dir()
    base = act.parent.name if act else None
    name = "v" + time.strftime("%m%d-%H%M")
    args = ["scripts.train", "--name", name, "--extra", str(out), "--extra", str(REVIEWER_TARGETS)]
    if base:
        arch = json.loads((act.parent / "meta.json").read_text()).get("base", "jhu-clsp/mmBERT-small")
        args += ["--init", base, "--base", arch, "--epochs", "2", "--lr", "2.5e-5", "--head-lr", "2e-4"]
    args += ["--bs", "8", "--accum", "2", "--max-len", "640", "--long-max-len", "1536", "--long-bs", "4"]
    run(*args)
    run("scripts.export_onnx", "--name", name)
    run("scripts.calibrate", "--name", name)
    run("scripts.evaluate", "--model", f"runs/{name}/onnx")
    run("scripts.eval_clio", "--model", f"runs/{name}/onnx")
    run("scripts.eval_user", "--model", f"runs/{name}/onnx")
    run("scripts.eval_hardlike", "--model", f"runs/{name}/onnx")
    run("scripts.eval_jevbench", "--model", f"runs/{name}/onnx", "--calibration", "jevbench", "--probes")

    new_meta = json.loads(Path(f"runs/{name}/meta.json").read_text())
    promote, why = True, "no active model"
    if act:
        read = lambda d, f, k: (json.loads((d / f).read_text()).get(k) if (d / f).exists() else None)  # noqa: E731
        def measures(d: Path) -> dict:
            meta = read(d, "meta.json", "val") or {}
            return {"kl": (meta.get("all") or {}).get("kl"), "test_acc": read(d, "eval_td_test.json", "accuracy"),
                    "clio": read(d, "eval_clio.json", "mean"), "hardlike": read(d, "eval_hardlike.json", "accuracy"),
                    "flip": (read(d, "eval_jevbench.json", "probes") or {}).get("rotation_flip_rate")}
        promote, why = promotion(measures(Path(f"runs/{name}")), measures(act.parent))
        jev = read(Path(f"runs/{name}"), "eval_jevbench.json", "accuracy")
        why = f"{why} vs {base}; jevbench public {jev} (reported only)"
    new_meta["notes"] = ("promoted: " if promote else "kept previous: ") + why
    Path(f"runs/{name}/meta.json").write_text(json.dumps(new_meta, indent=2))
    if promote:
        set_active(Path(f"runs/{name}/onnx"))
    print(f"[retrain] {'PROMOTED' if promote else 'not promoted'} {name}: {why}", flush=True)
    for step in (["scripts.rescore_queue"] if promote else []) + [
            ["scripts.import_queue", "--min-disagree", "0.35", "--max", "400"], ["scripts.prune_runs", "--apply"]]:
        step = [step] if isinstance(step, str) else step
        try:
            run(*step)
        except subprocess.CalledProcessError as e:
            print(f"[retrain] {step[0]} failed: {e}", flush=True)


if __name__ == "__main__":
    main()
