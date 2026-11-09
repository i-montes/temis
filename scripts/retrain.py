"""Retrain with the user's labels + replay of all previous targets; promote only if not worse.

  python -m scripts.retrain

0. Jev labels any synthetic or imported case it has not seen yet; targets are rebuilt
1. user labels (data/decisor.db) -> data/user_targets.jsonl (gold, x3 weight by repetition);
   reviewer answers -> data/reviewer_targets.jsonl (soft, weight 1)
2. fine-tune a new version starting from the active model, replaying data/targets.jsonl
3. export ONNX, evaluate on typed-decisions test
4. activate it only if validation KL and test accuracy are not worse than the active version's
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

from decisor.service import active_model_dir, set_active
from decisor.userlabels import REVIEWER_TARGETS, USER_TARGETS, export_reviewer_targets, export_user_targets


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
    args += ["--bs", "8", "--accum", "2", "--max-len", "640"]
    run(*args)
    run("scripts.export_onnx", "--name", name)
    run("scripts.calibrate", "--name", name)
    run("scripts.evaluate", "--model", f"runs/{name}/onnx")
    run("scripts.eval_clio", "--model", f"runs/{name}/onnx")
    run("scripts.eval_user", "--model", f"runs/{name}/onnx")

    new_meta = json.loads(Path(f"runs/{name}/meta.json").read_text())
    new_eval = json.loads(Path(f"runs/{name}/eval_td_test.json").read_text())
    promote, why = True, "no active model"
    if act:
        old_meta = json.loads((act.parent / "meta.json").read_text())
        old_eval_p = act.parent / "eval_td_test.json"
        old_eval = json.loads(old_eval_p.read_text()) if old_eval_p.exists() else {"accuracy": 0}
        dkl = new_meta["val"]["all"]["kl"] - old_meta["val"]["all"]["kl"]
        dacc = new_eval["accuracy"] - old_eval["accuracy"]
        clio = lambda d: json.loads((d / "eval_clio.json").read_text())["mean"] if (d / "eval_clio.json").exists() else 0.0  # noqa: E731
        dclio = clio(Path(f"runs/{name}")) - clio(act.parent)
        promote = dkl <= 0.01 and dacc >= -0.005 and dclio >= -0.01
        why = f"val KL {dkl:+.4f}, test acc {dacc:+.4f}, clio {dclio:+.4f} vs {base}"
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
