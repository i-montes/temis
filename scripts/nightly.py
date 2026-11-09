"""Autonomous loop: whenever enough new labeled synthetic cases exist, rebuild targets, train a
fresh version, export, evaluate on typed-decisions test, promote it if it is better, and refresh
the review queue.

  nohup python -m scripts.nightly --every 600 &

Promotion rule: higher mean of typed-decisions test accuracy and Clio benchmark mean (or no active model).
State in runs/nightly_state.json; log lines start with [nightly].
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from decisor.data.sources import load_synth, read_jsonl
from decisor.service import active_model_dir, set_active
from decisor.userlabels import REVIEWER_TARGETS, USER_TARGETS, export_reviewer_targets, export_user_targets

STATE = Path("runs/nightly_state.json")
STOP = Path("runs/nightly.stop")  # while this file exists the loop exits at its next turn (off by the user's choice)
USER_TEST_MIN = 150  # questions in Iván's personal test before it takes part in promotions


def log(msg: str) -> None:
    print(f"[nightly] {time.strftime('%H:%M:%S')} {msg}", flush=True)


ENV = {**__import__("os").environ, "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.5", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.4"}


def run(*args: str) -> None:
    log("$ " + " ".join(args))
    subprocess.run(["nice", "-n", "10", sys.executable, "-m", *args], check=True, env=ENV)


GGUF = "external/gguf/Qwen3.5-9B-Q4_K_M.gguf"
PROCS: dict = {}


def start_generation() -> None:
    """llama-server + generator, only while no training runs (they would not fit in memory together)."""
    if PROCS.get("gen") and PROCS["gen"].poll() is None:
        return
    if subprocess.run(["pgrep", "-f", r"python.* -m scripts\.generate"], capture_output=True).returncode == 0:
        return  # a generator from before a reload is still running
    Path("runs/logs").mkdir(parents=True, exist_ok=True)
    up = subprocess.run(["curl", "-sf", "http://127.0.0.1:8089/health"], capture_output=True).returncode == 0
    if not up:
        PROCS["llama"] = subprocess.Popen(
        ["nice", "-n", "5", "llama-server", "-m", GGUF, "--host", "127.0.0.1", "--port", "8089", "-np", "2",
         "-c", "12288", "-ngl", "99", "--jinja", "-fa", "on", "-t", "6", "--cache-type-k", "q8_0",
         "--cache-type-v", "q8_0", "--alias", "qwen3.5-9b"],
        stdout=open("runs/logs/llama-server.log", "a"), stderr=subprocess.STDOUT)
    for _ in range(90):
        try:
            import urllib.request
            urllib.request.urlopen("http://127.0.0.1:8089/health", timeout=2)
            break
        except Exception:  # noqa: BLE001
            time.sleep(2)
    PROCS["gen"] = subprocess.Popen(
        ["nice", "-n", "10", sys.executable, "-m", "scripts.generate", "--model", "qwen3.5-9b", "--target", "100000",
         "--workers", "2"], stdout=open("runs/logs/generate.log", "a"), stderr=subprocess.STDOUT)
    log("generation started")


def stop_generation() -> None:
    for k in ("gen", "llama"):
        p = PROCS.pop(k, None)
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
    subprocess.run(["pkill", "-f", "scripts.generate"], check=False)
    subprocess.run(["pkill", "-f", f"llama-server -m {GGUF}"], check=False)
    time.sleep(5)
    log("generation stopped")


def labeled_synth() -> int:
    ids = {c["id"] for c in load_synth()}
    return sum(1 for r in read_jsonl(Path("data/labels/jev.jsonl")) if r["id"] in ids)


def score(run_dir: Path) -> dict | None:
    p = run_dir / "eval_td_test.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    c = run_dir / "eval_clio.json"
    r["clio"] = json.loads(c.read_text())["mean"] if c.exists() else 0.0
    u = run_dir / "eval_user.json"
    r["user"] = json.loads(u.read_text()) if u.exists() else {"n": 0, "accuracy": None}
    parts = [r["accuracy"], r["clio"]]
    if (r["user"].get("n") or 0) >= USER_TEST_MIN:  # Iván's own test counts once it is big enough
        parts.append(r["user"]["accuracy"])
    r["combined"] = sum(parts) / len(parts)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=600, help="new labeled synthetic cases between trainings")
    ap.add_argument("--train-args", default="--epochs 2 --lr 2.5e-5 --head-lr 2e-4",
                    help="extra arguments for scripts.train (the recipe)")
    ap.add_argument("--from-scratch", action="store_true",
                    help="train a fresh model instead of fine-tuning the active one (same base)")
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--fresh-hour", type=int, default=1,
                    help="once a night from this hour (local), train a fresh model for longer instead of fine-tuning")
    ap.add_argument("--fresh-args", default="--epochs 5 --lr 3e-5 --head-lr 5e-4")
    a = ap.parse_args()
    st = json.loads(STATE.read_text()) if STATE.exists() else {"last_n": 0, "runs": []}
    if not st["last_n"]:
        st["last_n"] = labeled_synth()
    me = Path(__file__)
    started_mtime = me.stat().st_mtime
    while True:
        if STOP.exists():
            stop_generation()
            log(f"stopped: {STOP} exists (training is manual now: python -m scripts.retrain)")
            return
        if me.stat().st_mtime != started_mtime:  # code changed: restart with the new version
            log("code changed, reloading")
            import os
            os.execv(sys.executable, [sys.executable, "-m", "scripts.nightly", *sys.argv[1:]])
        n = labeled_synth()
        now = time.localtime()
        today = time.strftime("%Y-%m-%d")
        fresh = a.fresh_hour <= now.tm_hour < a.fresh_hour + 5 and st.get("last_fresh") != today
        if not fresh and n - st["last_n"] < a.every:
            start_generation()
            time.sleep(a.poll)
            continue
        stop_generation()
        name = "v" + time.strftime("%m%d-%H%M") + ("-fresh" if fresh else "")
        act = active_model_dir()
        extra = []
        if act:
            base = json.loads((act.parent / "meta.json").read_text()).get("base", "jhu-clsp/mmBERT-small")
            can_init = (act.parent / "model.pt").exists()  # pruned versions keep only their ONNX int8
            extra = ["--base", base] + ([] if (a.from_scratch or fresh or not can_init) else ["--init", act.parent.name])
        recipe = (a.fresh_args if fresh else a.train_args).split()
        if fresh:
            st["last_fresh"] = today
            STATE.write_text(json.dumps(st, indent=2))
            log("nightly fresh run: training from scratch for longer")
        try:
            run("scripts.build_targets")
            n_user = export_user_targets()
            log(f"{n_user} questions labeled by Iván go into this training (x3 weight)")
            log(f"{export_reviewer_targets()} reviewer answers go in as a second opinion (weight 1)")
            run("scripts.train", "--name", name, "--bs", "8", "--accum", "2", "--max-len", "640",
                *recipe, *extra, "--extra", str(USER_TARGETS), "--extra", str(REVIEWER_TARGETS))
            run("scripts.export_onnx", "--name", name)
            run("scripts.calibrate", "--name", name)
            run("scripts.evaluate", "--model", f"runs/{name}/onnx")
            run("scripts.eval_clio", "--model", f"runs/{name}/onnx")
            run("scripts.eval_user", "--model", f"runs/{name}/onnx")
        except subprocess.CalledProcessError as e:
            log(f"run {name} failed: {e}")
            time.sleep(a.poll)
            continue
        new = score(Path("runs") / name)
        old = score(act.parent) if act else None
        better = old is None or new["combined"] > old["combined"]
        if better:
            set_active(Path("runs") / name / "onnx")
            try:
                run("scripts.rescore_queue")
            except subprocess.CalledProcessError as e:
                log(f"rescore failed: {e}")
        log(f"{name}: synth={n} test acc={new['accuracy']:.4f} kl={new['kl_from_gold']:.4f} "
            f"clio={new['clio']:.4f} user={new['user'].get('accuracy')} (n={new['user'].get('n')}) {'PROMOTED' if better else 'kept ' + (act.parent.name if act else '-')}")
        try:
            run("scripts.import_queue", "--min-disagree", "0.35", "--max", "400")
        except subprocess.CalledProcessError as e:
            log(f"queue import failed: {e}")
        try:
            run("scripts.prune_runs", "--apply")  # old versions keep only their ONNX int8
        except subprocess.CalledProcessError as e:
            log(f"prune failed: {e}")
        st["last_n"] = n
        st["runs"].append({"name": name, "synth": n, "acc": new["accuracy"], "kl": new["kl_from_gold"], "clio": new["clio"], "user": new["user"].get("accuracy"), "user_n": new["user"].get("n"),
                          "fresh": fresh, "promoted": better})
        STATE.write_text(json.dumps(st, indent=2))


if __name__ == "__main__":
    main()
