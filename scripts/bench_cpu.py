"""CPU latency benchmark: replays calls shaped like real ones and measures what a user waits for.

    python -m scripts.bench_cpu --model runs/v1/onnx                       # int8, threads 1,2,4,8
    python -m scripts.bench_cpu --model runs/v1/onnx --precision int8,fp32 --threads 4 --processes 3

Calls: questions about one state, as an agent asks them. Most come from the held-out short set
(`data/cliolike.jsonl`: 1 to 18 questions about a short message); a fraction (`--long-frac`) from
hard-like (`data/hardlike.jsonl`: long structured states). Every configuration (precision x threads)
runs in `--processes` fresh processes, so the cold start and peak memory are real and one noisy
process doesn't decide the result; the report keeps the median of the processes.

Per call: the end-to-end time of `Decider.decide` (tokenise, run, calibrate). Reported: p50/p90/p95/
p99, cold start (load + first call), peak RSS, real tokens per second and the share of computed
tokens that are padding. The result goes to <model>/../bench_cpu.json with the machine's details.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

CLIOLIKE, HARDLIKE = Path("data/cliolike.jsonl"), Path("data/hardlike.jsonl")
METRICS = ("p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms", "cold_ms", "rss_mb", "tokens_per_s", "padding")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_calls(short: list[dict], long: list[dict], n: int, long_frac: float, seed: int) -> list[dict]:
    """n calls, round(n * long_frac) of them from `long`, in a reproducible order."""
    rnd = random.Random(seed)
    n_long = round(n * long_frac)
    pick = [("long", r) for r in rnd.choices(long, k=n_long)] if n_long else []
    pick += [("short", r) for r in rnd.choices(short, k=n - n_long)]
    rnd.shuffle(pick)
    return [{"kind": k, "id": r["id"], "state": r["state"], "questions": r["questions"]} for k, r in pick]


def summarize(times_ms: list[float], real_tokens: list[int], padded_tokens: list[int]) -> dict:
    t = np.asarray(times_ms, dtype=np.float64)
    p = {f"p{q}_ms": float(np.percentile(t, q)) for q in (50, 90, 95, 99)}
    return {**p, "mean_ms": float(t.mean()), "calls": len(t),
            "tokens_per_s": sum(real_tokens) / (t.sum() / 1000),
            "padding": 1 - sum(real_tokens) / sum(padded_tokens)}


def aggregate(runs: list[dict]) -> list[dict]:
    """One row per (precision, threads): the median of each metric over its processes."""
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault((r["precision"], r["threads"]), []).append(r)
    out = []
    for (prec, th), rs in groups.items():
        row = {"precision": prec, "threads": th, "processes": len(rs)}
        row.update({m: median(r[m] for r in rs) for m in METRICS if all(m in r for r in rs)})
        out.append(row)
    return out


def peak_rss_mb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024 / (1024 if sys.platform == "darwin" else 1)  # bytes on macOS, KiB on Linux


def machine() -> dict:
    import onnxruntime as ort
    info = {"platform": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
            "python": platform.python_version(), "onnxruntime": ort.__version__}
    try:
        info["cpu"] = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except (OSError, StopIteration):
        info["cpu"] = platform.processor() or subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov.exists():
        info["governor"] = gov.read_text().strip()
    return info


def worker(a: argparse.Namespace) -> dict:
    from temis.runtime import Decider
    calls = load_calls(read_jsonl(Path(a.cliolike)), read_jsonl(Path(a.hardlike)), a.calls, a.long_frac, a.seed)
    t0 = time.perf_counter()
    d = Decider(a.model, int8=a.precision == "int8", provider="CPUExecutionProvider", threads=a.threads)
    d.decide(calls[0]["state"], calls[0]["questions"])
    cold_ms = (time.perf_counter() - t0) * 1000
    for c in calls[:a.warmup]:
        d.decide(c["state"], c["questions"])
    times, real, padded = [], [], []
    for c in calls:
        t = time.perf_counter()
        d.decide(c["state"], c["questions"])
        times.append((time.perf_counter() - t) * 1000)
        real.append(d.last_input_tokens)
        padded.append(d.last_padded_tokens)
    return {"precision": a.precision, "threads": a.threads, "cold_ms": cold_ms, "rss_mb": peak_rss_mb(),
            **summarize(times, real, padded)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="ONNX directory (runs/<name>/onnx)")
    ap.add_argument("--precision", default="int8", help="comma list: int8, fp32")
    ap.add_argument("--threads", default="1,2,4,8", help="comma list of intra-op thread counts")
    ap.add_argument("--processes", type=int, default=3, help="fresh processes per configuration")
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--long-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cliolike", default=str(CLIOLIKE))
    ap.add_argument("--hardlike", default=str(HARDLIKE))
    ap.add_argument("--out", default=None, help="default: <model>/../bench_cpu.json")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        a.threads = int(a.threads)
        print(json.dumps(worker(a)))
        return
    shared = ["--model", a.model, "--calls", str(a.calls), "--warmup", str(a.warmup), "--long-frac", str(a.long_frac),
              "--seed", str(a.seed), "--cliolike", a.cliolike, "--hardlike", a.hardlike, "--worker"]
    runs = []
    for prec in a.precision.split(","):
        for th in a.threads.split(","):
            for i in range(a.processes):
                out = subprocess.run([sys.executable, "-m", "scripts.bench_cpu", *shared, "--precision", prec,
                                      "--threads", th], capture_output=True, text=True, check=True).stdout
                r = json.loads(out.strip().splitlines()[-1])
                runs.append(r)
                print(f"{prec} x{th} #{i + 1}: p50 {r['p50_ms']:.1f} ms · p95 {r['p95_ms']:.1f} · cold {r['cold_ms']:.0f}",
                      flush=True)
    table = aggregate(runs)
    res = {"model": a.model, "machine": machine(), "workload": {"calls": a.calls, "warmup": a.warmup,
           "long_frac": a.long_frac, "seed": a.seed}, "configs": table, "processes": runs}
    out = Path(a.out) if a.out else Path(a.model).parent / "bench_cpu.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\n{res['machine']['cpu']} · onnxruntime {res['machine']['onnxruntime']}")
    print("precision threads   p50    p95    p99   cold ms  RSS MB  tok/s  padding")
    for r in table:
        print(f"{r['precision']:9s} {r['threads']:>7} {r['p50_ms']:6.1f} {r['p95_ms']:6.1f} {r['p99_ms']:6.1f} "
              f"{r['cold_ms']:8.0f} {r['rss_mb']:7.0f} {r['tokens_per_s']:6.0f} {r['padding']:7.1%}")


if __name__ == "__main__":
    main()
