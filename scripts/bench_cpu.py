"""CPU latency benchmark over the same suites as the quality benchmarks, CPU only.

    python -m scripts.bench_cpu --model runs/v1/onnx                       # int8, threads 1,2,4,8
    python -m scripts.bench_cpu --model runs/v1/onnx --precision int8,fp32 --threads 4 --processes 3
    python -m scripts.bench_cpu --model runs/v1/onnx --workload w.jsonl    # same calls on another machine

Suites (a seeded sample of `--calls` cases each, the same for every configuration):
  td        typed-decisions test split (general decisions, 1-several questions per state)
  hardlike  data/hardlike.jsonl (long structured states)
  jevbench  JevBench public items (one question each; `scripts.eval_jevbench --fetch` first)
  cliolike  data/cliolike.jsonl (short messages, 1-18 questions, as an agent asks them)
The calls go to <out dir>/bench_workload.jsonl; `--workload` replays a saved file, so a phone or a
VPS without `datasets` runs exactly the same calls.

Every configuration (precision x threads) runs in `--processes` fresh processes on the CPU provider
alone, so cold start and peak memory are real and one noisy process doesn't decide the result; the
report keeps the median. Per suite and for all of them it measures the four things that matter:
- speed: p50/p90/p95/p99 per call, ms per question, real tokens per second, share of padding, cold
  start (load + first call) and peak RSS;
- accuracy: each suite's own rule, on CPU (what int8 and the thread count actually give);
- confidence: ECE (does a 70 % answer hit ~70 % of the time), with the production calibration;
- stability: spread of the p50 between processes, answers identical in every process, and answers
  equal to the most exact configuration (fp32, most threads).
The result goes to <model>/../bench_cpu.json with the machine.
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

SUITES = ("td", "hardlike", "jevbench", "cliolike")
CALL_METRICS = ("p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms", "ms_per_question", "tokens_per_s", "padding",
                "accuracy", "ece")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sample_calls(rows: list[dict], suite: str, n: int, seed: int) -> list[dict]:
    """Up to n distinct rows of one suite, in a reproducible order."""
    picked = random.Random(seed).sample(rows, min(n, len(rows)))
    return [{"suite": suite, "id": r["id"], "state": r["state"], "questions": r["questions"], "gold": r["gold"]}
            for r in picked]


def jevbench_rows(items: list[dict]) -> list[dict]:
    from temis.jevbench import to_request
    return [{"id": it["id"], **dict(zip(("state", "questions"), to_request(it))),
             "gold": {"decision": {k: v for k, v in it.items() if k != "state"}}} for it in items]


def grade(call: dict, answers: dict) -> list[tuple[bool, float, str | None]]:
    """(right, confidence, predicted label) per gold question, with each suite's own rule: argmax of
    the gold distribution (td, hardlike), Clio's use of the answer (cliolike: yes from 0.5), the item's
    labels (jevbench). A missing answer is wrong."""
    from scripts.eval_cliolike import right
    from scripts.evaluate import _noul_gold, score_one
    from temis.jevbench import probs_for_labels, score_item
    out = []
    for qid, g in call["gold"].items():
        a = answers.get(qid)
        if a is None:
            out.append((False, 0.0, None))
        elif call["suite"] == "jevbench":
            r = score_item(g, probs_for_labels(g, a))
            out.append((bool(r["correct"]), float(r["conf"]), r["predicted"]))
        elif call["suite"] == "cliolike":
            out.append((right(a.to_typesafe(), g), a.confidence, a.label))
        else:
            ok, _, _, conf = score_one(a.probabilities, g if "probabilities" in g else _noul_gold(g))
            out.append((bool(ok), float(conf), a.label))
    return out


def suite_rows(suite: str) -> list[dict]:
    if suite == "td":
        from temis.data.sources import load_td
        return load_td("test")
    if suite == "jevbench":
        from temis.jevbench import load_public
        return jevbench_rows(load_public())
    return read_jsonl(Path(f"data/{suite}.jsonl"))


def summarize(times_ms: list[float], real_tokens: list[int], padded_tokens: list[int], questions: list[int],
              graded: list[tuple] = ()) -> dict:
    from temis.jevbench import ece
    t = np.asarray(times_ms, dtype=np.float64)
    quality = {"accuracy": sum(g[0] for g in graded) / len(graded),
               "ece": ece([g[0] for g in graded], [g[1] for g in graded])} if graded else {}
    p = {f"p{q}_ms": float(np.percentile(t, q)) for q in (50, 90, 95, 99)}
    return {**p, "mean_ms": float(t.mean()), "ms_per_question": float(t.sum() / sum(questions)),
            "calls": len(t), "questions": sum(questions),
            "tokens_per_s": sum(real_tokens) / (t.sum() / 1000),
            "padding": 1 - sum(real_tokens) / sum(padded_tokens), **quality}


def by_suite(records: list[dict]) -> dict:
    """{suite: summary} for each suite in `records`, plus "all" over every call."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["suite"], []).append(r)
    groups["all"] = records
    return {k: summarize([r["ms"] for r in rs], [r["real"] for r in rs], [r["padded"] for r in rs],
                         [r["questions"] for r in rs], [g for r in rs for g in r.get("graded", ())])
            for k, rs in groups.items()}


def predictions(records: list[dict]) -> dict:
    """{suite: predicted labels in workload order}, plus "all"."""
    out: dict[str, list] = {}
    for r in records:
        out.setdefault(r["suite"], []).extend(g[2] for g in r["graded"])
    out["all"] = [g[2] for r in records for g in r["graded"]]
    return out


def _agree(a: list, b: list) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a) if a else 1.0


def aggregate(processes: list[dict], reference: tuple | None = None) -> list[dict]:
    """One row per (suite, precision, threads): the median of each metric over its processes, plus
    stability: p50_cv (spread of the p50 between processes), repeatable (share of answers identical in
    every process) and agree_ref (share equal to the reference configuration, default the first)."""
    groups: dict[tuple, list[tuple[dict, dict]]] = {}
    for p in processes:
        for suite, summ in p["suites"].items():
            groups.setdefault((suite, p["precision"], p["threads"]), []).append((p, summ))
    reference = reference or (processes[0]["precision"], processes[0]["threads"])
    out = []
    for (suite, prec, th), items in groups.items():
        row = {"suite": suite, "precision": prec, "threads": th, "processes": len(items)}
        row.update({m: median(s[m] for _, s in items) for m in CALL_METRICS if all(m in s for _, s in items)})
        row.update({m: median(p[m] for p, _ in items) for m in ("cold_ms", "rss_mb")})
        p50 = [s["p50_ms"] for _, s in items]
        row["p50_cv"] = float(np.std(p50) / np.mean(p50)) if len(p50) > 1 else 0.0
        preds = [p["preds"][suite] for p, _ in items if suite in p.get("preds", {})]
        ref = groups.get((suite, *reference))
        if preds and ref and suite in ref[0][0].get("preds", {}):
            row["repeatable"] = sum(len(set(col)) == 1 for col in zip(*preds)) / len(preds[0]) if preds[0] else 1.0
            row["agree_ref"] = _agree(preds[0], ref[0][0]["preds"][suite])
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
    calls = read_jsonl(Path(a.workload))
    t0 = time.perf_counter()
    d = Decider(a.model, int8=a.precision == "int8", provider="CPUExecutionProvider", threads=a.threads)
    d.decide(calls[0]["state"], calls[0]["questions"])
    cold_ms = (time.perf_counter() - t0) * 1000
    for c in calls[:a.warmup]:
        d.decide(c["state"], c["questions"])
    recs = []
    for c in calls:
        t = time.perf_counter()
        answers = d.decide(c["state"], c["questions"])
        ms = (time.perf_counter() - t) * 1000
        recs.append({"suite": c["suite"], "ms": ms, "real": d.last_input_tokens, "padded": d.last_padded_tokens,
                     "questions": len(c["questions"]), "graded": grade(c, answers)})
    return {"precision": a.precision, "threads": a.threads, "cold_ms": cold_ms, "rss_mb": peak_rss_mb(),
            "suites": by_suite(recs), "preds": predictions(recs)}


def build_workload(suites: list[str], n: int, seed: int) -> list[dict]:
    calls = []
    for suite in suites:
        try:
            calls += sample_calls(suite_rows(suite), suite, n, seed)
        except (FileNotFoundError, ImportError) as e:
            print(f"skipping {suite}: {e}", file=sys.stderr)
    random.Random(seed).shuffle(calls)  # suites interleaved: none gets a warmer or cooler CPU than the rest
    return calls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="ONNX directory (runs/<name>/onnx)")
    ap.add_argument("--precision", default="int8", help="comma list: int8, fp32")
    ap.add_argument("--threads", default="1,2,4,8", help="comma list of intra-op thread counts")
    ap.add_argument("--processes", type=int, default=3, help="fresh processes per configuration")
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--calls", type=int, default=100, help="cases sampled per suite")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workload", default=None, help="replay a saved bench_workload.jsonl instead of sampling")
    ap.add_argument("--out", default=None, help="default: <model>/../bench_cpu.json")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        a.threads = int(a.threads)
        print(json.dumps(worker(a)))
        return
    out = Path(a.out) if a.out else Path(a.model).parent / "bench_cpu.json"
    if a.workload:
        workload = Path(a.workload)
    else:
        workload = out.parent / "bench_workload.jsonl"
        workload.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n"
                                    for c in build_workload(a.suites.split(","), a.calls, a.seed)))
    calls = read_jsonl(workload)
    print(f"{len(calls)} calls: " + ", ".join(f"{s} {sum(c['suite'] == s for c in calls)}"
                                              for s in dict.fromkeys(c["suite"] for c in calls)), flush=True)
    shared = ["--model", a.model, "--workload", str(workload), "--warmup", str(a.warmup), "--worker"]
    procs = []
    for prec in a.precision.split(","):
        for th in a.threads.split(","):
            for i in range(a.processes):
                res = subprocess.run([sys.executable, "-m", "scripts.bench_cpu", *shared, "--precision", prec,
                                      "--threads", th], capture_output=True, text=True, check=True).stdout
                r = json.loads(res.strip().splitlines()[-1])
                procs.append(r)
                al = r["suites"]["all"]
                print(f"{prec} x{th} #{i + 1}: p50 {al['p50_ms']:.1f} ms · {al['ms_per_question']:.1f} ms/question · "
                      f"acc {al['accuracy']:.3f} · cold {r['cold_ms']:.0f}", flush=True)
    precs, ths = a.precision.split(","), a.threads.split(",")
    reference = ("fp32" if "fp32" in precs else precs[0], int(ths[-1]))  # the most exact configuration
    table = aggregate(procs, reference)
    res = {"model": a.model, "machine": machine(), "workload": {"file": str(workload), "calls": len(calls),
           "per_suite": a.calls, "warmup": a.warmup, "seed": a.seed}, "configs": table, "processes": procs}
    out.write_text(json.dumps(res, indent=2))
    print(f"\n{res['machine']['cpu']} · onnxruntime {res['machine']['onnxruntime']}")
    print(f"stability: p50_cv = spread of p50 between processes; repeatable = answers equal in every process; "
          f"agree = answers equal to {reference[0]} x{reference[1]}")
    print("suite     prec  thr   p50    p95    p99  ms/q  tok/s  pad    acc    ECE  p50_cv  repeat  agree  cold  RSS")
    for r in sorted(table, key=lambda r: (r["suite"], r["precision"], int(r["threads"]))):
        print(f"{r['suite']:9s} {r['precision']:5s} {r['threads']:>3} {r['p50_ms']:6.1f} {r['p95_ms']:6.1f} "
              f"{r['p99_ms']:6.1f} {r['ms_per_question']:5.1f} {r['tokens_per_s']:6.0f} {r['padding']:4.0%} "
              f"{r['accuracy']:6.3f} {r['ece']:6.3f} {r['p50_cv']:6.1%} {r.get('repeatable', 1):7.1%} "
              f"{r.get('agree_ref', 1):6.1%} {r['cold_ms']:5.0f} {r['rss_mb']:4.0f}")


if __name__ == "__main__":
    main()
