"""Label cases with the teacher (Jev). Resumable; appends to data/labels/jev.jsonl.

  python -m scripts.label --sources td-train,synth --workers 4 --loop
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from decisor.data.sources import load_synth, load_td, read_jsonl
from decisor.teachers import BudgetExceeded, Jev

LOCK = threading.Lock()


def cases_for(sources: list[str]) -> list[dict]:
    out = []
    for s in sources:
        if s == "td-train":
            out += load_td("train")
        elif s == "synth":
            out += load_synth()
        else:
            raise SystemExit(f"unknown source {s} (td-test is never labeled for training)")
    return out


def run_once(teacher: str, sources: list[str], workers: int, limit: int) -> int:
    path = Path(f"data/labels/{teacher}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(path)}
    todo = [c for c in cases_for(sources) if c["id"] not in done][:limit]
    if not todo:
        return 0
    jev = Jev()
    stop = threading.Event()
    n = {"ok": 0, "err": 0}

    def one(c: dict) -> None:
        if stop.is_set():
            return
        try:
            r = jev.distributions(c["state"], c["questions"])
            rec = {"id": c["id"], "teacher": r["model"], "dists": r["dists"]}
            with LOCK, path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            n["ok"] += 1
            if n["ok"] % 50 == 0:
                extra = f" spent ${jev.spent:.3f}"
                print(f"[label:{teacher}] {n['ok']}/{len(todo)}{extra}", flush=True)
        except BudgetExceeded as e:
            print(f"[label:{teacher}] {e}", flush=True)
            stop.set()
        except Exception as e:  # noqa: BLE001
            n["err"] += 1
            print(f"[label:{teacher}] error on {c['id']}: {e}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    print(f"[label:{teacher}] batch done ok={n['ok']} err={n['err']}", flush=True)
    return n["ok"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=["jev"], default="jev")
    ap.add_argument("--sources", default="td-train,synth")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=10**9)
    ap.add_argument("--loop", action="store_true", help="keep polling for new synthetic cases")
    a = ap.parse_args()
    while True:
        got = run_once(a.teacher, a.sources.split(","), a.workers, a.limit)
        if not a.loop:
            break
        if got == 0:
            time.sleep(300)


if __name__ == "__main__":
    main()
