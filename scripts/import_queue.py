"""Fill the review queue with the training cases that would teach the model the most.

  python -m scripts.import_queue --min-disagree 0.35 --max 400

Doubt of a case = max over its questions of:
  - disagreement between the active model and the teacher target (total variation),
  - the teacher target's own uncertainty (1 - its top probability, scaled to [0, 1]).
Iván's answer on these becomes gold and enters the next retraining.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from temis.data.sources import load_synth, read_jsonl
from temis.schema import parse_questions
from temis.service import Service


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-disagree", type=float, default=0.35)
    ap.add_argument("--max", type=int, default=400)
    a = ap.parse_args()
    cal_p = Path("data/teacher_calibration.json")
    cal = json.loads(cal_p.read_text()) if cal_p.exists() else {}
    jev = {r["id"]: r["dists"] for r in read_jsonl(Path("data/labels/jev.jsonl"))}
    targets: dict[str, dict] = defaultdict(dict)
    for r in read_jsonl(Path("data/targets.jsonl")):
        if r.get("source") == "synth":
            targets[r["id"]][r["qid"]] = r["target"]
    cases = {c["id"]: c for c in load_synth()}
    svc = Service()
    scored = []
    for cid, tg in targets.items():
        c = cases.get(cid)
        if c is None or svc.store.get(cid):
            continue
        try:
            qs = {q.qid: q for q in parse_questions(c["questions"])}
        except Exception:  # noqa: BLE001
            continue
        doubt = 0.0
        for qid, t in tg.items():
            n = len(t)
            if n > 1:
                doubt = max(doubt, (1 - max(t)) / (1 - 1 / n) * 0.8)
        answers = {}
        if svc.decider is not None:
            answers = svc.decide(c["state"], c["questions"], log=False, use_precedents=False)["answers"]
            for qid, t in tg.items():
                q = qs.get(qid)
                ans = answers.get(qid)
                if q and ans:
                    p = [ans["probabilities"].get(o, 0.0) for o in q.options]
                    doubt = max(doubt, 0.5 * sum(abs(x - y) for x, y in zip(p, t)))
        scored.append((doubt, cid, answers))
    scored.sort(key=lambda x: -x[0])
    n = 0
    for doubt, cid, answers in scored:
        if doubt < a.min_disagree or n >= a.max:
            break
        c = cases[cid]
        teacher = {"jev": jev.get(cid, {}), "mix": targets.get(cid, {}),
                   "calibration": cal.get("temperature")}
        svc.store.add_decision(id=cid, origin="teachers", workflow=c.get("workflow", ""), state=c["state"],
                               questions=c["questions"], answers=answers, model_version=svc.version,
                               uncertainty=round(min(1.0, doubt), 4), teacher=teacher)
        n += 1
    print(f"[queue] imported {n} doubtful cases (of {len(scored)} scored)")


if __name__ == "__main__":
    main()
