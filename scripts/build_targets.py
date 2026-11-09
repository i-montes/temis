"""Calibrate the teacher (Jev) against typed-decisions train gold and build soft training targets.

Jev is the only teacher. Its raw distributions are overconfident (KL 1.44 on the leaderboard), so one
temperature per question type is fitted against the gold of typed-decisions `train` and applied to
every synthetic case.

Writes data/targets.jsonl ({id, qid, kind, source, schema_id, target, teachers}) and
data/teacher_calibration.json.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from decisor.data.balance import copies
from decisor.data.sources import load_synth, load_td, read_jsonl
from decisor.schema import normalize_probs, parse_questions

EPS = 1e-4
TEACHER = "jev"


def temper(p: list[float], t: float) -> list[float]:
    w = [max(x, EPS) ** (1.0 / t) for x in p]
    s = sum(w)
    return [x / s for x in w]


def kl(g: list[float], p: list[float]) -> float:
    return sum(gi * math.log(gi / max(pi, 1e-9)) for gi, pi in zip(g, p) if gi > 0)


def main() -> None:
    labels = {r["id"]: r["dists"] for r in read_jsonl(Path(f"data/labels/{TEACHER}.jsonl"))}
    td = load_td("train")
    pairs: dict[str, list[tuple[list[float], list[float]]]] = defaultdict(list)
    for c in td:
        qs = {q.qid: q for q in parse_questions(c["questions"])}
        for qid, g in c["gold"].items():
            q = qs.get(qid)
            d = labels.get(c["id"], {}).get(qid)
            if q and d and len(d) == q.n:
                pairs[q.kind].append((normalize_probs(q, g.get("probabilities") or {}), d))
    calib: dict = {"teacher": TEACHER, "temperature": {}}
    for kind, ps in pairs.items():
        best = min((sum(kl(g, temper(p, T)) for g, p in ps) / len(ps), T) for T in [x / 10 for x in range(3, 61)])
        calib["temperature"][kind] = best[1]
        raw = sum(kl(g, p) for g, p in ps) / len(ps)
        acc = sum(max(range(len(g)), key=g.__getitem__) == max(range(len(p)), key=p.__getitem__) for g, p in ps) / len(ps)
        print(f"[calib] {TEACHER} {kind:6s} n={len(ps):4d} acc={acc:.3f} KL raw={raw:.3f} -> T={best[1]} KL={best[0]:.3f}")
    Path("data/teacher_calibration.json").write_text(json.dumps(calib, indent=2))

    n = dropped = 0
    with Path("data/targets.jsonl").open("w") as f:
        for c in td:  # typed-decisions train: its own gold
            qs = {q.qid: q for q in parse_questions(c["questions"])}
            for qid, g in c["gold"].items():
                if qid in qs:
                    q = qs[qid]
                    f.write(json.dumps({"id": c["id"], "qid": qid, "kind": q.kind, "source": "td-train",
                                        "schema_id": c["workflow"], "target": normalize_probs(q, g.get("probabilities") or {}),
                                        "teachers": ["gold"]}) + "\n")
                    n += 1
        for c in load_synth():  # synthetic: calibrated Jev
            try:
                qs = parse_questions(c["questions"])
            except Exception:  # noqa: BLE001
                continue
            for q in qs:
                d = labels.get(c["id"], {}).get(q.qid)
                if not d or len(d) != q.n:
                    continue
                k = copies(c, list(q.options), d)  # rebalance memory_op (see decisor/data/balance.py)
                dropped += k == 0
                tgt = temper(d, calib["temperature"].get(q.kind, 1.0))
                for _ in range(k):
                    f.write(json.dumps({"id": c["id"], "qid": q.qid, "kind": q.kind, "source": "synth",
                                        "schema_id": c.get("schema_id", c["id"]), "target": tgt,
                                        "teachers": [TEACHER]}) + "\n")
                    n += 1
    print(f"[targets] wrote {n} question targets ({dropped} left out to rebalance memory_op)")


if __name__ == "__main__":
    main()
