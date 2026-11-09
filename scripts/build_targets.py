"""Calibrate the teacher (Jev) against typed-decisions train gold and build soft training targets.

Jev is the only teacher. Its raw distributions are overconfident (KL 1.44 on the leaderboard), so one
temperature per question type is fitted against the gold of typed-decisions `train` and applied to
every synthetic case.

tasksource/procedural-typed-decisions (exact gold, no teacher) is added with its case embedded in
data/procedural/cases.jsonl and capped at PROCEDURAL_SHARE of all targets; the share of every source is printed.

Writes data/targets.jsonl ({id, qid, kind, source, schema_id, target, teachers, case?}) and
data/teacher_calibration.json.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from temis.data.balance import NO_OPINION, cap_share, copies, hard_target, write_held_back
from temis.data.procedural import budget_filter, gold_targets, load_procedural, write_cases
from temis.data.sources import load_synth, load_td, read_jsonl
from scripts.second_opinion import load_author_gold, load_second_opinions
from temis.schema import normalize_probs, parse_questions

EPS = 1e-4
TEACHER = "jev"
PROCEDURAL_SHARE = 0.24  # at most this share of all targets (above it the Clio families fall under their 35 % floor)
CLIO_SHARE_MIN = 0.35    # warn when the agent families (memory, tools, security, notes) fall below this


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

    rows: list[dict] = []
    dropped = 0
    for c in td:  # typed-decisions train: its own gold
        qs = {q.qid: q for q in parse_questions(c["questions"])}
        for qid, g in c["gold"].items():
            if qid in qs:
                q = qs[qid]
                rows.append({"id": c["id"], "qid": qid, "kind": q.kind, "source": "td-train",
                             "schema_id": c["workflow"], "target": normalize_probs(q, g.get("probabilities") or {}),
                             "teachers": ["gold"]})
    clio = 0
    held: list[tuple[str, str]] = []
    second = load_second_opinions()  # {(case id, qid): label | None} from another model, hard cases only
    gold = load_author_gold()        # {(case id, qid): label}: the answers of the model that wrote the hard cases
    for c in load_synth():  # synthetic: calibrated Jev
        try:
            qs = parse_questions(c["questions"])
        except Exception:  # noqa: BLE001
            continue
        for q in qs:
            d = labels.get(c["id"], {}).get(q.qid)
            if not d or len(d) != q.n:
                continue
            tgt = temper(d, calib["temperature"].get(q.kind, 1.0))
            if str(c.get("domain", "")).startswith("hard:"):
                key = (c["id"], q.qid)
                tgt = hard_target(d, list(q.options), tgt, surface=(c.get("surface_answer") or {}).get(q.qid),
                                  second=second.get(key, NO_OPINION), author=gold.get(key))
                if tgt is None:
                    held.append(key)  # trap, disagreement or unverified: not a target; import_queue puts it first
                    continue
            k = copies(c, list(q.options), d)  # rebalance memory_op (see temis/data/balance.py)
            dropped += k == 0
            for _ in range(k):
                rows.append({"id": c["id"], "qid": q.qid, "kind": q.kind, "source": "synth",
                             "schema_id": c.get("schema_id", c["id"]), "target": tgt, "teachers": [TEACHER]})
                clio += str(c.get("domain", "")).startswith("agent:")
    # procedural: exact gold; only states that fit the runtime's budget (a cut state with an exact gold
    # teaches guessing); the sampled cases are kept in data/procedural/
    proc = {c["id"]: c for c in load_procedural(fits=budget_filter())}
    for c in proc.values():
        for t in gold_targets(c):
            rows.append({"id": c["id"], "qid": t["qid"], "kind": t["kind"], "source": "procedural",
                         "schema_id": f"proc-{c['workflow']}-{c['level']}", "target": t["target"], "teachers": ["gold"]})
    rows = cap_share(rows, "procedural", PROCEDURAL_SHARE)
    write_held_back(held)
    write_cases([proc[i] for i in dict.fromkeys(r["id"] for r in rows if r["source"] == "procedural")])
    with Path("data/targets.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    print(f"[targets] wrote {n} question targets ({dropped} left out to rebalance memory_op, "
          f"{len(held)} hard questions held back: trap, disagreement or unverified)")
    by = defaultdict(int)
    for r in rows:
        by[r["source"]] += 1
    for src, k in sorted(by.items()):
        print(f"[mix] {src:12s} {k:6d} ({k / n:5.1%})")
    print(f"[mix] {'agent:*':12s} {clio:6d} ({clio / n:5.1%}) Clio families inside synth"
          + ("  <-- below the %.0f %% floor" % (CLIO_SHARE_MIN * 100) if clio / n < CLIO_SHARE_MIN else ""))


if __name__ == "__main__":
    main()
