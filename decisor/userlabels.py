"""Iván's labels as training targets.

A user label is the most reliable signal there is, so it is written as a near one-hot target
(0.9 on the chosen option) and repeated `repeat` times: in one epoch it weighs like three teacher
targets. Each row carries its case, so training does not need the case to exist anywhere else.
One in five labeled cases is held out as Iván's personal test set (`is_user_test`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from decisor.schema import parse_questions
from decisor.store import Store

USER_TARGETS = Path("data/user_targets.jsonl")


def is_user_test(decision_id: str) -> bool:
    """1 in 5 of Iván's labeled cases is his personal test set: never trained on (fixed by id)."""
    return int(hashlib.md5(decision_id.encode()).hexdigest(), 16) % 5 == 0


def export_user_targets(store: Store | None = None, path: Path = USER_TARGETS, repeat: int = 3) -> int:
    store = store or Store()
    rows = []
    for d in store.user_labeled():
        if is_user_test(d["id"]):
            continue
        qs = {q.qid: q for q in parse_questions(d["questions"])}
        case = {"id": f"user-{d['id']}", "state": d["state"], "questions": d["questions"],
                "workflow": d.get("workflow", "")}
        for qid, lab in d["labels"].items():
            q = qs.get(qid)
            if not q or lab not in q.options:
                continue
            tgt = [0.9 if o == lab else 0.1 / max(q.n - 1, 1) for o in q.options]
            rows.append({"id": case["id"], "qid": qid, "kind": q.kind, "source": "user", "target": tgt,
                         "schema_id": f"user-{d['id']}", "case": case})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            for _ in range(repeat):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


REVIEWER_TARGETS = Path("data/reviewer_targets.jsonl")
REVIEWER_CONFIDENCE = 0.7  # softer than Iván's 0.9: a reviewer is a second opinion, not the truth


def export_reviewer_targets(store: Store | None = None, path: Path = REVIEWER_TARGETS,
                            confidence: float = REVIEWER_CONFIDENCE) -> int:
    """Answers from reviewers (Claude, or another model filling revision.csv) as soft training targets.

    Weight 1, like one Jev target: they sit beside Jev's own target for the same case, so where the two
    disagree the student learns the doubt. Left out: questions Iván already labeled (his label wins),
    "I would not answer" (null), every case in Iván's personal test set, so it stays clean, and cases in
    the training validation split, so validation keeps measuring against Jev as before.
    When several reviewers answered one question their votes are averaged.
    """
    store = store or Store()
    with store.lock:
        rows = store.db.execute(
            "SELECT s.decision_id, s.qid, s.label FROM suggestions s WHERE s.label IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM labels l WHERE l.decision_id = s.decision_id AND l.qid = s.qid AND l.labeler = 'user')"
        ).fetchall()
    votes: dict[str, dict[str, list[str]]] = {}
    for did, qid, lab in rows:
        if not is_user_test(did):
            votes.setdefault(did, {}).setdefault(qid, []).append(lab)
    from decisor.data.sources import load_synth
    schema = {c["id"]: c.get("schema_id", c["id"]) for c in load_synth(clean=False)}  # same val split as Jev's row
    out = []
    for did, qv in votes.items():
        d = store.get(did)
        if not d:
            continue
        key = schema.get(did, did)
        if int(hashlib.md5(key.encode()).hexdigest(), 16) % 10 == 0:
            continue  # this case is in training's validation split (scripts/train.py is_val): keep val as it was
        qs = {q.qid: q for q in parse_questions(d["questions"])}
        case = {"id": did, "state": d["state"], "questions": d["questions"], "workflow": d.get("workflow", "")}
        for qid, labs in qv.items():
            q = qs.get(qid)
            labs = [x for x in labs if q and x in q.options]
            if not labs:
                continue
            rest = (1 - confidence) / max(q.n - 1, 1)
            tgt = [sum((confidence if o == x else rest) for x in labs) / len(labs) for o in q.options]
            out.append({"id": did, "qid": qid, "kind": q.kind, "source": "reviewer", "target": tgt,
                        "schema_id": key, "case": case})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(out)
