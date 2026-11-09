"""Recompute the stored suggestion and uncertainty of every open queue item with the active model
(for example after a new version or a new calibration).

  python -m scripts.rescore_queue
"""

from __future__ import annotations

import json

from decisor.data.families import queue_bonus
from decisor.service import Service, uncertainty
from decisor.runtime import make_answer
from decisor.schema import parse_questions, render_state


def main() -> None:
    svc = Service()
    if svc.decider is None:
        raise SystemExit("no active model")
    with svc.store.lock:
        rows = svc.store.db.execute("SELECT id, state, questions, teacher, workflow FROM decisions WHERE status='open'").fetchall()
    updates = []
    for r in rows:
        qs = parse_questions(json.loads(r["questions"]))
        st = render_state(json.loads(r["state"]))
        answers = {q.qid: make_answer(q, svc.decider.probabilities(q, lg))
                   for q, lg in zip(qs, svc.decider.logits(qs, st))}
        out = {k: a.to_typesafe() for k, a in answers.items()}
        # priority = the model's own doubt, or how far it is from the teacher (a confident answer that
        # contradicts Jev is the most useful thing to label), or the teacher's own doubt
        doubt = uncertainty(answers)
        teacher = json.loads(r["teacher"]) if r["teacher"] else {}
        targets = teacher.get("mix") or {}
        for q in qs:
            t = targets.get(q.qid)
            if t and len(t) == q.n:
                p = [answers[q.qid].probabilities[o] for o in q.options]
                doubt = max(doubt, 0.5 * sum(abs(x - y) for x, y in zip(p, t)),
                            (1 - max(t)) / (1 - 1 / q.n) * 0.8 if q.n > 1 else 0.0)
        u = round(min(doubt, 1.0), 4)
        updates.append((json.dumps(out, ensure_ascii=False), u, svc.version, round(u + queue_bonus(r["workflow"]), 4), r["id"]))
    # compute everything first, then one short write: the web server can keep saving labels meanwhile
    with svc.store.lock:
        svc.store.db.executemany("UPDATE decisions SET answers=?, uncertainty=?, model_version=?, priority=? WHERE id=?", updates)
        svc.store.db.commit()
    n = len(updates)
    print(f"[rescore] {n} open decisions rescored with {svc.version} ({svc.decider.calibration} calibration)")


if __name__ == "__main__":
    main()
