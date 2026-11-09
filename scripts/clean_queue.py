"""Apply the quality filter to the open review queue: rewrite command-like yes/no questions into
statements, drop the ones that cannot be yes/no, and remove open cases with an almost empty state
(or with no questions left). Labeled cases are never touched.

  python -m scripts.clean_queue
"""
from __future__ import annotations

import json

from decisor.data.quality import clean_questions, state_ok
from decisor.store import Store


def main() -> None:
    st = Store()
    with st.lock:
        rows = st.db.execute("SELECT id, state, questions, answers FROM decisions WHERE status='open'").fetchall()
    updates, deletes, rew, drop = [], [], 0, 0
    for r in rows:
        qs = json.loads(r["questions"])
        cleaned, a, b = clean_questions(qs)
        rew += a
        drop += b
        if not cleaned or not state_ok(json.loads(r["state"])):
            deletes.append(r["id"])
            continue
        if a or b:
            ans = {k: v for k, v in (json.loads(r["answers"]) if r["answers"] else {}).items() if k in cleaned}
            updates.append((json.dumps(cleaned, ensure_ascii=False), json.dumps(ans, ensure_ascii=False), r["id"],
                            [q for q in qs if q not in cleaned]))
    with st.lock:
        st.db.executemany("UPDATE decisions SET questions=?, answers=? WHERE id=?", [u[:3] for u in updates])
        for _, _, did, gone in updates:
            st.db.executemany("DELETE FROM suggestions WHERE decision_id=? AND qid=?", [(did, g) for g in gone])
        st.db.executemany("DELETE FROM suggestions WHERE decision_id=?", [(d,) for d in deletes])
        st.db.executemany("DELETE FROM decisions WHERE id=? AND status='open'", [(d,) for d in deletes])
        st.db.commit()
    print(f"[clean] {len(rows)} open cases: {rew} yes/no questions rewritten as statements, {drop} dropped, "
          f"{len(updates)} cases updated, {len(deletes)} cases removed")


if __name__ == "__main__":
    main()
