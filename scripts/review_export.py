"""Export the head of the review queue in a compact text form, for a human or Claude to review.

  python -m scripts.review_export --n 40 --offset 0 > review.txt
"""
from __future__ import annotations

import argparse

from decisor.schema import parse_questions, render_state
from decisor.store import Store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--only-unreviewed", action="store_true")
    a = ap.parse_args()
    st = Store()
    items = st.queue(a.offset + a.n * 3)
    if a.only_unreviewed:
        items = [it for it in items if not it.get("suggestions")]
    for it in items[a.offset:a.offset + a.n]:
        print(f"### {it['id']}  [{it.get('workflow', '')}]")
        print(render_state(it["state"])[:1400])
        for q in parse_questions(it["questions"]):
            opts = " | ".join(f"{o}={d}" if q.kind != "noul" else f"{o}" for o, d in zip(q.options, q.descriptions))
            print(f"  - {q.qid} ({q.kind}): {q.instructions}  ->  {opts}")
        print()


if __name__ == "__main__":
    main()
