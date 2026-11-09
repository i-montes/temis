"""Import Claude's review of queue cases as advisory suggestions (never labels).

  python -m scripts.import_suggestions data/reviews/claude_001.json

File: {"<decision_id>": {"<qid>": ["<option>" | null, "<short reason>"], ...}, ...}
null means "I would not answer this one" (ambiguous question, missing data, broken case).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from temis.schema import parse_questions
from temis.store import Store


def main() -> None:
    st = Store()
    n = bad = 0
    for path in sys.argv[1:]:
        for did, qs in json.loads(Path(path).read_text()).items():
            d = st.get(did)
            if not d:
                print(f"[suggest] unknown decision {did}")
                bad += 1
                continue
            opts = {q.qid: q.options for q in parse_questions(d["questions"])}
            for qid, (label, reason) in qs.items():
                if qid not in opts or (label is not None and label not in opts[qid]):
                    print(f"[suggest] invalid {did}/{qid}: {label!r}")
                    bad += 1
                    continue
                st.suggest(did, qid, label, reason)
                n += 1
    print(f"[suggest] imported {n} suggestions ({bad} rejected)")


if __name__ == "__main__":
    main()
