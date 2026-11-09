"""Import cases written by another model (see docs/prompt_generar_casos.md) into the training pool.

  python -m scripts.import_cases salida.jsonl --source gpt-5

Each line: {"family", "language", "workflow"?, "state", "questions"} (TypeSafe question format).
Lines that fail validation are reported and skipped. Valid cases go to data/synth/external.jsonl;
Jev labels them on its next pass and the next training uses them. Nothing here is a label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from decisor.data.quality import clean_questions, state_ok
from decisor.data.sources import EXTERNAL, load_synth
from decisor.schema import SchemaError, parse_questions

FAMILIES = {"memory_op", "tool_choice", "memory_relation", "note_link", "needs_memory", "asks_to_forget",
            "tools", "injection", "command_safety", "worth_remembering", "note_tags", "go_no_go", "other"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--source", required=True, help="which model wrote them, e.g. gpt-5")
    a = ap.parse_args()
    seen = {json.dumps(c["state"], sort_keys=True, ensure_ascii=False) for c in load_synth(clean=False)}
    ok = bad = dup = 0
    out = []
    for i, line in enumerate(Path(a.path).read_text().splitlines(), 1):
        line = line.strip().strip(",")
        if not line or line.startswith(("```", "//", "#")):
            continue
        try:
            c = json.loads(line)
            fam = c.get("family", "other")
            if fam not in FAMILIES:
                raise SchemaError(f"unknown family {fam!r}")
            if c.get("language") not in ("es", "en"):
                raise SchemaError("language must be es or en")
            if not state_ok(c.get("state")):
                raise SchemaError("state too short (min 120 characters)")
            qs, rewritten, dropped = clean_questions(c.get("questions") or {})
            if dropped:
                raise SchemaError(f"{dropped} yes/no question(s) are not statements")
            parse_questions(qs)
        except (json.JSONDecodeError, SchemaError, AttributeError, TypeError) as e:
            print(f"[import] line {i}: skipped ({e})")
            bad += 1
            continue
        key = json.dumps(c["state"], sort_keys=True, ensure_ascii=False)
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        h = hashlib.sha1(key.encode()).hexdigest()[:10]
        # the review queue shows `workflow`, and generated names often give the answer away
        # ("memory_duplicate_..."): show the family instead and keep the original name aside
        out.append({"id": f"ext-{h}", "schema_id": f"ext-{fam}-{h}", "workflow": fam,
                    "external_workflow": c.get("workflow"),
                    "domain": f"agent:{fam}" if fam != "other" else f"external:{a.source}",
                    "language": c["language"], "state": c["state"], "questions": qs,
                    "source": f"external:{a.source}"})
        ok += 1
    EXTERNAL.parent.mkdir(parents=True, exist_ok=True)
    with EXTERNAL.open("a") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[import] {ok} cases added to {EXTERNAL} · {bad} skipped · {dup} duplicates")


if __name__ == "__main__":
    main()
