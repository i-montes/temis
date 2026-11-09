"""Import cases written by another model (see docs/prompt_generar_casos.md) into the training pool.

  python -m scripts.import_cases salida.jsonl --source gpt-5

Each line: {"family", "language", "workflow"?, "state", "questions"} (TypeSafe question format).
Hard families (temis/data/families.py::HARD_FAMILIES) also carry "rationale" (required) and may
carry "surface_answer" {qid: option} and "contrast_of" (the workflow of their contrast twin);
those fields are kept with the case (the rationale feeds training later, never the model's input).
Lines that fail validation are reported and skipped. Valid cases go to data/synth/external.jsonl;
Jev labels them on its next pass and the next training uses them. Nothing here is a label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from temis.data.families import HARD_FAMILIES
from temis.data.quality import clean_questions, state_ok
from temis.data.sources import EXTERNAL, load_synth
from temis.schema import SchemaError, parse_questions

AGENT_FAMILIES = {"memory_op", "tool_choice", "memory_relation", "note_link", "needs_memory", "asks_to_forget",
                  "tools", "injection", "command_safety", "worth_remembering", "note_tags", "go_no_go"}
FAMILIES = AGENT_FAMILIES | set(HARD_FAMILIES) | {"other"}
MIN_RATIONALE = 40


def parse_line(line: str, source: str) -> dict | None:
    """One JSONL line from the other model -> a normalized case, or None (the reason is printed)."""
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
        parsed = {q.qid: q for q in parse_questions(qs)}
        extra: dict = {}
        if fam in HARD_FAMILIES:
            rationale = " ".join(str(c.get("rationale") or "").split())
            if len(rationale) < MIN_RATIONALE:
                raise SchemaError(f"hard family {fam!r} needs a rationale of at least {MIN_RATIONALE} characters")
            extra["rationale"] = rationale
            surface = c.get("surface_answer")
            if surface:
                if not isinstance(surface, dict):
                    raise SchemaError("surface_answer must be {question_id: option}")
                for qid, opt in surface.items():
                    if qid not in parsed or str(opt) not in parsed[qid].options:
                        raise SchemaError(f"surface_answer {qid!r}: {opt!r} is not one of its options")
                extra["surface_answer"] = {k: str(v) for k, v in surface.items()}
            if c.get("contrast_of"):
                extra["contrast_of"] = str(c["contrast_of"])
    except (json.JSONDecodeError, SchemaError, AttributeError, TypeError) as e:
        print(f"[import] skipped ({e})")
        return None
    key = json.dumps(c["state"], sort_keys=True, ensure_ascii=False)
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    # the review queue shows `workflow`, and generated names often give the answer away
    # ("memory_duplicate_..."): show the family instead and keep the original name aside
    domain = f"agent:{fam}" if fam in AGENT_FAMILIES else (f"hard:{fam}" if fam in HARD_FAMILIES else f"external:{source}")
    return {"id": f"ext-{h}", "schema_id": f"ext-{fam}-{h}", "workflow": fam, "external_workflow": c.get("workflow"),
            "domain": domain, "language": c["language"], "state": c["state"], "questions": qs,
            "source": f"external:{source}", **extra}


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
        c = parse_line(line, a.source)
        if c is None:
            print(f"[import]   ... at line {i}")
            bad += 1
            continue
        key = json.dumps(c["state"], sort_keys=True, ensure_ascii=False)
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        out.append(c)
        ok += 1
    EXTERNAL.parent.mkdir(parents=True, exist_ok=True)
    with EXTERNAL.open("a") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[import] {ok} cases added to {EXTERNAL} · {bad} skipped · {dup} duplicates")


if __name__ == "__main__":
    main()
