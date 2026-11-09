"""A second opinion on the hard cases, from another model (docs/prompt_segunda_opinion.md).

  python -m scripts.second_opinion export pendientes.jsonl [--family long_policy,trap] [--all]
  python -m scripts.second_opinion import respuestas.jsonl --by gpt-5

export: hard cases (domain hard:*) that have no second opinion yet, one per line:
        {"id", "family", "state", "questions", "rationale"} (no labels of any kind).
import: {"id", "answers": {qid: option | null}} per line -> data/second_opinion.jsonl rows
        {"id", "qid", "label", "by"}. build_targets keeps a hard question only when this label
        matches Jev's top option (temis/data/balance.py::agree); null means "the rationale does not
        justify any option", which also holds the question back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from temis.data.sources import load_synth, read_jsonl
from temis.schema import parse_questions

SECOND_OPINION = Path("data/second_opinion.jsonl")


def load_second_opinions(path: Path = SECOND_OPINION) -> dict[tuple[str, str], str | None]:
    return {(r["id"], r["qid"]): r.get("label") for r in read_jsonl(path)}


def hard_cases(families: set[str] | None = None) -> list[dict]:
    out = []
    for c in load_synth():
        dom = str(c.get("domain", ""))
        if dom.startswith("hard:") and (not families or dom[5:] in families):
            out.append(c)
    return out


def export(path: Path, families: set[str] | None, include_done: bool) -> int:
    done = {cid for cid, _ in load_second_opinions()}
    n = 0
    with path.open("w") as f:
        for c in hard_cases(families):
            if c["id"] in done and not include_done:
                continue
            f.write(json.dumps({"id": c["id"], "family": c["workflow"], "state": c["state"],
                                "questions": c["questions"], "rationale": c.get("rationale", "")},
                               ensure_ascii=False) + "\n")
            n += 1
    return n


def import_answers(path: Path, by: str) -> tuple[int, int]:
    cases = {c["id"]: c for c in hard_cases()}
    ok = bad = 0
    SECOND_OPINION.parent.mkdir(parents=True, exist_ok=True)
    with SECOND_OPINION.open("a") as f:
        for r in read_jsonl(path):
            c = cases.get(r.get("id"))
            if not c:
                print(f"[second] unknown case {r.get('id')!r}")
                bad += 1
                continue
            opts = {q.qid: q.options for q in parse_questions(c["questions"])}
            for qid, label in (r.get("answers") or {}).items():
                if qid not in opts or (label is not None and str(label) not in opts[qid]):
                    print(f"[second] invalid {r['id']}/{qid}: {label!r}")
                    bad += 1
                    continue
                f.write(json.dumps({"id": r["id"], "qid": qid, "label": None if label is None else str(label), "by": by}) + "\n")
                ok += 1
    return ok, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("path")
    ex.add_argument("--family", default=None, help="comma-separated families; default all hard families")
    ex.add_argument("--all", action="store_true", help="include cases that already have a second opinion")
    im = sub.add_parser("import")
    im.add_argument("path")
    im.add_argument("--by", required=True, help="which model answered")
    a = ap.parse_args()
    if a.cmd == "export":
        fams = set(a.family.split(",")) if a.family else None
        n = export(Path(a.path), fams, a.all)
        print(f"[second] exported {n} cases to {a.path}")
    else:
        ok, bad = import_answers(Path(a.path), a.by)
        print(f"[second] imported {ok} answers into {SECOND_OPINION} ({bad} rejected)")


if __name__ == "__main__":
    main()
