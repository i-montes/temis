"""A second opinion on the hard cases, from another model (docs/prompt_segunda_opinion.md).

  python -m scripts.second_opinion export pendientes.jsonl [--family long_policy,trap] [--all] [--only-disagreeing]
  python -m scripts.second_opinion import respuestas.jsonl --by gpt-5 [--as gold]

--as gold: the answers are the AUTHOR's own (the model that wrote the cases) -> data/author_gold.jsonl;
build_targets then uses them as the target when Jev or the second opinion agrees (temis/data/balance.py::
hard_target). --only-disagreeing: export only questions where Jev's top option differs from the
author's gold (the ones a second opinion can settle), in any family that carries it.

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
AUTHOR_GOLD = Path("data/author_gold.jsonl")


def load_second_opinions(path: Path = SECOND_OPINION) -> dict[tuple[str, str], str | None]:
    return {(r["id"], r["qid"]): r.get("label") for r in read_jsonl(path)}


def load_author_gold(path: Path = AUTHOR_GOLD) -> dict[tuple[str, str], str]:
    """The author's answers: from data/author_gold.jsonl plus the `author_gold` field of the cases."""
    out = {(r["id"], r["qid"]): str(r["label"]) for r in read_jsonl(path) if r.get("label") is not None}
    for c in load_synth():
        for qid, label in (c.get("author_gold") or {}).items():
            out.setdefault((c["id"], qid), str(label))
    return out


def jev_top(case: dict) -> dict[str, str]:
    """Jev's most likely option per question, when labeled."""
    jev = {r["id"]: r["dists"] for r in read_jsonl(Path("data/labels/jev.jsonl"))} if not hasattr(jev_top, "_c") else jev_top._c
    jev_top._c = jev
    d = jev.get(case["id"]) or {}
    out = {}
    for q in parse_questions(case["questions"]):
        dist = d.get(q.qid)
        if dist and len(dist) == q.n:
            out[q.qid] = q.options[max(range(q.n), key=dist.__getitem__)]
    return out


def hard_cases(families: set[str] | None = None) -> list[dict]:
    out = []
    for c in load_synth():
        dom = str(c.get("domain", ""))
        if dom.startswith("hard:") and (not families or dom[5:] in families):
            out.append(c)
    return out


def export(path: Path, families: set[str] | None, include_done: bool, only_disagreeing: bool = False) -> int:
    done = {cid for cid, _ in load_second_opinions()}
    gold = load_author_gold() if only_disagreeing else {}
    n = 0
    with path.open("w") as f:
        # settling a disagreement with the author covers every family that carries author gold (the short
        # batches too); the plain export stays on the hard cases
        pool = [c for c in load_synth() if not families or c.get("workflow") in families] if only_disagreeing \
            else hard_cases(families)
        for c in pool:
            if c["id"] in done and not include_done:
                continue
            questions = c["questions"]
            if only_disagreeing:
                top = jev_top(c)
                questions = {qid: q for qid, q in questions.items()
                             if (c["id"], qid) in gold and top.get(qid) != gold[(c["id"], qid)]}
                if not questions:
                    continue
            f.write(json.dumps({"id": c["id"], "family": c["workflow"], "state": c["state"],
                                "questions": questions, "rationale": c.get("rationale", "")},
                               ensure_ascii=False) + "\n")
            n += 1
    return n


def import_answers(path: Path, by: str, as_gold: bool = False) -> tuple[int, int]:
    cases = {c["id"]: c for c in load_synth()}
    ok = bad = 0
    target = AUTHOR_GOLD if as_gold else SECOND_OPINION
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as f:
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
    ex.add_argument("--only-disagreeing", action="store_true", help="only questions where Jev differs from the author's gold")
    im = sub.add_parser("import")
    im.add_argument("path")
    im.add_argument("--by", required=True, help="which model answered")
    im.add_argument("--as", dest="as_", choices=["second", "gold"], default="second",
                    help="gold: these are the author's own answers (data/author_gold.jsonl)")
    a = ap.parse_args()
    if a.cmd == "export":
        fams = set(a.family.split(",")) if a.family else None
        n = export(Path(a.path), fams, a.all, a.only_disagreeing)
        print(f"[second] exported {n} cases to {a.path}")
    else:
        ok, bad = import_answers(Path(a.path), a.by, as_gold=(a.as_ == "gold"))
        print(f"[second] imported {ok} answers into {AUTHOR_GOLD if a.as_ == 'gold' else SECOND_OPINION} ({bad} rejected)")


if __name__ == "__main__":
    main()
