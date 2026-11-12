"""Build the held-out short set (data/cliolike.jsonl): cases shaped like Clio's production calls, written
by one model and answered blind by another; only the answers they share are kept. Never trained on.

  python -m scripts.build_cliolike blind autor/*.jsonl > ciego.jsonl      # the cases without their answers
  python -m scripts.build_cliolike merge --author autor/*.jsonl --second respuestas.jsonl

The author's files are compact lines (docs/prompt_casos_cortos.md). The second labeler gets `blind` and
returns, per line, {"n": <number>, <the answer fields of its family>}. A question stays when both gave
the same answer; a tool that either of them marked debatable, or needed for one and not for the other,
is not asked. scripts/eval_cliolike.py scores a model on the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from scripts.expand_short import check, expand, read_compact

CLIOLIKE = Path("data/cliolike.jsonl")
ANSWERS = {"pre": ("tools", "skip", "mem"), "rel": ("helps",), "guard": ("inseguro", "guardar", "cat"),
           "relation": ("rel",), "preserved": ("ok",), "place": ("lugar",), "op": ("op",), "del": ("pide",)}


def blind(author: list[dict]) -> list[dict]:
    """The cases as the second labeler sees them: numbered, without any answer."""
    return [{"n": i, **{k: v for k, v in line.items() if k not in ANSWERS[line["f"]]}} for i, line in enumerate(author)]


def merge(author: list[dict], second: list[dict]) -> tuple[list[dict], int]:
    """(held-out cases, how many were dropped whole). Each case keeps the questions both answered alike."""
    theirs = {r.get("n"): r for r in second}
    out, dropped = [], 0
    for i, line in enumerate(author):
        other = {**{k: v for k, v in line.items() if k not in ANSWERS[line["f"]]},
                 **{k: v for k, v in (theirs.get(i) or {}).items() if k in ANSWERS[line["f"]]}}
        if i not in theirs or check(other):
            dropped += 1
            continue
        a, b = expand(line, every=True), expand(other, every=True)
        gold = {q: v for q, v in a["expected"].items() if b["expected"].get(q) == v}
        if not gold:
            dropped += 1
            continue
        key = json.dumps(a["state"], sort_keys=True, ensure_ascii=False)
        out.append({"id": "cl-" + hashlib.sha1(key.encode()).hexdigest()[:10], "family": a["family"],
                    "language": a["language"], "state": a["state"],
                    "questions": {q: a["questions"][q] for q in gold}, "gold": gold})
    return out, dropped


def _read(paths: list[str]) -> list[dict]:
    lines: list[dict] = []
    for p in paths:
        good, bad = read_compact(Path(p))
        for b in bad:
            print(f"[cliolike] {p} {b}", file=sys.stderr)
        lines += good
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    bl = sub.add_parser("blind")
    bl.add_argument("paths", nargs="+")
    me = sub.add_parser("merge")
    me.add_argument("--author", nargs="+", required=True)
    me.add_argument("--second", required=True)
    me.add_argument("--out", default=str(CLIOLIKE))
    a = ap.parse_args()
    if a.cmd == "blind":
        for line in blind(_read(a.paths)):
            print(json.dumps(line, ensure_ascii=False))
        return
    author = _read(a.author)
    second = [json.loads(s) for s in Path(a.second).read_text().splitlines() if s.strip().startswith("{")]
    cases, dropped = merge(author, second)
    seen: set[str] = set()
    with Path(a.out).open("w") as f:
        for c in cases:
            if c["id"] not in seen:
                seen.add(c["id"])
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    asked = Counter(c["family"] for c in cases for _ in c["gold"])
    print(f"[cliolike] {len(seen)} cases, {sum(asked.values())} questions -> {a.out} ({dropped} cases dropped: no shared answer)")
    for fam, n in sorted(asked.items()):
        print(f"[cliolike]   {fam:18s} {n:5d} questions")


if __name__ == "__main__":
    main()
