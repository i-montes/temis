"""Label the review queue somewhere else (a spreadsheet, another computer, another model) and import it.

  python -m scripts.review_sheet export revision.csv --n 150     # the 150 most doubtful, max 20 per family
  python -m scripts.review_sheet import revision.csv              # your answers become labels
  python -m scripts.review_sheet import revision.csv --by gpt-5   # someone else's: advisory only
  python -m scripts.review_sheet export revision.csv --skip-reviewed-by gpt-5   # the next batch for gpt-5

The CSV has one row per question; the case text is on the first row of each case. Fill the
`respuesta` column with the option key, its number (1, 2, 3…), or sí/no for yes/no questions.
Leave it empty (or write "?") to skip a question. Nothing else in the file is read back except
`caso_id`, `pregunta_id` and `respuesta`.

With --by other than "user", answers are stored as suggestions (like Claude's): shown in the queue,
never used as gold, precedent or training data.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from temis.schema import parse_questions, render_state
from temis.store import Store

COLUMNS = ["caso_id", "familia", "caso", "pregunta_id", "tipo", "pregunta", "opciones",
           "sugerida_modelo", "claude", "respuesta"]
YES = {"si", "sí", "s", "yes", "y", "true", "verdadero", "1"}
NO = {"no", "n", "false", "falso", "0"}
SKIP = {"", "?", "no se", "no sé", "skip", "-"}


def model_pick(ans: dict | None) -> str:
    """The model's top option, or empty when the top two are within 0.04 (the queue shows no pick then)."""
    probs = (ans or {}).get("probabilities") or {}
    if not probs:
        return ""
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.04:
        return ""
    return str(ranked[0][0])


def show(kind: str, option: str) -> str:
    if kind == "noul":
        return {"true": "sí", "false": "no"}.get(option, option)
    return option


def export(path: Path, n: int, per_family: int, skip_by: str | None = None) -> None:
    """The most doubtful open cases, at most `per_family` from any one family so all of them get answers.
    With `skip_by`, cases that reviewer already answered are left out (the next batch)."""
    st = Store()
    done: set[str] = set()
    if skip_by:
        with st.lock:
            done = {r[0] for r in st.db.execute("SELECT DISTINCT decision_id FROM suggestions WHERE by=?", (skip_by,))}
    items, seen = [], defaultdict(int)
    for it in st.queue(n * 40 + len(done)):
        if it["id"] in done:
            continue
        fam = it.get("workflow", "")
        if seen[fam] < per_family:
            seen[fam] += 1
            items.append(it)
            if len(items) == n:
                break
    rows = 0
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # BOM: Excel reads the accents
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for it in items:
            first = True
            for q in parse_questions(it["questions"]):
                if q.kind == "noul":
                    opts = "sí | no"
                else:
                    opts = " | ".join(f"{i}={o}: {d}" for i, (o, d) in enumerate(zip(q.options, q.descriptions), 1))
                sug = (it.get("suggestions") or {}).get(q.qid) or {}
                claude = ""
                if sug:
                    claude = (show(q.kind, sug["label"]) if sug.get("label") else "no respondería") + \
                             (f" — {sug['reason']}" if sug.get("reason") else "")
                w.writerow([it["id"], it.get("workflow", ""), render_state(it["state"]) if first else "",
                            q.qid, q.kind, q.instructions, opts,
                            show(q.kind, model_pick((it.get("answers") or {}).get(q.qid))), claude, ""])
                first = False
                rows += 1
    print(f"[sheet] {len(items)} cases, {rows} questions -> {path}")


def parse_answer(raw: str, q) -> str | None:
    """Option key from what was written: key, 1-based number, or sí/no. None = skip. ValueError = invalid."""
    v = raw.strip().strip(".").lower()
    if v in SKIP:
        return None
    if q.kind == "noul":
        if v in YES:
            return "true"
        if v in NO:
            return "false"
        raise ValueError(raw)
    if v in q.options:
        return v
    for o in q.options:  # "2=repetida: …" pasted whole, or the key in another case
        if v.split(":")[0].split("=")[-1].strip() == o.lower():
            return o
    if v.split("=")[0].isdigit() and 1 <= int(v.split("=")[0]) <= q.n:
        return q.options[int(v.split("=")[0]) - 1]
    raise ValueError(raw)


def do_import(path: Path, by: str) -> None:
    st = Store()
    answers: dict[str, dict[str, str]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("caso_id") and r.get("pregunta_id"):
                answers[r["caso_id"].strip()][r["pregunta_id"].strip()] = r.get("respuesta") or ""
    cases = labels = bad = 0
    for did, raw in answers.items():
        d = st.get(did)
        if not d:
            print(f"[sheet] unknown case {did}")
            bad += 1
            continue
        qs = parse_questions(d["questions"])
        by_id = {q.qid: q for q in qs}
        picked, how = {}, {}
        for qid, v in raw.items():
            q = by_id.get(qid)
            if q is None:
                continue
            try:
                lab = parse_answer(v, q)
            except ValueError:
                print(f"[sheet] {did}/{qid}: cannot read {v!r} (options: {', '.join(q.options)})")
                bad += 1
                continue
            if lab is not None:
                picked[qid] = lab
                how[qid] = "accept" if lab == model_pick((d.get("answers") or {}).get(qid)) else "pick"
        if not picked:
            continue
        if by == "user":
            st.label(did, picked, qs, render_state(d["state"]), how=how)
        else:
            for qid, lab in picked.items():
                st.suggest(did, qid, lab, f"de {by}", by=by)
        cases += 1
        labels += len(picked)
    kind = "labels" if by == "user" else f"suggestions by {by}"
    print(f"[sheet] {labels} {kind} in {cases} cases ({bad} problems)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("path")
    e.add_argument("--n", type=int, default=150, help="how many cases, most doubtful first")
    e.add_argument("--per-family", type=int, default=20, help="at most this many cases from one family")
    e.add_argument("--skip-reviewed-by", help="leave out cases this reviewer already answered, e.g. claude-hoja")
    i = sub.add_parser("import")
    i.add_argument("path")
    i.add_argument("--by", default="user", help='"user" (you: real labels) or who answered (advisory only)')
    a = ap.parse_args()
    if a.cmd == "export":
        export(Path(a.path), a.n, a.per_family, a.skip_reviewed_by)
    else:
        do_import(Path(a.path), a.by)


if __name__ == "__main__":
    sys.exit(main())
