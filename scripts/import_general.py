"""General decision data from public sets, as an --extra file for scripts.train.

  python -m scripts.import_general --sn92 <dir with train-*.jsonl.gz> --tdes <train.es.jsonl> \
      --out runs/_data/general_extra.jsonl [--per-category 3000]

Two sources:
- decisionmodels/sn92-generalist-train-v1 (CC-BY-4.0): policy, finance and insurance, support, tool routing
  and knowledge items with one correct option. Only items made for training are kept: anything built to
  imitate a benchmark (`component` jev / hfdi / aadi, or a `benchmark` tag) is dropped, and so is any item
  sharing a 6-word phrase with JevBench's public items or with our own suites (typed-decisions test,
  cliolike, hard-like). The hard label becomes a soft target (1 - eps on the gold option).
- telepatia-ai/typed-decisions-pt-es (Apache-2.0): the Spanish translation of typed-decisions `train`. It
  reuses the English targets and val side of each case; `test` cases are never read.

Every row carries its case, so scripts.train needs nothing else.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

from scripts.import_cases import _texts_of, public_shingles, shingles
from temis.schema import SchemaError, parse_questions

BENCH_COMPONENTS = {"jev", "hfdi", "aadi"}
CATEGORIES = ("rules_policy_law", "finance_commerce_insurance", "support_ops_logistics",
              "tools_agent_routing", "knowledge_reasoning")
EPS = 0.1


def soft_target(gold: int, n: int, eps: float = EPS) -> list[float]:
    rest = eps / (n - 1)
    return [1.0 - eps if i == gold else rest for i in range(n)]


def sn92_row(it: dict, leaks: set) -> dict | None:
    if it.get("component") in BENCH_COMPONENTS or it.get("benchmark"):
        return None
    if it.get("lang") not in ("en", "es"):
        return None
    qid = "decision"
    case = {"id": f"sn92-{it['id']}", "state": it["state"], "questions": {qid: it["question"]}}
    try:
        (q,) = parse_questions(case["questions"])
    except (SchemaError, ValueError):
        return None
    if str(it.get("expected")) not in q.options:   # parse_questions already refuses more than 12 options
        return None
    if leaks and shingles(_texts_of(case)) & leaks:
        return None
    root = (it.get("ancestry") or {}).get("root") or it["id"]
    return {"id": case["id"], "qid": qid, "kind": q.kind, "source": "sn92", "schema_id": f"sn92-{root}",
            "target": soft_target(q.options.index(str(it["expected"])), q.n), "teachers": ["gold"],
            "case": case, "category": it.get("category"), "lang": it["lang"]}


def sample_by_category(rows: list[dict], per_category: int, rng: random.Random) -> list[dict]:
    """Every Spanish row, then English rows at random up to `per_category` per category."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    out = []
    for cat in sorted(by):
        es = [r for r in by[cat] if r["lang"] == "es"]
        en = [r for r in by[cat] if r["lang"] != "es"]
        rng.shuffle(en)
        out += es + en[:max(0, per_category - len(es))]
    return out


def tdes_rows(cases: list[dict], en_targets: dict, en_options: dict | None = None) -> list[dict]:
    """Spanish typed-decisions `train` cases with the English targets of the same (case, question)."""
    out = []
    for c in cases:
        if not c["id"].startswith("tr_"):
            continue
        cid = f"td-{c['id']}"
        for q in parse_questions(c["questions"]):
            t = en_targets.get((cid, q.qid))
            if t is None or len(t) != q.n:
                continue
            if en_options is not None and en_options.get((cid, q.qid)) != q.options:
                continue
            out.append({"id": f"{cid}-es", "qid": q.qid, "kind": q.kind, "source": "td-train-es", "schema_id": cid,
                        "target": t, "teachers": ["gold"],
                        "case": {"id": f"{cid}-es", "state": c["state"], "questions": c["questions"]}})
    return out


def eval_shingles() -> set:
    """Phrases of every suite we measure with: JevBench public, typed-decisions test, cliolike, hard-like."""
    from temis.data.sources import load_td
    texts: list[str] = []
    for c in load_td("test"):
        texts += _texts_of(c)
    for path in ("data/cliolike.jsonl", "data/hardlike.jsonl"):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                texts += _texts_of(json.loads(line))
    return public_shingles() | shingles(texts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sn92", required=True, help="directory with the train-*.jsonl.gz shards")
    ap.add_argument("--tdes", required=True, help="train.es.jsonl of telepatia-ai/typed-decisions-pt-es")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-category", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    leaks = eval_shingles()
    print(f"[general] {len(leaks)} phrases from the evaluation suites")
    items = [json.loads(line) for f in sorted(glob.glob(f"{a.sn92}/train-*.jsonl.gz"))
             for line in gzip.open(f, "rt", encoding="utf-8") if line.strip()]
    wanted = [it for it in items if it.get("category") in CATEGORIES]
    kept = [r for r in (sn92_row(it, leaks) for it in wanted) if r is not None]
    picked = sample_by_category(kept, a.per_category, random.Random(a.seed))
    print(f"[general] sn92: {len(items)} items, {len(wanted)} in the categories, {len(kept)} kept, {len(picked)} sampled")

    from temis.data.sources import load_td, read_jsonl
    en_targets = {(r["id"], r["qid"]): r["target"] for r in read_jsonl(Path("data/targets.jsonl"))
                  if r.get("source") == "td-train"}
    en_options = {(c["id"], q.qid): q.options for c in load_td("train") for q in parse_questions(c["questions"])}
    es_cases = [json.loads(line) for line in Path(a.tdes).read_text().splitlines() if line.strip()]
    es = tdes_rows(es_cases, en_targets, en_options)
    print(f"[general] typed-decisions es: {len(es_cases)} cases, {len(es)} questions")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in picked + es:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[general] {len(picked) + len(es)} rows -> {out}")


if __name__ == "__main__":
    main()
