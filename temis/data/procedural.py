"""`tasksource/procedural-typed-decisions` (Apache-2.0): 396k deterministic typed decisions in Temis's
own format (state, questions, answers) over 18 procedural tasks (ledger arithmetic, deadlines with
clock times, event-driven state, evidence sufficiency, adjudication between versions, graph
navigation…), levels 0–4. The gold is exact, so these cost no teacher calls and teach what the
synthetic cases never cover: counting, dates, thresholds, chained lookups, "not enough evidence".

Two tasks are held out entirely (HOLDOUT_TASKS) so the internal hard-like set measures transfer,
not memorization. Everything here is English; the mix is capped in build_targets.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from temis.schema import normalize_probs, parse_questions

DATASET = "tasksource/procedural-typed-decisions"
HOLDOUT_TASKS = ("multi_view_adjudication", "policy_under_uncertainty")
PER_TASK = 2000
LEVEL_WEIGHT = {0: 0.5, 1: 1.0, 2: 1.0, 3: 1.5, 4: 1.5}
SMOOTH = 0.9  # a sure gold becomes 0.9 on the answer, the rest shared (like Iván's labels)
MAX_OPTIONS = 12  # the schema's limit; questions above it are dropped


def _questions(raw: Any) -> dict:
    qs = json.loads(raw) if isinstance(raw, str) else dict(raw)
    out = {}
    for qid, q in qs.items():
        q = dict(q)
        crit = q.get("criteria")
        if q.get("type") == "choice" and isinstance(crit, dict):
            q["criteria"] = {k: (v if v else str(k)) for k, v in crit.items()}
        if q.get("type") == "noul" and not crit:
            q.pop("criteria", None)
        if len(q.get("criteria") or ()) > MAX_OPTIONS:
            continue  # Temis (and its API) answer at most 12 options
        out[qid] = q
    return out


def to_case(row: dict) -> dict:
    state = row["state"]
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except json.JSONDecodeError:
            pass
    gold = row["answers"]
    return {"id": f"proc-{row['id']}", "workflow": row["task"], "level": int(row["level"]), "state": state,
            "questions": _questions(row["questions"]), "gold": json.loads(gold) if isinstance(gold, str) else gold,
            "source": "procedural", "language": "en"}


def _smooth(vec: list[float]) -> list[float]:
    n = len(vec)
    if n < 2 or max(vec) < 1.0:
        return vec
    j = vec.index(max(vec))
    return [SMOOTH if i == j else round((1 - SMOOTH) / (n - 1), 6) for i in range(n)]


def gold_targets(case: dict) -> list[dict]:
    out = []
    for q in parse_questions(case["questions"]):
        ans = case["gold"].get(q.qid)
        if not ans:
            continue
        if q.kind == "noul":
            p = float(ans.get("noul", 0.0))
            vec = [1 - p, p]
        else:
            vec = normalize_probs(q, ans.get("probabilities") or {})
        out.append({"qid": q.qid, "kind": q.kind, "target": _smooth(vec)})
    return out


STATE_BUDGET = 1024 - 160  # the runtime reads 1,024 tokens; ~160 go to the question and its options


def state_counter(base: str = "jhu-clsp/mmBERT-base"):
    """text -> number of tokens, with the base model's tokenizer (downloaded once)."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(base, "tokenizer.json"))
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


def budget_filter(max_tokens: int = STATE_BUDGET, base: str = "jhu-clsp/mmBERT-base"):
    count = state_counter(base)
    return lambda case: fits_budget(case, count, max_tokens)


def fits_budget(case: dict, count, max_tokens: int) -> bool:
    """True when the rendered state fits in max_tokens (count: text -> number of tokens). Cases that would
    be cut at inference time are not worth training on with an exact gold: the answer may be in the cut."""
    from temis.schema import render_state

    return count(render_state(case["state"])) <= max_tokens


def _weighted_sample(rows: list[dict], k: int, rng: random.Random) -> list[dict]:
    """k rows without replacement, level-weighted (Efraimidis–Spirakis keys), deterministic."""
    keyed = sorted(rows, key=lambda r: -(rng.random() ** (1.0 / LEVEL_WEIGHT.get(int(r["level"]), 1.0))))
    return keyed[:k]


def _load(split: str):
    from datasets import load_dataset

    return load_dataset(DATASET, "all", split=split)


def load_procedural(split: str = "train", per_task: int = PER_TASK, seed: int = 0,
                    exclude: tuple[str, ...] = HOLDOUT_TASKS, fits=None) -> list[dict]:
    """fits: optional predicate on a case (see fits_budget); cases failing it are skipped before sampling."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in _load(split):
        if r["task"] not in exclude:
            by_task[r["task"]].append(r)
    rng = random.Random(seed)
    out = []
    for task in sorted(by_task):
        cases = [c for c in map(to_case, by_task[task]) if c["questions"] and (fits is None or fits(c))]
        out += _weighted_sample(cases, per_task, rng)
    return out


def load_hardlike(n: int = 2000, seed: int = 0, fits=None) -> list[dict]:
    """The internal hard-like set, from the `test` split: half from the held-out tasks (all levels),
    half from levels 3–4 of the other tasks. Never trained on."""
    held: dict[str, list[dict]] = defaultdict(list)
    hard: dict[str, list[dict]] = defaultdict(list)
    for r in _load("test"):
        if r["task"] in HOLDOUT_TASKS:
            held[r["task"]].append(r)
        elif int(r["level"]) >= 3:
            hard[r["task"]].append(r)
    rng = random.Random(seed)
    out = []
    for group, share in ((held, n // 2), (hard, n - n // 2)):
        per = max(1, share // max(len(group), 1))
        for task in sorted(group):
            cases = [c for c in map(to_case, group[task]) if c["questions"] and (fits is None or fits(c))]
            rng.shuffle(cases)
            out += cases[:per]
    return out


__all__ = ["DATASET", "HOLDOUT_TASKS", "PER_TASK", "LEVEL_WEIGHT", "to_case", "gold_targets", "load_procedural",
           "load_hardlike", "CASES", "write_cases", "read_cases", "fits_budget", "budget_filter", "STATE_BUDGET"]


CASES = Path("data/procedural/cases.jsonl")  # the sampled cases, rebuilt by build_targets (not versioned)


def write_cases(cases: list[dict], path: Path = CASES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def read_cases(path: Path = CASES) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
