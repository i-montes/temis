import json

from temis.data.balance import cap_share
from temis.data.procedural import gold_targets, to_case
from temis.schema import parse_questions

ROW = {"id": "arithmetic:train:0", "task": "arithmetic", "level": 2,
       "state": json.dumps({"opening_balance": 35}),
       "questions": json.dumps({"final": {"type": "choice", "instructions": "Closing balance?", "criteria": {"£0": None, "£20": None}},
                                "neg": {"type": "noul", "instructions": "Does it go negative?"},
                                "cnt": {"type": "score", "instructions": "Count.", "criteria": ["0", "1", "2"]}}),
       "answers": json.dumps({"final": {"type": "choice", "choice": "£20", "probabilities": {"£0": 0.0, "£20": 1.0}},
                              "neg": {"type": "noul", "noul": 0.0},
                              "cnt": {"type": "score", "score": 1.0, "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}}})}


def test_procedural_null_criteria_become_keys():
    c = to_case(ROW)
    qs = {q.qid: q for q in parse_questions(c["questions"])}
    assert qs["final"].descriptions == ("£0", "£20") and c["id"] == "proc-arithmetic:train:0" and c["source"] == "procedural"
    assert c["workflow"] == "arithmetic" and c["level"] == 2 and c["state"] == {"opening_balance": 35}


def test_gold_targets_are_smoothed_one_hot():
    t = {x["qid"]: x for x in gold_targets(to_case(ROW))}
    assert t["final"]["target"] == [0.1, 0.9] and t["neg"]["target"] == [0.9, 0.1]
    assert [round(v, 6) for v in t["cnt"]["target"]] == [0.05, 0.9, 0.05]
    assert t["cnt"]["kind"] == "score"


def test_cap_share_limits_one_source():
    rows = [{"source": "procedural"}] * 80 + [{"source": "synth"}] * 20
    out = cap_share(rows, "procedural", 0.35)
    n = sum(r["source"] == "procedural" for r in out)
    assert n <= round(0.35 * len(out)) + 1 and sum(r["source"] == "synth" for r in out) == 20
    assert cap_share(rows[80:], "procedural", 0.35) == rows[80:]   # nothing to cap: unchanged


def test_questions_with_more_than_12_options_are_dropped():
    row = {**ROW, "questions": json.dumps({"where": {"type": "choice", "instructions": "Where?",
                                                     "criteria": {f"room_{i}": None for i in range(13)}},
                                           "neg": {"type": "noul", "instructions": "Does it go negative?"}})}
    c = to_case(row)
    assert list(c["questions"]) == ["neg"]        # Temis answers at most 12 options


def test_cases_file_roundtrip(tmp_path):
    from temis.data.procedural import read_cases, write_cases
    p = tmp_path / "cases.jsonl"
    cases = [to_case(ROW)]
    write_cases(cases, p)
    assert {c["id"]: c for c in read_cases(p)} == {"proc-arithmetic:train:0": cases[0]}
    assert read_cases(tmp_path / "missing.jsonl") == []


def test_fits_budget_counts_state_tokens():
    from temis.data.procedural import fits_budget
    c = to_case(ROW)
    assert fits_budget(c, count=lambda text: len(text), max_tokens=100)
    assert not fits_budget(c, count=lambda text: len(text), max_tokens=10)
