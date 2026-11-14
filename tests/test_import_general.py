"""General decision data from public sets (scripts/import_general.py): only items made for training,
never items built to imitate a benchmark or copying one of our evaluation suites."""

import random

import pytest

from scripts.import_general import sn92_row, soft_target, sample_by_category, tdes_rows


def item(**kw):
    base = {"id": "x1", "category": "rules_policy_law", "family": "legal.requirement", "lang": "en",
            "component": None, "benchmark": None, "qtype": "choice", "gold_basis": "programmatic",
            "state": {"policy": "Refunds need a receipt. The customer has a receipt and asks for a refund."},
            "question": {"type": "choice", "instructions": "What should support do?",
                         "criteria": {"refund": "Refund the order", "deny": "Deny the refund", "ask": "Ask for more"}},
            "expected": "refund", "ancestry": {"root": "r-7"}}
    return {**base, **kw}


def test_soft_target_puts_most_of_the_mass_on_the_gold_option():
    assert soft_target(1, 3, eps=0.1) == pytest.approx([0.05, 0.9, 0.05])
    assert soft_target(0, 2, eps=0.2) == pytest.approx([0.8, 0.2])
    assert sum(soft_target(4, 7)) == pytest.approx(1.0)


def test_a_training_item_becomes_one_row_with_its_case():
    r = sn92_row(item(), leaks=set())
    assert r["source"] == "sn92" and r["kind"] == "choice" and r["teachers"] == ["gold"]
    assert r["target"] == pytest.approx([0.9, 0.05, 0.05])            # refund is the first option
    assert r["case"]["state"] == item()["state"] and list(r["case"]["questions"]) == [r["qid"]]
    assert r["schema_id"] == "sn92-r-7"                                # a counterfactual group stays on one side of val


def test_score_and_yes_no_items_map_their_expected_answer_to_the_right_option():
    score = sn92_row(item(qtype="score", expected=2, question={"type": "score", "instructions": "How severe?",
                                                              "criteria": ["none", "low", "high", "critical"]}), set())
    assert score["target"].index(max(score["target"])) == 2
    noul = sn92_row(item(qtype="noul", expected="false", question={"type": "noul", "instructions": "It is allowed."}), set())
    assert noul["target"] == pytest.approx([0.9, 0.1])                 # options are (false, true)


@pytest.mark.parametrize("bad", [
    {"component": "jev"}, {"component": "hfdi"}, {"component": "aadi"},   # built to imitate a benchmark
    {"benchmark": "jev:ordinal"},
    {"lang": "de"},                                                      # Temis reads Spanish and English
    {"expected": "maybe"},                                               # not one of the options
    {"question": {"type": "choice", "instructions": "Pick", "criteria": {str(i): f"o{i}" for i in range(13)}},
     "expected": "0"},                                                   # more options than the head holds
])
def test_items_that_must_not_enter_training_are_dropped(bad):
    assert sn92_row(item(**bad), set()) is None


def test_an_item_sharing_a_phrase_with_an_evaluation_suite_is_dropped():
    from scripts.import_cases import shingles
    leaks = shingles(["the customer has a receipt and asks for a refund"])
    assert sn92_row(item(), leaks) is None
    assert sn92_row(item(), shingles(["nothing in common with this one at all"])) is not None


def test_sampling_caps_each_category_and_keeps_every_spanish_item():
    rows = [{"category": c, "lang": "en", "id": f"{c}{i}"} for c in "ab" for i in range(50)]
    rows += [{"category": "a", "lang": "es", "id": f"es{i}"} for i in range(5)]
    got = sample_by_category(rows, per_category=20, rng=random.Random(0))
    count = {c: sum(r["category"] == c for r in got) for c in "ab"}
    assert count == {"a": 20, "b": 20}
    assert {r["id"] for r in got if r["lang"] == "es"} == {f"es{i}" for i in range(5)}
    assert got == sample_by_category(rows, per_category=20, rng=random.Random(0))
    b = {r["id"] for r in got if r["category"] == "b"}
    assert b != {f"b{i}" for i in range(20)}                           # a random draw, not the first rows


def test_spanish_typed_decisions_reuse_the_english_targets_and_their_val_side():
    from scripts.train import is_val
    case = {"id": "tr_wf_000001", "state": {"tarea": "Revise la cola"},
            "questions": {"action": {"type": "choice", "instructions": "¿Qué hacer?",
                                     "criteria": {"go": "Siga", "stop": "Pare"}},
                          "urgent": {"type": "noul", "instructions": "Es urgente."}}}
    en = {("td-tr_wf_000001", "action"): [0.3, 0.7], ("td-tr_wf_000001", "urgent"): [0.6, 0.4]}
    rows = tdes_rows([case], en)
    assert {r["qid"]: r["target"] for r in rows} == {"action": [0.3, 0.7], "urgent": [0.6, 0.4]}
    assert all(r["schema_id"] == "td-tr_wf_000001" and r["source"] == "td-train-es" for r in rows)
    assert is_val(rows[0]["schema_id"]) == is_val("td-tr_wf_000001")   # same side of val as its English twin


def test_spanish_typed_decisions_skip_test_cases_and_reordered_options():
    test_case = {"id": "te_wf_000001", "state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}}
    assert tdes_rows([test_case], {("td-te_wf_000001", "q"): [0.5, 0.5]}) == []
    moved = {"id": "tr_wf_2", "state": "s", "questions": {"a": {"type": "choice", "instructions": "x",
                                                               "criteria": {"stop": "Pare", "go": "Siga"}}}}
    assert tdes_rows([moved], {("td-tr_wf_2", "a"): [0.3, 0.7]}, en_options={("td-tr_wf_2", "a"): ("go", "stop")}) == []
