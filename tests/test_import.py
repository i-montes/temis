import json

from scripts.import_cases import parse_line


def line(**kw):
    base = {"family": "trap", "language": "en", "state": "x" * 200,
            "questions": {"ok": {"type": "noul", "instructions": "Is this return eligible for the instant refund?",
                                 "criteria": {"true": "Eligible.", "false": "Not eligible."}}},
            "rationale": "The seal is split, so the item is not unopened under the stated definition.",
            "surface_answer": {"ok": "true"}}
    d = {**base, **kw}
    return json.dumps({k: v for k, v in d.items() if v is not None})


def test_import_keeps_question_form_noul():
    c = parse_line(line(), "gpt")
    assert c and c["workflow"] == "trap" and c["domain"] == "hard:trap" and "ok" in c["questions"]
    assert c["rationale"].startswith("The seal") and c["surface_answer"] == {"ok": "true"}


def test_import_hard_family_requires_rationale_and_valid_surface():
    assert parse_line(line(rationale=""), "gpt") is None
    assert parse_line(line(surface_answer={"ok": "maybe"}), "gpt") is None


def test_import_agent_families_unchanged():
    c = parse_line(line(family="memory_op", rationale=None, surface_answer=None), "gpt")
    assert c["domain"] == "agent:memory_op" and "rationale" not in c


def test_import_keeps_contrast_pair_reference():
    assert parse_line(line(contrast_of="refund_seal_01"), "gpt")["contrast_of"] == "refund_seal_01"


def test_agree_with_second_opinion():
    from temis.data.balance import agree
    assert agree([0.2, 0.7, 0.1], ["a", "b", "c"], "b") and not agree([0.2, 0.7, 0.1], ["a", "b", "c"], "a")
    assert agree([0.2, 0.7, 0.1], ["a", "b", "c"], None)  # no second opinion: the case goes in
