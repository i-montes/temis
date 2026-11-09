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
    from temis.data.balance import NO_OPINION, agree
    assert agree([0.2, 0.7, 0.1], ["a", "b", "c"], "b") and not agree([0.2, 0.7, 0.1], ["a", "b", "c"], "a")
    assert agree([0.2, 0.7, 0.1], ["a", "b", "c"], NO_OPINION)   # nobody asked: the case goes in on Jev alone
    assert not agree([0.2, 0.7, 0.1], ["a", "b", "c"], None)     # explicit null: "no option is justified" -> held back


def test_import_rejects_lines_that_copy_public_benchmark_text():
    from scripts.import_cases import shingles
    public = shingles(["Select the primary requested action. A mention without a request does not establish intent."])
    q = {"intent": {"type": "choice", "instructions": "Select the primary requested action. A mention without a request does not establish intent.",
                    "criteria": {"status": "Learn delivery progress", "other": "Nothing is requested", "cancel": "End it"}}}
    assert parse_line(line(questions=q, surface_answer={"intent": "status"}), "gpt", leaks=public) is None
    assert parse_line(line(questions=q, surface_answer={"intent": "status"}), "gpt") is not None  # no public items at hand: no check
    assert parse_line(line(), "gpt", leaks=public) is not None          # our own wording passes


def test_held_back_roundtrip(tmp_path):
    from temis.data.balance import read_held_back, write_held_back
    p = tmp_path / "held.jsonl"
    write_held_back([("ext-1", "q1"), ("ext-2", "q2")], p)
    assert read_held_back(p) == {("ext-1", "q1"), ("ext-2", "q2")}
    assert read_held_back(tmp_path / "missing.jsonl") == set()


def test_hard_target_uses_author_gold_when_a_teacher_agrees():
    from temis.data.balance import NO_OPINION, hard_target
    jev = [0.2, 0.7, 0.1]; opts = ["a", "b", "c"]; jev_t = [0.25, 0.6, 0.15]
    assert hard_target(jev, opts, jev_t, surface=None, second=NO_OPINION, author="b") == [0.05, 0.9, 0.05]   # Jev agrees
    assert hard_target(jev, opts, jev_t, surface=None, second="a", author="a") == [0.9, 0.05, 0.05]          # second agrees
    assert hard_target(jev, opts, jev_t, surface=None, second="c", author="a") is None                       # nobody agrees: held


def test_hard_target_without_author_gold_holds_the_surface_answer():
    from temis.data.balance import NO_OPINION, hard_target
    jev = [0.2, 0.7, 0.1]; opts = ["a", "b", "c"]; jev_t = [0.25, 0.6, 0.15]
    assert hard_target(jev, opts, jev_t, surface="b", second=NO_OPINION, author=None) is None   # Jev fell for the trap
    assert hard_target(jev, opts, jev_t, surface="a", second=NO_OPINION, author=None) == jev_t  # Jev alone, not the trap
    assert hard_target(jev, opts, jev_t, surface="a", second="b", author=None) == jev_t         # second agrees with Jev
    assert hard_target(jev, opts, jev_t, surface="a", second="c", author=None) is None          # teachers disagree


def test_import_keeps_author_expected_answers():
    c = parse_line(line(expected={"ok": "false"}), "gpt")
    assert c["author_gold"] == {"ok": "false"}
    assert parse_line(line(expected={"ok": "maybe"}), "gpt") is None
    assert parse_line(line(expected={"ok": "true"}), "gpt") is None   # the gold cannot be the surface answer


def test_second_opinion_import_as_gold_writes_author_file(tmp_path, monkeypatch):
    import json
    from scripts import second_opinion as so
    monkeypatch.setattr(so, "SECOND_OPINION", tmp_path / "second.jsonl")
    monkeypatch.setattr(so, "AUTHOR_GOLD", tmp_path / "gold.jsonl")
    case = parse_line(line(), "gpt")
    monkeypatch.setattr(so, "hard_cases", lambda families=None: [case])
    ans = tmp_path / "answers.jsonl"
    ans.write_text(json.dumps({"id": case["id"], "answers": {"ok": "false"}}) + "\n")
    assert so.import_answers(ans, "gpt", as_gold=True) == (1, 0)
    assert so.load_author_gold(tmp_path / "gold.jsonl") == {(case["id"], "ok"): "false"}
    assert not (tmp_path / "second.jsonl").exists()
