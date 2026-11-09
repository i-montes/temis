import random

import pytest

from temis.encoding import CTX, EOS, OPT, Encoder
from temis.schema import SchemaError, normalize_probs, parse_questions, render_state
from temis.store import Store, signature

Q = {
    "action": {"type": "choice", "instructions": "What now?", "criteria": {"go": "Proceed.", "stop": "Halt."}},
    "risk": {"type": "score", "instructions": "How risky?", "criteria": ["none", "low", "high"]},
    "urgent": {"type": "noul", "instructions": "It is urgent."},
}


def tok_path():
    from huggingface_hub import hf_hub_download
    return hf_hub_download("jhu-clsp/mmBERT-small", "tokenizer.json")


def test_parse_all_types():
    qs = {q.qid: q for q in parse_questions(Q)}
    assert qs["action"].options == ("go", "stop")
    assert qs["risk"].options == ("0", "1", "2") and qs["risk"].descriptions[2] == "high"
    assert qs["urgent"].options == ("false", "true")


@pytest.mark.parametrize("bad", [
    {"x": {"type": "scroe", "instructions": "a", "criteria": ["a", "b"]}},
    {"x": {"type": "choice", "instructions": "a", "criteria": {"only": "one"}}},
    {"x": {"type": "score", "instructions": "a", "criteria": ["one"]}},
    {"x": {"type": "noul", "instructions": ""}},
    {},
])
def test_invalid_schemas_are_rejected(bad):
    with pytest.raises(SchemaError):
        parse_questions(bad)


def test_noul_accepts_yes_no_and_renormalizes():
    q = parse_questions(Q)[2]
    assert normalize_probs(q, {"yes": 0.6, "no": 0.2}) == pytest.approx([0.25, 0.75])


def test_render_state_flattens_json():
    s = render_state({"trace_summary": {"tool_errors": 1}, "tags": ["a", "b"]})
    assert "trace summary · tool errors: 1" in s and "tags: a; b" in s


def test_encoding_puts_markers_before_state_and_never_truncates_options():
    enc = Encoder(tok_path(), max_len=64)
    q = parse_questions(Q)[0]
    e = enc.encode(q, "word " * 500, context="ctx " * 100)
    assert len(e.ids) <= 64 and e.ids[-1] == EOS
    assert [e.ids[p] for p in e.opt_pos] == [OPT, OPT]
    assert max(e.opt_pos) < e.ids.index(EOS)


def test_encoding_shuffle_keeps_order_mapping_and_score_order():
    enc = Encoder(tok_path())
    qs = parse_questions(Q)
    e = enc.encode(qs[0], "s", rng=random.Random(3))
    assert sorted(e.order) == [0, 1]
    s = enc.encode(qs[1], "s", rng=random.Random(3))
    assert s.order == [0, 1, 2]
    c = enc.encode(qs[0], "s", context="c")
    assert CTX in c.ids


def test_store_precedents_match_same_signature_only(tmp_path):
    st = Store(tmp_path / "d.db")
    qs = parse_questions(Q)
    did = st.add_decision(origin="ui", state={"a": 1}, questions=Q, answers={}, model_version="t", uncertainty=0.9)
    st.label(did, {"action": "stop", "risk": "2", "urgent": "true"}, qs, "agent deleted the production database")
    hits = st.precedents(qs[0], "the agent deleted a database")
    assert hits and hits[0]["label"] == "stop"
    other = parse_questions({"a": {"type": "choice", "instructions": "Other?", "criteria": {"go": "Proceed.", "stop": "Halt."}}})[0]
    assert signature(other) != signature(qs[0])
    assert st.precedents(other, "the agent deleted a database") == []
    with pytest.raises(ValueError):
        st.label(did, {"action": "nope"}, qs, "x")
    assert st.stats()["labeled"] == 1


def test_answer_typesafe_shape():
    from temis.runtime import make_answer
    qs = parse_questions(Q)
    a = make_answer(qs[2], [0.3, 0.7]).to_typesafe()
    assert a["type"] == "noul" and a["noul"] == pytest.approx(0.7)
    s = make_answer(qs[1], [0.2, 0.3, 0.5]).to_typesafe()
    assert s["score"] == pytest.approx(1.3) and s["confidence"] == pytest.approx(0.5)


def test_agreement_separates_picked_from_accepted(tmp_path):
    from temis.store import agreement
    st = Store(tmp_path / "d.db")
    qs = parse_questions(Q)
    answers = {"action": {"type": "choice", "probabilities": {"go": 0.8, "stop": 0.2}},
               "risk": {"type": "score", "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
               "urgent": {"type": "noul", "probabilities": {"false": 0.4, "true": 0.6}}}
    did = st.add_decision(origin="ui", state={"a": 1}, questions=Q, answers=answers, model_version="t",
                          uncertainty=0.5, teacher={"jev": {"action": [0.1, 0.9]}})
    st.label(did, {"action": "stop", "risk": "2", "urgent": "true"}, qs, "x",
             how={"action": "pick", "risk": "accept", "urgent": "pick"})
    a = agreement(st)
    assert a["all"]["n"] == 3 and a["all"]["model"] == pytest.approx(2 / 3, abs=1e-3)
    assert a["by_how"]["pick"]["n"] == 2 and a["by_how"]["pick"]["model"] == 0.5
    assert a["all"]["jev"] == 1.0
    assert a["disagreements"][0]["model"] == "go" and a["disagreements"][0]["user"] == "stop"


def test_suggestions_are_advisory_and_reviewed_cases_come_first(tmp_path):
    st = Store(tmp_path / "d.db")
    qs = parse_questions(Q)
    a = st.add_decision(origin="t", state={"a": 1}, questions=Q, answers={}, model_version="t", uncertainty=0.9)
    b = st.add_decision(origin="t", state={"b": 1}, questions=Q, answers={}, model_version="t", uncertainty=0.2)
    st.suggest(b, "action", "stop", "por qué")
    st.suggest(b, "risk", None, "ambigua")
    assert st.queue(1)[0]["id"] == b                      # reviewed first, even with less doubt
    assert st.get(b)["suggestions"]["risk"]["label"] is None
    assert st.stats()["user_labels"] == 0                 # a suggestion is never a label
    assert st.precedents(qs[0], "b") == []                # nor a precedent


@pytest.mark.parametrize("raw,expected", [
    ("Determine if the command performs an irreversible action.", "The command performs an irreversible action."),
    ("Verifique si la nota menciona nombres propios.", "La nota menciona nombres propios."),
    ("Assess whether forgetting this memory breaks a safety check", "Forgetting this memory breaks a safety check."),
    ("The note should carry the tag 'music'.", "The note should carry the tag 'music'."),
    ("¿La propuesta incluye un plan de mitigación?", "¿La propuesta incluye un plan de mitigación?"),
    ("Evaluate the emotional weight of the content.", None),
    ("Verificación de recursos disponibles.", None),
    ("Indique si la fecha es fija o si existen ventanas de flexibilidad.", None),
    ("¿Qué tipo de memoria es?", None),
    ("Si la alerta indica una caída total, elige A.", None),
])
def test_noul_quality_filter(raw, expected):
    from temis.data.quality import normalize_noul
    assert normalize_noul(raw) == expected


def test_clean_questions_keeps_other_types_and_drops_bad_noul():
    from temis.data.quality import clean_questions, state_ok
    qs = {"a": {"type": "noul", "instructions": "Evaluate the urgency."},
          "b": {"type": "noul", "instructions": "Determina si hay que responder hoy."},
          "c": {"type": "choice", "instructions": "Elige.", "criteria": {"x": "1", "y": "2"}}}
    out, rewritten, dropped = clean_questions(qs)
    assert set(out) == {"b", "c"} and rewritten == 1 and dropped == 1
    assert out["b"]["instructions"] == "Hay que responder hoy."
    assert not state_ok("Layer 30 failed, print halted automatically.")


def test_user_test_holdout_is_never_exported(tmp_path):
    import json as _json
    from temis.userlabels import export_user_targets, is_user_test
    st = Store(tmp_path / "d.db")
    qs = parse_questions(Q)
    ids = [st.add_decision(origin="t", state={"i": i}, questions=Q, answers={}, model_version="t", uncertainty=0.5)
           for i in range(30)]
    for did in ids:
        st.label(did, {"action": "go", "risk": "1", "urgent": "false"}, qs, "x")
    out = tmp_path / "u.jsonl"
    export_user_targets(st, out, repeat=1)
    exported = {_json.loads(l)["id"] for l in out.open()}
    held = [d for d in ids if is_user_test(d)]
    assert held and all(f"user-{d}" not in exported for d in held)
    assert len(exported) == len(ids) - len(held)


def test_weak_families_get_queue_priority(tmp_path):
    from temis.data.families import queue_bonus
    st = Store(tmp_path / "d.db")
    a = st.add_decision(origin="t", state={"a": 1}, questions=Q, answers={}, model_version="t", uncertainty=0.6, workflow="note_tags")
    b = st.add_decision(origin="t", state={"b": 1}, questions=Q, answers={}, model_version="t", uncertainty=0.5, workflow="memory_op")
    assert queue_bonus("memory_op") > queue_bonus("evaluar_obsolescencia_memoria") > queue_bonus("note_tags") == 0
    assert st.queue(1)[0]["id"] == b


def test_temis_versions(tmp_path, monkeypatch):
    import temis.naming as N
    monkeypatch.setattr(N, "REGISTRY", tmp_path / "registry.json")
    base = "jhu-clsp/mmBERT-base"
    assert N.version_of("v1108-1157") == "1.0.6"                       # backfilled
    assert N.assign("a", base, fine_tune=True) == "1.0.7"               # short fine-tune -> patch
    assert N.assign("b", base, fine_tune=False) == "1.1.0"              # fresh run -> minor
    assert N.assign("c", base, fine_tune=True) == "1.1.1"
    assert N.assign("b", base, fine_tune=False) == "1.1.0"              # idempotent
    assert N.assign("d", "some/new-base", fine_tune=False) == "2.0.0"   # new base -> major
    assert N.display("c") == "Temis 1.1.1" and N.model_id("c") == "temis-1.1.1"


def test_memory_op_rebalance():
    from temis.data.balance import copies
    opts = ["nueva", "repetida", "sospechosa", "corrige_m1"]
    other = {"id": "syn-x", "workflow": "tools"}
    assert copies(other, opts, [0, 0, 0, 1]) == 1
    assert copies({"id": "ext-1", "workflow": "memory_op"}, opts, [0, 0, 0, 1]) == 2
    assert copies({"id": "syn-1", "workflow": "memory_op"}, opts, [0, 1, 0, 0]) == 1
    kept = sum(copies({"id": f"syn-{i}", "workflow": "memory_op"}, opts, [0, 0, 0, 1]) for i in range(1000))
    assert 230 < kept < 370  # about 30 % of corrections from Qwen stay


def test_review_sheet_answers():
    import pytest
    from temis.schema import parse_questions
    from scripts.review_sheet import parse_answer
    qs = {q.qid: q for q in parse_questions({
        "op": {"type": "choice", "instructions": "x", "criteria": {"nueva": "a", "repetida": "b", "corrige_m1": "c"}},
        "ok": {"type": "noul", "instructions": "Es seguro.", "criteria": {"false": "no", "true": "sí"}}})}
    assert parse_answer("2", qs["op"]) == "repetida"
    assert parse_answer("corrige_m1", qs["op"]) == "corrige_m1"
    assert parse_answer("3=corrige_m1: c", qs["op"]) == "corrige_m1"
    assert parse_answer("Sí", qs["ok"]) == "true" and parse_answer("no", qs["ok"]) == "false"
    assert parse_answer("", qs["op"]) is None and parse_answer("?", qs["ok"]) is None
    with pytest.raises(ValueError):
        parse_answer("7", qs["op"])


def test_reviewer_targets(tmp_path):
    from temis.schema import parse_questions, render_state
    from temis.store import Store
    from temis.userlabels import export_reviewer_targets, is_user_test
    st = Store(tmp_path / "t.db")
    qs = {"op": {"type": "choice", "instructions": "x", "criteria": {"nueva": "a", "repetida": "b", "corrige_m1": "c"}}}
    ids = []
    for i in range(40):
        did = st.add_decision(state={"t": f"caso {i}"}, questions=qs, answers={}, model_version="t",
                              uncertainty=0.5, workflow="memory_op", origin="test")
        ids.append(did)
        st.suggest(did, "op", "repetida", "r", by="claude-hoja")
        st.suggest(did, "op", "nueva", "r", by="claude")
    st.label(ids[0], {"op": "corrige_m1"}, parse_questions(qs), "x")  # Iván's label wins
    out = tmp_path / "r.jsonl"
    n = export_reviewer_targets(st, out)
    import json
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert n == len(rows) and 0 < n < 40
    assert all(r["id"] != ids[0] and not is_user_test(r["id"]) and r["source"] == "reviewer" for r in rows)
    assert abs(sum(rows[0]["target"]) - 1) < 1e-9
    assert abs(rows[0]["target"][0] - rows[0]["target"][1]) < 1e-9  # two reviewers split: half and half
