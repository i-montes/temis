from temis.jevbench import probs_for_labels, score_item, summarize, to_request
from temis.schema import Answer


def item(kind, labels, expected, **extra):
    q = {"choice": {"type": "choice", "instructions": "Pick.", "criteria": {l: l for l in labels}},
         "noul": {"type": "noul", "instructions": "Is it?", "criteria": {"true": "yes", "false": "no"}},
         "score": {"type": "score", "instructions": "Rate.", "criteria": [f"level {i}" for i in labels]}}[kind]
    return {"id": f"hard-x-{kind}-01", "family": "x", "state": "s", "labels": labels, "expected": expected,
            "question": q, "provenance": extra, "tier": "hard"}


def test_to_request_wraps_question_as_decision():
    st, qs = to_request(item("choice", ["a", "b"], "a"))
    assert st == "s" and list(qs) == ["decision"] and qs["decision"]["type"] == "choice"


def test_noul_band_edges():
    it = item("noul", ["no", "yes"], "yes")
    assert score_item(it, {"yes": 0.80, "no": 0.20})["v15_correct"] is True
    r = score_item(it, {"yes": 0.7999, "no": 0.2001})
    assert r["correct"] is True and r["v15_correct"] is False


def test_score_uses_argmax_for_v12_and_nmae_for_v15():
    it = item("score", ["0", "1", "2", "3"], 1)
    r = score_item(it, {"0": 0.1, "1": 0.6, "2": 0.2, "3": 0.1})
    assert r["correct"] and abs(r["nmae"] - abs(1.3 - 1) / 3) < 1e-9 and abs(r["nmae_chance"] - 0.5 / 3) < 1e-9


def test_probs_for_labels_maps_noul_and_score():
    a = Answer("noul", "true", {"false": 0.3, "true": 0.7}, 0.7)
    assert probs_for_labels(item("noul", ["no", "yes"], "yes"), a) == {"no": 0.3, "yes": 0.7}
    a = Answer("score", "2", {"0": 0.1, "1": 0.2, "2": 0.7}, 0.7, expected=1.6)
    assert probs_for_labels(item("score", ["0", "1", "2"], 2), a) == {"0": 0.1, "1": 0.2, "2": 0.7}


def test_summarize_tier_weights_and_intelligence():
    rows = [{**score_item(item("choice", ["a", "b", "c", "d"], "a"), {"a": 1.0, "b": 0, "c": 0, "d": 0}),
             "tier": "hard", "family": "x", "type": "choice", "id": "1", "ms": 10.0}]
    s = summarize(rows)
    assert s["accuracy"] == 1.0 and s["by_tier"]["hard"]["accuracy"] == 1.0
    assert s["v15"]["cc_by_tier"]["hard"]["choice"] == 100.0 and s["v15"]["intelligence"] == 100.0
