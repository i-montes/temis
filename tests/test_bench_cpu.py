import numpy as np
import pytest

from scripts import bench_cpu as b


def rows(prefix, n, nq=1):
    return [{"id": f"{prefix}{i}", "state": {"m": f"{prefix} {i}"}, "gold": {"g": f"{prefix}{i}"},
             "questions": {f"q{j}": {"type": "noul", "instructions": "x"} for j in range(nq)}} for i in range(n)]


def test_sample_calls_is_reproducible_without_repeats_and_tagged_with_its_suite():
    calls = b.sample_calls(rows("x", 50), "td", n=10, seed=1)
    assert len(calls) == 10 and len({c["id"] for c in calls}) == 10
    assert {c["suite"] for c in calls} == {"td"}
    assert all(set(c) == {"suite", "id", "state", "questions", "gold"} and c["gold"] == {"g": c["id"]} for c in calls)
    assert calls == b.sample_calls(rows("x", 50), "td", n=10, seed=1)
    assert calls != b.sample_calls(rows("x", 50), "td", n=10, seed=2)


def test_sample_calls_takes_every_row_when_asked_for_more_than_there_are():
    assert sorted(c["id"] for c in b.sample_calls(rows("x", 3), "td", n=10, seed=0)) == ["x0", "x1", "x2"]


def test_jevbench_items_become_one_question_calls():
    item = {"id": "j1", "state": {"a": 1}, "question": {"type": "noul", "instructions": "x"}}
    row = b.jevbench_rows([item])[0]
    assert (row["id"], row["state"], row["questions"]) == ("j1", {"a": 1}, {"decision": item["question"]})


def test_summarize_reports_percentiles_per_question_time_throughput_and_padding():
    s = b.summarize(times_ms=[10.0] * 98 + [100.0, 200.0], real_tokens=[25] * 100, padded_tokens=[100] * 100,
                    questions=[2] * 100, graded=[(True, 0.9, "a"), (False, 0.9, "b")] * 100)
    assert s["p50_ms"] == 10.0
    assert s["p99_ms"] == pytest.approx(np.percentile([10.0] * 98 + [100.0, 200.0], 99))
    assert s["mean_ms"] == pytest.approx(12.8)
    assert s["ms_per_question"] == pytest.approx(6.4)
    assert s["tokens_per_s"] == pytest.approx(2500 / 1.28)
    assert s["padding"] == pytest.approx(0.75)
    assert s["calls"] == 100 and s["questions"] == 200
    assert s["accuracy"] == pytest.approx(0.5)
    assert s["ece"] == pytest.approx(0.4)  # says 90 % and is right half of the time


def test_by_suite_summarizes_each_suite_and_all_of_them_together():
    recs = [{"suite": "td", "ms": 10.0, "real": 10, "padded": 10, "questions": 1, "graded": [(True, 1.0, "a")]}] * 3
    recs += [{"suite": "jevbench", "ms": 30.0, "real": 10, "padded": 20, "questions": 1, "graded": [(False, 0.5, "b")]}]
    out = b.by_suite(recs)
    assert out["td"]["p50_ms"] == 10.0 and out["td"]["calls"] == 3
    assert out["jevbench"]["padding"] == pytest.approx(0.5)
    assert out["all"]["calls"] == 4 and out["all"]["mean_ms"] == pytest.approx(15.0)
    assert out["td"]["accuracy"] == 1.0 and out["all"]["accuracy"] == pytest.approx(0.75)


def test_aggregate_takes_the_median_of_independent_processes_per_suite_and_configuration():
    def proc(prec, v, preds=("a", "b")):
        return {"precision": prec, "threads": 2, "cold_ms": 1000 + v, "rss_mb": 300 + v,
                "suites": {"td": {"p50_ms": v, "accuracy": 0.5}, "all": {"p50_ms": v + 1}},
                "preds": {"td": list(preds), "all": list(preds)}}
    agg = b.aggregate([proc("int8", 10.0), proc("int8", 30.0), proc("int8", 20.0), proc("fp32", 5.0, ("a", "c"))])
    rows_ = {(a["suite"], a["precision"]): a for a in agg}
    assert set(rows_) == {("td", "int8"), ("all", "int8"), ("td", "fp32"), ("all", "fp32")}
    assert rows_[("td", "int8")]["p50_ms"] == 20.0 and rows_[("all", "int8")]["p50_ms"] == 21.0
    assert rows_[("td", "int8")]["accuracy"] == 0.5
    assert rows_[("td", "int8")]["processes"] == 3 and rows_[("td", "fp32")]["processes"] == 1
    assert rows_[("td", "int8")]["cold_ms"] == 1020.0 and rows_[("td", "int8")]["rss_mb"] == 320.0
    assert rows_[("td", "int8")]["p50_cv"] == pytest.approx(np.std([10, 30, 20]) / 20)
    assert rows_[("td", "fp32")]["p50_cv"] == 0.0


def test_aggregate_measures_answer_stability_across_processes_and_against_the_reference():
    def proc(prec, preds):
        return {"precision": prec, "threads": 1, "cold_ms": 1, "rss_mb": 1,
                "suites": {"td": {"p50_ms": 1.0}}, "preds": {"td": preds}}
    agg = b.aggregate([proc("fp32", ["a", "b", "c", "d"]), proc("int8", ["a", "b", "c", "x"]),
                       proc("int8", ["a", "b", "y", "x"])], reference=("fp32", 1))
    rows_ = {a["precision"]: a for a in agg}
    assert rows_["fp32"]["repeatable"] == 1.0 and rows_["fp32"]["agree_ref"] == 1.0
    assert rows_["int8"]["repeatable"] == pytest.approx(0.75)  # one answer changed between two int8 processes
    assert rows_["int8"]["agree_ref"] == pytest.approx(0.75)   # first int8 process vs fp32


def test_grade_scores_each_suite_with_its_own_rule():
    from temis.schema import Answer

    def yes(p):
        return Answer(kind="noul", label="true" if p >= 0.5 else "false",
                      probabilities={"false": 1 - p, "true": p}, confidence=max(p, 1 - p))
    choice = Answer(kind="choice", label="go", probabilities={"go": 0.7, "stop": 0.3}, confidence=0.7)
    td = {"suite": "td", "gold": {"q": {"probabilities": {"go": 0.9, "stop": 0.1}}, "n": {"noul": 0.2}}}
    assert b.grade(td, {"q": choice, "n": yes(0.8)}) == [(True, 0.7, "go"), (False, 0.8, "true")]
    clio = {"suite": "cliolike", "gold": {"t": "true", "c": "stop"}}
    assert b.grade(clio, {"t": yes(0.6), "c": choice}) == [(True, 0.6, "true"), (False, 0.7, "go")]
    item = {"question": {"type": "noul"}, "labels": ["yes", "no"], "expected": "no"}
    jev = {"suite": "jevbench", "gold": {"decision": item}}
    ok, conf, pred = b.grade(jev, {"decision": yes(0.3)})[0]
    assert (ok, pred) == (True, "no") and conf == pytest.approx(0.7)
    assert b.grade(clio, {"t": yes(0.6)})[1] == (False, 0.0, None)  # missing answer counts as wrong


def test_jevbench_items_keep_what_grading_needs_as_gold():
    item = {"id": "j1", "state": {"a": 1}, "question": {"type": "noul"}, "labels": ["yes", "no"], "expected": "no"}
    row = b.jevbench_rows([item])[0]
    assert row["gold"]["decision"]["expected"] == "no" and "state" not in row["gold"]["decision"]


def test_cpu_provider_alone_when_asked_for_cpu(monkeypatch):
    import onnxruntime as ort

    from temis import runtime
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert runtime._providers("CPUExecutionProvider") == ["CPUExecutionProvider"]
    assert runtime._providers(None) == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_logits_record_real_and_padded_tokens():
    from huggingface_hub import hf_hub_download

    from temis.encoding import Encoder
    from temis.runtime import Decider
    from temis.schema import parse_questions

    class Session:
        def run(self, _, feed):
            return [np.zeros((feed["input_ids"].shape[0], feed["opt_pos"].shape[1]), np.float32)]

    d = object.__new__(Decider)
    d.encoder = Encoder(hf_hub_download("jhu-clsp/mmBERT-small", "tokenizer.json"), 1024)
    d.long_max_len, d.session = 1024, Session()
    qs = parse_questions({"a": {"type": "noul", "instructions": "x"},
                          "b": {"type": "noul", "instructions": "a much longer instruction with many more words in it"}})
    d.logits(qs, "state")
    lens = [len(d.encoder.encode(q, "state", None, max_len=1024).ids) for q in qs]
    assert d.last_input_tokens == sum(lens)
    assert d.last_padded_tokens == 2 * max(lens) > sum(lens)
