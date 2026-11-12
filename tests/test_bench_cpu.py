import numpy as np
import pytest

from scripts import bench_cpu as b


def rows(prefix, n, nq=1):
    return [{"id": f"{prefix}{i}", "state": {"m": f"{prefix} {i}"},
             "questions": {f"q{j}": {"type": "noul", "instructions": "x"} for j in range(nq)}} for i in range(n)]


def test_load_calls_mixes_short_and_long_in_the_asked_proportion_and_is_reproducible():
    calls = b.load_calls(rows("short", 50), rows("long", 50), n=40, long_frac=0.25, seed=1)
    assert len(calls) == 40
    assert sum(c["kind"] == "long" for c in calls) == 10
    assert {c["kind"] for c in calls[:10]} == {"short", "long"}  # mixed, not all long calls first
    assert all(c["state"]["m"].startswith(c["kind"]) for c in calls)
    assert calls == b.load_calls(rows("short", 50), rows("long", 50), n=40, long_frac=0.25, seed=1)
    assert calls != b.load_calls(rows("short", 50), rows("long", 50), n=40, long_frac=0.25, seed=2)


def test_load_calls_repeats_rows_when_asked_for_more_calls_than_there_are():
    calls = b.load_calls(rows("short", 3), rows("long", 1), n=10, long_frac=0.0, seed=0)
    assert len(calls) == 10 and {c["state"]["m"] for c in calls} == {"short 0", "short 1", "short 2"}


def test_summarize_reports_percentiles_throughput_and_padding():
    s = b.summarize(times_ms=[10.0] * 98 + [100.0, 200.0], real_tokens=[25] * 100, padded_tokens=[100] * 100)
    assert s["p50_ms"] == 10.0
    assert s["p99_ms"] == pytest.approx(np.percentile([10.0] * 98 + [100.0, 200.0], 99))
    assert s["mean_ms"] == pytest.approx(12.8)
    assert s["tokens_per_s"] == pytest.approx(2500 / 1.28)
    assert s["padding"] == pytest.approx(0.75)
    assert s["calls"] == 100


def test_aggregate_takes_the_median_of_independent_processes_per_configuration():
    runs = [{"precision": "int8", "threads": 2, "p50_ms": v, "rss_mb": 300 + v} for v in (10.0, 30.0, 20.0)]
    runs += [{"precision": "fp32", "threads": 2, "p50_ms": 5.0, "rss_mb": 600.0}]
    agg = b.aggregate(runs)
    assert [(a["precision"], a["threads"], a["processes"]) for a in agg] == [("int8", 2, 3), ("fp32", 2, 1)]
    assert agg[0]["p50_ms"] == 20.0 and agg[0]["rss_mb"] == 320.0


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
