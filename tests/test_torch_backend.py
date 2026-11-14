"""The evaluation backend on PyTorch (GPU when there is one): same answers as the ONNX fp32 model,
and its results never overwrite the ONNX ones the promotion gate reads."""

import numpy as np

from temis.schema import parse_questions
from tests.test_core import Q


def test_torch_backend_gives_the_onnx_fp32_logits(tmp_path):
    from temis.runtime import Decider
    from tests.tiny import tiny_run
    d = tiny_run(tmp_path)
    onnx = Decider(d, int8=False, backend="onnx")
    torch_ = Decider(d, backend="torch")
    assert torch_.backend == "torch" and onnx.backend == "onnx"
    qs = parse_questions(Q)
    for state in ("short state", "a longer state " * 30):
        a, b = onnx.logits(qs, state), torch_.logits(qs, state)
        for x, y in zip(a, b):
            assert np.abs(x - y).max() < 1e-3


def test_backend_comes_from_the_environment_unless_given(tmp_path, monkeypatch):
    from temis.runtime import Decider
    from tests.tiny import tiny_run
    d = tiny_run(tmp_path)
    monkeypatch.setenv("TEMIS_BACKEND", "torch")
    assert Decider(d, int8=False).backend == "torch"
    assert Decider(d, int8=False, backend="onnx").backend == "onnx"  # bench_cpu always measures ONNX


def test_results_of_the_torch_backend_go_to_their_own_files(tmp_path, monkeypatch):
    from temis.runtime import eval_path
    model = tmp_path / "run" / "onnx"
    monkeypatch.delenv("TEMIS_BACKEND", raising=False)
    assert eval_path(model, "eval_clio") == tmp_path / "run" / "eval_clio.json"
    monkeypatch.setenv("TEMIS_BACKEND", "torch")
    assert eval_path(model, "eval_clio") == tmp_path / "run" / "eval_clio.torch.json"
