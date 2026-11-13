"""Banded local attention: the sliding-window layers read only the keys inside their window instead of
the whole T x T matrix. It must give the eager model's answers at every length, in torch and in ONNX."""

import numpy as np
import torch


def pair(tmp_path, local_attention=16):
    """The same weights twice: eager (the reference) and banded."""
    from temis.model import Decisor
    from tests.tiny import tiny_base
    base = tiny_base(tmp_path, layers=4, local_attention=local_attention)
    torch.manual_seed(1)
    eager = Decisor(base, attn="eager").eval()
    banded = Decisor(base, attn="banded").eval()
    banded.load_state_dict(eager.state_dict())
    return eager, banded


def inputs(B, T, pad_last=0, K=3):
    g = torch.Generator().manual_seed(T * 7 + B)
    ids = torch.randint(10, 1000, (B, T), generator=g)
    am = torch.ones(B, T, dtype=torch.long)
    if pad_last:
        am[-1, T - pad_last:] = 0
    pos = torch.tensor([[1, 2, 3][:K]] * B)
    return ids, am, pos, torch.ones(B, K, dtype=torch.bool)


def test_banded_gives_the_eager_logits_at_every_length(tmp_path):
    eager, banded = pair(tmp_path)  # window 16: half 8, so blocks of 8 tokens
    with torch.no_grad():
        for T in (4, 8, 9, 16, 23, 40, 67):
            for B, pad in ((1, 0), (3, min(T - 4, 30) if T > 4 else 0)):
                x = inputs(B, T, pad)
                assert torch.allclose(eager(*x), banded(*x), atol=1e-5), (T, B, pad)


def test_banded_states_match_on_every_real_token(tmp_path):
    eager, banded = pair(tmp_path)
    ids, am, _, _ = inputs(2, 45, pad_last=20)
    with torch.no_grad():
        a = eager.encoder(input_ids=ids, attention_mask=am).last_hidden_state
        b = banded.encode(ids, am)
    real = am.bool()
    assert torch.allclose(a[real], b[real], atol=1e-5)


def test_banded_onnx_export_runs_at_lengths_it_was_not_traced_with(tmp_path):
    import onnxruntime as ort
    from scripts.export_onnx import Wrapped
    eager, banded = pair(tmp_path)
    x = inputs(2, 24, pad_last=5)
    path = tmp_path / "banded.onnx"
    torch.onnx.export(Wrapped(banded).eval(), (x[0], x[1], x[2], x[3].long()), str(path),
                      input_names=["input_ids", "attention_mask", "opt_pos", "opt_mask"], output_names=["logits"],
                      opset_version=18, dynamo=False,
                      dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"},
                                    "opt_pos": {0: "b", 1: "k"}, "opt_mask": {0: "b", 1: "k"}, "logits": {0: "b", 1: "k"}})
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for B, T, pad in ((1, 5, 0), (3, 37, 11), (2, 100, 60), (1, 64, 0)):
        ids, am, pos, om = inputs(B, T, pad)
        got = s.run(None, {"input_ids": ids.numpy(), "attention_mask": am.numpy(), "opt_pos": pos.numpy(),
                           "opt_mask": om.long().numpy()})[0]
        with torch.no_grad():
            ref = eager(ids, am, pos, om).numpy()
        assert np.abs(got - ref).max() < 1e-4, (B, T, pad)

