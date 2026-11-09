import torch

from temis.model import align_to_options, consistency_loss


def test_align_to_options_inverts_shuffle():
    p = torch.tensor([[0.2, 0.5, 0.3, 0.0]])
    assert torch.allclose(align_to_options(p, [[2, 0, 1]], [3]), torch.tensor([[0.5, 0.3, 0.2, 0.0]]))


def test_consistency_loss_zero_when_same_distribution_in_option_order():
    la = torch.tensor([[1.0, 2.0, 3.0, -1e4]])
    lb = torch.tensor([[3.0, 1.0, 2.0, -1e4]])
    m = torch.tensor([[True, True, True, False]])
    assert consistency_loss(la, lb, m, m, [[0, 1, 2]], [[2, 0, 1]], [3]).item() < 1e-6
    assert consistency_loss(la, la, m, m, [[0, 1, 2]], [[2, 0, 1]], [3]).item() > 0.1


def test_consistency_loss_score_same_order():
    la = torch.tensor([[1.0, 2.0, 3.0]])
    lb = torch.tensor([[1.1, 2.0, 2.9]])
    m = torch.ones(1, 3, dtype=torch.bool)
    assert 0 < consistency_loss(la, lb, m, m, [[0, 1, 2]], [[0, 1, 2]], [3]).item() < 0.05


def test_export_clamps_inference_long_budget():
    from scripts.export_onnx import inference_long_max_len
    assert inference_long_max_len({"long_max_len": 1536}) == 1024
    assert inference_long_max_len({"long_max_len": 768}) == 768
    assert inference_long_max_len({}) == 1024


def test_batches_split_long_examples_by_token_length():
    import random
    from scripts.train import batches
    short = [{"state": "s" * 300, "ntok": 100} for _ in range(8)]
    long = [{"state": "s" * 300, "ntok": 900} for _ in range(4)]   # short in characters, long in tokens
    for b in batches(short + long, 4, random.Random(0), long_bs=2, max_len=640):
        if any(e["ntok"] > 640 for e in b):
            assert len(b) <= 2


def test_encode_twins_share_context_and_noul_descriptions():
    import random
    from scripts.train import encode_twins
    from temis.encoding import Encoder
    from temis.schema import parse_questions
    from tests.test_core import Q, tok_path
    enc = Encoder(tok_path(), 256)
    q = [x for x in parse_questions(Q) if x.kind == "noul"][0]
    exs = [{"q": q, "state": f"case {i} " * 20, "target": [0.3, 0.7], "kind": "noul"} for i in range(30)]
    a, b = encode_twins(enc, exs, random.Random(1), ["distractor " * 30], 256, None)
    with_ctx = [(enc.ctx in x["ids"]) for x in a]
    assert with_ctx == [(enc.ctx in x["ids"]) for x in b] and any(with_ctx)
    assert all(sorted(x["ids"]) == sorted(y["ids"]) for x, y in zip(a, b))   # same tokens, only the order differs


def test_promotion_gate_compares_flip_rate_with_the_active_run():
    from scripts.retrain import promotion
    base = {"kl": 0.14, "test_acc": 0.76, "clio": 0.86, "hardlike": 0.5, "flip": 0.16}
    ok, why = promotion({**base, "flip": 0.12}, base)
    assert ok
    ok, _ = promotion({**base, "flip": 0.22}, base)
    assert not ok
    ok, _ = promotion({**base, "flip": 0.12}, {**base, "flip": None})
    assert ok
    ok, _ = promotion({**base, "hardlike": 0.48}, base)
    assert not ok
