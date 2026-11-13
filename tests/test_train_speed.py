"""Training-loop knobs: the seed and gradient checkpointing only where memory needs it."""

import torch


def test_seed_defaults_to_zero_and_can_be_changed():
    from scripts.train import parser
    assert parser().parse_args([]).seed == 0
    assert parser().parse_args(["--seed", "7"]).seed == 7


def test_checkpointing_only_for_batches_over_the_token_budget():
    from scripts.train import needs_checkpointing
    assert not needs_checkpointing(16, 256, budget=5120)  # 4 096 tokens: fits without it
    assert needs_checkpointing(16, 448, budget=5120)      # 7 168 tokens
    assert not needs_checkpointing(16, 320, budget=5120)  # exactly at the budget
    assert needs_checkpointing(2, 64, budget=0)           # 0 = always (the old behaviour)


def test_checkpointing_can_be_switched_per_batch_and_both_ways_train_the_layers(tmp_path):
    from scripts.train import set_checkpointing
    from temis.model import Decisor, decision_loss
    from tests.tiny import tiny_base
    m = Decisor(tiny_base(tmp_path), attn="eager")
    m.encoder.get_input_embeddings().weight.requires_grad_(False)  # frozen, as in train.py
    m.train()
    ids = torch.randint(10, 1000, (2, 24))
    am = torch.ones(2, 24, dtype=torch.long)
    pos = torch.tensor([[1, 4]] * 2)
    om = torch.ones(2, 2, dtype=torch.bool)
    tgt = torch.tensor([[0.3, 0.7]] * 2)
    for on in (True, False, True):
        set_checkpointing(m, on)
        assert all(layer.gradient_checkpointing == on for layer in m.encoder.layers)
        m.zero_grad()
        decision_loss(m(ids, am, pos, om), tgt, om, torch.zeros(2, dtype=torch.bool)).backward()
        g = m.encoder.layers[0].attn.Wqkv.weight.grad
        assert g is not None and g.abs().sum() > 0
