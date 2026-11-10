import types

import torch
from torch import nn

import temis.model as model_mod
from temis.model import Decisor


class FakeEncoder(nn.Module):
    """Stands in for the pretrained encoder: fixed hidden states times `scale`."""

    def __init__(self, h: int = 16) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=h)
        self.scale = 1.0
        self.states = torch.randn(2, 10, h, generator=torch.Generator().manual_seed(0))

    def forward(self, input_ids, attention_mask):
        return types.SimpleNamespace(last_hidden_state=self.states * self.scale)


def decisor(monkeypatch, **kw) -> Decisor:
    monkeypatch.setattr(model_mod.AutoModel, "from_pretrained", lambda *a, **k: FakeEncoder())
    torch.manual_seed(0)
    return Decisor("fake", **kw).eval()


def logits(m: Decisor, scale: float) -> torch.Tensor:
    m.encoder.scale = scale
    ids = torch.zeros(2, 10, dtype=torch.long)
    pos = torch.tensor([[1, 4, 7], [2, 5, 8]])
    return m(ids, torch.ones_like(ids), pos, torch.ones(2, 3, dtype=torch.bool))


def test_head_norm_makes_the_head_blind_to_the_encoder_scale(monkeypatch):
    # MrBERT-es hidden states are ~7x larger than mmBERT's at the head input; with the same head LR the
    # head saturates (every pre-activation negative) and the model answers the same for every option.
    m = decisor(monkeypatch, head_norm=True)
    assert torch.allclose(logits(m, 1.0), logits(m, 7.0), atol=1e-4)


def test_without_head_norm_the_scale_reaches_the_head(monkeypatch):
    m = decisor(monkeypatch)
    assert not torch.allclose(logits(m, 1.0), logits(m, 7.0), atol=1e-2)
    assert not any(k.startswith("head_norm") for k in m.state_dict())  # old checkpoints still load
