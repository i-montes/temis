"""The student: a multilingual encoder whose option-marker states are scored by a small MLP."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel

KIND_ID = {"choice": 0, "score": 1, "noul": 2}


class Decisor(nn.Module):
    def __init__(self, base: str = "jhu-clsp/mmBERT-small", attn: str | None = None) -> None:
        super().__init__()
        kw = {"attn_implementation": attn} if attn else {}
        self.encoder = AutoModel.from_pretrained(base, **kw)
        h = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, input_ids: Tensor, attention_mask: Tensor, opt_pos: Tensor, opt_mask: Tensor) -> Tensor:
        """input_ids/attention_mask (B, T); opt_pos/opt_mask (B, K) -> logits (B, K)."""
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        opt = torch.gather(hs, 1, opt_pos.unsqueeze(-1).expand(-1, -1, hs.size(-1)))  # (B, K, H)
        cls = hs[:, :1].expand_as(opt)
        logits = self.head(torch.cat([opt, opt * cls], -1)).squeeze(-1)
        return logits.masked_fill(~opt_mask, -1e4)


def decision_loss(logits: Tensor, target: Tensor, mask: Tensor, is_score: Tensor,
                  brier_w: float = 0.25, ordinal_w: float = 0.25) -> Tensor:
    logp = logits.float().masked_fill(~mask, -1e4).log_softmax(-1)
    loss = -(target * logp).sum(-1)
    p = logp.exp()
    loss = loss + brier_w * (((p - target) ** 2) * mask).sum(-1)
    if is_score.any():
        k = p.size(-1)
        lv = torch.arange(k, device=p.device, dtype=p.dtype)
        dist = (lv[:, None] - lv[None, :]).abs()
        span = (mask.sum(-1) - 1).clamp(min=1).to(p.dtype)
        ordinal = torch.einsum("bi,ij,bj->b", p, dist, target) / span
        loss = loss + ordinal_w * ordinal * is_score.float()
    return loss.mean()


def collate(batch: list[dict], device: str | torch.device = "cpu") -> dict[str, Tensor]:
    """batch items: {ids: list[int], opt_pos: list[int], target: list[float] (in marker order), kind}."""
    T = max(len(b["ids"]) for b in batch)
    T = ((T + 63) // 64) * 64  # few distinct shapes: keeps the MPS/CUDA allocator caches small
    K = max(len(b["opt_pos"]) for b in batch)
    B = len(batch)
    ids = torch.zeros(B, T, dtype=torch.long)
    am = torch.zeros(B, T, dtype=torch.long)
    pos = torch.zeros(B, K, dtype=torch.long)
    om = torch.zeros(B, K, dtype=torch.bool)
    tgt = torch.zeros(B, K)
    for i, b in enumerate(batch):
        n = len(b["ids"])
        ids[i, :n] = torch.tensor(b["ids"])
        am[i, :n] = 1
        k = len(b["opt_pos"])
        pos[i, :k] = torch.tensor(b["opt_pos"])
        om[i, :k] = True
        if "target" in b:
            tgt[i, :k] = torch.tensor(b["target"])
    out = {"input_ids": ids, "attention_mask": am, "opt_pos": pos, "opt_mask": om, "target": tgt,
           "is_score": torch.tensor([b["kind"] == "score" for b in batch])}
    return {k: v.to(device) for k, v in out.items()}


def fit_temperature(logits: Tensor, target: Tensor, mask: Tensor) -> float:
    """One temperature minimizing soft cross-entropy on held-out data (grid + refine)."""
    best, best_t = float("inf"), 1.0
    for t in [x / 20 for x in range(6, 81)]:
        lp = (logits / t).masked_fill(~mask, -1e4).log_softmax(-1)
        ce = float(-(target * lp).sum(-1).mean())
        if ce < best:
            best, best_t = ce, t
    return best_t


__all__ = ["Decisor", "decision_loss", "collate", "fit_temperature", "KIND_ID", "F"]
