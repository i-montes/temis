"""Banded local attention for ModernBERT (an experiment; the export does not use it).

Measured on 2.4.0 int8 (Ryzen 5700X, 4 threads, bench_cpu workload): only long single sequences get
faster (JevBench p95 754 -> 620 ms, hard-like p50 1449 -> 1394 ms); Clio's short calls get slower
(p50 173 -> 211 ms) and memory grows by 220 MB, so it is kept here, not shipped.

ModernBERT alternates global layers (every token reads every token) with sliding-window layers where a
token reads only the keys within `w` positions (w = 64 in MrBERT-es and mmBERT: 14 of their 22 layers).
Eager attention still computes those layers' full T x T score matrix and masks most of it: at 1 024
tokens attention was 41-55 % of the int8 model's CPU time. Here the sequence is cut into blocks of w
tokens and each block of queries reads only its own block and the two next to it (3w keys), which hold
its whole window. Same keys, same softmax: the answers are the eager ones up to float rounding.

The mask is passed as key validity (B, 1, 1, T), not as the (B, 1, T, T) additive matrix transformers
builds, so the global layers do not pay for building it either.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

NAME = "temis_banded"


def _masked_softmax_attention(query, key, value, allowed, scaling):
    scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
    # a finite minimum, not -inf: a padding query whose window holds only padding gets a uniform row
    # instead of NaN, exactly as the eager mask does (and NaN would leak into the next layer)
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    return torch.matmul(F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype), value)


def banded_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, sliding_window=None, **_):
    """transformers attention interface: query/key/value (B, H, T, D); attention_mask: key validity
    (B, 1, 1, T) bool. Returns ((B, T, H, D), None)."""
    if sliding_window is None:  # global layer
        out = _masked_softmax_attention(query, key, value, attention_mask, scaling)
        return out.transpose(1, 2).contiguous(), None
    w = int(sliding_window) - 1  # transformers stores the half window + 1 (flash attention's convention)
    B, H, T, D = query.shape
    pad = (w - T % w) % w
    q = F.pad(query, (0, 0, 0, pad)).reshape(B, H, -1, w, D)                   # (B, H, n, w, D)
    k = F.pad(key, (0, 0, w, w + pad)).reshape(B, H, -1, w, D)                 # (B, H, n + 2, w, D)
    v = F.pad(value, (0, 0, w, w + pad)).reshape(B, H, -1, w, D)
    k = torch.cat([k[:, :, :-2], k[:, :, 1:-1], k[:, :, 2:]], dim=3)          # (B, H, n, 3w, D)
    v = torch.cat([v[:, :, :-2], v[:, :, 1:-1], v[:, :, 2:]], dim=3)
    m = F.pad(attention_mask.reshape(B, T).to(query.dtype), (w, w + pad)).reshape(B, -1, w)
    m = torch.cat([m[:, :-2], m[:, 1:-1], m[:, 2:]], dim=-1)                   # (B, n, 3w)
    # query i of a block sits at key column w + i; its window is the columns within w of it
    i = torch.arange(w, device=query.device).unsqueeze(1)
    j = torch.arange(3 * w, device=query.device).unsqueeze(0)
    band = (j - w - i).abs() <= w                                              # (w, 3w), the same for every block
    allowed = (m > 0.5)[:, None, :, None, :] & band                            # (B, 1, n, w, 3w)
    out = _masked_softmax_attention(q, k, v, allowed, scaling)                 # (B, H, n, w, D)
    out = out.reshape(B, H, -1, D)[:, :, :T]
    return out.transpose(1, 2).contiguous(), None


def register() -> None:
    from transformers import AttentionInterface
    AttentionInterface.register(NAME, banded_attention)


def key_mask(attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """The mask mapping ModernBERT takes instead of building its own: key validity for both layer types."""
    m = attention_mask.bool()[:, None, None, :]
    return {"full_attention": m, "sliding_attention": m}
