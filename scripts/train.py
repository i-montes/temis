"""Train the student on data/targets.jsonl.

  python -m scripts.train --name v1 --epochs 3

Saves runs/<name>/{model.pt, meta.json}. meta.json holds per-kind temperatures and val metrics.
Extra labeled data (user labels, replay) can be passed with --extra data/user_targets.jsonl.

Twin batches (default): every question is encoded twice, with different option shuffles and
different dropout, and a symmetric KL between the two answers (in option order) is added to the
loss (--consistency). That removes the option-position bias and doubles as R-Drop. Long states
(truncated at --max-len) are re-encoded at --long-max-len and batched --long-bs at a time.
On CUDA the forward runs in bf16 with SDPA attention; MPS/CPU stay in fp32.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from temis.data.procedural import read_cases as read_procedural_cases
from temis.data.sources import load_synth, load_td, read_jsonl
from temis.encoding import Encoder
from temis.model import Decisor, collate, consistency_loss, decision_loss, fit_temperature
from temis.schema import NOUL_DEFAULT, parse_questions, render_state

BASE = "jhu-clsp/mmBERT-small"  # default; --base overrides (saved in meta.json)


def tokenizer_path(base: str = BASE) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(base, "tokenizer.json")


def is_val(key: str) -> bool:
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 10 == 0


def load_examples(extra: list[str]) -> tuple[list[dict], list[dict]]:
    cases = {c["id"]: c for c in load_td("train") + load_synth() + read_procedural_cases()}
    rows = list(read_jsonl(Path("data/targets.jsonl")))
    for e in extra:
        rows += list(read_jsonl(Path(e)))
    for r in rows:
        c = r.get("case") or cases.get(r["id"])
        if c is not None:
            cases[r["id"]] = c
    train, val = [], []
    for r in rows:
        c = cases.get(r["id"])
        if c is None:
            continue
        q = next((q for q in parse_questions(c["questions"]) if q.qid == r["qid"]), None)
        if q is None or len(r["target"]) != q.n:
            continue
        ex = {"q": q, "state": render_state(c["state"]), "target": r["target"], "kind": q.kind,
              "source": r.get("source", "synth"), "weight": float(r.get("weight", 1.0))}
        key = r["id"] if r.get("source") == "td-train" else str(r.get("schema_id", r["id"]))
        (val if is_val(key) and r.get("source") != "user" else train).append(ex)
    return train, val


def augment(ex: dict, rng: random.Random | None, ctx_pool: list[str] | None):
    """The training-time augmentation of one example: (question, context). Drawn once per example so
    twin encodings differ only in option order and dropout."""
    ctx = None
    q = ex["q"]
    if rng is not None and ctx_pool and rng.random() < 0.15:
        ctx = rng.choice(ctx_pool)[:600]
    if rng is not None and q.kind == "noul" and rng.random() < 0.3:
        # callers often send noul questions without criteria: learn to decide from the statement alone
        q = dataclasses.replace(q, descriptions=(NOUL_DEFAULT["false"], NOUL_DEFAULT["true"]))
    return q, ctx


def encode_one(enc: Encoder, ex: dict, q, ctx, rng: random.Random | None, max_len: int,
               long_max_len: int | None = None) -> dict:
    """A state that gets truncated at max_len is re-encoded at long_max_len (when given)."""
    e = enc.encode(q, ex["state"], ctx, rng=rng, max_len=max_len)
    if long_max_len and long_max_len > max_len and len(e.ids) >= max_len:
        e = enc.encode(q, ex["state"], ctx, rng=rng, max_len=long_max_len)
    return {"ids": e.ids, "opt_pos": e.opt_pos, "target": [ex["target"][i] for i in e.order],
            "kind": ex["kind"], "order": e.order, "n": q.n}


def encode_batch(enc: Encoder, exs: list[dict], rng: random.Random | None, ctx_pool: list[str] | None,
                 max_len: int, long_max_len: int | None = None) -> list[dict]:
    return [encode_one(enc, ex, *augment(ex, rng, ctx_pool), rng, max_len, long_max_len) for ex in exs]


def encode_twins(enc: Encoder, exs: list[dict], rng: random.Random, ctx_pool: list[str] | None,
                 max_len: int, long_max_len: int | None = None) -> tuple[list[dict], list[dict]]:
    """Two encodings of the same examples with the same augmentation and different option shuffles."""
    a, b = [], []
    for ex in exs:
        q, ctx = augment(ex, rng, ctx_pool)
        a.append(encode_one(enc, ex, q, ctx, rng, max_len, long_max_len))
        b.append(encode_one(enc, ex, q, ctx, rng, max_len, long_max_len))
    return a, b


def batches(exs: list[dict], bs: int, rng: random.Random, long_bs: int | None = None,
            max_len: int | None = None) -> list[list[dict]]:
    """Batches of similar length (by state tokens, `ntok`); the ones holding a state longer than
    max_len are split into long_bs."""
    length = lambda e: e.get("ntok", len(e["state"]) // 3)  # noqa: E731
    exs = sorted(exs, key=lambda e: length(e) + rng.random() * 120)
    bl = [exs[i:i + bs] for i in range(0, len(exs), bs)]
    if long_bs and max_len and long_bs < bs:
        split = []
        for b in bl:
            if max(length(e) for e in b) > max_len:
                split += [b[i:i + long_bs] for i in range(0, len(b), long_bs)]
            else:
                split.append(b)
        bl = split
    rng.shuffle(bl)
    return bl


@torch.no_grad()
def predict(model: Decisor, enc: Encoder, exs: list[dict], device, max_len: int, bs: int = 32,
            long_max_len: int | None = None, autocast=None):
    model.eval()
    logits, targets, masks, kinds, srcs = [], [], [], [], []
    for i in range(0, len(exs), bs):
        chunk = exs[i:i + bs]
        b = collate(encode_batch(enc, chunk, None, None, max_len, long_max_len), device)
        with autocast() if autocast else contextlib.nullcontext():
            lg = model(b["input_ids"], b["attention_mask"], b["opt_pos"], b["opt_mask"]).float().cpu()
        K = 12
        pad = lambda t, v: torch.nn.functional.pad(t, (0, K - t.size(1)), value=v)  # noqa: E731
        logits.append(pad(lg, -1e4)); targets.append(pad(b["target"].cpu(), 0.0)); masks.append(pad(b["opt_mask"].cpu(), False))
        kinds += [e["kind"] for e in chunk]; srcs += [e["source"] for e in chunk]
    return torch.cat(logits), torch.cat(targets), torch.cat(masks), kinds, srcs


def metrics(logits, target, mask, temps: dict[str, float] | None, kinds) -> dict:
    t = torch.tensor([(temps or {}).get(k, 1.0) for k in kinds]).unsqueeze(-1)
    lp = (logits / t).masked_fill(~mask, -1e4).log_softmax(-1)
    p = lp.exp()
    acc = (p.argmax(-1) == target.argmax(-1)).float()
    ce = -(target * lp).sum(-1)
    ent = -(target * target.clamp(min=1e-9).log()).sum(-1)
    brier = ((p - target) ** 2 * mask).sum(-1)
    return {"acc": float(acc.mean()), "kl": float((ce - ent).mean()), "brier": float(brier.mean()), "n": len(kinds)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--init", default=None, help="start from runs/<name>/model.pt")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--train-emb", action="store_true", help="also fine-tune the token embeddings")
    ap.add_argument("--twin", action=argparse.BooleanOptionalAction, default=True,
                    help="encode every question twice (different shuffles/dropout) and add the consistency loss")
    ap.add_argument("--consistency", type=float, default=0.5, help="weight of the symmetric KL between the twins")
    ap.add_argument("--long-max-len", type=int, default=1536, help="budget for states that overflow --max-len")
    ap.add_argument("--long-bs", type=int, default=4, help="batch size for batches holding a long state")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None, help="default: on CUDA only")
    ap.add_argument("--head-norm", action="store_true",
                    help="LayerNorm the encoder states before the head (bases whose states are large, e.g. MrBERT-es)")
    ap.add_argument("--val-limit", type=int, default=None, help="validate on the first N questions (smoke tests)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = (device == "cuda") if a.bf16 is None else a.bf16
    autocast = (lambda: torch.autocast(device_type=device, dtype=torch.bfloat16)) if bf16 else None
    attn = "sdpa" if device == "cuda" else None
    out = Path("runs") / a.name
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    torch.manual_seed(0)

    enc = Encoder(tokenizer_path(a.base), a.max_len)
    train, val = load_examples(a.extra)
    val = val[: a.val_limit] if a.val_limit else val
    for ex in train:
        ex["ntok"] = len(enc._ids(ex["state"]))  # for length batching (tokenizers: a few seconds)
    print(f"[train] {len(train)} train / {len(val)} val questions on {device} "
          f"(twin={a.twin}, consistency={a.consistency}, long_max_len={a.long_max_len}, bf16={bf16}, attn={attn or 'eager'})",
          flush=True)
    ctx_pool = [e["state"] for e in train[:5000]]

    model = Decisor(a.base, attn=attn, head_norm=a.head_norm)
    if a.init:
        model.load_state_dict(torch.load(Path("runs") / a.init / "model.pt", map_location="cpu"))
    model.to(device)
    # token embeddings stay frozen (98M of the 140M parameters): less memory, keeps multilinguality
    emb = model.encoder.get_input_embeddings()
    emb.weight.requires_grad_(a.train_emb)
    model.encoder.gradient_checkpointing_enable()  # ~7 GB -> ~2 GB on MPS at the same speed
    opt = torch.optim.AdamW([
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": a.lr},
        {"params": model.head.parameters(), "lr": a.head_lr},
    ], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / (a.bs * a.accum))
    total = int(steps_per_epoch * a.epochs)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 1 - max(0, s - warm) / max(1, total - warm)))

    best, step, t0 = float("inf"), 0, time.time()
    log = []
    while step < total:
        model.train()
        micro = 0
        for chunk in batches(train, a.bs, rng, a.long_bs, a.max_len):
            if step >= total:
                break
            if a.twin:
                enc_a, enc_b = encode_twins(enc, chunk, rng, ctx_pool, a.max_len, a.long_max_len)
            else:
                enc_a, enc_b = encode_batch(enc, chunk, rng, ctx_pool, a.max_len, a.long_max_len), []
            b = collate(enc_a + enc_b, device)
            with autocast() if autocast else contextlib.nullcontext():
                logits = model(b["input_ids"], b["attention_mask"], b["opt_pos"], b["opt_mask"])
            loss = decision_loss(logits, b["target"], b["opt_mask"], b["is_score"])
            if a.twin and a.consistency > 0:
                h = len(enc_a)
                loss = loss + a.consistency * consistency_loss(
                    logits[:h], logits[h:], b["opt_mask"][:h], b["opt_mask"][h:],
                    [e["order"] for e in enc_a], [e["order"] for e in enc_b], [e["n"] for e in enc_a])
            (loss / a.accum).backward()
            micro += 1
            if micro % a.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            opt.zero_grad(set_to_none=True)
            if step % 50 == 0:
                if device == "mps":
                    torch.mps.empty_cache()
                elif device == "cuda":
                    torch.cuda.empty_cache()
                print(f"[train] step {step}/{total} loss {loss.item():.4f} {(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % steps_per_epoch == 0 or step == total:
                lg, tg, mk, kinds, srcs = predict(model, enc, val, device, a.max_len, long_max_len=a.long_max_len, autocast=autocast)
                m = metrics(lg, tg, mk, None, kinds)
                log.append({"step": step, **m})
                print(f"[val] step {step} {m}", flush=True)
                score = m["kl"]
                if score < best:
                    best = score
                    torch.save(model.state_dict(), out / "model.pt")
                model.train()

    # calibration on val with the best weights
    model.load_state_dict(torch.load(out / "model.pt", map_location="cpu"))
    model.to(device)
    lg, tg, mk, kinds, srcs = predict(model, enc, val, device, a.max_len, long_max_len=a.long_max_len, autocast=autocast)
    temps = {}
    for k in ("choice", "score", "noul"):
        idx = [i for i, x in enumerate(kinds) if x == k]
        if idx:
            temps[k] = fit_temperature(lg[idx], tg[idx], mk[idx])
    by = defaultdict(list)
    for i, s in enumerate(srcs):
        by[s].append(i)
    report = {"all": metrics(lg, tg, mk, temps, kinds)}
    for s, idx in by.items():
        report[s] = metrics(lg[idx], tg[idx], mk[idx], temps, [kinds[i] for i in idx])
    meta = {"base": a.base, "init": a.init, "max_len": a.max_len, "long_max_len": a.long_max_len, "twin": a.twin,
            "consistency": a.consistency, "head_norm": a.head_norm, "bf16": bf16, "attn": attn or "eager",
            "temperatures": temps, "val": report, "log": log,
            "n_train": len(train), "n_val": len(val), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] temps={temps} val={json.dumps(report)}", flush=True)


if __name__ == "__main__":
    main()
