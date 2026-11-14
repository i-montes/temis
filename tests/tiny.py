"""A tiny random ModernBERT with MrBERT-es's vocabulary, saved to disk like a real base, so tests can
build a Decisor (and export it) in seconds without downloading weights."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

MRBERT = "BSC-LT/MrBERT-es"


def mrbert_tokenizer() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(MRBERT, "tokenizer.json")


def tiny_base(root: Path, layers: int = 4, local_attention: int = 16) -> str:
    """A 4-layer ModernBERT (hidden 32) with global attention every 3rd layer and a short local
    window, so sequences of a few dozen tokens already exercise the sliding-window layers."""
    import torch
    from transformers import ModernBertConfig, ModernBertModel
    torch.manual_seed(0)
    cfg = ModernBertConfig(vocab_size=51200, hidden_size=32, intermediate_size=48, num_hidden_layers=layers,
                           num_attention_heads=2, local_attention=local_attention, global_attn_every_n_layers=3,
                           max_position_embeddings=512, pad_token_id=1, bos_token_id=0, eos_token_id=2,
                           cls_token_id=0, sep_token_id=2)
    path = root / "tiny-base"
    ModernBertModel(cfg).save_pretrained(path)
    return str(path)


def tiny_run(root: Path, export: bool = True) -> Path:
    """runs/<tiny>/{model.pt, meta.json, onnx/{decisor.onnx, tokenizer.json, config.json}}; returns the onnx dir."""
    import torch
    from scripts.export_onnx import Wrapped
    from temis.model import Decisor
    base = tiny_base(root)
    run = root / "run"
    (run / "onnx").mkdir(parents=True)
    torch.manual_seed(1)
    m = Decisor(base, attn="eager").eval()
    torch.save(m.state_dict(), run / "model.pt")
    cfg = {"version": "tiny", "base": base, "max_len": 256, "long_max_len": 256,
           "temperatures": {"choice": 1.0, "score": 1.0, "noul": 1.0}}
    (run / "onnx" / "config.json").write_text(json.dumps(cfg))
    shutil.copyfile(mrbert_tokenizer(), run / "onnx" / "tokenizer.json")
    if export:
        B, T, K = 2, 24, 3
        args = (torch.randint(10, 1000, (B, T)), torch.ones(B, T, dtype=torch.long),
                torch.tensor([[1, 4, 7]] * B), torch.ones(B, K, dtype=torch.long))
        torch.onnx.export(Wrapped(m).eval(), args, str(run / "onnx" / "decisor.onnx"),
                          input_names=["input_ids", "attention_mask", "opt_pos", "opt_mask"], output_names=["logits"],
                          opset_version=18, dynamo=False,
                          dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"},
                                        "opt_pos": {0: "b", 1: "k"}, "opt_mask": {0: "b", 1: "k"},
                                        "logits": {0: "b", 1: "k"}})
    return run / "onnx"
