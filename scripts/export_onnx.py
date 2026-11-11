"""Export runs/<name>/model.pt to ONNX (fp32 + int8) for the torch-free runtime.

  python -m scripts.export_onnx --name v1

Writes runs/<name>/onnx/{decisor.onnx, decisor.int8.onnx, tokenizer.json, config.json}.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from temis.model import Decisor
from scripts.train import BASE, tokenizer_path


INFER_LONG_MAX_LEN = 1024  # the runtime budget for long states (CPU cost); training may use more


def inference_long_max_len(meta: dict, cap: int = INFER_LONG_MAX_LEN) -> int:
    return min(int(meta.get("long_max_len", cap)), cap)


OUTLIER_LAYERS = 5  # mlp.Wo of this many layers stays fp32 in the int8 model (about +14 MB)


def outlier_layers(model: torch.nn.Module, batches: list[dict], k: int = OUTLIER_LAYERS) -> list[int]:
    """The k encoder layers whose mlp.Wo input is most extreme (max |x| / mean |x|), most extreme first.
    That input comes out of the gated MLP activation and a few channels blow up (×10⁴ in MrBERT-es
    layers 2 and 12, mmBERT-base layer 12); dynamic int8 uses one scale per tensor, so those channels
    crush the rest. Which layers it hits depends on the base and the run, hence measured, not listed."""
    layers = getattr(getattr(model, "encoder", None), "layers", None) or []
    stats: dict[int, list[float]] = {}

    def hook(i: int):
        def f(_mod, inp, _out):
            x = inp[0].detach().float().abs()
            s = stats.setdefault(i, [0.0, 0.0, 0.0])
            s[0] = max(s[0], float(x.max()))
            s[1] += float(x.mean())
            s[2] += 1
        return f

    handles = [layer.mlp.Wo.register_forward_hook(hook(i)) for i, layer in enumerate(layers)
               if hasattr(getattr(layer, "mlp", None), "Wo")]
    try:
        with torch.no_grad():
            for b in batches:
                model(b["input_ids"], b["attention_mask"], b["opt_pos"], b["opt_mask"])
    finally:
        for h in handles:
            h.remove()
    ratio = {i: s[0] / max(s[1] / s[2], 1e-12) for i, s in stats.items()}
    return sorted(ratio, key=ratio.get, reverse=True)[:k]


def quantize_int8(src: Path, dst: Path, fp32_layers: list[int] | tuple[int, ...] = ()) -> None:
    """Dynamic int8 for the encoder; the scoring head (2 small MatMuls, +4 MB) and the mlp.Wo of
    `fp32_layers` stay fp32.
    Quantizing the head turns its error straight into the logits: on v1109-gpu it moved typed-decisions
    KL from 0.119 (fp32) to 0.186, and to 0.135 with the head left out. The outlier layers (see
    outlier_layers) cost MrBERT-es 4 points: v1110-mrbert-h5 typed-decisions 0.774 fp32, 0.734 int8,
    0.7655 int8 with its 5 outlier layers in fp32 (169 MB). Per-channel is avoided on purpose: without
    reduce_range it saturates on x86 CPUs without VNNI (accuracy fell to 0.31)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import onnx
    graph = onnx.load(str(src), load_external_data=False).graph
    wo = tuple(f"/layers.{i}/mlp/Wo/" for i in fp32_layers)
    keep = [n.name for n in graph.node if n.op_type in ("MatMul", "Gemm")
            and (n.name.startswith("/m/head/") or any(w in n.name for w in wo))]
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, nodes_to_exclude=keep)


def calibration_batches(base: str, meta: dict, n: int = 64, bs: int = 16) -> list[dict]:
    """n validation questions (never test) encoded like at training time, to find the outlier layers."""
    from scripts.train import Encoder, collate, encode_batch, load_examples
    enc = Encoder(tokenizer_path(base), meta["max_len"])
    _, val = load_examples([])
    val = val[:n]
    return [collate(encode_batch(enc, val[i:i + bs], None, None, meta["max_len"], meta.get("long_max_len")), "cpu")
            for i in range(0, len(val), bs)]


class Wrapped(torch.nn.Module):
    def __init__(self, m: Decisor) -> None:
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, opt_pos, opt_mask):
        return self.m(input_ids, attention_mask, opt_pos, opt_mask.bool())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    run = Path("runs") / a.name
    out = run / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run / "meta.json").read_text())
    base = meta.get("base", BASE)
    model = Decisor(base, attn="eager")
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    model.eval()
    w = Wrapped(model).eval()
    B, T, K = 2, 64, 4
    am = torch.ones(B, T, dtype=torch.long)
    am[1, 40:] = 0
    om = torch.ones(B, K, dtype=torch.long)
    om[1, 3] = 0
    args = (torch.randint(10, 1000, (B, T)), am, torch.tensor([[1, 5, 9, 13]] * B), om)
    path = out / "decisor.onnx"
    torch.onnx.export(
        w, args, str(path), input_names=["input_ids", "attention_mask", "opt_pos", "opt_mask"],
        output_names=["logits"], opset_version=18, dynamo=False,
        dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"},
                      "opt_pos": {0: "b", 1: "k"}, "opt_mask": {0: "b", 1: "k"}, "logits": {0: "b", 1: "k"}})
    fp32_layers = outlier_layers(model, calibration_batches(base, meta))
    print(f"[export] int8: mlp.Wo of layers {fp32_layers} stay fp32 (most extreme inputs)")
    quantize_int8(path, out / "decisor.int8.onnx", fp32_layers)
    (out / "tokenizer.json").unlink(missing_ok=True)
    shutil.copyfile(tokenizer_path(base), out / "tokenizer.json")
    from temis.naming import NAME, assign
    semver = assign(a.name, base, fine_tune=bool(meta.get("init")))
    (out / "config.json").write_text(json.dumps({"version": a.name, "name": NAME, "semver": semver, "base": base,
                                                 "max_len": meta["max_len"], "long_max_len": inference_long_max_len(meta),
                                                 "temperatures": meta["temperatures"],
                                                 "int8_fp32_layers": fp32_layers},
                                                indent=2))
    print(f"[export] {NAME} {semver} ({a.name})")
    # parity check
    import numpy as np
    import onnxruntime as ort
    with torch.no_grad():
        ref = w(*args).numpy()
    for f in ["decisor.onnx", "decisor.int8.onnx"]:
        s = ort.InferenceSession(str(out / f), providers=["CPUExecutionProvider"])
        got = s.run(None, {"input_ids": args[0].numpy(), "attention_mask": args[1].numpy(),
                           "opt_pos": args[2].numpy(), "opt_mask": args[3].numpy()})[0]
        print(f"[export] {f}: {(out / f).stat().st_size / 1e6:.0f} MB, max |diff| vs torch = {np.abs(got - ref).max():.4f}")


if __name__ == "__main__":
    main()
