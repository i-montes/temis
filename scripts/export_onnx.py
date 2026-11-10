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


def quantize_int8(src: Path, dst: Path) -> None:
    """Dynamic int8 for the encoder; the scoring head (2 small MatMuls, +4 MB) stays fp32.
    Quantizing the head turns its error straight into the logits: on v1109-gpu it moved typed-decisions
    KL from 0.119 (fp32) to 0.186, and to 0.135 with the head left out. Per-channel is avoided on
    purpose: without reduce_range it saturates on x86 CPUs without VNNI (accuracy fell to 0.31)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import onnx
    graph = onnx.load(str(src), load_external_data=False).graph
    head = [n.name for n in graph.node if n.op_type in ("MatMul", "Gemm") and n.name.startswith("/m/head/")]
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, nodes_to_exclude=head)


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
    model = Decisor(base, attn="eager", head_norm=bool(meta.get("head_norm", False)))
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
    quantize_int8(path, out / "decisor.int8.onnx")
    (out / "tokenizer.json").unlink(missing_ok=True)
    shutil.copyfile(tokenizer_path(base), out / "tokenizer.json")
    from temis.naming import NAME, assign
    semver = assign(a.name, base, fine_tune=bool(meta.get("init")))
    (out / "config.json").write_text(json.dumps({"version": a.name, "name": NAME, "semver": semver, "base": base,
                                                 "max_len": meta["max_len"], "long_max_len": inference_long_max_len(meta),
                                                 "temperatures": meta["temperatures"]},
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
