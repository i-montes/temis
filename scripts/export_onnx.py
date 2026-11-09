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

from decisor.model import Decisor
from scripts.train import BASE, tokenizer_path


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
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(path), str(out / "decisor.int8.onnx"), weight_type=QuantType.QInt8)
    (out / "tokenizer.json").unlink(missing_ok=True)
    shutil.copyfile(tokenizer_path(), out / "tokenizer.json")
    from decisor.naming import NAME, assign
    semver = assign(a.name, base, fine_tune=bool(meta.get("init")))
    (out / "config.json").write_text(json.dumps({"version": a.name, "name": NAME, "semver": semver, "base": base,
                                                 "max_len": meta["max_len"], "temperatures": meta["temperatures"]},
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
