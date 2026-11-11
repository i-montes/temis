import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from scripts.export_onnx import quantize_int8

ENC = "/m/encoder/layers.0/mlp/Wi/MatMul"
HEAD = "/m/head/head.0/MatMul"


def tiny_graph(path):
    """x -> encoder MatMul -> head MatMul -> y, named like the exported Decisor."""
    rng = np.random.default_rng(0)
    w1 = numpy_helper.from_array(rng.normal(size=(32, 32)).astype(np.float32), "w_enc")
    w2 = numpy_helper.from_array(rng.normal(size=(32, 32)).astype(np.float32), "w_head")
    g = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w_enc"], ["h"], name=ENC),
         helper.make_node("MatMul", ["h", "w_head"], ["y"], name=HEAD)],
        "tiny", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 32])], [w1, w2])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]), str(path))


def test_int8_keeps_the_head_in_fp32(tmp_path):
    src, dst = tmp_path / "decisor.onnx", tmp_path / "decisor.int8.onnx"
    tiny_graph(src)
    quantize_int8(src, dst)
    ops = {n.name: n.op_type for n in onnx.load(str(dst)).graph.node}
    assert ops.get(HEAD) == "MatMul"  # the head decides the logits: quantizing it costs most of the KL
    assert "MatMulInteger" in ops.values()  # the encoder is still quantized
    assert ENC not in ops or ops[ENC] != "MatMul"


WO = "/m/encoder/layers.3/mlp/Wo/MatMul"


def graph_with_wo(path):
    """x -> layer-3 mlp.Wo MatMul -> head MatMul -> y."""
    rng = np.random.default_rng(1)
    w1 = numpy_helper.from_array(rng.normal(size=(32, 32)).astype(np.float32), "w_wo")
    w2 = numpy_helper.from_array(rng.normal(size=(32, 32)).astype(np.float32), "w_head")
    g = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w_wo"], ["h"], name=WO),
         helper.make_node("MatMul", ["h", "w_head"], ["y"], name=HEAD)],
        "wo", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 32])], [w1, w2])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]), str(path))


def test_int8_keeps_the_mlp_output_of_outlier_layers_in_fp32(tmp_path):
    src = tmp_path / "decisor.onnx"
    graph_with_wo(src)
    quantize_int8(src, tmp_path / "keep.onnx", fp32_layers=[3])
    quantize_int8(src, tmp_path / "quant.onnx", fp32_layers=[5])
    assert {n.name: n.op_type for n in onnx.load(str(tmp_path / "keep.onnx")).graph.node}.get(WO) == "MatMul"
    assert {n.name: n.op_type for n in onnx.load(str(tmp_path / "quant.onnx")).graph.node}.get(WO) != "MatMul"


def test_outlier_layers_are_the_ones_with_the_most_extreme_mlp_output_inputs():
    import torch
    from torch import nn
    from scripts.export_onnx import outlier_layers

    class Layer(nn.Module):
        def __init__(self, spike: float) -> None:
            super().__init__()
            self.spike = spike
            self.mlp = nn.Module()
            self.mlp.Wo = nn.Linear(8, 8)

        def forward(self, x):
            h = x.clone()
            h[..., 0] *= self.spike  # one channel blown up, like MrBERT's layers 2 and 12
            return self.mlp.Wo(h)

    class Fake(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layers = nn.ModuleList([Layer(s) for s in (1, 1, 5000, 1, 300, 1)])

        def forward(self, input_ids, attention_mask, opt_pos, opt_mask):
            x = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(0))
            for layer in self.encoder.layers:
                layer(x)
            return x

    batch = {"input_ids": None, "attention_mask": None, "opt_pos": None, "opt_mask": None}
    assert outlier_layers(Fake(), [batch], k=2) == [2, 4]
    assert outlier_layers(Fake(), [batch], k=1) == [2]
