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
