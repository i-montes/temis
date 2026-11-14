"""On CPU every question gets its own forward pass, at its own length: no padding, and an answer that
does not depend on which other questions came in the same call."""

import numpy as np

from temis.schema import parse_questions

QS = {
    "short": {"type": "noul", "instructions": "Is it urgent?"},
    "long": {"type": "choice", "instructions": "Which team should take this request? " * 4,
             "criteria": {"billing": "Invoices, refunds, payment methods and anything about money " * 3,
                          "tech": "Bugs, outages, errors and anything about the product not working " * 3,
                          "sales": "Prices, plans, demos and anything about buying more " * 3}},
}


def int8_run(tmp_path):
    from scripts.export_onnx import quantize_int8
    from tests.tiny import tiny_run
    d = tiny_run(tmp_path)
    quantize_int8(d / "decisor.onnx", d / "decisor.int8.onnx")
    return d


def test_an_answer_does_not_depend_on_the_other_questions_of_the_call(tmp_path):
    from temis.runtime import Decider
    d = Decider(int8_run(tmp_path), backend="onnx")
    short, long_ = parse_questions(QS)
    state = "The customer says the app crashes every time they open an invoice."
    together = d.logits([short, long_], state)
    assert np.array_equal(together[0], d.logits([short], state)[0])
    assert np.array_equal(together[1], d.logits([long_], state)[0])


def test_no_padding_on_cpu(tmp_path):
    from temis.runtime import Decider
    d = Decider(int8_run(tmp_path), backend="onnx")
    d.logits(parse_questions(QS), "state")
    assert d.last_padded_tokens == d.last_input_tokens


def test_the_gpu_backend_keeps_batching(tmp_path):
    from temis.runtime import Decider
    from tests.tiny import tiny_run
    assert not Decider(tiny_run(tmp_path), backend="torch").per_question
