"""Packed layout in training and in the runtime."""

import json
import random

import numpy as np
import torch

from temis.schema import parse_questions
from tests.test_core import Q

QS = parse_questions(Q)  # choice (2), score (3), noul


def ex(q, case="c1", target=None, source="synth", state="the state"):
    t = target or [1.0 / q.n] * q.n
    return {"q": q, "state": state, "target": t, "kind": q.kind, "source": source, "weight": 1.0, "case": case}


def test_questions_of_a_case_go_in_one_pack_and_copies_in_another():
    from scripts.train import group_packs
    rows = [ex(QS[0]), ex(QS[1]), ex(QS[2]), ex(QS[2]),          # noul duplicated (rebalancing copy)
            ex(QS[0], case="c2", state="other")]
    packs = group_packs(rows)
    by_case = {}
    for p in packs:
        by_case.setdefault(p["case"], []).append(sorted(q.qid for q in p["qs"]))
    assert sorted(by_case["c1"]) == [["action", "risk", "urgent"], ["urgent"]]
    assert by_case["c2"] == [["action"]]


def test_twin_packs_hold_the_same_tokens_and_targets_in_option_order():
    from scripts.train import encode_pack_twins, group_packs
    from temis.encoding import Encoder
    from tests.test_core import tok_path
    enc = Encoder(tok_path(), 512)
    rows = [ex(QS[0], target=[0.2, 0.8]), ex(QS[1], target=[0.1, 0.3, 0.6]), ex(QS[2], target=[0.9, 0.1])]
    packs = group_packs(rows)
    a, b = encode_pack_twins(enc, packs, random.Random(4), ["noise " * 20], 512)
    assert sorted(a[0]["ids"]) == sorted(b[0]["ids"])
    assert a[0]["targets"] == b[0]["targets"]
    t = {q.qid: tg for q, tg in zip(packs[0]["qs"], a[0]["targets"])}
    assert t == {"action": [0.2, 0.8], "risk": [0.1, 0.3, 0.6], "urgent": [0.9, 0.1]}


def test_packed_training_step_learns_every_question(tmp_path):
    from scripts.train import collate_packs, encode_pack_twins, group_packs, pack_loss
    from temis.encoding import Encoder
    from temis.model import Decisor
    from tests.tiny import mrbert_tokenizer, tiny_base
    enc = Encoder(mrbert_tokenizer(), 256)
    m = Decisor(tiny_base(tmp_path), attn="eager")
    rows = [ex(QS[0], target=[0.0, 1.0]), ex(QS[1], target=[0.0, 0.0, 1.0]), ex(QS[2], target=[0.0, 1.0])]
    packs = group_packs(rows)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for step in range(30):
        a, b = encode_pack_twins(enc, packs, random.Random(step), None, 256)
        batch = collate_packs(a + b)
        loss = pack_loss(m, batch, n_twin=len(a), consistency=0.5)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.5


def test_runtime_reads_a_packed_model(tmp_path):
    from temis.runtime import Decider, slots_to_logits
    from tests.tiny import tiny_run
    d = tiny_run(tmp_path)
    cfg = json.loads((d / "config.json").read_text())
    (d / "config.json").write_text(json.dumps({**cfg, "layout": "packed"}))
    dec = Decider(d, int8=False, backend="onnx")
    got = dec.logits(QS, "a state")
    p = dec.encoder.encode_pack(QS, "a state", max_len=dec.long_max_len)
    marker = dec.session.run(None, {"input_ids": np.array([p.ids]), "attention_mask": np.ones((1, len(p.ids)), np.int64),
                                    "opt_pos": np.array([p.opt_pos]), "opt_mask": np.ones((1, len(p.opt_pos)), np.int64)})[0][0]
    for x, y in zip(got, slots_to_logits(marker, p.slots)):
        assert np.allclose(x, y, atol=1e-5)
    assert dec.last_input_tokens == len(p.ids)   # the state was read once
    torch_ = Decider(d, backend="torch")
    for x, y in zip(got, torch_.logits(QS, "a state")):
        assert np.abs(x - y).max() < 1e-3


def test_runtime_splits_questions_that_do_not_fit_in_one_pack(tmp_path):
    from temis.runtime import Decider
    from tests.tiny import tiny_run
    d = tiny_run(tmp_path)
    cfg = json.loads((d / "config.json").read_text())
    (d / "config.json").write_text(json.dumps({**cfg, "layout": "packed", "long_max_len": 128}))
    qs = parse_questions({f"q{i}": {"type": "noul", "instructions": f"Question {i} " * 8} for i in range(10)})
    dec = Decider(d, int8=False, backend="onnx")
    out = dec.logits(qs, "state " * 50)
    assert len(out) == 10 and all(x.shape == (2,) and x[0] == 0 for x in out)


def test_twin_consistency_adds_to_the_loss_when_the_twins_disagree(tmp_path):
    from scripts.train import collate_packs, encode_pack_twins, group_packs, pack_loss
    from temis.encoding import Encoder
    from temis.model import Decisor
    from tests.tiny import mrbert_tokenizer, tiny_base
    enc = Encoder(mrbert_tokenizer(), 256)
    m = Decisor(tiny_base(tmp_path), attn="eager").eval()   # no dropout: only the shuffles differ
    torch.manual_seed(0)
    for lin in (m.head[0], m.head[3]):  # a random head that reacts to where each option sits
        torch.nn.init.normal_(lin.weight, std=3.0)
    packs = group_packs([ex(QS[0]), ex(QS[1]), ex(QS[2])])
    a, b = encode_pack_twins(enc, packs, random.Random(2), None, 256)
    batch = collate_packs(a + b)
    with torch.no_grad():
        plain = pack_loss(m, batch, n_twin=len(a), consistency=0.0).item()
        with_kl = pack_loss(m, batch, n_twin=len(a), consistency=0.5).item()
    assert with_kl > plain


def test_a_split_twin_holds_the_same_questions_in_several_packs():
    from scripts.train import encode_pack_twins, group_packs
    from temis.encoding import Encoder
    from tests.test_core import tok_path
    enc = Encoder(tok_path(), 512)
    packs = group_packs([ex(QS[0]), ex(QS[1]), ex(QS[2]), ex(QS[0], case="c2")])
    a, b = encode_pack_twins(enc, packs, random.Random(0), None, 512, split=1.0)
    assert len(a) == 2 and len(b) == 3          # the 3-question pack is split in two, the single one is not
    keys = lambda items: sorted(k for it in items for k in it["keys"])  # noqa: E731
    assert keys(a) == keys(b) == [(0, 0), (0, 1), (0, 2), (1, 0)]
    a, b = encode_pack_twins(enc, packs, random.Random(0), None, 512, split=0.0)
    assert len(b) == 2
    for seed in range(20):   # never an empty half
        _, b = encode_pack_twins(enc, packs, random.Random(seed), None, 512, split=1.0)
        assert all(it["targets"] for it in b)


def test_twin_consistency_compares_the_same_question_across_a_split(tmp_path):
    from scripts.train import twin_alignment
    ka = [(0, 0), (0, 1), (0, 2), (1, 0)]
    kb = [(0, 2), (0, 0), (1, 0), (0, 1)]
    assert [kb[i] for i in twin_alignment(ka, kb)] == ka


def test_split_twin_consistency_matches_each_question_with_itself(tmp_path):
    from scripts.train import collate_packs, encode_pack_twins, group_packs, pack_logits, pack_loss, twin_alignment
    from temis.encoding import Encoder
    from temis.model import Decisor, consistency_loss
    from tests.tiny import mrbert_tokenizer, tiny_base
    enc = Encoder(mrbert_tokenizer(), 256)
    m = Decisor(tiny_base(tmp_path), attn="eager").eval()
    torch.manual_seed(0)
    for lin in (m.head[0], m.head[3]):
        torch.nn.init.normal_(lin.weight, std=3.0)
    packs = group_packs([ex(QS[0]), ex(QS[1]), ex(QS[2])])
    a, b = encode_pack_twins(enc, packs, random.Random(5), None, 256, split=1.0)
    batch = collate_packs(a + b)
    h = sum(len(x["targets"]) for x in a)
    with torch.no_grad():
        lg, mask = pack_logits(m, batch)
        perm = twin_alignment(batch["keys"][:h], batch["keys"][h:])
        same = [list(range(n)) for n in batch["n"][:h]]
        want = consistency_loss(lg[:h], lg[h:][perm], mask[:h], mask[h:][perm], same, same, batch["n"][:h]).item()
        got = pack_loss(m, batch, len(a), 1.0).item() - pack_loss(m, batch, len(a), 0.0).item()
    assert batch["keys"][h:] != batch["keys"][:h]   # the split really reordered the questions
    assert abs(got - want) < 1e-5
