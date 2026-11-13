"""Packed layout: every question of a call in one sequence, the state read once.

    <bos> {q1: kind question: instructions <opt> option ...} <eos> {q2 ...} <eos> {state} [<ctx> context] <eos>

A yes/no question gets a single marker (its logit is P(yes) against a fixed 0 for no), so the 17 tool
questions of a Clio call do not pay two option texts each.
"""

import random

import numpy as np
import pytest
import torch

from temis.schema import parse_questions
from tests.test_core import Q, tok_path

LONG_STATE = "the customer wrote again about the same invoice " * 40


def enc(max_len=1024):
    from temis.encoding import Encoder
    return Encoder(tok_path(), max_len)


def test_the_state_appears_once_after_every_question():
    e = enc()
    qs = parse_questions(Q)
    p = e.encode_pack(qs, "state words here")
    state = e._ids("state words here")
    s = p.ids
    starts = [i for i in range(len(s)) if s[i:i + len(state)] == state]
    assert len(starts) == 1
    assert all(pos < starts[0] for pos in p.opt_pos)   # questions first: truncation only cuts the state
    assert s[0] == e.bos and s[-1] == e.eos


def test_markers_per_question_one_for_noul():
    e = enc()
    qs = parse_questions(Q)  # choice (2 options), score (3 levels), noul
    p = e.encode_pack(qs, "s")
    assert len(p.opt_pos) == 2 + 3 + 1
    assert all(p.ids[i] == e.opt for i in p.opt_pos)
    noul = next(i for i, q in enumerate(qs) if q.kind == "noul")
    assert p.slots[noul][0] == -1 and p.slots[noul][1] >= 0   # "false" is the fixed 0, "true" a marker


def test_slots_follow_the_question_order_of_options_whatever_the_shuffle():
    e = enc()
    qs = parse_questions(Q)
    for seed in range(5):
        p = e.encode_pack(qs, "s", rng=random.Random(seed))
        for q, slots in zip(qs, p.slots):
            assert len(slots) == q.n
            if q.kind == "noul":
                continue  # one marker, no option text after it
            for o, m in enumerate(slots):
                if m < 0:
                    continue
                # the text right after marker m is option o's own text
                after = p.ids[p.opt_pos[m] + 1: p.opt_pos[m] + 4]
                from temis.encoding import option_text
                assert after == e._ids(" " + option_text(q, o))[:3]


def test_score_levels_keep_their_order():
    e = enc()
    qs = [q for q in parse_questions(Q) if q.kind == "score"]
    p = e.encode_pack(qs, "s", rng=random.Random(3))
    assert p.slots[0] == sorted(p.slots[0])


def test_truncation_cuts_only_the_state():
    e = enc(max_len=96)
    qs = parse_questions(Q)
    p = e.encode_pack(qs, LONG_STATE)
    assert len(p.ids) == 96 and p.ids[-1] == e.eos
    assert len(p.opt_pos) == 6


def test_too_many_questions_for_one_sequence_are_split():
    from temis.encoding import split_for_packs
    e = enc(max_len=256)
    qs = parse_questions({f"q{i}": {"type": "noul", "instructions": f"Question number {i} " * 6} for i in range(12)})
    groups = split_for_packs(e, qs, max_len=256)
    assert sorted(i for g in groups for i in g) == list(range(12))
    for g in groups:
        p = e.encode_pack([qs[i] for i in g], LONG_STATE, max_len=256)
        assert len(p.ids) - len(e.question_ids_pack([qs[i] for i in g])) >= 64   # room left for the state
    assert len(groups) > 1


def test_gather_turns_marker_logits_into_per_question_logits():
    from temis.model import pack_question_logits
    # two packs: pack 0 has markers [a, b, c], pack 1 has [d]
    marker = torch.tensor([[1.0, 2.0, 3.0], [4.0, -1e4, -1e4]])
    slots = [[[2, 0], [-1, 1]], [[-1, 0]]]   # q0: options (marker 2, marker 0); q1 noul; q2 noul in pack 1
    lg, mask = pack_question_logits(marker, slots, k=3)
    assert lg.shape == (3, 3)
    assert torch.equal(lg[0, :2], torch.tensor([3.0, 1.0]))
    assert torch.equal(lg[1, :2], torch.tensor([0.0, 2.0]))
    assert torch.equal(lg[2, :2], torch.tensor([0.0, 4.0]))
    assert mask.tolist() == [[True, True, False]] * 3


def test_pack_slots_numpy_and_torch_agree():
    from temis.model import pack_question_logits
    from temis.runtime import slots_to_logits
    marker = np.array([0.5, -0.3, 1.2, 2.0], np.float32)
    slots = [[1, 3, 0], [-1, 2]]
    a = slots_to_logits(marker, slots)
    b, _ = pack_question_logits(torch.from_numpy(marker)[None], [slots], k=3)
    for i, x in enumerate(a):
        assert np.allclose(x, b[i, :len(x)].numpy())


@pytest.mark.parametrize("ctx", [None, "some passage"])
def test_context_after_the_state(ctx):
    e = enc()
    p = e.encode_pack(parse_questions(Q), "state", ctx)
    assert (e.ctx in p.ids) == bool(ctx)
