"""Turns (question, state, context) into token ids. Shared by training (torch) and the runtime
(ONNX), so it only depends on `tokenizers` and the standard library.

Layout of one sequence (one question):

    <bos> {kind} question: {instructions} <opt> option_1 <opt> option_2 ... <eos>
          {state text} [<ctx> context: {passages}] <eos>

Each <opt> marker's final hidden state is scored to give that option's logit. Options go before the
state so that truncation only ever cuts the state or the context, never an option.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from tokenizers import Tokenizer

from temis.schema import NOUL_DEFAULT, Question

BOS, EOS, PAD, OPT, CTX = 2, 1, 0, 7, 8  # mmBERT/Gemma ids; <unused0> = option marker, <unused1> = context
# (bos, eos, pad, option marker, context marker) by vocabulary, looked up in the model's own tokenizer.
# The markers are reserved tokens the base model never saw, so they carry no prior meaning.
SPECIAL_TOKENS = [
    ("<bos>", "<eos>", "<pad>", "<unused0>", "<unused1>"),                    # mmBERT (Gemma vocabulary)
    ("<s>", "</s>", "<pad>", "<|reserved_token_1|>", "<|reserved_token_2|>"),  # MrBERT, MrBERT-es
]
MAX_INSTR, MAX_OPTION = 160, 80
MIN_STATE = 64  # packed layout: tokens always left for the state (split_for_packs)


@dataclass
class Encoded:
    ids: list[int]
    opt_pos: list[int]   # position of each option's marker, in `order`
    order: list[int]     # order[i] = index into question.options of the i-th marker


def option_text(q: Question, i: int) -> str:
    k, d = q.options[i], q.descriptions[i]
    if q.kind == "noul":
        return f"{'no' if k == 'false' else 'yes'}: {d}"
    if q.kind == "score":
        return f"level {k} of {q.n - 1}: {d}"
    return k if k == d else f"{k}: {d}"


def special_ids(tok: Tokenizer) -> tuple[int, int, int, int, int]:
    for names in SPECIAL_TOKENS:
        ids = [tok.token_to_id(n) for n in names]
        if all(i is not None for i in ids):
            return tuple(ids)  # type: ignore[return-value]
    raise ValueError("unknown vocabulary: add its special tokens to SPECIAL_TOKENS")


@dataclass
class Pack:
    """Every question of a call in one sequence (packed layout, see Encoder.encode_pack)."""
    ids: list[int]
    opt_pos: list[int]         # every marker, in sequence order
    slots: list[list[int]]     # per question, per option in the question's own order: index into
                               # opt_pos, or -1 for a fixed logit of 0 (the "no" of a yes/no question)


def noul_note(q: Question) -> str:
    """A yes/no question has one marker; its criteria go into the text when they say something."""
    if q.descriptions == (NOUL_DEFAULT["false"], NOUL_DEFAULT["true"]):
        return ""
    return f" (yes: {q.descriptions[1]} no: {q.descriptions[0]})"


class Encoder:
    def __init__(self, tokenizer_path: str, max_len: int = 1024) -> None:
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.tok.no_padding()
        self.tok.no_truncation()
        self.max_len = max_len
        self.bos, self.eos, self.pad, self.opt, self.ctx = special_ids(self.tok)
        # SentencePiece vocabularies that add the word-start mark themselves would turn our leading
        # space into an extra token; Gemma's (mmBERT) does not, so there the space is kept.
        self.prefix_space = (len(self.tok.encode(" context", add_special_tokens=False).ids)
                             > len(self.tok.encode("context", add_special_tokens=False).ids))

    def _ids(self, text: str) -> list[int]:
        if self.prefix_space and text.startswith(" "):
            text = text[1:]
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode(self, q: Question, state_text: str, context: str | None = None,
               rng: random.Random | None = None, max_len: int | None = None) -> Encoded:
        max_len = max_len or self.max_len
        order = list(range(q.n))
        if rng is not None and q.kind != "score":
            rng.shuffle(order)
        ids = [self.bos] + self._ids(f"{q.kind} question: {q.instructions}")[:MAX_INSTR]
        opt_pos = []
        for i in order:
            opt_pos.append(len(ids))
            ids.append(self.opt)
            ids += self._ids(" " + option_text(q, i))[:MAX_OPTION]
        ids.append(self.eos)
        budget = max_len - len(ids) - 1
        state = self._ids(state_text)
        ctx = ([self.ctx] + self._ids(" context: " + context)) if context else []
        if ctx:
            keep_state = min(len(state), max(budget * 2 // 3, budget - len(ctx)))
            state = state[:keep_state]
            ctx = ctx[: max(budget - len(state), 0)]
        else:
            state = state[:max(budget, 0)]
        ids += state + ctx + [self.eos]
        return Encoded(ids[:max_len], opt_pos, order)

    def question_ids_pack(self, qs: list[Question], rng: random.Random | None = None) -> list[int]:
        return self._pack_questions(qs, rng)[0]

    def _pack_questions(self, qs: list[Question], rng: random.Random | None = None):
        """(ids after <bos>, marker positions relative to them, slots) for the question blocks."""
        ids: list[int] = []
        opt_pos: list[int] = []
        slots: list[list[int]] = [[] for _ in qs]
        qorder = list(range(len(qs)))
        if rng is not None:
            rng.shuffle(qorder)
        for qi in qorder:
            q = qs[qi]
            if q.kind == "noul":
                ids += self._ids(f"noul question: {q.instructions}")[:MAX_INSTR] + self._ids(noul_note(q))[:MAX_OPTION]
                slots[qi] = [-1, len(opt_pos)]
                opt_pos.append(len(ids))
                ids.append(self.opt)
            else:
                ids += self._ids(f"{q.kind} question: {q.instructions}")[:MAX_INSTR]
                order = list(range(q.n))
                if rng is not None and q.kind != "score":
                    rng.shuffle(order)
                slots[qi] = [0] * q.n
                for o in order:
                    slots[qi][o] = len(opt_pos)
                    opt_pos.append(len(ids))
                    ids.append(self.opt)
                    ids += self._ids(" " + option_text(q, o))[:MAX_OPTION]
            ids.append(self.eos)
        return ids, opt_pos, slots

    def encode_pack(self, qs: list[Question], state_text: str, context: str | None = None,
                    rng: random.Random | None = None, max_len: int | None = None) -> Pack:
        """<bos> {question blocks, each ending in <eos>} {state} [<ctx> context: ...] <eos>.
        Questions first, so truncation only cuts the state or the context (split_for_packs keeps
        at least MIN_STATE tokens for them)."""
        max_len = max_len or self.max_len
        qids, rel, slots = self._pack_questions(qs, rng)
        ids = [self.bos] + qids
        opt_pos = [1 + r for r in rel]
        budget = max_len - len(ids) - 1
        state = self._ids(state_text)
        ctx = ([self.ctx] + self._ids(" context: " + context)) if context else []
        if ctx:
            keep_state = min(len(state), max(budget * 2 // 3, budget - len(ctx)))
            state = state[:keep_state]
            ctx = ctx[: max(budget - len(state), 0)]
        else:
            state = state[:max(budget, 0)]
        ids += state + ctx + [self.eos]
        return Pack(ids[:max_len], opt_pos, slots)


def split_for_packs(enc: Encoder, qs: list[Question], max_len: int) -> list[list[int]]:
    """Question indices grouped so each group's blocks leave at least MIN_STATE tokens for the state
    (greedy, in the given order); a call whose questions fit is a single group."""
    groups: list[list[int]] = []
    used = 0
    budget = max_len - 2 - MIN_STATE
    for i, q in enumerate(qs):
        n = len(enc.question_ids_pack([q]))
        if groups and used + n <= budget:
            groups[-1].append(i)
            used += n
        else:
            groups.append([i])
            used = n
    return groups
