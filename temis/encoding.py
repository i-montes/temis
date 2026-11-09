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

from temis.schema import Question

BOS, EOS, PAD, OPT, CTX = 2, 1, 0, 7, 8  # mmBERT/Gemma ids; <unused0> = option marker, <unused1> = context
MAX_INSTR, MAX_OPTION = 160, 80


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


class Encoder:
    def __init__(self, tokenizer_path: str, max_len: int = 1024) -> None:
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.tok.no_padding()
        self.tok.no_truncation()
        self.max_len = max_len

    def _ids(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode(self, q: Question, state_text: str, context: str | None = None,
               rng: random.Random | None = None, max_len: int | None = None) -> Encoded:
        max_len = max_len or self.max_len
        order = list(range(q.n))
        if rng is not None and q.kind != "score":
            rng.shuffle(order)
        ids = [BOS] + self._ids(f"{q.kind} question: {q.instructions}")[:MAX_INSTR]
        opt_pos = []
        for i in order:
            opt_pos.append(len(ids))
            ids.append(OPT)
            ids += self._ids(" " + option_text(q, i))[:MAX_OPTION]
        ids.append(EOS)
        budget = max_len - len(ids) - 1
        state = self._ids(state_text)
        ctx = ([CTX] + self._ids(" context: " + context)) if context else []
        if ctx:
            keep_state = min(len(state), max(budget * 2 // 3, budget - len(ctx)))
            state = state[:keep_state]
            ctx = ctx[: max(budget - len(state), 0)]
        else:
            state = state[:max(budget, 0)]
        ids += state + ctx + [EOS]
        return Encoded(ids[:max_len], opt_pos, order)
