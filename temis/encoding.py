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
# (bos, eos, pad, option marker, context marker) by vocabulary, looked up in the model's own tokenizer.
# The markers are reserved tokens the base model never saw, so they carry no prior meaning.
SPECIAL_TOKENS = [
    ("<bos>", "<eos>", "<pad>", "<unused0>", "<unused1>"),                    # mmBERT (Gemma vocabulary)
    ("<s>", "</s>", "<pad>", "<|reserved_token_1|>", "<|reserved_token_2|>"),  # MrBERT, MrBERT-es
]
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


def special_ids(tok: Tokenizer) -> tuple[int, int, int, int, int]:
    for names in SPECIAL_TOKENS:
        ids = [tok.token_to_id(n) for n in names]
        if all(i is not None for i in ids):
            return tuple(ids)  # type: ignore[return-value]
    raise ValueError("unknown vocabulary: add its special tokens to SPECIAL_TOKENS")


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
