"""Generate synthetic decision cases (schema + states) with a local Ollama model.

Appends to data/synth/cases.jsonl, resumable. Each line:
  {"id", "workflow", "domain", "language", "state", "questions", "source": "synth"}

Usage: python -m scripts.generate --target 4000 --workers 2
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from decisor.data import agent_schemas
from decisor.data.quality import clean_questions, state_ok
from decisor.data.domains import BROAD, CORE
from decisor.llm import chat_json
from decisor.schema import SchemaError, parse_questions

OUT = Path("data/synth/cases.jsonl")
LOCK = threading.Lock()
TEMPLATE_P = 0.8

SCHEMA_FORMAT = {
    "type": "object",
    "properties": {
        "workflow": {"type": "string"},
        "description": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": ["choice", "score", "noul"]},
                    "instructions": {"type": "string"},
                    "options": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}, "description": {"type": "string"}},
                        "required": ["key", "description"]}},
                    "levels": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "type", "instructions"],
            },
        },
    },
    "required": ["workflow", "description", "questions"],
}

STATES_FORMAT = {
    "type": "object",
    "properties": {"states": {"type": "array", "items": {"type": "object", "properties": {
        "kind": {"type": "string"}, "state": {}}, "required": ["state"]}}},
    "required": ["states"],
}

SYS_SCHEMA = """You design typed decision schemas for a small decision model.
A schema has 3 to 5 questions about one piece of input ("state"). Question types:
- choice: pick one of 2-6 options. Each option has a short snake_case key and a one-sentence description.
- score: an ordered rubric of 3-5 levels, lowest first. Each level is a natural-language sentence
  (NOT snake_case, NOT a key) describing what that level means.
- noul: a yes/no STATEMENT about the state that is either true or false, e.g. "The request is urgent." or
  "El remitente ya trabajó antes con nosotros.". NEVER a command or a topic: not "Determine if...",
  not "Evaluate the urgency", not "Verificación de recursos". Include options for "false" and "true"
  describing what each means.
Mix the types (at least one of each when there are 3+ questions). Questions must be answerable from the
state alone, practical, and specific to the domain. Write instructions and descriptions as natural sentences. Write everything (workflow name and keys excepted) in the
requested language. The workflow name is snake_case English."""

SYS_STATES = """You write realistic inputs ("states") for a typed decision schema.
Write varied cases: some clear-cut in each direction, some genuinely ambiguous, some with numbers,
dates, quantities, logs or quotes that matter for the decision, and occasionally misleading details.
A state is either a JSON object with nested fields (like an API payload or record) or a plain text
(an email, a message, a log excerpt, a note). Use the requested language for natural-language text.
Never include the answers. Do not mention the questions."""


def to_typesafe(qs: list[dict]) -> dict:
    out = {}
    for q in qs:
        qid = "".join(c if c.isalnum() or c == "_" else "_" for c in str(q.get("id", "")).strip().lower())[:40] or f"q{len(out)}"
        t = q.get("type")
        item = {"type": t, "instructions": str(q.get("instructions", "")).strip()}
        if t == "choice":
            item["criteria"] = {str(o["key"]).strip(): str(o["description"]).strip()
                                for o in q.get("options") or [] if o.get("key")}
        elif t == "score":
            levels = q.get("levels") or [o.get("description", "") for o in q.get("options") or []]
            item["criteria"] = [str(x).strip() for x in levels if str(x).strip()]
        else:
            opts = {str(o.get("key", "")).lower(): str(o.get("description", "")) for o in q.get("options") or []}
            if "true" in opts and "false" in opts:
                item["criteria"] = {"false": opts["false"], "true": opts["true"]}
        out[qid] = item
    return out


def make_schema(model: str, rng: random.Random) -> dict | None:
    domain = rng.choice(CORE) if rng.random() < 0.45 else rng.choice(BROAD)
    lang = "Spanish" if rng.random() < 0.5 else "English"
    nq = rng.choice([3, 4, 4, 5])
    msg = (f"Domain: {domain}\nLanguage: {lang}\nNumber of questions: {nq}\n"
           f"Variation seed: {rng.randint(0, 10**6)} (make this schema different from typical ones).")
    raw = chat_json(model, [{"role": "system", "content": SYS_SCHEMA}, {"role": "user", "content": msg}],
                    SCHEMA_FORMAT, temperature=0.9, num_predict=1500)
    questions, _, _ = clean_questions(to_typesafe(raw.get("questions") or []))
    if not questions:
        return None
    try:
        parse_questions(questions)
    except SchemaError:
        return None
    return {"workflow": str(raw.get("workflow", "custom"))[:60], "description": raw.get("description", ""),
            "domain": domain, "language": "es" if lang == "Spanish" else "en", "questions": questions}


def make_states(model: str, schema: dict, n: int) -> list:
    qtext = json.dumps(schema["questions"], ensure_ascii=False)
    lang = "Spanish" if schema["language"] == "es" else "English"
    hint = schema.get("state_hint")
    msg = (f"Workflow: {schema['workflow']} — {schema['description']}\nLanguage: {lang}\n"
           + (f"State format: {hint}\n" if hint else "")
           + f"The states will later be judged with these questions (do not answer them): {qtext}\n\n"
           f"Write {n} different states. Make about 3 of every 4 clear-cut: someone careful would pick the "
           f"same answer for every question. Make the rest genuinely hard, not vague. Balance them: for yes/no questions, roughly half the states should "
           f"clearly lead to NO and half to YES; for choices, aim at different options in different states.")
    raw = chat_json(model, [{"role": "system", "content": SYS_STATES}, {"role": "user", "content": msg}],
                    STATES_FORMAT, temperature=0.95, num_predict=3000)
    out = []
    for s in raw.get("states") or []:
        st = s.get("state")
        if isinstance(st, (dict, str)) and state_ok(st):
            out.append(st)
    return out[:n]


def count() -> int:
    return sum(1 for _ in OUT.open()) if OUT.exists() else 0


def worker(model: str, seed: int, states_per_schema: int, target: int) -> None:
    rng = random.Random(seed)
    while count() < target:
        try:
            schema = agent_schemas.sample(rng) if rng.random() < TEMPLATE_P else make_schema(model, rng)
            if not schema:
                continue
            states = []
            for _ in range(max(1, states_per_schema // 4)):
                states += make_states(model, schema, 4)
            sid = uuid.uuid4().hex[:10]
            with LOCK, OUT.open("a") as f:
                for i, st in enumerate(states):
                    f.write(json.dumps({"id": f"syn-{sid}-{i}", "schema_id": sid, "workflow": schema["workflow"],
                                        "domain": schema["domain"], "language": schema["language"],
                                        "state": st, "questions": schema["questions"], "source": "synth"},
                                       ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"[gen] error: {e}", flush=True)
            time.sleep(5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:9b-32k")
    ap.add_argument("--target", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--states-per-schema", type=int, default=8)
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0, c0 = time.time(), count()
    with ThreadPoolExecutor(a.workers) as ex:
        for w in range(a.workers):
            ex.submit(worker, a.model, int(time.time()) + w, a.states_per_schema, a.target)
        while any(not f.done() for f in []) or count() < a.target:
            time.sleep(60)
            c = count()
            rate = (c - c0) / max(time.time() - t0, 1) * 3600
            print(f"[gen] {c}/{a.target} cases ({rate:.0f}/h)", flush=True)


if __name__ == "__main__":
    main()
