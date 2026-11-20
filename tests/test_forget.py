from scripts.build_forget import ASK, BLIND_ASK, BLIND_FACTS, FACTS, QID, blind, build
from scripts.eval_clio import load_cases
from temis.schema import parse_questions


def test_build_is_balanced_and_apart_from_the_blind_set_and_from_clios_test():
    rows = build(400, seed=3)
    assert sum(r["target"][1] > 0.5 for r in rows) == 200
    states = [r["case"]["state"] for r in rows]
    assert all(parse_questions(r["case"]["questions"])[0].qid == QID for r in rows)
    assert all("{" not in s["mensaje_de_la_persona"] + s["memoria"] and " de el " not in s["mensaje_de_la_persona"] for s in states)
    assert not {s["mensaje_de_la_persona"] for s in states} & {c["state"]["mensaje_de_la_persona"] for c in blind()}
    assert not set(FACTS) & set(BLIND_FACTS) and not {t for v in ASK.values() for t in v} & {t for v in BLIND_ASK.values() for t in v}
    test = [c["state"] for c in load_cases().build() if c["task"] == "del"]
    assert not {s["mensaje_de_la_persona"].lower().rstrip(".") for s in states} & {s["mensaje_de_la_persona"].lower().rstrip(".") for s in test}
    assert not {s["memoria"].split(": ", 1)[-1] for s in states} & {s["memoria"].split(": ", 1)[-1] for s in test}
