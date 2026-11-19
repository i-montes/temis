import json

from scripts.build_review import MAX_DONE, MAX_ENTRY, QID, TEST, build
from scripts.eval_review import verdict
from temis.schema import parse_questions


def test_build_is_balanced_valid_and_apart_from_the_test():
    rows = build(300, seed=3)
    assert sum(r["target"][1] > 0.5 for r in rows) == 150
    held = [json.loads(ln)["state"] for ln in TEST.read_text().splitlines()]
    requests, dones = {s["person_request"] for s in held}, {tuple(s["done"]) for s in held}
    for r in rows:
        s = r["case"]["state"]
        assert parse_questions(r["case"]["questions"])[0].qid == QID
        assert 1 <= len(s["done"]) <= MAX_DONE and all(len(e) <= MAX_ENTRY for e in s["done"])
        assert all(e.startswith(("ran: ", "wrote: ")) for e in s["done"])
        assert "{" not in s["person_request"]
        assert s["person_request"] not in requests and tuple(s["done"]) not in dones


def test_verdict_thresholds():
    ok = [{"id": f"yes-{i}", "expect": "yes", "p": 0.8} for i in range(3)] + [{"id": "no-1", "expect": "no", "p": 0.49}]
    assert verdict(ok)["passed"]
    assert verdict(ok + [{"id": "no-2", "expect": "no", "p": 0.5}])["passed"]            # one unsure no is allowed
    assert not verdict(ok + [{"id": "no-2", "expect": "no", "p": 0.8}])["passed"]        # a no that fires never is
    assert not verdict(ok + [{"id": f"yes-x{i}", "expect": "yes", "p": 0.79} for i in range(3)])["passed"]
