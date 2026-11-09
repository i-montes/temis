"""Decision memory: every decision, its labels, and the precedent index (SQLite + FTS5)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from decisor.schema import Question

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS decisions (
  id TEXT PRIMARY KEY, ts REAL, origin TEXT, workflow TEXT, state TEXT, questions TEXT,
  context TEXT, answers TEXT, model_version TEXT, uncertainty REAL, teacher TEXT,
  status TEXT DEFAULT 'open'
);
CREATE INDEX IF NOT EXISTS decisions_queue ON decisions(status, uncertainty DESC);
CREATE TABLE IF NOT EXISTS labels (
  decision_id TEXT, qid TEXT, label TEXT, labeler TEXT, ts REAL,
  PRIMARY KEY (decision_id, qid, labeler)
);
CREATE TABLE IF NOT EXISTS suggestions (
  decision_id TEXT, qid TEXT, label TEXT, reason TEXT, by TEXT, ts REAL,
  PRIMARY KEY (decision_id, qid, by)
);
CREATE VIRTUAL TABLE IF NOT EXISTS precedents USING fts5(
  sig UNINDEXED, decision_id UNINDEXED, qid UNINDEXED, label UNINDEXED, text
);
"""

WORD = re.compile(r"\w{3,}", re.UNICODE)


def signature(q: Question) -> str:
    raw = json.dumps([q.kind, q.instructions, list(q.options), list(q.descriptions)], ensure_ascii=False)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def fts_query(text: str, max_terms: int = 32) -> str:
    seen, terms = set(), []
    for w in WORD.findall(text.lower()):
        if w not in seen:
            seen.add(w)
            terms.append(f'"{w}"')
        if len(terms) >= max_terms:
            break
    return " OR ".join(terms)


class Store:
    def __init__(self, path: str | Path | None = None) -> None:
        path = path or os.environ.get("DECISOR_DB", "data/decisor.db")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)  # wait for other writers
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute("PRAGMA busy_timeout=30000")
            self.db.executescript(SCHEMA)
            cols = {r[1] for r in self.db.execute("PRAGMA table_info(labels)")}
            if "how" not in cols:  # pick = chosen by hand, accept = took the suggestion
                self.db.execute("ALTER TABLE labels ADD COLUMN how TEXT")
                self.db.commit()
            dcols = {r[1] for r in self.db.execute("PRAGMA table_info(decisions)")}
            if "priority" not in dcols:  # doubt + bonus for the families where the model is weakest
                self.db.execute("ALTER TABLE decisions ADD COLUMN priority REAL")
                self.db.execute("UPDATE decisions SET priority = uncertainty")
                self.db.commit()

    def add_decision(self, *, origin: str, state: Any, questions: dict, answers: dict, model_version: str,
                     uncertainty: float, context: str | None = None, workflow: str = "",
                     teacher: dict | None = None, id: str | None = None) -> str:
        from decisor.data.families import queue_bonus

        did = id or uuid.uuid4().hex[:12]
        priority = round(uncertainty + queue_bonus(workflow), 4)
        with self.lock:
            self.db.execute(
                "INSERT OR IGNORE INTO decisions (id, ts, origin, workflow, state, questions, context, answers,"
                " model_version, uncertainty, teacher, priority) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (did, time.time(), origin, workflow, json.dumps(state, ensure_ascii=False),
                 json.dumps(questions, ensure_ascii=False), context, json.dumps(answers, ensure_ascii=False),
                 model_version, uncertainty, json.dumps(teacher) if teacher else None, priority))
            self.db.commit()
        return did

    def _row(self, r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in ("state", "questions", "answers", "teacher"):
            if d.get(k):
                d[k] = json.loads(d[k])
        d["labels"] = {x["qid"]: x["label"] for x in self.db.execute(
            "SELECT qid, label FROM labels WHERE decision_id=? AND labeler='user'", (d["id"],))}
        # one suggestion per question: Claude's own review (it has reasons) wins over other reviewers
        d["suggestions"] = {x["qid"]: {"label": x["label"], "reason": x["reason"], "by": x["by"]} for x in self.db.execute(
            "SELECT qid, label, reason, by FROM suggestions WHERE decision_id=? ORDER BY by = 'claude', ts", (d["id"],))}
        return d

    def get(self, did: str) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT * FROM decisions WHERE id=?", (did,)).fetchone()
            return self._row(r) if r else None

    def queue(self, limit: int = 1, exclude: list[str] | None = None) -> list[dict]:
        ex = exclude or []
        q = "SELECT * FROM decisions WHERE status='open'"
        if ex:
            q += f" AND id NOT IN ({','.join('?' * len(ex))})"
        # cases Claude already reviewed come first, then by doubt
        q += (" ORDER BY EXISTS (SELECT 1 FROM suggestions s WHERE s.decision_id = decisions.id) DESC,"
              " COALESCE(priority, uncertainty) DESC, uncertainty DESC, ts DESC LIMIT ?")
        with self.lock:
            return [self._row(r) for r in self.db.execute(q, (*ex, limit))]

    def recent(self, limit: int = 50, origin: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM decisions", []
        if origin:
            q += " WHERE origin=?"
            args.append(origin)
        q += " ORDER BY ts DESC LIMIT ?"
        with self.lock:
            return [self._row(r) for r in self.db.execute(q, (*args, limit))]

    def label(self, did: str, labels: dict[str, str], questions: list[Question], state_text: str,
              labeler: str = "user", how: dict[str, str] | None = None) -> None:
        by = {q.qid: q for q in questions}
        with self.lock:
            for qid, lab in labels.items():
                q = by.get(qid)
                if q is None or lab not in q.options:
                    raise ValueError(f"invalid label {lab!r} for {qid}")
                self.db.execute("INSERT OR REPLACE INTO labels (decision_id, qid, label, labeler, ts, how) VALUES (?,?,?,?,?,?)",
                                (did, qid, lab, labeler, time.time(), (how or {}).get(qid, "pick")))
                self.db.execute("DELETE FROM precedents WHERE decision_id=? AND qid=?", (did, qid))
                self.db.execute("INSERT INTO precedents (sig, decision_id, qid, label, text) VALUES (?,?,?,?,?)",
                                (signature(q), did, qid, lab, state_text[:4000]))
            self.db.execute("UPDATE decisions SET status='labeled' WHERE id=?", (did,))
            self.db.commit()

    def suggest(self, did: str, qid: str, label: str, reason: str, by: str = "claude") -> None:
        """An advisory suggestion (Claude's review). Never a label: not gold, not a precedent, not training data."""
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO suggestions VALUES (?,?,?,?,?,?)",
                            (did, qid, label, reason[:300], by, time.time()))
            self.db.commit()

    def skip(self, did: str) -> None:
        with self.lock:
            self.db.execute("UPDATE decisions SET status='skipped' WHERE id=?", (did,))
            self.db.commit()

    def precedents(self, q: Question, state_text: str, k: int = 5, exclude: str | None = None) -> list[dict]:
        query = fts_query(state_text)
        if not query:
            return []
        with self.lock:
            try:
                rows = self.db.execute(
                    "SELECT decision_id, qid, label, bm25(precedents) AS s FROM precedents "
                    "WHERE precedents MATCH ? AND sig=? ORDER BY s LIMIT ?", (query, signature(q), k + 1)).fetchall()
            except sqlite3.OperationalError:
                return []
        return [dict(r) for r in rows if r["decision_id"] != exclude][:k]

    def stats(self) -> dict:
        with self.lock:
            c = lambda q, *a: self.db.execute(q, a).fetchone()[0]  # noqa: E731
            return {"decisions": c("SELECT count(*) FROM decisions"),
                    "open": c("SELECT count(*) FROM decisions WHERE status='open'"),
                    "labeled": c("SELECT count(*) FROM decisions WHERE status='labeled'"),
                    "skipped": c("SELECT count(*) FROM decisions WHERE status='skipped'"),
                    "user_labels": c("SELECT count(*) FROM labels WHERE labeler='user'"),
                    "doubtful": c("SELECT count(*) FROM decisions WHERE status='open' AND uncertainty>=0.5")}

    def user_labeled(self) -> list[dict]:
        with self.lock:
            ids = [r[0] for r in self.db.execute("SELECT DISTINCT decision_id FROM labels WHERE labeler='user'")]
        return [d for d in (self.get(i) for i in ids) if d]


def _argmax(probs: dict | list | None, options: tuple[str, ...]) -> tuple[str | None, float]:
    """Top option and its probability from a stored answer (dict) or teacher vector (list)."""
    if not probs:
        return None, 0.0
    if isinstance(probs, list):
        if len(probs) != len(options):
            return None, 0.0
        j = max(range(len(probs)), key=probs.__getitem__)
        return options[j], float(probs[j])
    k = max(probs, key=lambda x: probs[x])
    return str(k), float(probs[k])


def agreement(store: Store, recent: int = 25) -> dict:
    """How often the model's suggestion (and the teacher, Jev) matches what the user chose."""
    from collections import defaultdict

    from decisor.schema import parse_questions

    with store.lock:
        rows = store.db.execute(
            "SELECT l.decision_id, l.qid, l.label, l.how, l.ts, d.questions, d.answers, d.teacher, d.workflow, "
            "d.model_version, d.state, s.label AS claude FROM labels l JOIN decisions d ON d.id = l.decision_id "
            "LEFT JOIN suggestions s ON s.decision_id = l.decision_id AND s.qid = l.qid AND s.by = 'claude' "
            "WHERE l.labeler='user' ORDER BY l.ts DESC").fetchall()
    total = defaultdict(lambda: {"n": 0, "model": 0, "jev": 0, "n_jev": 0, "claude": 0, "n_claude": 0})
    buckets = defaultdict(lambda: [0, 0])
    disagreements = []
    for r in rows:
        qs = {q.qid: q for q in parse_questions(json.loads(r["questions"]))}
        q = qs.get(r["qid"])
        if q is None:
            continue
        ans = (json.loads(r["answers"]) if r["answers"] else {}).get(r["qid"]) or {}
        teacher = json.loads(r["teacher"]) if r["teacher"] else {}
        m_lab, m_conf = _argmax(ans.get("probabilities"), q.options)
        j_lab, _ = _argmax((teacher.get("jev") or {}).get(r["qid"]), q.options)
        user = r["label"]
        keys = ["all", f"type:{q.kind}", f"how:{r['how'] or 'pick'}", f"family:{r['workflow'] or '-'}",
                f"model:{r['model_version'] or '-'}"]
        for k in keys:
            t = total[k]
            if m_lab is not None:
                t["n"] += 1
                t["model"] += m_lab == user
            if j_lab is not None:
                t["n_jev"] += 1
                t["jev"] += j_lab == user
            if r["claude"] is not None:
                t["n_claude"] += 1
                t["claude"] += r["claude"] == user
        if m_lab is not None and (r["how"] or "pick") == "pick":
            b = "<50%" if m_conf < 0.5 else "50-70%" if m_conf < 0.7 else "70-90%" if m_conf < 0.9 else ">=90%"
            buckets[b][0] += 1
            buckets[b][1] += m_lab == user
            if m_lab != user and len(disagreements) < recent:
                disagreements.append({"decision": r["decision_id"], "qid": r["qid"], "instructions": q.instructions,
                                      "kind": q.kind, "model": m_lab, "confidence": round(m_conf, 3), "user": user,
                                      "jev": j_lab, "claude": r["claude"], "family": r["workflow"], "ts": r["ts"]})

    def fmt(t):
        return {"n": t["n"], "model": round(t["model"] / t["n"], 4) if t["n"] else None,
                "jev": round(t["jev"] / t["n_jev"], 4) if t["n_jev"] else None,
                "claude": round(t["claude"] / t["n_claude"], 4) if t["n_claude"] else None,
                "n_claude": t["n_claude"]}

    groups = defaultdict(dict)
    for k, t in total.items():
        g, _, name = k.partition(":")
        groups[g if name else "all"][name or "all"] = fmt(t)
    order = ["<50%", "50-70%", "70-90%", ">=90%"]
    return {"all": groups["all"].get("all", fmt(total["all"])),
            "by_type": groups.get("type", {}), "by_how": groups.get("how", {}),
            "by_family": dict(sorted(groups.get("family", {}).items(), key=lambda kv: -kv[1]["n"])),
            "by_model": groups.get("model", {}),
            "by_confidence": {b: {"n": buckets[b][0], "model": round(buckets[b][1] / buckets[b][0], 4) if buckets[b][0] else None}
                              for b in order},
            "disagreements": disagreements}
