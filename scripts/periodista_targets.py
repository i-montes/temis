"""Temis periodista: objetivos de entrenamiento a partir de las respuestas de Jev.

  python -m scripts.periodista_targets               # después de scripts.periodista_label
  TEMIS_TARGETS=data/periodista/targets.jsonl python -m scripts.train --name per-v1 --init v1112-mrbert-cortos

Escribe data/periodista/targets.jsonl con el mismo formato de data/targets.jsonl, con el caso dentro de cada fila
(`case`, solo en la primera fila de cada caso), así scripts.train no necesita saber dónde están los casos:
  - cada pregunta de cada caso de data/periodista/cases.jsonl, con la distribución de Jev calibrada con las
    temperaturas de data/teacher_calibration.json (las mismas de Temis);
  - `schema_id` = la nota, para que la validación de scripts.train (10 % por schema_id) aparte notas enteras;
  - un repaso de typed-decisions `train` (filas source=td-train de data/targets.jsonl) hasta --repaso del total,
    para que el modelo no olvide el formato general.
Las notas del benchmark (data/periodista/bench_cases.jsonl) nunca entran.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from scripts.build_targets import temper
from temis.schema import parse_questions

DIR = Path("data/periodista")


def leer(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repaso", type=float, default=0.2, help="parte del total que es typed-decisions (0 = nada)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    calib = json.loads(Path("data/teacher_calibration.json").read_text(encoding="utf-8"))["temperature"]
    dists = {r["id"]: r for r in leer(DIR / "labels" / "jev.jsonl")}
    bench = {c["meta"]["post_id"] for c in leer(DIR / "bench_cases.jsonl")}
    filas, sin = [], 0
    for c in leer(DIR / "cases.jsonl"):
        if c["meta"]["post_id"] in bench:
            continue
        r = dists.get(c["id"])
        if not r:
            sin += 1
            continue
        # el caso va solo en la primera fila de cada id: scripts.train junta los casos de todas las filas antes de usarlas
        caso = {k: c[k] for k in ("id", "state", "questions")}
        for q in parse_questions(c["questions"]):
            d = r["dists"].get(q.qid)
            if not d or len(d) != q.n:
                continue
            fila = {"id": c["id"], "qid": q.qid, "kind": q.kind, "source": "periodista",
                    "schema_id": f"periodista-{c['meta']['post_id']}",
                    "target": temper(d, calib.get(q.kind, 1.0)), "teachers": [r["teacher"]]}
            if caso:
                fila["case"], caso = caso, None
            filas.append(fila)
    propias = len(filas)
    if a.repaso > 0:
        td = [r for r in leer(Path("data/targets.jsonl")) if r.get("source") == "td-train"]
        n = min(len(td), int(propias * a.repaso / (1 - a.repaso)))
        filas += random.Random(a.seed).sample(td, n)
    with (DIR / "targets.jsonl").open("w", encoding="utf-8") as f:
        for r in filas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"periodista: {propias} preguntas ({sin} casos sin respuesta de Jev); repaso td-train: {len(filas) - propias}")


if __name__ == "__main__":
    main()
