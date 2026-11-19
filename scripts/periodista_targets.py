"""Temis periodista: objetivos de entrenamiento a partir de las respuestas de Jev.

  python -m scripts.periodista_targets               # después de scripts.periodista_label
  TEMIS_TARGETS=data/periodista/targets.jsonl python -m scripts.train --name per-v1 --init v1112-mrbert-cortos ...

Escribe data/periodista/targets.jsonl con el mismo formato de data/targets.jsonl, con el caso dentro de la fila
(`case`, solo en la primera fila de cada caso), así scripts.train no necesita saber dónde están los casos:
  - cada pregunta de los casos de entrenamiento (set 1, set 2 y trazas, los que existan), con la distribución de
    Jev calibrada con las temperaturas de data/teacher_calibration.json (las mismas de Temis);
  - `schema_id` = la nota, para que la validación de scripts.train (10 % por schema_id) aparte notas enteras;
  - balance por copias, como temis/data/balance.py (scripts.train no usa `weight`): por pregunta, la respuesta más
    probable de Jev que pase de --tope (0,35 en choice, 0,6 en noul) entra solo esa parte de las veces, y la que
    no llegue a --rara (0,08) entra dos veces; score no se toca;
  - en los casos difíciles (meta.dificil), la pregunta difícil con Jev por debajo de --conf-dificil (0,7) no
    entra: queda en data/periodista/held_back.jsonl para revisión (como data/held_back.jsonl en Temis);
  - un repaso de typed-decisions `train` (filas source=td-train de data/targets.jsonl) hasta --repaso del total.
Nunca entran las notas de un benchmark, ni un caso que traiga un pasaje de una nota de un benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from scripts.build_targets import temper
from scripts.periodista_label import huella
from temis.data.balance import _unit
from temis.schema import parse_questions

DIR = Path("data/periodista")
CASOS = [DIR / "cases.jsonl", DIR / "set2" / "cases.jsonl", DIR / "trazas" / "cases.jsonl"]
BENCHS = [DIR / "bench_cases.jsonl", DIR / "set2" / "bench_estilos_cases.jsonl"]


def leer(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def familia(qid: str) -> str:
    for pre in ("ent_fuente_vigente_", "ent_fuente_"):
        if qid.startswith(pre):
            return pre.rstrip("_")
    return qid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repaso", type=float, default=0.2, help="parte del total que es typed-decisions (0 = nada)")
    ap.add_argument("--tope", type=float, default=0.35, help="parte máxima de una respuesta en una pregunta choice")
    ap.add_argument("--tope-noul", type=float, default=0.6)
    ap.add_argument("--rara", type=float, default=0.08, help="por debajo de esta parte, la respuesta entra dos veces")
    ap.add_argument("--sin-balance", action="store_true")
    ap.add_argument("--conf-dificil", type=float, default=0.7,
                    help="en los casos difíciles, la pregunta difícil solo entra si Jev la responde con esta confianza")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    calib = json.loads(Path("data/teacher_calibration.json").read_text(encoding="utf-8"))["temperature"]
    dists = {r["id"]: r for r in leer(DIR / "labels" / "jev.jsonl")}
    bench = {c["meta"]["post_id"] for p in BENCHS for c in leer(p) if "post_id" in c["meta"]}

    # 1. filas sin balance (una por pregunta), y la respuesta más probable de cada una
    filas, sin, fuera, lotes, apartadas = [], 0, 0, Counter(), []
    for ruta in CASOS:
        for c in leer(ruta):
            m = c["meta"]
            if m.get("post_id") in bench or any(f.get("post_id") in bench for f in (m.get("fuente") or {}).values()):
                fuera += 1   # nota de un benchmark, o un pasaje de una (fuga)
                continue
            r = dists.get(c["id"])
            if not r or r.get("qhash", huella(c)) != huella(c):   # sin respuesta, o Jev respondió otras preguntas
                sin += 1
                continue
            lotes[m.get("lote", "set1")] += 1
            caso = {k: c[k] for k in ("id", "state", "questions")}
            for q in parse_questions(c["questions"]):
                d = r["dists"].get(q.qid)
                if not d or len(d) != q.n:
                    continue
                # la pregunta que hace difícil al caso, con Jev dudando: no entra (revisión del 2026-11-19: con
                # confianza >= 0,7 Jev acierta 97-100 % en sin_correcto y apellido_solo; por debajo, 73-88 %)
                clave_dificil = {"sin_correcto": "ent_candidato", "apellido_solo": "ent_candidato",
                                 "vigencia": "ent_fuente_vigente_"}.get(m.get("dificil") or "")
                if clave_dificil and q.qid.startswith(clave_dificil) and max(d) < a.conf_dificil:
                    apartadas.append({"id": c["id"], "qid": q.qid, "dificil": m["dificil"], "jev_confianza": round(max(d), 3)})
                    continue
                top = q.options[max(range(q.n), key=d.__getitem__)]
                # en candidato y oración clave las opciones son ids de posición (c2, o3), no clases: se balancea
                # «ninguno» contra «alguno»
                if q.qid == "ent_candidato":
                    top = "ninguno" if top == "ninguno" else "alguno"
                elif q.qid == "ent_oracion_clave":
                    top = "ninguna" if top == "ninguna" else "alguna"
                filas.append({"fila": {"id": c["id"], "qid": q.qid, "kind": q.kind, "source": "periodista",
                                       "schema_id": f"periodista-{m.get('post_id') or c['id']}",
                                       "target": temper(d, calib.get(q.kind, 1.0)), "teachers": [r["teacher"]]},
                              "caso": caso, "fam": familia(q.qid), "top": top})

    # 2. balance por copias
    reparto = defaultdict(Counter)
    for f in filas:
        reparto[f["fam"]][f["top"]] += 1
    antes = {k: {o: round(n / sum(v.values()), 2) for o, n in v.most_common(3)} for k, v in reparto.items()}
    salida, vistos = [], set()
    for f in filas:
        k = 1
        if not a.sin_balance and f["fila"]["kind"] != "score":
            share = reparto[f["fam"]][f["top"]] / sum(reparto[f["fam"]].values())
            tope = a.tope_noul if f["fila"]["kind"] == "noul" else a.tope
            if share > tope:
                k = 1 if _unit(f"{f['fila']['id']}|{f['fila']['qid']}") < tope / share else 0
            elif share < a.rara:
                k = 2
        for _ in range(k):
            fila = dict(f["fila"])
            if fila["id"] not in vistos:
                fila["case"] = f["caso"]
                vistos.add(fila["id"])
            salida.append(fila)
    propias = len(salida)
    # reparto después (contando las copias)
    despues = defaultdict(Counter)
    tops = {(f["fila"]["id"], f["fila"]["qid"]): (f["fam"], f["top"]) for f in filas}
    for fila in salida:
        fam, top = tops[(fila["id"], fila["qid"])]
        despues[fam][top] += 1
    if a.repaso > 0:
        td = [r for r in leer(Path("data/targets.jsonl")) if r.get("source") == "td-train"]
        n = min(len(td), int(propias * a.repaso / (1 - a.repaso)))
        salida += random.Random(a.seed).sample(td, n)
    with (DIR / "targets.jsonl").open("w", encoding="utf-8") as out:
        for r in salida:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (DIR / "held_back.jsonl").open("w", encoding="utf-8") as out:   # para la cola de revisión, no para entrenar
        for r in apartadas:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"casos por lote: {dict(lotes)}; fuera por benchmark o fuga: {fuera}; sin respuesta de Jev: {sin}")
    print(f"preguntas difíciles apartadas (Jev con confianza < {a.conf_dificil}): {len(apartadas)} -> {DIR / 'held_back.jsonl'}")
    print(f"preguntas: {len(filas)} -> {propias} con el balance; repaso td-train: {len(salida) - propias}")
    for k in sorted(antes):
        d = despues[k]
        print(f"  {k:24s} antes {antes[k]}  después {{{', '.join(f'{o}: {n / sum(d.values()):.2f}' for o, n in d.most_common(3))}}}")


if __name__ == "__main__":
    main()
