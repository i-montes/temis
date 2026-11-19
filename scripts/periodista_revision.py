"""Temis periodista: ¿Jev resuelve los casos difíciles? Si ni Jev los resuelve, no sirve de profesor para ellos.

  python -m scripts.periodista_revision                  # medición automática + hoja para revisar a mano
  python -m scripts.periodista_revision --calificar      # después de llenar la hoja: acierto de Jev por categoría

1. Automática, donde la respuesta se sabe por construcción (scripts/periodista_build2.py):
     sin_correcto   se quitó el perfil con el nombre exacto: la respuesta correcta es «ninguno»
     apellido_solo  la entidad solo por su apellido: la correcta es el perfil con el nombre completo de la nota
2. A mano: data/periodista/revision/revision_jev.csv, una muestra por categoría (vigencia, roles raros, si un
   cargo es la misma persona, entidades descartadas, y los fallos de Jev en la parte automática). Se llenan las
   columnas `jev_acierta` (si / no / dudoso), `respuesta_correcta` (si Jev falló) y `nota`.

Con --calificar, el acierto de Jev por categoría sale en pantalla y en revision/resumen.json. Una categoría donde
Jev acierte poco se saca del entrenamiento (o se entrena solo con las respuestas de la revisión).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from scripts.periodista_build import Indices, norm
from temis.schema import parse_questions

DIR = Path("data/periodista")
CASOS = DIR / "set2" / "cases.jsonl"
REV = DIR / "revision"
HOJA = REV / "revision_jev.csv"
COLUMNAS = ["n", "categoria", "id", "pregunta_id", "seccion", "nota", "fecha", "link", "entidad", "pregunta",
            "texto_para_decidir", "opciones", "jev_respuesta", "jev_confianza", "jev_acierta", "respuesta_correcta",
            "comentario"]
ROLES_RAROS = {"senalado", "afectado", "autoridad", "aliado_o_rival"}
PERSONA_RAROS = {"fuerza_publica", "armado_ilegal", "justicia_control", "empresario", "sociedad_civil", "ciudadano",
                 "periodista", "experto"}


def leer(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def respuesta(c: dict, dist: dict, qid: str) -> tuple[str, float] | None:
    q = next((q for q in parse_questions(c["questions"]) if q.qid == qid), None)
    d = dist.get(qid)
    if not q or not d:
        return None
    k = max(range(len(d)), key=d.__getitem__)
    return (str(k) if q.kind == "score" else q.options[k]), round(d[k], 2)


def correcta_apellido(c: dict, base: dict | None, perfiles: dict) -> str | None:
    if not base:
        return None
    completo = norm(base["state"]["entidad"]["nombre"])
    for cid, m in (c["meta"].get("candidatos") or {}).items():
        if m["clase"] == "perfil" and norm(perfiles.get(m["id"], "")) == completo:
            return cid
    return None


def texto_para(c: dict, qid: str) -> str:
    s = c["state"]
    partes = [f"[{k}] {v}" for k, v in s.get("oraciones", {}).items()]
    if qid.startswith("ent_fuente_vigente_") or qid.startswith("ent_fuente_"):
        f = s["fuente"][qid.rsplit("_", 1)[1]]
        partes.append(f"PASAJE ({f['fecha']}, «{f['titulo']}»): {f['texto']}")
    if qid == "ent_candidato":
        partes += [f"{k}: {v}" for k, v in s.get("candidatos", {}).items()]
    return "\n".join(partes)


def opciones(c: dict, qid: str) -> str:
    q = c["questions"][qid]
    if q["type"] == "noul":
        return "true = sí / false = no"
    crit = q["criteria"]
    if isinstance(crit, list):
        return " | ".join(f"{i}: {t}" for i, t in enumerate(crit))
    if qid in ("ent_candidato", "ent_oracion_clave"):
        return ", ".join(crit)
    return " | ".join(f"{k}: {v}" for k, v in crit.items())


def hoja_y_automatica(por_cat: int, seed: int) -> None:
    rng = random.Random(seed)
    casos = leer(CASOS)
    dists = {r["id"]: r["dists"] for r in leer(DIR / "labels" / "jev.jsonl")}
    por_id = {c["id"]: c for c in casos}
    idx = Indices(json.loads((DIR / "raw" / "perfiles.json").read_text(encoding="utf-8")),
                  json.loads((DIR / "raw" / "etiquetas.json").read_text(encoding="utf-8")))
    perfiles = {p["id"]: p["nombre"] for p in idx.perfiles}

    # 1. automática
    auto = defaultdict(lambda: {"n": 0, "ok": 0, "conf": []})
    fallos = defaultdict(list)
    for c in casos:
        dif = c["meta"].get("dificil")
        if dif not in ("sin_correcto", "apellido_solo") or c["id"] not in dists:
            continue
        esperado = "ninguno" if dif == "sin_correcto" else correcta_apellido(c, por_id.get(c["id"].rsplit("-", 1)[0]), perfiles)
        r = respuesta(c, dists[c["id"]], "ent_candidato")
        if not esperado or not r:
            continue
        a = auto[dif]
        a["n"] += 1
        a["ok"] += int(r[0] == esperado)
        a["conf"].append(r[1])
        if r[0] != esperado:
            fallos[dif].append((c, esperado))
    print("Medición automática (respuesta conocida por construcción):")
    resumen = {}
    for k, a in auto.items():
        acc = a["ok"] / a["n"] if a["n"] else 0
        resumen[k] = {"n": a["n"], "acierto_jev": round(acc, 3), "confianza_media": round(sum(a["conf"]) / len(a["conf"]), 2)}
        print(f"  {k:14s} n={a['n']:4d}  Jev acierta {acc:.0%}  (confianza media {resumen[k]['confianza_media']})")

    # 2. hoja a mano
    filas = []

    def agregar(cat: str, c: dict, qid: str, esperado: str | None = None) -> None:
        r = respuesta(c, dists.get(c["id"], {}), qid)
        if not r:
            return
        s = c["state"]
        filas.append({"categoria": cat, "id": c["id"], "pregunta_id": qid, "seccion": c["meta"]["seccion"],
                      "nota": s["documento"]["titulo"], "fecha": s["documento"]["fecha"], "link": c["meta"]["link"],
                      "entidad": s["entidad"]["nombre"], "pregunta": c["questions"][qid]["instructions"],
                      "texto_para_decidir": texto_para(c, qid), "opciones": opciones(c, qid),
                      "jev_respuesta": r[0], "jev_confianza": r[1],
                      "comentario": f"esperado por construcción: {esperado}" if esperado else ""})

    def muestra(cat: str, pares: list[tuple[dict, str]]) -> None:
        pares.sort(key=lambda x: (x[0]["id"], x[1]))
        rng.shuffle(pares)
        for c, qid in pares[:por_cat]:
            agregar(cat, c, qid)

    con = [c for c in casos if c["id"] in dists]
    muestra("vigencia", [(c, q) for c in con for q in c["questions"] if q.startswith("ent_fuente_vigente_")])
    muestra("misma_mencion_si", [(c, "ent_misma_mencion") for c in con if "ent_misma_mencion" in c["questions"]
                                 and (respuesta(c, dists[c["id"]], "ent_misma_mencion") or ("",))[0] == "true"])
    muestra("misma_mencion_no", [(c, "ent_misma_mencion") for c in con if "ent_misma_mencion" in c["questions"]
                                 and (respuesta(c, dists[c["id"]], "ent_misma_mencion") or ("",))[0] == "false"])
    muestra("rol_documento_raro", [(c, "ent_rol_documento") for c in con if "ent_rol_documento" in c["questions"]
                                   and (respuesta(c, dists[c["id"]], "ent_rol_documento") or ("",))[0] in ROLES_RAROS])
    muestra("rol_persona_raro", [(c, "ent_rol_persona") for c in con if "ent_rol_persona" in c["questions"]
                                 and (respuesta(c, dists[c["id"]], "ent_rol_persona") or ("",))[0] in PERSONA_RAROS])
    muestra("no_es_entidad", [(c, "ent_es_entidad") for c in con if "ent_es_entidad" in c["questions"]
                              and (respuesta(c, dists[c["id"]], "ent_es_entidad") or ("",))[0] == "false"])
    for dif, lista in fallos.items():   # los fallos de la parte automática: ¿falla Jev o falla la construcción?
        lista.sort(key=lambda x: x[0]["id"])
        rng.shuffle(lista)
        for c, esperado in lista[:por_cat]:
            agregar(f"{dif}_fallo_jev", c, "ent_candidato", esperado)
    REV.mkdir(parents=True, exist_ok=True)
    with HOJA.open("w", encoding="utf-8-sig", newline="") as f:   # con BOM: Excel la abre con tildes
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        for n, fila in enumerate(filas, 1):
            w.writerow({**{k: "" for k in COLUMNAS}, **fila, "n": n})
    (REV / "automatica.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nHoja para revisar a mano: {HOJA} ({len(filas)} filas)")
    print("  por categoría:", dict(Counter(f["categoria"] for f in filas)))


def calificar() -> None:
    with HOJA.open(encoding="utf-8-sig", newline="") as f:
        filas = list(csv.DictReader(f))
    por = defaultdict(Counter)
    for r in filas:
        v = (r.get("jev_acierta") or "").strip().lower()
        if v in ("si", "sí", "no", "dudoso"):
            por[r["categoria"]]["si" if v in ("si", "sí") else v] += 1
    resumen = {}
    print(f"{'categoría':24s} {'revisadas':>9s} {'Jev acierta':>11s} {'dudosas':>7s}")
    for k, c in sorted(por.items()):
        n = sum(c.values())
        claras = c["si"] + c["no"]
        resumen[k] = {"revisadas": n, "acierto_jev": round(c["si"] / claras, 3) if claras else None, "dudosas": c["dudoso"]}
        print(f"{k:24s} {n:9d} {resumen[k]['acierto_jev'] if claras else '-':>11} {c['dudoso']:7d}")
    (REV / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calificar", action="store_true")
    ap.add_argument("--por-categoria", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    calificar() if a.calificar else hoja_y_automatica(a.por_categoria, a.seed)


if __name__ == "__main__":
    main()
