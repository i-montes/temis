"""Temis periodista, set 2: más notas, más estilos, casos raros y casos difíciles.

  python -m scripts.periodista_fetch --solo set2       # antes: notas de varias secciones -> raw/notas_set2.jsonl
  python -m scripts.periodista_build2

Usa las funciones del set 1 (scripts/periodista_build.py, que no cambia) y añade:

- Estilos: notas de Nacional (otras que las del set 1), En Vivo, Opinión, Red de Expertos, Detector de Mentiras,
  Silla Académica y Amazonía. `documento.tipo` dice qué clase de texto es.
- Casos raros: al elegir las entidades de una nota pesan más las que aparecen en oraciones de roles poco
  frecuentes (acusados, víctimas, fuerza pública, grupos armados, jueces, empresarios, líderes sociales…);
  `ent_misma_mencion` usa la mitad de las veces el cargo más cercano a la entidad (en el set 1 casi siempre era
  de otra persona); más candidatos dudosos para `ent_es_entidad`.
- Casos difíciles armados por código (`meta.dificil`):
    sin_correcto    el perfil con el nombre exacto se quita de los candidatos: quedan homónimos (respuesta: ninguno)
    apellido_solo   la entidad solo por su apellido, con todos los perfiles que lo comparten
    vigencia        pasajes de hace 2+ años con la pregunta ent_fuente_vigente_fN (¿sigue vigente su cargo?)
- Benchmark de estilos (data/periodista/set2/bench_estilos_cases.jsonl): notas de 2025-2026 apartadas de cada
  sección nueva; nunca se entrenan.
- Sin fugas: ningún pasaje de la fuente sale de una nota de un benchmark (ni el del set 1 ni el de estilos).

Escribe data/periodista/set2/cases.jsonl y bench_estilos_cases.jsonl (sin respuestas: scripts.periodista_label).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from scripts.periodista_build import (CARGO, LARGO_BAJADA, LARGO_CANDIDATO, LARGO_FUENTE, LARGO_ORACION, MAX_CANDIDATOS,
                                      MAX_FUENTE, MAX_ORACIONES, TAREA, Indices, candidatos_de_nota, cortar, leer_jsonl,
                                      noul, norm, preguntas, tipo_codigo, trozo_con)

RAW = Path("data/periodista/raw")
OUT = Path("data/periodista/set2")
SET1 = Path("data/periodista")

TIPO_DOC = {
    "nacional": "nota",
    "en_vivo": "nota corta de En Vivo (cubrimiento minuto a minuto)",
    "opinion": "columna de opinión",
    "red_de_expertos": "columna de un experto (Red de Expertos)",
    "detector": "chequeo del Detector de Mentiras",
    "silla_academica": "resumen de una investigación académica (Silla Académica)",
    "amazonia": "nota de Silla Amazonía",
}
# notas de entrenamiento por sección y notas apartadas para el benchmark de estilos
CUOTA = {"nacional": 300, "en_vivo": 150, "opinion": 120, "red_de_expertos": 120, "detector": 120,
         "silla_academica": 45, "amazonia": 45}
BENCH_POR_SECCION = 6
MAX_ENTIDADES, MAX_ENTIDADES_BENCH = 8, 5

RAROS = re.compile(
    r"(acusad|investigad|capturad|imputad|denunci|condenad|procesad|señalad|cuestionad|sancionad|destituid|"
    r"suspendid|inhabilitad|v[ií]ctima|asesinad|amenazad|desplazad|secuestrad|herid|afectad|damnificad|"
    r"alias|disidencia|guerriller|paramilitar|clan del golfo|\beln\b|coronel|general|comandante|polic[ií]a|"
    r"ej[eé]rcito|contratista|empresari|gerente|l[ií]der social|lideresa|sindica|comunidad|ind[ií]gena|"
    r"campesin|magistrad|juez|jueza|fiscal)", re.I)


class Archivo:
    """Pasajes de notas anteriores con un índice invertido (el set 2 tiene ~10 mil notas: recorrerlas todas por
    cada entidad era lento). Nunca devuelve notas de un benchmark."""

    def __init__(self, notas: list[dict], prohibidas: set[int]):
        self.notas = [n for n in notas if n["id"] not in prohibidas]
        self.texto = [norm(" ".join([n["titulo"]] + n["parrafos"])) for n in self.notas]
        self.idx = defaultdict(set)
        for i, t in enumerate(self.texto):
            for w in set(t.split()):
                if len(w) >= 4:
                    self.idx[w].add(i)

    def pasajes(self, g: dict, nota: dict, rng: random.Random) -> list[dict]:
        clave = g["norm"]
        toks = [w for w in clave.split() if len(w) >= 4]
        if len(clave) < 4 or not toks:
            return []
        cand = set.intersection(*(self.idx.get(w, set()) for w in toks))
        patron = re.compile(rf"(?<![\w]){re.escape(clave)}(?![\w])")
        en_titulo, en_cuerpo = [], []
        for i in cand:
            n = self.notas[i]
            if n["id"] == nota["id"] or n["fecha"] >= nota["fecha"] or not patron.search(self.texto[i]):
                continue
            (en_titulo if patron.search(norm(n["titulo"])) else en_cuerpo).append(n)
        en_cuerpo.sort(key=lambda n: n["id"])
        rng.shuffle(en_cuerpo)
        en_titulo.sort(key=lambda n: n["fecha"], reverse=True)
        elegidas = en_titulo[:1] + en_cuerpo[: MAX_FUENTE - min(1, len(en_titulo))]
        sal = []
        for n in elegidas:
            trozo = next((p for p in n["parrafos"] if patron.search(norm(p))), n["parrafos"][0])
            sal.append({"titulo": n["titulo"], "fecha": n["fecha"], "texto": trozo_con(trozo, g["menciones"], LARGO_FUENTE),
                        "_id": n["id"], "_en_titulo": n in en_titulo})
        rng.shuffle(sal)
        return sal


def dias(a: str, b: str) -> int:
    return (date.fromisoformat(a) - date.fromisoformat(b)).days


def estado_base(nota: dict, g: dict, ors: list[str], ks: list[int]) -> dict:
    primera = ors[g["oraciones"][0]]
    return {
        "tarea": TAREA,
        "documento": {"tipo": TIPO_DOC.get(nota.get("seccion", "nacional"), "nota"), "titulo": nota["titulo"],
                      "fecha": nota["fecha"], "medio": "La Silla Vacía", "bajada": cortar(nota["bajada"], LARGO_BAJADA)},
        "entidad": {"mencion": next((m for m in g["menciones"] if m in primera), g["nombre"]), "nombre": g["nombre"],
                    "otras_menciones": [m for m in dict.fromkeys(g["menciones"]) if m != g["nombre"]][:3]},
        "oraciones": {f"o{k}": cortar(ors[k], LARGO_ORACION) for k in ks},
    }


def poner_candidatos(estado: dict, cands: list[tuple[str, dict]], rng: random.Random) -> dict:
    rng.shuffle(cands)
    estado["candidatos"], meta = {}, {}
    for n, (clase, c) in enumerate(cands, 1):
        txt = (f"Perfil del «Quién es quién»: {c['nombre']}. {c['descripcion']}" if clase == "perfil"
               else f"Etiqueta de La Silla «{c['nombre']}» ({c['total']} notas). {c['descripcion']}")
        estado["candidatos"][f"c{n}"] = cortar(txt, LARGO_CANDIDATO)
        meta[f"c{n}"] = {"clase": clase, "id": c["id"], "link": c["link"]}
    if not estado["candidatos"]:
        del estado["candidatos"]
    return meta


def cargo_para(g: dict, ors: list[str], rng: random.Random) -> str | None:
    """La mitad de las veces el cargo más cercano a la entidad (misma oración o la siguiente), si lo hay."""
    cerca = []
    for k in g["oraciones"][:3]:
        for kk in (k, k + 1):
            if kk < len(ors):
                cerca += [m.group(0) for m in CARGO.finditer(ors[kk])]
    todos = sorted({m.group(0) for o in ors for m in CARGO.finditer(o)},
                    key=lambda s: (-len(s), s))   # desempate por texto: sin él el orden dependía de PYTHONHASHSEED
    if cerca and rng.random() < 0.5:
        return rng.choice(cerca)
    if todos and rng.random() < 0.6:
        return rng.choice(todos[:4])
    return None


def casos_de_nota(nota: dict, idx: Indices, archivo: Archivo, rng: random.Random, max_ent: int,
                  dificiles: bool) -> list[dict]:
    grupos, ors, _ = candidatos_de_nota(nota, idx)
    if not grupos:
        return []
    for g in grupos:   # casos raros: más peso a las entidades de oraciones con roles poco frecuentes
        if any(RAROS.search(ors[k]) for k in g["oraciones"][:MAX_ORACIONES]):
            g["puntos"] += 3
    grupos.sort(key=lambda g: (-g["puntos"], g["oraciones"][0]))
    elegidos = grupos[:max_ent]
    dudosos = [g for g in grupos[max_ent:] if g["puntos"] <= 0]
    if dudosos and rng.random() < 0.8:
        elegidos.append(rng.choice(dudosos))
    pref = "per2"
    sal = []
    for j, g in enumerate(elegidos):
        ks = g["oraciones"][:MAX_ORACIONES]
        estado = estado_base(nota, g, ors, ks)
        cands = [("perfil", p) for p in g["parecidos"][:MAX_CANDIDATOS]]
        if g["etiqueta"]:
            cands.append(("etiqueta", g["etiqueta"]))
        meta_c = poner_candidatos(estado, list(cands), rng)
        pas = archivo.pasajes(g, nota, rng)
        estado["fuente"] = {f"f{n}": {k: v for k, v in p.items() if not k.startswith("_")} for n, p in enumerate(pas)}
        if not estado["fuente"]:
            del estado["fuente"]
        tipo = tipo_codigo(g, idx)
        cargo = cargo_para(g, ors, rng) if tipo in ("persona", None) else None
        qs = preguntas(estado, tipo, cargo)
        viejos = [f"f{n}" for n, p in enumerate(pas) if dias(nota["fecha"], p["fecha"]) >= 730]
        if tipo in ("persona", "organizacion", None):
            for k in viejos:
                qs[f"ent_fuente_vigente_{k}"] = noul(
                    f"Lo que `fuente.{k}` dice del cargo o la situación de `entidad` sigue vigente en `documento.fecha`.")
        meta = {"lote": "set2", "seccion": nota.get("seccion", "nacional"), "post_id": nota["id"], "link": nota["link"],
                "fecha": nota["fecha"], "tipo_codigo": tipo, "etiqueta_de_la_nota": g["de_nota"], "candidatos": meta_c,
                "fuente": {f"f{n}": {"post_id": p["_id"], "nombre_en_titulo": p["_en_titulo"]} for n, p in enumerate(pas)},
                "dificil": "vigencia" if viejos and tipo in ("persona", "organizacion", None) else None}
        sal.append(caso(f"{pref}-{nota['id']}-{j}", estado, qs, meta))
        if not dificiles:
            continue
        # sin_correcto: se quita el perfil con el nombre exacto (y la etiqueta con ese nombre); quedan homónimos
        exactos = [p for p in g["parecidos"] if norm(p["nombre"]) == g["norm"]]
        otros = [p for p in g["parecidos"] if norm(p["nombre"]) != g["norm"]]
        if exactos and otros:
            e2 = estado_base(nota, g, ors, ks)
            m2 = poner_candidatos(e2, [("perfil", p) for p in otros[:MAX_CANDIDATOS]], rng)
            q2 = {k: v for k, v in preguntas(e2, None, None).items() if k in ("ent_candidato", "ent_tipo")}
            sal.append(caso(f"{pref}-{nota['id']}-{j}-sc", e2, q2, dict(meta, candidatos=m2, fuente={}, dificil="sin_correcto")))
        # apellido_solo: oraciones donde aparece solo el apellido, con todos los perfiles que lo comparten
        toks = g["nombre"].split()
        if tipo == "persona" and len(toks) >= 2:
            apellido = toks[-1]
            solo = [k for k in g["oraciones"] if apellido in ors[k] and g["nombre"] not in ors[k]][:MAX_ORACIONES]
            mismos = idx.por_token.get(norm(apellido), [])
            if solo and len(mismos) >= 2:
                g2 = dict(g, nombre=apellido, menciones=[apellido])
                e3 = estado_base(nota, g2, ors, solo)
                e3["entidad"]["otras_menciones"] = []
                elegidos_p = ([p for p in mismos if norm(p["nombre"]) == g["norm"]][:1]
                              + [p for p in mismos if norm(p["nombre"]) != g["norm"]][:MAX_CANDIDATOS])
                m3 = poner_candidatos(e3, [("perfil", p) for p in elegidos_p], rng)
                q3 = {k: v for k, v in preguntas(e3, None, None).items()
                      if k in ("ent_candidato", "ent_es_entidad", "ent_oracion_clave")}
                sal.append(caso(f"{pref}-{nota['id']}-{j}-ap", e3, q3, dict(meta, candidatos=m3, fuente={}, dificil="apellido_solo")))
    return sal


def mal_armados(casos: list[dict], idx: Indices) -> set[str]:
    """Casos difíciles que la construcción deja mal (revisión de 2026-11-19, scripts.periodista_revision):
    sin_correcto  queda otro perfil con el nombre completo de la misma persona («Ana Teresa Bernal» -> «Ana Teresa
                  Bernal Montañez»): «ninguno» no es la respuesta. 66 de 312.
    apellido_solo el apellido quedó fuera del trozo visible de las oraciones (cortadas a LARGO_ORACION). 6 de 407.
    Se quitan al final, sin tocar el azar: los demás casos salen idénticos."""
    por_id = {c["id"]: c for c in casos}
    nombres = {p["id"]: p["nombre"] for p in idx.perfiles}
    malos = set()
    for c in casos:
        d = c["meta"].get("dificil")
        if d == "sin_correcto":
            base = set(norm(por_id[c["id"].rsplit("-", 1)[0]]["state"]["entidad"]["nombre"]).split())
            if any(base <= set(norm(nombres.get(m["id"], "")).split())
                   for m in c["meta"]["candidatos"].values() if m["clase"] == "perfil"):
                malos.add(c["id"])
        elif d == "apellido_solo":
            ap = c["state"]["entidad"]["nombre"]
            if not any(ap in o for o in c["state"]["oraciones"].values()):
                malos.add(c["id"])
    return malos


def caso(cid: str, estado: dict, qs: dict, meta: dict) -> dict:
    return {"id": cid, "schema_id": "periodista_entidad", "family": "periodista_entidad",
            "workflow": f"entidad_{meta['seccion']}", "language": "es",
            "domain": "periodismo: entidades de una nota de La Silla Vacía, su contexto y su rol",
            "state": estado, "questions": qs, "meta": meta}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--escala", type=float, default=1.0, help="multiplica las cuotas de notas por sección")
    ap.add_argument("--salida", default=str(OUT), help="carpeta de salida (para comparar dos armados)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    idx = Indices(json.loads((RAW / "perfiles.json").read_text(encoding="utf-8")),
                  json.loads((RAW / "etiquetas.json").read_text(encoding="utf-8")))
    set1 = leer_jsonl(RAW / "notas.jsonl")
    nuevas = leer_jsonl(RAW / "notas_set2.jsonl")
    usadas1 = {c["meta"]["post_id"] for f in ("cases.jsonl", "bench_cases.jsonl") for c in leer_jsonl(SET1 / f)}
    bench1 = {c["meta"]["post_id"] for c in leer_jsonl(SET1 / "bench_cases.jsonl")}

    por_sec = defaultdict(list)
    for n in nuevas:
        if n["id"] not in usadas1 and len(n["parrafos"]) <= 60:
            por_sec[n["seccion"]].append(n)
    for v in por_sec.values():
        v.sort(key=lambda n: n["id"])
        rng.shuffle(v)
    # benchmark de estilos: notas recientes de cada sección nueva (Nacional ya tiene el del set 1)
    bench2 = []
    for sec in CUOTA:
        if sec == "nacional":
            continue
        recientes = [n for n in por_sec[sec] if n["fecha"] >= "2025-01-01"]
        bench2 += recientes[:BENCH_POR_SECCION]
    b2 = {n["id"] for n in bench2}
    # entrenamiento: cuota por sección, repartida por año
    docs = []
    for sec, cuota in CUOTA.items():
        resto = [n for n in por_sec[sec] if n["id"] not in b2]
        anios = sorted({n["fecha"][:4] for n in resto})
        colas = {y: [n for n in resto if n["fecha"][:4] == y] for y in anios}
        tomadas, objetivo = [], int(cuota * a.escala)
        while len(tomadas) < objetivo and any(colas.values()):
            for y in anios:
                if colas[y] and len(tomadas) < objetivo:
                    tomadas.append(colas[y].pop())
        docs += tomadas
    # archivo de pasajes: todas las notas, menos las de los dos benchmarks
    archivo = Archivo(set1 + nuevas, prohibidas=bench1 | b2)
    train = [c for n in docs for c in casos_de_nota(n, idx, archivo, rng, MAX_ENTIDADES, dificiles=True)]
    malos = mal_armados(train, idx)
    train = [c for c in train if c["id"] not in malos]
    print(f"difíciles mal armados que se quitan: {len(malos)}")
    bench = [c for n in bench2 for c in casos_de_nota(n, idx, archivo, rng, MAX_ENTIDADES_BENCH, dificiles=False)]
    for c in bench:
        c["id"] = c["id"].replace("per2-", "per2b-", 1)
    for nombre, cs in (("cases.jsonl", train), ("bench_estilos_cases.jsonl", bench)):
        with (out / nombre).open("w", encoding="utf-8") as f:
            for c in cs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"entrenamiento: {len(docs)} notas, {len(train)} casos, {sum(len(c['questions']) for c in train)} preguntas")
    print("  por sección:", dict(Counter(c["meta"]["seccion"] for c in train)))
    print("  difíciles:", dict(Counter(c["meta"]["dificil"] for c in train if c["meta"]["dificil"])))
    print(f"benchmark de estilos: {len(bench2)} notas, {len(bench)} casos, {sum(len(c['questions']) for c in bench)} preguntas")
    print("  por sección:", dict(Counter(c["meta"]["seccion"] for c in bench)))


if __name__ == "__main__":
    main()
