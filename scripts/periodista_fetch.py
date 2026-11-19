"""Temis periodista: baja de la API pública del WordPress de La Silla Vacía lo que necesita la base de entidades.

  python -m scripts.periodista_fetch                 # todo (unas 60 peticiones, ~2 minutos)
  python -m scripts.periodista_fetch --solo notas    # solo una parte: perfiles | etiquetas | notas
  python -m scripts.periodista_fetch --solo set2     # set 2: más notas de Nacional y otras secciones (raw/notas_set2.jsonl)

Deja en data/periodista/raw/ (no se sube al repo; se regenera con este script):
  perfiles.json   perfiles del «Quién es quién» (categoría 5027): id, slug, link, nombre, descripción, fecha
  etiquetas.json  etiquetas con más notas: id, nombre, slug, link, total, descripción
  notas.jsonl     notas de Nacional y sus regionales, repartidas por año: id, fecha, link, título, bajada,
                  etiquetas y párrafos en texto plano

Solo usa la biblioteca estándar. Va despacio a propósito (una petición cada ~1,5 s) y reintenta los 429.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

WP = "https://www.lasillavacia.com/wp-json/wp/v2"
RAW = Path("data/periodista/raw")
QUIEN = 5027
# Nacional y sus regionales (Bogotá, Caribe, Santanderes, Pacífico, Antioquia, Región Sur)
NACIONAL = [4924, 4926, 4925, 4929, 4927, 4928, 4930]
PAUSA = 1.5


def get(path: str, reintentos: int = 5) -> tuple[list | dict, dict]:
    url = f"{WP}/{path}"
    for i in range(reintentos):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (temis-periodista)",
                                                   "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                datos = json.loads(r.read().decode("utf-8"))
                time.sleep(PAUSA)
                return datos, dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504) and i < reintentos - 1:
                time.sleep(10 * (i + 1))
                continue
            if e.code == 400:   # página fuera de rango
                return [], {}
            raise
    raise RuntimeError(url)


def texto(h: str) -> str:
    h = re.sub(r"<(script|style|figure|blockquote)[^>]*>.*?</\1>", " ", h or "", flags=re.S | re.I)
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", h)).split())


RUIDO = re.compile(r"^(foto|fotos|imagen|video|lea también|le puede interesar|lea:|vea:|escuche|suscríbase|"
                   r"este artículo|esta nota|únase|haga parte|apoye)", re.I)


def parrafos(h: str) -> list[str]:
    h = re.sub(r"<(script|style|figure|blockquote)[^>]*>.*?</\1>", " ", h or "", flags=re.S | re.I)
    sal = []
    for bloque in re.findall(r"<(?:p|li|h2|h3)[^>]*>(.*?)</(?:p|li|h2|h3)>", h, flags=re.S | re.I):
        t = texto(bloque)
        if len(t) >= 60 and not RUIDO.match(t):
            sal.append(t)
    return sal


def perfiles() -> None:
    out, page = [], 1
    while True:
        d, _ = get(f"posts?categories={QUIEN}&per_page=100&page={page}&_fields=id,date,slug,link,title,excerpt")
        if not d:
            break
        for p in d:
            out.append({"id": p["id"], "slug": p["slug"], "link": p["link"], "fecha": p["date"][:10],
                        "titulo": texto(p["title"]["rendered"]), "descripcion": texto(p["excerpt"]["rendered"])})
        print(f"[perfiles] {len(out)}", flush=True)
        page += 1
    (RAW / "perfiles.json").write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")


def etiquetas(paginas: int = 20) -> None:
    out = []
    for page in range(1, paginas + 1):
        d, _ = get(f"tags?orderby=count&order=desc&per_page=100&page={page}&_fields=id,name,slug,link,count,description")
        if not d:
            break
        out += [{"id": t["id"], "nombre": texto(t["name"]), "slug": t["slug"], "link": t["link"], "total": t["count"],
                 "descripcion": texto(t.get("description") or "")} for t in d]
        print(f"[etiquetas] {len(out)}", flush=True)
        if d[-1]["count"] < 3:
            break
    (RAW / "etiquetas.json").write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")


def notas(desde: int, hasta: int, por_anio: int) -> None:
    cats = ",".join(map(str, NACIONAL))
    vistos, n = set(), 0
    with (RAW / "notas.jsonl").open("w", encoding="utf-8") as f:
        for anio in range(hasta, desde - 1, -1):
            for page in range(1, por_anio // 100 + 1):
                d, _ = get(f"posts?categories={cats}&categories_exclude={QUIEN}&per_page=100&page={page}"
                           f"&after={anio}-01-01T00:00:00&before={anio + 1}-01-01T00:00:00"
                           "&_fields=id,date,link,title,excerpt,tags,categories,content")
                for p in d:
                    if p["id"] in vistos:
                        continue
                    vistos.add(p["id"])
                    ps = parrafos(p["content"]["rendered"])
                    if len(ps) < 4:
                        continue
                    f.write(json.dumps({"id": p["id"], "fecha": p["date"][:10], "link": p["link"],
                                        "titulo": texto(p["title"]["rendered"]), "bajada": texto(p["excerpt"]["rendered"]),
                                        "etiquetas": p["tags"], "categorias": p["categories"], "parrafos": ps},
                                       ensure_ascii=False) + "\n")
                    n += 1
                print(f"[notas] {anio} p{page}: {n}", flush=True)
                if len(d) < 100:
                    break


# --- set 2: más estilos -------------------------------------------------------------------------------------
# sección -> (categorías padre, páginas de 100 por año, desde qué página, mínimo de párrafos)
# Nacional sigue desde la página 4: las páginas 1-3 de cada año son las del set 1.
SECCIONES = {
    "nacional": (NACIONAL, 2, 4, 4),
    "en_vivo": ([4932], 1, 1, 2),
    "opinion": ([4952], 1, 1, 4),
    "red_de_expertos": ([4933], 1, 1, 4),
    "detector": ([4984], 1, 1, 3),
    "silla_academica": ([5008], 1, 1, 3),
    "amazonia": ([137184], 1, 1, 3),
}


def con_hijas(padres: list[int]) -> list[int]:
    """Las categorías y sus subcategorías (la API no incluye las hijas sola)."""
    ids = list(padres)
    for p in padres:
        d, _ = get(f"categories?parent={p}&per_page=100&_fields=id")
        ids += [c["id"] for c in d]
    return ids


def secciones(desde: int, hasta: int, solo: list[str] | None = None) -> None:
    """data/periodista/raw/notas_set2.jsonl: notas de varias secciones, cada una con su `seccion`."""
    vistos = {json.loads(l)["id"] for l in (RAW / "notas.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    n = 0
    with (RAW / "notas_set2.jsonl").open("w", encoding="utf-8") as f:
        for sec, (padres, paginas, primera, min_p) in SECCIONES.items():
            if solo and sec not in solo:
                continue
            cats = ",".join(map(str, padres if sec == "nacional" else con_hijas(padres)))
            for anio in range(hasta, desde - 1, -1):
                for page in range(primera, primera + paginas):
                    d, _ = get(f"posts?categories={cats}&categories_exclude={QUIEN}&per_page=100&page={page}"
                               f"&after={anio}-01-01T00:00:00&before={anio + 1}-01-01T00:00:00"
                               "&_fields=id,date,link,title,excerpt,tags,categories,content")
                    for p in d:
                        if p["id"] in vistos:
                            continue
                        vistos.add(p["id"])
                        ps = parrafos(p["content"]["rendered"])
                        if len(ps) < min_p:
                            continue
                        f.write(json.dumps({"id": p["id"], "fecha": p["date"][:10], "link": p["link"], "seccion": sec,
                                            "titulo": texto(p["title"]["rendered"]), "bajada": texto(p["excerpt"]["rendered"]),
                                            "etiquetas": p["tags"], "categorias": p["categories"], "parrafos": ps},
                                           ensure_ascii=False) + "\n")
                        n += 1
                    print(f"[set2] {sec} {anio} p{page}: {n}", flush=True)
                    if len(d) < 100:
                        break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", choices=["perfiles", "etiquetas", "notas", "set2"])
    ap.add_argument("--desde", type=int, default=2019)
    ap.add_argument("--hasta", type=int, default=2026)
    ap.add_argument("--por-anio", type=int, default=300, help="notas por año (múltiplo de 100)")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if a.solo in (None, "perfiles"):
        perfiles()
    if a.solo in (None, "etiquetas"):
        etiquetas()
    if a.solo in (None, "notas"):
        notas(a.desde, a.hasta, a.por_anio)
    if a.solo == "set2":
        secciones(a.desde, a.hasta)


if __name__ == "__main__":
    main()
