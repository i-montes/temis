"""Temis periodista: casos a partir de las trazas del modo depuración del verificador (extension-verificador-x).

Las trazas son peticiones reales de producción: textos de redes sociales, de otros medios y de páginas web
(otros estilos que las notas de La Silla), con los candidatos que el verificador sacó del «Quién es quién», de
las etiquetas y del archivo. Aquí cada candidato de cada petición de entidades a Jev se vuelve un caso por entidad
con las preguntas de Temis periodista; Jev los responde otra vez con scripts.periodista_label.

Cómo sacar las trazas: en el verificador, opciones -> encender «Modo depuración», navegar (redes, notas, PDF…),
abrir la página de depuración y «Descargar todo» -> depuracion-verificador-AAAAMMDD-HHMM.json.

  python -m scripts.periodista_trazas ~/Downloads/depuracion-verificador-*.json

Escribe data/periodista/trazas/cases.jsonl (añade sin repetir: la clave es nombre + párrafo). Las respuestas
originales de Jev a las preguntas del verificador (es, perfil, persona, trata_persona, tipo_etiqueta) quedan en
`meta.jev_verificador` para comparar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from scripts.periodista_build import (LARGO_CANDIDATO, LARGO_FUENTE, LARGO_ORACION, MAX_ORACIONES, TAREA, cortar,
                                      oraciones, preguntas)

OUT = Path("data/periodista/trazas/cases.jsonl")
ETAPAS = re.compile(r"entidades|perfiles|etiquetas|contexto")   # peticiones que llevan candidatos o etiquetas


def separar(nombre: str) -> tuple[str, str]:
    """«Sanguino (Antonio Sanguino)» -> («Sanguino», «Antonio Sanguino»): así manda el verificador la superficie."""
    m = re.match(r"^(.*?)\s*\((.+)\)\s*$", nombre or "")
    return (m.group(1).strip(), m.group(2).strip()) if m else (nombre, nombre)


def tipo_documento(etapa: str, estado: dict) -> str:
    if etapa.startswith("redes"):
        return "post de una red social"
    if estado.get("titulo_de_la_pagina") is not None:
        return "página web (no es una nota de prensa)"
    return "nota de prensa de un medio"


def casos_de_evento(ev: dict) -> list[dict]:
    estado = ev.get("estado") or {}
    etapa = ev.get("etapa") or ""
    respuesta = ((ev.get("respuesta") or {}).get("answers")) or {}
    fecha = datetime.fromtimestamp((ev.get("t") or 0) / 1000, tz=timezone.utc).date().isoformat() if ev.get("t") else None
    titulo = estado.get("titulo_de_la_nota") or estado.get("titulo_de_la_pagina")
    doc = {"tipo": tipo_documento(etapa, estado), "titulo": titulo, "fecha": fecha}
    sal = []
    entradas = [("candidatos", k, c) for k, c in enumerate(estado.get("candidatos") or [])]
    entradas += [("etiquetas", k, c) for k, c in enumerate(estado.get("etiquetas") or [])]
    for lista, k, c in entradas:
        sup, nombre = separar(c.get("nombre", ""))
        parrafo = c.get("parrafo") or ""
        if not sup or not parrafo:
            continue
        ors = [o for o in oraciones([parrafo]) if sup in o or nombre in o] or [parrafo]
        st = {"tarea": TAREA.replace("una nota de La Silla Vacía", "un texto (nota, post o página)"),
              "documento": {k2: v for k2, v in doc.items() if v},
              "entidad": {"mencion": sup, "nombre": nombre, "otras_menciones": [sup] if sup != nombre else []},
              "oraciones": {f"o{i}": cortar(o, LARGO_ORACION) for i, o in enumerate(ors[:MAX_ORACIONES])}}
        cands = {}
        for n, p in enumerate(c.get("perfiles") or [], 1):
            txt = f"Perfil del «Quién es quién»: {p.get('nombre')}. {p.get('descripcion') or ''}"
            cands[f"c{n}"] = cortar(txt, LARGO_CANDIDATO)
        if lista == "etiquetas":
            cands[f"c{len(cands) + 1}"] = cortar(f"Etiqueta de La Silla «{nombre}».", LARGO_CANDIDATO)
        if cands:
            st["candidatos"] = cands
        fuente = {f"f{j}": {"titulo": f.get("titulo"), "fecha": f.get("fecha"), "texto": cortar(f.get("texto") or "", LARGO_FUENTE)}
                  for j, f in enumerate(c.get("notas") or [])}
        if fuente:
            st["fuente"] = fuente
        # las preguntas del verificador numeran aparte los candidatos (es, perfil, persona, trata_persona) y las
        # etiquetas (tipo_etiqueta)
        patron = (rf"^(es|perfil|persona)\.{k}$|^trata_persona\.{k}\.\d+$" if lista == "candidatos"
                  else rf"^tipo_etiqueta\.{k}$")
        original = {q: a for q, a in respuesta.items() if re.match(patron, q)}
        clave = hashlib.sha1(f"{nombre}|{parrafo}".encode("utf-8")).hexdigest()[:12]
        sal.append({"id": f"trz-{clave}", "schema_id": "periodista_entidad", "family": "periodista_entidad",
                    "workflow": f"entidad_traza_{etapa}", "language": "es",
                    "domain": "periodismo: entidades de un texto (traza real del verificador de La Silla Vacía)",
                    "state": st, "questions": preguntas(st, None, None),
                    "meta": {"lote": "trazas", "etapa": etapa, "lista": lista, "fecha": fecha, "jev_verificador": original}})
    return sal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archivos", nargs="+", help="exportaciones de la página de depuración (formato vjx-depuracion)")
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    vistos = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()} if OUT.exists() else set()
    nuevos, eventos = [], 0
    for ruta in a.archivos:
        d = json.loads(Path(ruta).read_text(encoding="utf-8"))
        evs = d.get("eventos") or ([d["traza"]] if d.get("traza") else [])
        for ev in evs:
            if ev.get("tipo") != "jev" or not ETAPAS.search(ev.get("etapa") or "") or ev.get("error"):
                continue
            eventos += 1
            for c in casos_de_evento(ev):
                if c["id"] not in vistos:
                    vistos.add(c["id"])
                    nuevos.append(c)
    with OUT.open("a", encoding="utf-8") as f:
        for c in nuevos:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{eventos} peticiones de entidades a Jev -> {len(nuevos)} casos nuevos ({len(vistos)} en total) en {OUT}")


if __name__ == "__main__":
    main()
