"""Temis periodista: corre el benchmark de entidades contra una API con la forma de TypeSafe / Temis y lo califica.

  export TEMIS_TOKEN=...                                   # nunca en el repo
  python -m scripts.periodista_bench                       # API alojada de Temis, modelo temis-latest
  python -m scripts.periodista_bench --model temis-2.4.0
  python -m scripts.periodista_bench --url http://127.0.0.1:8765/v1/systemone --token-env NONE   # Temis local
  python -m scripts.periodista_bench --url https://api.typesafe.ai/v1/systemone --token-env TYPESAFE_API_KEY --model jev

Lee data/periodista/bench_cases.jsonl (casos) y data/periodista/bench_gold.jsonl (respuestas correctas).
Escribe benchmarks/periodista/<nombre>-results.jsonl (respuesta por caso) y summary-<nombre>.json, e imprime el
acierto por pregunta. Con --solo-calificar no llama a la API: vuelve a calificar el results.jsonl que ya existe.

Calificación: choice y noul por acierto (noul: probabilidad de sí ≥ 0,5); score por acierto del nivel más
probable y error absoluto medio del nivel esperado. Las preguntas ent_fuente_fN se juntan en «ent_fuente».
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CASOS = Path("data/periodista/bench_cases.jsonl")
ORO = Path("data/periodista/bench_gold.jsonl")
SALIDA = Path("benchmarks/periodista")
URL = "https://api.imontes.dev/temis/v1/decide"


def leer(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def familia(qid: str) -> str:
    return "ent_fuente" if qid.startswith("ent_fuente_") else qid


def pedir(url: str, token: str | None, cuerpo: dict, reintentos: int = 8) -> tuple[dict, float]:
    datos = json.dumps(cuerpo, ensure_ascii=False).encode("utf-8")
    # Cloudflare (API alojada) rechaza el agente por defecto de urllib con «error code: 1010»
    h = {"Content-Type": "application/json; charset=utf-8", "User-Agent": "temis-periodista-bench/0.1"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    for i in range(reintentos):
        t0 = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=datos, headers=h), timeout=120) as r:
                return json.loads(r.read().decode("utf-8")), (time.time() - t0) * 1000
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 529) and i < reintentos - 1:
                time.sleep(min(90, 15 * (i + 1)))   # 503 «overloaded»: el servidor necesita aire
                continue
            raise RuntimeError(f"{e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError):   # red o servidor sin responder a tiempo
            if i == reintentos - 1:
                raise
            time.sleep(min(90, 15 * (i + 1)))
    raise RuntimeError("sin respuesta")


def prediccion(q: dict, a: dict) -> tuple[str | None, float | None]:
    """(etiqueta elegida, nivel esperado para score)."""
    if not a:
        return None, None
    if q["type"] == "noul":
        return ("true" if float(a.get("noul", 0)) >= 0.5 else "false"), None
    if q["type"] == "choice":
        return a.get("choice"), None
    probs = a.get("probabilities") or {}
    if probs:
        claves = list(probs)
        mejor = max(claves, key=lambda k: probs[k])
        # las claves pueden ser el índice del nivel o su texto
        nivel = str(claves.index(mejor)) if not mejor.isdigit() else mejor
        return nivel, float(a.get("score", 0))
    return str(round(float(a.get("score", 0)))), float(a.get("score", 0))


def calificar(casos: list[dict], oro: dict, res: dict) -> dict:
    por = defaultdict(lambda: {"n": 0, "ok": 0, "err_abs": 0.0})
    for c in casos:
        g = oro.get(c["id"])
        r = res.get(c["id"])
        if not g or not r:
            continue
        for qid, q in c["questions"].items():
            if qid not in g["gold"]:
                continue
            esperado = str(g["gold"][qid]).lower()
            pred, nivel = prediccion(q, (r.get("answers") or {}).get(qid))
            f = por[familia(qid)]
            f["n"] += 1
            f["ok"] += int(pred is not None and str(pred).lower() == esperado)
            if q["type"] == "score" and nivel is not None:
                f["err_abs"] += abs(nivel - float(esperado))
    filas = {}
    for k, f in sorted(por.items()):
        filas[k] = {"n": f["n"], "acierto": round(f["ok"] / f["n"], 3) if f["n"] else None}
        if k == "ent_centralidad" and f["n"]:
            filas[k]["mae"] = round(f["err_abs"] / f["n"], 3)
    tot_n = sum(f["n"] for f in por.values())
    tot_ok = sum(f["ok"] for f in por.values())
    # media por familia (cada pregunta pesa igual, sin importar cuántas veces aparece)
    medias = [v["acierto"] for v in filas.values() if v["acierto"] is not None]
    return {"preguntas": tot_n, "acierto_total": round(tot_ok / tot_n, 3) if tot_n else None,
            "acierto_medio_por_pregunta": round(sum(medias) / len(medias), 3) if medias else None, "por_pregunta": filas}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--model", default="temis-latest")
    ap.add_argument("--token-env", default="TEMIS_TOKEN", help="variable de entorno con el token (NONE: sin token)")
    ap.add_argument("--nombre", default=None, help="nombre de los archivos de salida (por defecto, el modelo que responde)")
    ap.add_argument("--solo-calificar", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="peticiones a la vez. La API alojada (VPS de 4 núcleos) se satura con más de 1: 4 a la vez "
                         "tardan más que en serie. Sirve con un Temis local o con GPU.")
    a = ap.parse_args()
    casos = leer(CASOS)
    oro = {g["id"]: g for g in leer(ORO)} if ORO.exists() else {}
    SALIDA.mkdir(parents=True, exist_ok=True)
    nombre = a.nombre or a.model
    archivo = SALIDA / f"{nombre}-results.jsonl"
    # se reanuda: lo que ya está en el archivo no se vuelve a pedir
    res = {r["id"]: r for r in leer(archivo)} if archivo.exists() else {}

    if not a.solo_calificar:
        token = None if a.token_env == "NONE" else os.environ.get(a.token_env)
        if a.token_env != "NONE" and not token:
            raise SystemExit(f"falta la variable de entorno {a.token_env}")
        pendientes = [c for c in casos if c["id"] not in res]
        print(f"[bench] {len(pendientes)} casos por pedir ({len(res)} ya estaban)", flush=True)
        lock = threading.Lock()

        def uno(c: dict) -> None:
            r, t = pedir(a.url, token, {"model": a.model, "state": c["state"], "questions": c["questions"]})
            fila = {"id": c["id"], "model": r.get("model"), "answers": r.get("answers"), "usage": r.get("usage"),
                    "ms": round(t)}
            with lock:
                res[c["id"]] = fila
                with archivo.open("a", encoding="utf-8") as f:   # cada caso apenas llega
                    f.write(json.dumps(fila, ensure_ascii=False) + "\n")
                hechos = len(res)
            if hechos % 10 == 0:
                print(f"[bench] {hechos}/{len(casos)}", flush=True)
            if a.workers == 1:
                time.sleep(0.5)   # la API alojada permite 120 peticiones por minuto

        with ThreadPoolExecutor(max(1, a.workers)) as ex:
            list(ex.map(uno, pendientes))
    resumen = calificar(casos, oro, res)
    resumen.update({"modelo": nombre, "casos": len(casos), "con_oro": sum(1 for c in casos if c["id"] in oro),
                    "respondidos": sum(1 for c in casos if c["id"] in res)})
    vals = [r for r in res.values() if r.get("usage")]
    if vals:
        ts = sorted((r["usage"] or {}).get("input_tokens", 0) for r in vals)
        ms = sorted(r.get("ms", 0) for r in vals)
        resumen.update({"tokens_entrada_mediana": ts[len(ts) // 2], "tokens_entrada_max": ts[-1],
                        "ms_mediana": ms[len(ms) // 2]})
    (SALIDA / f"summary-{nombre}.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
