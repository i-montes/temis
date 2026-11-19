"""Temis periodista: Jev responde los casos de entidades (entrenamiento y benchmark). Se puede reanudar.

  export TYPESAFE_API_KEY=...        # o en .env (nunca en el repo)
  python -m scripts.periodista_label --workers 6

Escribe:
  data/periodista/labels/jev.jsonl   {id, teacher, dists}: una distribución por pregunta, en el orden de las opciones
                                     del caso (el mismo formato de data/labels/jev.jsonl, así build_targets lo lee igual)
  data/periodista/bench_gold.jsonl   respuesta de referencia del benchmark: la opción más probable de Jev, con su
                                     confianza; `revisado: false` hasta que alguien de la redacción la confirme

Usa el cliente de Jev de Temis (temis/teachers.py): tope de gasto JEV_BUDGET_USD y registro de uso en
data/periodista/labels/jev_usage.json.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from temis.schema import parse_questions
from temis.teachers import BudgetExceeded, Jev, load_env

DIR = Path("data/periodista")
ETIQUETAS = DIR / "labels" / "jev.jsonl"
ORO = DIR / "bench_gold.jsonl"
LOCK = threading.Lock()


def leer(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def etiquetar(casos: list[dict], workers: int) -> None:
    hechos = {r["id"] for r in leer(ETIQUETAS)}
    pendientes = [c for c in casos if c["id"] not in hechos]
    print(f"[jev] {len(pendientes)} casos por responder ({len(hechos)} ya estaban)", flush=True)
    if not pendientes:
        return
    jev = Jev(usage_file=str(DIR / "labels" / "jev_usage.json"))
    parar = threading.Event()
    n = {"ok": 0, "err": 0}

    def uno(c: dict) -> None:
        if parar.is_set():
            return
        try:
            r = jev.distributions(c["state"], c["questions"])
            with LOCK, ETIQUETAS.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": c["id"], "teacher": r["model"], "dists": r["dists"]}) + "\n")
            n["ok"] += 1
            if n["ok"] % 50 == 0:
                print(f"[jev] {n['ok']}/{len(pendientes)} gastado ${jev.spent:.3f}", flush=True)
        except BudgetExceeded as e:
            print(f"[jev] {e}", flush=True)
            parar.set()
        except Exception as e:  # noqa: BLE001
            n["err"] += 1
            print(f"[jev] {c['id']}: {e}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(uno, pendientes))
    print(f"[jev] listo: {n['ok']} respondidos, {n['err']} errores, gastado ${jev.spent:.3f}", flush=True)


def oro_del_benchmark(bench: list[dict]) -> None:
    dists = {r["id"]: r for r in leer(ETIQUETAS)}
    revisados = {g["id"]: g for g in leer(ORO) if g.get("revisado")}
    with ORO.open("w", encoding="utf-8") as f:
        for c in bench:
            if c["id"] in revisados:          # lo que ya confirmó la redacción no se pisa
                f.write(json.dumps(revisados[c["id"]], ensure_ascii=False) + "\n")
                continue
            r = dists.get(c["id"])
            if not r:
                continue
            oro, conf = {}, {}
            for q in parse_questions(c["questions"]):
                d = r["dists"].get(q.qid)
                if not d:
                    continue
                k = max(range(len(d)), key=d.__getitem__)
                oro[q.qid] = str(k) if q.kind == "score" else q.options[k]
                conf[q.qid] = round(d[k], 3)
            f.write(json.dumps({"id": c["id"], "gold": oro, "confianza": conf, "fuente": r["teacher"],
                                "revisado": False}, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--solo", choices=["bench", "train"])
    a = ap.parse_args()
    load_env()
    (DIR / "labels").mkdir(parents=True, exist_ok=True)
    bench = leer(DIR / "bench_cases.jsonl")
    train = leer(DIR / "cases.jsonl")
    if a.solo != "train":
        etiquetar(bench, a.workers)
        oro_del_benchmark(bench)
    if a.solo != "bench":
        etiquetar(train, a.workers)


if __name__ == "__main__":
    main()
