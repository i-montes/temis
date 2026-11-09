"""Web server: TypeSafe-compatible API + the review/playground UI.

  python -m decisor.server --port 8765      ->  http://127.0.0.1:8765

API
  POST /v1/systemone        {state, questions, context?, workflow?, log?}   (same answers as TypeSafe)
  GET  /api/queue           next doubtful decisions to label
  POST /api/label           {decision_id, labels: {qid: option}}
  POST /api/skip            {decision_id}
  GET  /api/decision/{id}   one decision
  GET  /api/recent          latest decisions
  GET  /api/stats           counts + active model
  GET  /api/agreement       how often the suggestion (and each teacher) matches your labels
  GET  /api/models          trained versions with their metrics
  POST /api/models/activate {name}
  POST /api/retrain         start a retraining run (background)
  GET  /api/retrain         its status and log tail
  GET  /api/templates       example question schemas
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from decisor.schema import SchemaError, parse_questions, render_state
from decisor.service import Service, active_model_dir, set_active
from decisor.templates import TEMPLATES

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
app = FastAPI(title="Temis")
LOCAL = {"127.0.0.1", "::1", "localhost"}


def access_token() -> str:
    """Token required for requests that do not come from this Mac (phone on the Wi-Fi). Lives in .env."""
    import secrets
    env = ROOT / ".env"
    lines = env.read_text().splitlines() if env.exists() else []
    for ln in lines:
        if ln.startswith("DECISOR_TOKEN="):
            return ln.split("=", 1)[1].strip()
    tok = secrets.token_urlsafe(9)
    with env.open("a") as f:
        f.write(f"DECISOR_TOKEN={tok}\n")
    return tok


TOKEN = access_token()


@app.middleware("http")
async def lan_guard(request: Request, call_next):
    # pages (HTML, static files) load freely; the API needs the token unless the caller is this machine
    host = request.client.host if request.client else ""
    if host not in LOCAL and (request.url.path.startswith("/api") or request.url.path.startswith("/v1")):
        if request.headers.get("x-decisor-token") != TOKEN:
            return JSONResponse({"error": "clave incorrecta o ausente"}, status_code=401)
    return await call_next(request)


def lan_ip() -> str | None:
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return None
svc: Service | None = None
RETRAIN: dict[str, Any] = {"proc": None, "log": None, "started": None}
LISTEN: dict[str, Any] = {"lan": False}


def service() -> Service:
    global svc
    if svc is None:
        svc = Service()
    return svc


@app.exception_handler(SchemaError)
async def schema_error(_, exc: SchemaError):
    return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/v1/systemone")
def systemone(body: dict = Body(...)):
    try:
        return service().decide(body.get("state"), body.get("questions"), context=body.get("context"),
                                workflow=body.get("workflow", ""), origin=body.get("origin", "api"),
                                # benchmarks call with model "nolog-..." so they never reach the queue or training
                                log=body.get("log", True) and not str(body.get("model", "")).startswith("nolog"))
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e


@app.get("/api/queue")
def queue(limit: int = 1, exclude: str = ""):
    items = service().store.queue(limit, [x for x in exclude.split(",") if x])
    return {"items": items, "stats": service().store.stats()}


@app.post("/api/label")
def label(body: dict = Body(...)):
    s = service()
    d = s.store.get(body["decision_id"])
    if not d:
        raise HTTPException(404, "decision not found")
    qs = parse_questions(d["questions"])
    try:
        s.store.label(d["id"], body["labels"], qs, render_state(d["state"]), how=body.get("how"))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "stats": s.store.stats()}


@app.post("/api/skip")
def skip(body: dict = Body(...)):
    service().store.skip(body["decision_id"])
    return {"ok": True}


@app.get("/api/decision/{did}")
def decision(did: str):
    d = service().store.get(did)
    if not d:
        raise HTTPException(404, "not found")
    return d


@app.get("/api/agreement")
def agreement_stats():
    from decisor.store import agreement
    return agreement(service().store)


@app.get("/api/recent")
def recent(limit: int = 30, origin: str | None = None):
    return {"items": service().store.recent(limit, origin)}


@app.get("/api/stats")
def stats():
    s = service()
    s.maybe_reload()
    return {**s.store.stats(), "model": s.display, "run": s.version}


def _models() -> list[dict]:
    out = []
    act = active_model_dir()
    for meta in sorted(ROOT.glob("runs/*/meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        run = meta.parent
        if run.name == "smoke":
            continue
        m = json.loads(meta.read_text())
        ev = run / "eval_td_test.json"
        cl = run / "eval_clio.json"
        ue = run / "eval_user.json"
        from decisor.naming import display
        out.append({"name": run.name, "display": display(run.name), "created": m.get("created"), "val": m.get("val", {}).get("all"),
                    "val_by_source": {k: v for k, v in m.get("val", {}).items() if k != "all"},
                    "td_test": json.loads(ev.read_text()) if ev.exists() else None,
                    "clio": json.loads(cl.read_text()) if cl.exists() else None,
                    "user": json.loads(ue.read_text()) if ue.exists() else None,
                    "exported": (run / "onnx" / "config.json").exists(),
                    "active": act is not None and act.resolve() == (run / "onnx").resolve(),
                    "n_train": m.get("n_train"), "notes": m.get("notes")})
    return out


@app.get("/api/models")
def models():
    return {"items": _models()}


@app.post("/api/models/activate")
def activate(body: dict = Body(...)):
    d = ROOT / "runs" / body["name"] / "onnx"
    if not (d / "config.json").exists():
        raise HTTPException(400, "that version is not exported")
    set_active(d)
    service().reload()
    return {"ok": True, "model": service().display}


@app.post("/api/retrain")
def retrain():
    p = RETRAIN.get("proc")
    if p is not None and p.poll() is None:
        raise HTTPException(409, "a retraining run is already in progress")
    log = ROOT / "runs" / "logs" / f"retrain-{time.strftime('%Y%m%d-%H%M%S')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    f = log.open("w")
    env = {**os.environ, "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.5", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.4"}
    RETRAIN.update(proc=subprocess.Popen(["nice", "-n", "10", sys.executable, "-m", "scripts.retrain"], cwd=ROOT,
                                         stdout=f, stderr=subprocess.STDOUT, env=env), log=log, started=time.time())
    return {"ok": True, "log": log.name}


@app.get("/api/retrain")
def retrain_status():
    p, log = RETRAIN.get("proc"), RETRAIN.get("log")
    running = p is not None and p.poll() is None
    tail = ""
    if log and Path(log).exists():
        lines = [ln for ln in Path(log).read_text(errors="replace").splitlines()
                 if ln.startswith(("[", "Traceback", "  File")) or "Error" in ln]
        tail = "\n".join(lines[-25:])
    if p is not None and not running and RETRAIN.get("reloaded") != RETRAIN.get("started"):
        RETRAIN["reloaded"] = RETRAIN.get("started")
        service().reload()
    return {"running": running, "exit": None if running or p is None else p.returncode,
            "started": RETRAIN.get("started"), "tail": tail}


@app.get("/api/templates")
def templates():
    return TEMPLATES


@app.get("/api/lan-info")
def lan_info(request: Request):
    """Only for this Mac: the address and the token to open the phone page."""
    if (request.client.host if request.client else "") not in LOCAL:
        raise HTTPException(403, "solo desde el propio equipo")
    ip = lan_ip()
    port = request.url.port or 8765
    return {"url": f"http://{ip}:{port}/m" if ip else None, "token": TOKEN, "listening_lan": LISTEN.get("lan", False)}


@app.get("/m")
def mobile():
    return FileResponse(WEB / "m.html")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


if WEB.exists():
    app.mount("/web", StaticFiles(directory=WEB), name="web")


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--lan", action="store_true", help="also listen on the Wi-Fi so the phone can open /m")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    os.chdir(ROOT)
    host = "0.0.0.0" if a.lan else a.host
    LISTEN["lan"] = a.lan
    if a.lan:
        print(f"[decisor] celular: http://{lan_ip()}:{a.port}/m  (la clave está en la web del Mac: botón «Celular»)", flush=True)
    uvicorn.run(app, host=host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
