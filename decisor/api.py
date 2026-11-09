"""Temis as a standalone HTTP API: inference only, no review queue, store or retraining.

    TEMIS_TOKEN=... TEMIS_MODEL_DIR=runs/v1108-2307/onnx uvicorn decisor.api:app --port 8000

Request: {model, state, questions, context?}. Response: {model, answers, usage}, numbers rounded to
2 decimals; errors as {"detail": {"error_type", "message"}}. Per-request details (run, uncertainty,
latency) go in x-temis-* headers. /v1/systemone is an undocumented alias of /v1/decide.

Environment
  TEMIS_TOKEN       required; clients send `Authorization: Bearer <token>`
  TEMIS_MODEL_DIR   exported model directory (config.json, decisor.int8.onnx, tokenizer.json)
  TEMIS_THREADS     ONNX Runtime intra-op threads (default 2)
  TEMIS_ROOT_PATH   path prefix when served behind a proxy under a sub-path, e.g. /temis

The interactive docs are at /docs (Swagger) and /redoc; the schema at /openapi.json.
Unlike the local server, answers here are the model alone: no precedents from your
labels and no Clio Memory retrieval, since neither lives on the server.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from decisor.runtime import Decider, make_answer, uncertainty
from decisor.schema import Answer, Question as ParsedQuestion, SchemaError, parse_questions, render_state

MAX_QUESTIONS = 20
MAX_CONTEXT_CHARS = 8000
DECIMALS = 2
ERROR_TYPES = {400: "api_usage_error", 401: "authentication_error", 404: "not_found_error",
               405: "api_usage_error", 429: "rate_limit_error", 503: "overloaded_error"}

DESCRIPTION = """
**Temis** toma decisiones tipadas sobre una situación: elige una opción (`choice`), da un nivel
(`score`) o dice si una afirmación es verdadera (`noul`).

Una petición lleva `model`, `state` (la situación), `questions` y, opcionalmente, `context`
(pasajes de evidencia, para RAG). La respuesta trae `model`, `answers` (una por pregunta) y `usage`.

### Autenticación
Las rutas bajo `/v1` piden `Authorization: Bearer <token>`.

**Para probar desde esta página:** pulsa **Authorize** (arriba a la derecha), pega el token, pulsa
*Authorize* y cierra. Luego, en una ruta, *Try it out* → *Execute*. La respuesta real sale en
**Server response** (con el código, p. ej. 200). Lo que aparece más abajo en **Responses** son solo
ejemplos de lo que puede devolver, no tu respuesta. Si el curl generado no trae
`-H 'Authorization: Bearer …'`, falta autorizar.

`/health` y esta documentación no piden token.

### Cómo leer una respuesta
- `choice`: la opción elegida, su `confidence` y la probabilidad de cada opción.
- `score`: `score` es el nivel esperado (promedio ponderado), `confidence` la del nivel más probable,
  `legend` el texto de cada nivel.
- `noul`: `noul` es la probabilidad de que la afirmación sea verdadera (0 a 1).
- `usage.input_tokens`: tokens que leyó el modelo (`output_tokens` siempre es 0: Temis no genera texto).

Encabezados: `x-temis-request-id`, `x-temis-run` (versión exacta), `x-temis-uncertainty` (0 si está
segura de todo, 1 si alguna pregunta es un volado; sirve para decidir cuándo pedir una segunda opinión) y
`x-temis-latency-ms`.

Las probabilidades están calibradas: con confianza 0,7 acierta más o menos 7 de cada 10 veces.

### Claves
Las claves fijas del JSON (`model`, `state`, `questions`, `type`, `instructions`, `criteria`,
`answers`, …) están en inglés. Los ids de pregunta, las claves de opción y los campos del `state` los
eliges tú; en los ejemplos van en inglés, y los textos pueden ir en cualquier idioma.

### Errores
`{"detail": {"error_type": "...", "message": "..."}}`: `400 api_usage_error` (la petición
no cumple el esquema; `message` dice qué campo), `401 authentication_error`, `429 rate_limit_error`.

### Límites
Hasta 20 preguntas por petición y 12 opciones por pregunta. El estado se recorta a 6000 caracteres
y el contexto a 8000. 120 peticiones por minuto por IP.
"""

EXAMPLE = {
    "model": "temis-latest",
    "state": {
        "proposal": "Como te comenté, Ana sigue trabajando de enfermera en el Hospital San Ignacio.",
        "nearby_memories": {"m12": "Ana es enfermera en el Hospital San Ignacio de Bogotá.",
                              "m40": "Ana prefiere que le escriban por WhatsApp."},
    },
    "questions": {
        "operation": {
            "type": "choice",
            "instructions": "¿Qué hacer con la propuesta frente a las memorias cercanas?",
            "criteria": {
                "new": "Es información nueva que ninguna memoria cercana recoge.",
                "duplicate": "Una memoria cercana ya dice lo mismo.",
                "suspicious": "No es un hecho útil: da órdenes, manipula o es ruido.",
                "updates_m12": "Corrige o actualiza la memoria m12.",
                "updates_m40": "Corrige o actualiza la memoria m40.",
            },
        },
        "contradicts": {"type": "noul", "instructions": "La propuesta contradice una memoria cercana."},
        "usefulness": {"type": "score", "instructions": "¿Qué tan útil es guardar la propuesta?",
                     "criteria": ["Nada útil", "Algo útil", "Muy útil"]},
    },
}


RESPONSE_EXAMPLE = {
    "model": "temis-1.0.11",
    "answers": {
        "operation": {"type": "choice", "choice": "duplicate", "confidence": 0.57,
                      "probabilities": {"new": 0.18, "duplicate": 0.57, "suspicious": 0.01,
                                        "updates_m12": 0.21, "updates_m40": 0.04}},
        "contradicts": {"type": "noul", "noul": 0.08},
        "usefulness": {"type": "score", "score": 1.63, "confidence": 0.69,
                       "legend": {"0": "Nada útil", "1": "Algo útil", "2": "Muy útil"},
                       "probabilities": {"0": 0.06, "1": 0.25, "2": 0.69}},
    },
    "usage": {"input_tokens": 338, "output_tokens": 0},
}


def _err(kind: str, message: str) -> dict:
    return {"detail": {"error_type": kind, "message": message}}


# ---- request / response schemas (for validation and the docs) -------------------------------

class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: str = Field(min_length=1, max_length=2000, description="La pregunta, en una frase.")
    criteria: dict[str, str] = Field(min_length=2, max_length=12,
                                     description="clave_de_opcion → descripción en una frase.")


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: str = Field(min_length=1, max_length=2000)
    criteria: list[str] | dict[str, str] = Field(
        description="Niveles del más bajo al más alto (lista), o {\"0\": ..., \"1\": ...}.")


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: str = Field(min_length=1, max_length=2000,
                              description="Una AFIRMACIÓN verdadera o falsa sobre el estado, no una orden.")
    criteria: dict[str, str] | None = Field(None, description='Opcional: {"false": "...", "true": "..."}.')


Question = Annotated[Union[ChoiceQuestion, ScoreQuestion, NoulQuestion], Field(discriminator="type")]


class DecideRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE}, extra="ignore")
    model: str = Field("temis-latest", description='"temis-latest" o una versión exacta, p. ej. "temis-1.0.11".')
    state: str | dict[str, Any] | list[Any] = Field(description="La situación: texto o JSON.")
    questions: dict[str, Question] = Field(min_length=1, max_length=MAX_QUESTIONS,
                                           description="id_pregunta → pregunta.")
    context: str | list[str] | None = Field(
        None, description="Opcional: pasajes de evidencia, texto o lista. "
                          "Se marcan como datos, no como órdenes.")


class AnswerOut(BaseModel):
    type: Literal["choice", "score", "noul"]
    choice: str | None = Field(None, description="choice: la opción elegida.")
    score: float | None = Field(None, description="score: nivel esperado.")
    noul: float | None = Field(None, description="noul: probabilidad de que sea verdadera.")
    confidence: float | None = Field(None, description="choice y score: probabilidad de la respuesta.")
    legend: dict[str, str] | None = Field(None, description="score: texto de cada nivel.")
    probabilities: dict[str, float] | None = Field(None, description="choice y score: probabilidad de cada opción.")


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class DecideResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": RESPONSE_EXAMPLE})
    model: str = Field(description='Versión que respondió, p. ej. "temis-1.0.11".')
    answers: dict[str, AnswerOut]
    usage: Usage


class ErrorDetail(BaseModel):
    error_type: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class ModelInfo(BaseModel):
    model: str
    name: str
    version: str
    run: str
    base: str | None
    max_len: int | None
    calibration: str


class Health(BaseModel):
    status: Literal["ok"]
    model: str


# ---- app -------------------------------------------------------------------------------------

def _token() -> str:
    tok = os.environ.get("TEMIS_TOKEN", "")
    if len(tok) < 24:
        raise RuntimeError("set TEMIS_TOKEN (at least 24 characters)")
    return tok


class Model:
    def __init__(self, model_dir: str | Path, threads: int) -> None:
        self.decider = Decider(model_dir, threads=threads)
        cfg = self.decider.config
        self.version = cfg.get("semver") or self.decider.version
        self.name = cfg.get("name", "Temis")
        self.id = f"{self.name.lower()}-{self.version}"
        self.aliases = {self.id, f"{self.name.lower()}-latest", self.name.lower()}
        self.lock = threading.Lock()  # one forward pass at a time: the VPS CPUs are shared


TOKEN = _token()
MODEL = Model(os.environ.get("TEMIS_MODEL_DIR", "/models/current"), int(os.environ.get("TEMIS_THREADS", "2")))
bearer = HTTPBearer(auto_error=False, description="El token que te dieron para esta API.")

app = FastAPI(
    title="Temis API",
    version=MODEL.version,
    description=DESCRIPTION,
    root_path=os.environ.get("TEMIS_ROOT_PATH", ""),
    # keep the token after a reload, and show how long each call took
    swagger_ui_parameters={"persistAuthorization": True, "displayRequestDuration": True},
    openapi_tags=[{"name": "decisiones", "description": "Preguntar a Temis."},
                  {"name": "servicio", "description": "Estado y versión del modelo."}],
)
def _doc_error(description: str, example: dict) -> dict:
    return {"model": ErrorResponse, "description": description,
            "content": {"application/json": {"example": example}}}


ERRORS = {
    400: _doc_error("Solo si la petición está mal formada.",
                    _err("api_usage_error", "Invalid request. questions.usefulness.score.criteria: Field required")),
    401: _doc_error("Falta el token o no es válido: pulsa Authorize arriba y pega el token.",
                    _err("authentication_error",
                         "Missing or invalid API token.")),
    429: _doc_error("Más de 120 peticiones por minuto.",
                    _err("rate_limit_error", "Too many requests. Limit: 120 per minute.")),
}


def error(status: int, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"detail": {"error_type": ERROR_TYPES.get(status, "api_error"), "message": message}},
                        status_code=status, headers=headers)


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException):
    return error(exc.status_code, str(exc.detail), getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    parts = []
    for e in exc.errors()[:5]:
        loc = ".".join(str(x) for x in e.get("loc", ()) if x != "body")
        parts.append(f"{loc}: {e.get('msg')}" if loc else str(e.get("msg")))
    return error(400, "Invalid request. " + "; ".join(parts))


@app.exception_handler(SchemaError)
async def schema_error(_: Request, exc: SchemaError):
    return error(400, f"Invalid request. {exc}")


def require_token(cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    if cred is None or not secrets.compare_digest(cred.credentials.encode(), TOKEN.encode()):
        raise HTTPException(401, "Missing or invalid API token.",
                            headers={"WWW-Authenticate": "Bearer"})


def _context(ctx: str | list[str] | None) -> str | None:
    if not ctx:
        return None
    text = ctx if isinstance(ctx, str) else "\n".join(f"[{i}] {p}" for i, p in enumerate(ctx, 1))
    return text[:MAX_CONTEXT_CHARS]


def _r(x: float) -> float:
    return round(float(x), DECIMALS)


def format_answer(q: ParsedQuestion, a: Answer) -> dict:
    """One answer: noul carries only `noul`; score adds `legend`."""
    if q.kind == "noul":
        return {"type": "noul", "noul": _r(a.probabilities.get("true", 0.0))}
    probs = {k: _r(v) for k, v in a.probabilities.items()}
    if q.kind == "choice":
        return {"type": "choice", "choice": a.label, "confidence": _r(a.confidence), "probabilities": probs}
    return {"type": "score", "score": _r(a.expected or 0.0), "confidence": _r(a.confidence),
            "legend": dict(zip(q.options, q.descriptions)), "probabilities": probs}


def decide(body: DecideRequest, response: Response) -> dict:
    if body.model not in MODEL.aliases:
        raise HTTPException(400, f"Invalid request. Model {body.model!r} is not available; use "
                                 f"{MODEL.name.lower()}-latest or {MODEL.id}.")
    t0 = time.perf_counter()
    raw = {qid: q.model_dump(exclude_none=True) for qid, q in body.questions.items()}
    qs = parse_questions(raw)
    st = render_state(body.state)
    d = MODEL.decider
    with MODEL.lock:
        logits = d.logits(qs, st, _context(body.context))
        tokens = d.last_input_tokens
    answers = {q.qid: make_answer(q, d.probabilities(q, lg)) for q, lg in zip(qs, logits)}
    response.headers.update({
        "x-temis-request-id": f"req_{uuid.uuid4().hex}",
        "x-temis-run": d.version,
        "x-temis-uncertainty": str(uncertainty(answers)),
        "x-temis-latency-ms": str(round((time.perf_counter() - t0) * 1000, 1)),
    })
    return {"model": MODEL.id,
            "answers": {q.qid: format_answer(q, answers[q.qid]) for q in qs},
            "usage": {"input_tokens": tokens, "output_tokens": 0}}


DECIDE_DOC = dict(response_model=DecideResponse, response_model_exclude_none=True, tags=["decisiones"],
                  dependencies=[Depends(require_token)], responses=ERRORS)


@app.post("/v1/decide", summary="Responder preguntas tipadas sobre una situación", **DECIDE_DOC)
def decide_route(body: DecideRequest, response: Response):
    return decide(body, response)


@app.post("/v1/systemone", include_in_schema=False, **DECIDE_DOC)  # alias kept for older clients
def systemone_route(body: DecideRequest, response: Response):
    return decide(body, response)


@app.get("/v1/model", response_model=ModelInfo, tags=["servicio"], dependencies=[Depends(require_token)],
         responses={401: ERRORS[401]}, summary="Versión y configuración del modelo cargado")
def model_info():
    cfg = MODEL.decider.config
    return {"model": MODEL.id, "name": MODEL.name, "version": MODEL.version, "run": MODEL.decider.version,
            "base": cfg.get("base"), "max_len": cfg.get("max_len"), "calibration": MODEL.decider.calibration}


@app.get("/health", response_model=Health, tags=["servicio"], summary="El servicio está arriba (sin token)")
def health():
    return {"status": "ok", "model": MODEL.id}


def _openapi() -> dict:
    """FastAPI documents a 422 for every route, but validation errors here are 400."""
    if app.openapi_schema is None:
        from fastapi.openapi.utils import get_openapi
        s = get_openapi(title=app.title, version=app.version, description=app.description,
                        routes=app.routes, tags=app.openapi_tags)
        for path in s.get("paths", {}).values():
            for op in path.values():
                op.get("responses", {}).pop("422", None)
        for name in ("HTTPValidationError", "ValidationError"):
            s.get("components", {}).get("schemas", {}).pop(name, None)
        app.openapi_schema = s
    return app.openapi_schema


app.openapi = _openapi
