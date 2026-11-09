"""Temis as a standalone HTTP API: inference only, no review queue, store or retraining.

    TEMIS_TOKEN=... TEMIS_MODEL_DIR=runs/v1108-2307/onnx uvicorn temis.api:app --port 8000

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

from temis.runtime import Decider, make_answer, uncertainty
from temis.schema import Answer, Question as ParsedQuestion, SchemaError, parse_questions, render_state

MAX_QUESTIONS = 20
MAX_CONTEXT_CHARS = 8000
DECIMALS = 2
ERROR_TYPES = {400: "api_usage_error", 401: "authentication_error", 404: "not_found_error",
               405: "api_usage_error", 429: "rate_limit_error", 503: "overloaded_error"}

DESCRIPTION = """
**Temis** makes typed decisions about a situation: it picks an option (`choice`), gives a level
(`score`) or says whether a statement is true (`noul`).

A request carries `model`, `state` (the situation), `questions` and, optionally, `context`
(evidence passages, for RAG). The response has `model`, `answers` (one per question) and `usage`.

### Authentication
Routes under `/v1` require `Authorization: Bearer <token>`.

**To try it from this page:** click **Authorize** (top right), paste the token, click *Authorize*
and close. Then, on a route, *Try it out* → *Execute*. The real response shows up under
**Server response** (with its status code, e.g. 200). What appears further down under **Responses**
are only examples of what it can return, not your response. If the generated curl has no
`-H 'Authorization: Bearer …'`, you still need to authorize.

`/health` and these docs need no token.

### Reading a response
- `choice`: the chosen option, its `confidence` and the probability of every option.
- `score`: `score` is the expected level (weighted average), `confidence` that of the most likely
  level, `legend` the text of each level.
- `noul`: `noul` is the probability that the statement is true (0 to 1).
- `usage.input_tokens`: tokens the model read (`output_tokens` is always 0: Temis does not generate text).

Headers: `x-temis-request-id`, `x-temis-run` (exact version), `x-temis-uncertainty` (0 when it is sure
of everything, 1 when some question is a coin flip; useful to decide when to ask for a second
opinion) and `x-temis-latency-ms`.

Probabilities are calibrated: answers with confidence 0.7 are right about 7 times out of 10.

### Keys
The fixed JSON keys (`model`, `state`, `questions`, `type`, `instructions`, `criteria`, `answers`, …)
are in English. Question ids, option keys and the fields inside `state` are yours to choose. Texts
can be in English or Spanish: Temis was trained on both (62 % of its cases are Spanish, so it is
somewhat stronger there; the example below is in Spanish).

### Errors
`{"detail": {"error_type": "...", "message": "..."}}`: `400 api_usage_error` (the request does not
match the schema; `message` says which field), `401 authentication_error`, `429 rate_limit_error`.

### Limits
Up to 20 questions per request and 12 options per question. The state is cut at 40,000 characters
and the context at 8,000; the model reads up to `long_max_len` tokens of a long state (1,024 by
default) and the rest is dropped. 120 requests per minute per IP.
"""

EXAMPLE = {  # texts in Spanish on purpose: Temis reads both languages, and most of its memory cases are Spanish
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
    "model": "temis-2.0.0",
    "answers": {
        "operation": {"type": "choice", "choice": "duplicate", "confidence": 0.7,
                      "probabilities": {"new": 0.07, "duplicate": 0.7, "suspicious": 0.01,
                                        "updates_m12": 0.2, "updates_m40": 0.03}},
        "contradicts": {"type": "noul", "noul": 0.04},
        "usefulness": {"type": "score", "score": 1.77, "confidence": 0.81,
                       "legend": {"0": "Nada útil", "1": "Algo útil", "2": "Muy útil"},
                       "probabilities": {"0": 0.04, "1": 0.16, "2": 0.81}},
    },
    "usage": {"input_tokens": 341, "output_tokens": 0},
}


def _err(kind: str, message: str) -> dict:
    return {"detail": {"error_type": kind, "message": message}}


# ---- request / response schemas (for validation and the docs) -------------------------------

class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: str = Field(min_length=1, max_length=2000, description="The question, in one sentence.")
    criteria: dict[str, str] = Field(min_length=2, max_length=12,
                                     description="option_key → one-sentence description.")


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: str = Field(min_length=1, max_length=2000)
    criteria: list[str] | dict[str, str] = Field(
        description="Levels from lowest to highest (a list), or {\"0\": ..., \"1\": ...}.")


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: str = Field(min_length=1, max_length=2000,
                              description="A true-or-false STATEMENT about the state, not an instruction.")
    criteria: dict[str, str] | None = Field(None, description='Optional: {"false": "...", "true": "..."}.')


Question = Annotated[Union[ChoiceQuestion, ScoreQuestion, NoulQuestion], Field(discriminator="type")]


class DecideRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE}, extra="ignore")
    model: str = Field("temis-latest", description='"temis-latest" or an exact version, e.g. "temis-1.0.11".')
    state: str | dict[str, Any] | list[Any] = Field(description="The situation: text or JSON.")
    questions: dict[str, Question] = Field(min_length=1, max_length=MAX_QUESTIONS,
                                           description="question_id → question.")
    context: str | list[str] | None = Field(
        None, description="Optional: evidence passages, a string or a list. "
                          "They are marked as data, not instructions.")


class AnswerOut(BaseModel):
    type: Literal["choice", "score", "noul"]
    choice: str | None = Field(None, description="choice: the chosen option.")
    score: float | None = Field(None, description="score: expected level.")
    noul: float | None = Field(None, description="noul: probability that it is true.")
    confidence: float | None = Field(None, description="choice and score: probability of the answer.")
    legend: dict[str, str] | None = Field(None, description="score: text of each level.")
    probabilities: dict[str, float] | None = Field(None, description="choice and score: probability of each option.")


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class DecideResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": RESPONSE_EXAMPLE})
    model: str = Field(description='Version that answered, e.g. "temis-1.0.11".')
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
bearer = HTTPBearer(auto_error=False, description="The token you were given for this API.")

app = FastAPI(
    title="Temis API",
    version=MODEL.version,
    description=DESCRIPTION,
    root_path=os.environ.get("TEMIS_ROOT_PATH", ""),
    # keep the token after a reload, and show how long each call took
    swagger_ui_parameters={"persistAuthorization": True, "displayRequestDuration": True},
    openapi_tags=[{"name": "decisions", "description": "Ask Temis."},
                  {"name": "service", "description": "Model status and version."}],
)
def _doc_error(description: str, example: dict) -> dict:
    return {"model": ErrorResponse, "description": description,
            "content": {"application/json": {"example": example}}}


ERRORS = {
    400: _doc_error("Only when the request is malformed.",
                    _err("api_usage_error", "Invalid request. questions.usefulness.score.criteria: Field required")),
    401: _doc_error("Missing or invalid token: click Authorize above and paste the token.",
                    _err("authentication_error",
                         "Missing or invalid API token.")),
    429: _doc_error("More than 120 requests per minute.",
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


DECIDE_DOC = dict(response_model=DecideResponse, response_model_exclude_none=True, tags=["decisions"],
                  dependencies=[Depends(require_token)], responses=ERRORS)


@app.post("/v1/decide", summary="Answer typed questions about a situation", **DECIDE_DOC)
def decide_route(body: DecideRequest, response: Response):
    return decide(body, response)


@app.post("/v1/systemone", include_in_schema=False, **DECIDE_DOC)  # alias kept for older clients
def systemone_route(body: DecideRequest, response: Response):
    return decide(body, response)


@app.get("/v1/model", response_model=ModelInfo, tags=["service"], dependencies=[Depends(require_token)],
         responses={401: ERRORS[401]}, summary="Version and configuration of the loaded model")
def model_info():
    cfg = MODEL.decider.config
    return {"model": MODEL.id, "name": MODEL.name, "version": MODEL.version, "run": MODEL.decider.version,
            "base": cfg.get("base"), "max_len": cfg.get("max_len"), "calibration": MODEL.decider.calibration}


@app.get("/health", response_model=Health, tags=["service"], summary="The service is up (no token)")
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
