# Temis API

Temis makes typed decisions about a situation: it picks an option (`choice`), gives a level
(`score`) or says whether a statement is true (`noul`).

A request carries `model`, `state` (the situation), `questions` and, optionally, `context`
(evidence passages, for RAG). The response has `model`, `answers` (one per question) and `usage`,
plus `x-temis-*` headers with details about the request.

- **Base URL:** `http://149.130.184.223/temis`
- **Interactive docs:** <http://149.130.184.223/temis/docs> (Swagger: click *Authorize*, paste the
  token and try the routes) · <http://149.130.184.223/temis/redoc>
- **OpenAPI schema:** `http://149.130.184.223/temis/openapi.json`

> It is served over plain HTTP on the IP, unencrypted: the token travels in the clear. Use it from
> trusted networks and change it if you think it leaked (see "Changing the token").

The fixed JSON keys are in English. Question ids, option keys and the fields inside `state` are yours
to choose. Texts can be in English or Spanish: Temis was trained on both (62 % of its cases are
Spanish, so it is somewhat stronger there).

## Authentication

Every route under `/v1` requires:

```
Authorization: Bearer <token>
```

Without a token, or with a wrong one, they answer `401`. `/health` and the docs need no token.

## Routes

| Method | Route | What it does |
|---|---|---|
| POST | `/v1/decide` | Answers the questions about the state |
| GET | `/v1/model` | Version and calibration of the loaded model |
| GET | `/health` | `{"status":"ok","model":"temis-2.0.0"}` (no token) |

## Request

```json
{
  "model": "temis-latest",
  "state": {"request": "What is the latest numpy version published on PyPI?"},
  "questions": {
    "tool": {
      "type": "choice",
      "instructions": "Which tool does the agent use first?",
      "criteria": {
        "search_code": "Search the open repository.",
        "web": "Look it up on the internet.",
        "memory": "Check the stored memories.",
        "none": "Answer without tools."
      }
    },
    "needs_tool": {
      "type": "noul",
      "instructions": "The agent needs to use at least one tool to answer well."
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is the request?",
      "criteria": ["Not urgent", "Can wait a few hours", "Must be done now"]
    }
  }
}
```

- `model`: `"temis-latest"` or an exact version, e.g. `"temis-2.0.0"`, so an update does not catch
  you by surprise (if that version is not loaded it answers `400`). If omitted, `temis-latest`.
- `state`: text or JSON (cut at 6,000 characters).
- `questions`: `id → question`, up to 20 per request.
  - `choice`: `criteria` is `key → description`, 2 to 12 options.
  - `score`: `criteria` is the list of levels from lowest to highest (2 to 12).
  - `noul`: `instructions` is a true-or-false **statement** about the state, not an instruction
    ("The proposal corrects a stored fact.", not "Determine whether…"). `criteria` is optional.
- `context`: optional, a string or a list of passages (up to 8,000 characters). They are numbered
  and given to the model as data, not as instructions. See the second example below.

## Response

The real response to the request above:

```json
{
  "model": "temis-2.0.0",
  "answers": {
    "tool": {"type": "choice", "choice": "web", "confidence": 0.8,
             "probabilities": {"search_code": 0.11, "web": 0.8, "memory": 0.01, "none": 0.08}},
    "needs_tool": {"type": "noul", "noul": 0.5},
    "urgency": {"type": "score", "score": 0.43, "confidence": 0.72,
                "legend": {"0": "Not urgent", "1": "Can wait a few hours", "2": "Must be done now"},
                "probabilities": {"0": 0.72, "1": 0.14, "2": 0.15}}
  },
  "usage": {"input_tokens": 178, "output_tokens": 0}
}
```

`needs_tool` came out at 0.5 here: a coin flip, which `x-temis-uncertainty` reports as high.

- `choice`: the chosen option, its probability (`confidence`) and that of every option.
- `score`: the expected level (weighted average), the probability of the most likely level, the
  text of each level (`legend`) and the probability of each level.
- `noul`: the probability that the statement is true (no `probabilities`).
- `usage`: tokens the model read; `output_tokens` is always 0 because Temis does not generate text.
- Probabilities are calibrated: answers with confidence 0.7 are right about 7 times out of 10.

Response headers:

| Header | What it says |
|---|---|
| `x-temis-request-id` | request id |
| `x-temis-run` | the exact version (`v1108-2307`) |
| `x-temis-uncertainty` | 0 when it is sure of everything, 1 when some question is a coin flip: useful to decide when to ask for a second opinion |
| `x-temis-latency-ms` | how long the model took |

## Errors

All shaped as `{"detail": {"error_type": "...", "message": "..."}}`.

| Code | `error_type` | When |
|---|---|---|
| 400 | `api_usage_error` | The request does not match the schema (`message` says which field), or `model` is not loaded |
| 401 | `authentication_error` | Missing or invalid token |
| 413 | `api_usage_error` | Request larger than 256 KB |
| 429 | `rate_limit_error` | More than 120 requests per minute per IP (bursts of 20) |
| 503 | `overloaded_error` | The container is not responding |

Unlike the local server, this one answers with the model alone: it does not use the precedents
from your labels or Clio's memory, since neither lives on the server.

## Examples

With context (real answer: `{"contradicts": {"type": "noul", "noul": 0.9}}`):

```sh
curl -s http://149.130.184.223/temis/v1/decide \
  -H "Authorization: Bearer $TEMIS_API_TOKEN" -H 'content-type: application/json' \
  -d '{"model":"temis-latest",
       "state":"Iván says he no longer lives in Bogotá; he now lives in Medellín.",
       "context":["Memory m3: Iván lives in Bogotá."],
       "questions":{"contradicts":{"type":"noul","instructions":"The new information contradicts a stored memory."}}}'
```

```python
import os, requests
r = requests.post("http://149.130.184.223/temis/v1/decide",
                  headers={"Authorization": f"Bearer {os.environ['TEMIS_API_TOKEN']}"},
                  json={"model": "temis-latest", "state": {...}, "questions": {...}}, timeout=30)
r.raise_for_status()
print(r.json()["answers"])
```

## Operations (on the VPS, `~/containers/temis`)

- Container `temis-api` (image `temis-api`), listening only on `127.0.0.1:8093`; nginx publishes it
  at `/temis/` (`/etc/nginx/snippets/temis-location.conf`, included from `sites-enabled/laya`, with
  the limits in `/etc/nginx/conf.d/temis.conf`). Restart policy `unless-stopped`, at most 2 GB and
  2 CPUs.
- The model is mounted from `./models/<run>/` (read-only); `.env` (mode 600) holds `TEMIS_TOKEN` and
  `TEMIS_MODEL_DIR`.
- Logs: `docker logs -f temis-api`.

**Deploying another version** (from the Mac, in the project folder):

```sh
deploy/deploy.sh              # the version active locally
deploy/deploy.sh v1108-2307   # a given run
```

It copies the code and the int8 model, points `TEMIS_MODEL_DIR` at that version, rebuilds and waits
for `/health`. The token is kept.

**Changing the token:**

```sh
ssh vps 'cd ~/containers/temis && sed -i "s|^TEMIS_TOKEN=.*|TEMIS_TOKEN=$(python3 -c "import secrets;print(secrets.token_urlsafe(32))")|" .env && docker compose up -d && grep TEMIS_TOKEN .env'
```
