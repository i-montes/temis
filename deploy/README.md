# Temis API

Temis toma decisiones tipadas sobre una situación: elige una opción (`choice`), da un nivel (`score`)
o dice si una afirmación es verdadera (`noul`).

Una petición lleva `model`, `state` (la situación), `questions` y, opcionalmente, `context`
(pasajes de evidencia, para RAG). La respuesta trae `model`, `answers` (una por pregunta) y `usage`,
y los encabezados `x-temis-*` con datos de la petición.

- **URL base:** `http://149.130.184.223/temis`
- **Documentación interactiva:** <http://149.130.184.223/temis/docs> (Swagger: pulsa *Authorize*,
  pega el token y prueba las rutas) · <http://149.130.184.223/temis/redoc>
- **Esquema OpenAPI:** `http://149.130.184.223/temis/openapi.json`

> Va por HTTP sobre la IP, sin cifrar: el token viaja en claro. Úsalo desde redes de confianza y
> cámbialo si sospechas que se filtró (ver «Cambiar el token»).

Las claves fijas del JSON están en inglés. Los ids de pregunta, las claves de opción y
los campos del `state` los eliges tú; en los ejemplos van en inglés y los textos en español (Temis
entiende los dos idiomas).

## Autenticación

Todas las rutas bajo `/v1` piden:

```
Authorization: Bearer <token>
```

Sin token o con uno incorrecto responden `401`. `/health` y la documentación no piden token.

## Rutas

| Método | Ruta | Qué hace |
|---|---|---|
| POST | `/v1/decide` | Responde las preguntas sobre el estado |
| GET | `/v1/model` | Versión y calibración del modelo cargado |
| GET | `/health` | `{"status":"ok","model":"temis-1.0.11"}` (sin token) |

## Petición

```json
{
  "model": "temis-latest",
  "state": {"request": "¿Cuál es la última versión publicada de numpy en PyPI?"},
  "questions": {
    "tool": {
      "type": "choice",
      "instructions": "¿Qué herramienta usa primero el agente?",
      "criteria": {
        "search_code": "Buscar en el repositorio abierto.",
        "web": "Consultar internet.",
        "memory": "Consultar las memorias guardadas.",
        "none": "Responder sin herramientas."
      }
    },
    "needs_tool": {
      "type": "noul",
      "instructions": "El agente necesita usar al menos una herramienta para responder bien."
    },
    "urgency": {
      "type": "score",
      "instructions": "¿Qué tan urgente es la petición?",
      "criteria": ["Nada urgente", "Puede esperar unas horas", "Hay que hacerlo ya"]
    }
  },
  "context": ["opcional: pasajes de evidencia", "se marcan como datos, no como órdenes"]
}
```

- `model`: `"temis-latest"` o una versión exacta, p. ej. `"temis-1.0.11"`, para
  que una actualización no te tome por sorpresa (si no está cargada responde `400`). Si lo omites,
  usa `temis-latest`.
- `state`: texto o JSON (se recorta a 6000 caracteres).
- `questions`: `id → pregunta`, hasta 20 por petición.
  - `choice`: `criteria` es `clave → descripción`, de 2 a 12 opciones.
  - `score`: `criteria` es la lista de niveles del más bajo al más alto (2 a 12).
  - `noul`: `instructions` es una **afirmación** verdadera o falsa sobre el estado, no una orden
    («La propuesta corrige un dato ya guardado.», no «Determina si…»). `criteria` es opcional.
- `context`: opcional, texto o lista de pasajes (hasta
  8000 caracteres). Se numeran y se le pasan al modelo como datos, no como órdenes.

## Respuesta

```json
{
  "model": "temis-1.0.11",
  "answers": {
    "tool": {"type": "choice", "choice": "web", "confidence": 0.64,
                    "probabilities": {"search_code": 0.08, "web": 0.64, "memory": 0.05, "none": 0.23}},
    "needs_tool": {"type": "noul", "noul": 0.81},
    "urgency": {"type": "score", "score": 0.7, "confidence": 0.45,
                 "legend": {"0": "Nada urgente", "1": "Puede esperar unas horas", "2": "Hay que hacerlo ya"},
                 "probabilities": {"0": 0.45, "1": 0.4, "2": 0.15}}
  },
  "usage": {"input_tokens": 155, "output_tokens": 0}
}
```

- `choice`: la opción elegida, su probabilidad (`confidence`) y la de cada opción.
- `score`: el nivel esperado (promedio ponderado), la probabilidad del nivel más probable, el texto de
  cada nivel (`legend`) y la probabilidad de cada nivel.
- `noul`: la probabilidad de que la afirmación sea verdadera (sin `probabilities`).
- `usage`: tokens que leyó el modelo; `output_tokens` siempre es 0 porque Temis no genera texto.
- Las probabilidades están calibradas: con confianza 0,7 acierta más o menos 7 de cada 10 veces.

Encabezados de la respuesta:

| Encabezado | Qué dice |
|---|---|
| `x-temis-request-id` | id de la petición |
| `x-temis-run` | la versión exacta (`v1108-2307`) |
| `x-temis-uncertainty` | 0 si está segura de todo, 1 si alguna pregunta es un volado: sirve para decidir cuándo pedir una segunda opinión |
| `x-temis-latency-ms` | lo que tardó el modelo |

## Errores

Todos con la forma `{"detail": {"error_type": "...", "message": "..."}}`.

| Código | `error_type` | Cuándo |
|---|---|---|
| 400 | `api_usage_error` | La petición no cumple el esquema (`message` dice qué campo) o `model` no está cargado |
| 401 | `authentication_error` | Falta el token o no es válido |
| 413 | `api_usage_error` | Petición de más de 256 KB |
| 429 | `rate_limit_error` | Más de 120 peticiones por minuto por IP (ráfagas de 20) |
| 503 | `overloaded_error` | El contenedor no está respondiendo |

A diferencia del servidor local, aquí responde solo el modelo: no usa los
precedentes de tus etiquetas ni la memoria de Clio, porque no viven en el servidor.

## Ejemplos

```sh
curl -s http://149.130.184.223/temis/v1/decide \
  -H "Authorization: Bearer $TEMIS_API_TOKEN" -H 'content-type: application/json' \
  -d '{"model":"temis-latest","state":"Iván dice que ya no vive en Bogotá, ahora vive en Medellín.",
       "questions":{"contradicts":{"type":"noul","instructions":"La propuesta contradice una memoria anterior."}}}'
```

```python
import os, requests
r = requests.post("http://149.130.184.223/temis/v1/decide",
                  headers={"Authorization": f"Bearer {os.environ['TEMIS_API_TOKEN']}"},
                  json={"model": "temis-latest", "state": {...}, "questions": {...}}, timeout=30)
r.raise_for_status()
print(r.json()["answers"])
```

## Operación (en el VPS, `~/containers/temis`)

- Contenedor `temis-api` (imagen `temis-api`), escucha solo en `127.0.0.1:8093`; nginx lo publica en
  `/temis/` (`/etc/nginx/snippets/temis-location.conf`, incluido desde `sites-enabled/laya`, y los
  límites en `/etc/nginx/conf.d/temis.conf`). Reinicio `unless-stopped`, 2 GB y 2 CPU como máximo.
- El modelo se monta de `./models/<run>/` (solo lectura); `.env` (permisos 600) guarda
  `TEMIS_TOKEN` y `TEMIS_MODEL_DIR`.
- Logs: `docker logs -f temis-api`.

**Desplegar otra versión** (desde el Mac, en la carpeta del decisor):

```sh
deploy/deploy.sh              # la versión activa en local
deploy/deploy.sh v1108-2307   # una versión concreta
```

Copia el código y el modelo int8, apunta `TEMIS_MODEL_DIR` a esa versión, reconstruye y espera a
`/health`. Conserva el token.

**Cambiar el token:**

```sh
ssh vps 'cd ~/containers/temis && sed -i "s|^TEMIS_TOKEN=.*|TEMIS_TOKEN=$(python3 -c "import secrets;print(secrets.token_urlsafe(32))")|" .env && docker compose up -d && grep TEMIS_TOKEN .env'
```
