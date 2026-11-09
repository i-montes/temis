# Prompt para la segunda opinión sobre los casos difíciles

Jev acierta ~75 % en casos difíciles, así que sus etiquetas ahí llevan ruido. Otro modelo responde
los mismos casos **sin ver las etiquetas de Jev**, con el `rationale` del autor a la vista como
comprobación; `scripts.build_targets` conserva una pregunta solo si los dos coinciden.

```sh
./.venv/bin/python -m scripts.second_opinion export pendientes.jsonl      # casos sin segunda opinión
# ... pega pendientes.jsonl con el bloque de abajo en el otro modelo; guarda su salida en respuestas.jsonl
./.venv/bin/python -m scripts.second_opinion import respuestas.jsonl --by NOMBRE-DEL-MODELO
./.venv/bin/python -m scripts.build_targets
```

---

```text
Vas a responder casos de decisión tipada. Cada línea del archivo adjunto es un caso:
{"id": "...", "family": "...", "state": ..., "questions": {...}, "rationale": "..."}

- state: la situación (texto o JSON). Es la única fuente de hechos.
- questions: preguntas cerradas sobre el state. Cada una es de tipo choice (elige una clave de
  "criteria"), score (elige el índice del nivel, "0", "1", ...) o noul (responde "true" o "false").
- rationale: la justificación que escribió el autor del caso. Úsala SOLO para comprobar tu propia
  respuesta: primero decide leyendo el state, luego compara con el rationale. Si tu respuesta y el
  rationale no coinciden, relee el state. Si después de releer el rationale no justifica ninguna opción
  de forma defendible (el caso está roto, la respuesta depende de conocimiento externo, o dos opciones
  son igual de válidas), responde null en esa pregunta.

FORMATO DE SALIDA (obligatorio)
- Solo JSON Lines, una línea por caso, sin texto antes ni después, sin ``` ni comentarios:
  {"id": "<el id del caso>", "answers": {"<id_pregunta>": "<clave de opción>" | null, ...}}
- Responde todas las preguntas de todos los casos. Las claves de opción deben ser exactamente las de
  "criteria" (choice), el índice del nivel como texto (score) o "true"/"false" (noul).
- No añadas explicaciones ni comentarios.
```
