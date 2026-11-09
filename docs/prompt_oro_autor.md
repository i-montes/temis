# Prompt para pedirle al autor la respuesta correcta de sus casos

Los lotes escritos antes de que el formato llevara `expected` no traen la respuesta correcta como
campo (solo en prosa dentro de `rationale`). Este prompt se lo da al **mismo modelo que escribió los
casos** para que la entregue; se importa como oro del autor y `scripts.build_targets` lo usa como
objetivo cuando Jev o la segunda opinión coinciden con él (`temis/data/balance.py::hard_target`).

```sh
./.venv/bin/python -m scripts.second_opinion export casos.jsonl --all     # los casos difíciles
# ... el autor responde con el bloque de abajo; guarda su salida en oro.jsonl
./.venv/bin/python -m scripts.second_opinion import oro.jsonl --by NOMBRE-DEL-MODELO --as gold
./.venv/bin/python -m scripts.build_targets
```

---

```text
Tú escribiste estos casos de decisión tipada. Cada línea del archivo adjunto es uno:
{"id": "...", "family": "...", "state": ..., "questions": {...}, "rationale": "..."}

Para cada caso, entrega la respuesta CORRECTA de cada pregunta: la que justifica el rationale que
escribiste, comprobada contra el state. Si al releer el state ves que el rationale no sostiene
ninguna opción de forma defendible (el caso está mal escrito o dos opciones son igual de válidas),
responde null en esa pregunta.

FORMATO DE SALIDA (obligatorio)
- Solo JSON Lines, una línea por caso, sin texto antes ni después, sin ``` ni comentarios:
  {"id": "<el id del caso>", "answers": {"<id_pregunta>": "<clave de opción>" | null, ...}}
- Claves de opción exactamente como en "criteria" (choice), el índice del nivel como texto (score)
  o "true"/"false" (noul). Responde todas las preguntas de todos los casos. Sin explicaciones.
```
