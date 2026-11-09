# Temis: lo que debe estar listo antes de la próxima corrida — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dejar el código de Temis listo para la corrida limpia de la fase 1: medir JevBench localmente (v1.2 y v1.5), quitar los cortes de entrada, calibrar por tipo × nº de opciones, cargar `procedural-typed-decisions` con topes de mezcla y conjunto interno, entrenar con consistencia por permutación y entradas largas en CUDA/MPS, e importar casos difíciles escritos por un modelo privado con `rationale` y segunda opinión. **No genera datos ni entrena.**

**Architecture:** Lógica nueva como funciones puras en `temis/` (puntuación de JevBench, calibración, carga procedural, pérdida de consistencia) con scripts finos en `scripts/` que las orquestan; el runtime ONNX no cambia de forma, solo de longitud y temperaturas. Todo lo que lee datos externos vive bajo `external/` (ignorado por git) y se descarga bajo demanda.

**Tech Stack:** Python 3.12, torch 2.15, transformers 5.19, onnxruntime 1.31, `datasets` 5.1, `tokenizers`, numpy, pytest. Sin dependencias nuevas.

**Spec:** `docs/specs/2026-11-09-plan-jevbench.md` (secciones A, B, C1, C2–C4, C6, D1, D4, H). Léelo antes de empezar.

## Global Constraints

- CPU-first: el runtime (`temis/runtime.py`) sigue dependiendo solo de `onnxruntime`, `tokenizers`, `numpy`. Nada de torch ahí.
- Los ítems públicos de JevBench (`external/jevbench/public/*.jsonl`) **nunca** entran en `data/` ni en ningún target. Solo evaluación.
- Tamaño int8 ≤ 400 MB; latencia p50 en casos cortos ≤ 50 ms: nada de este plan añade cómputo en inferencia salvo leer hasta `long_max_len` (1 024) cuando el estado es largo.
- `typed-decisions test` sigue sin usarse para entrenar. El conjunto interno "hard-like" (`data/hardlike.jsonl`) tampoco.
- Topes de mezcla en `build_targets`: procedural ≤ 35 % de los targets; se imprime la cuota de cada fuente.
- Español e inglés: nada de lo nuevo asume inglés (los prompts piden ambos; el loader procedural marca `language: en`).
- Commits sin `Co-Authored-By` ni firmas de herramienta (CLAUDE.md global). Autor: Iván.
- Tests: `./.venv/bin/pytest -q tests` debe seguir en verde después de cada tarea; los tests nuevos no necesitan modelos entrenados (solo el tokenizer de mmBERT-small que ya usan los existentes).

## Review Focus

1. **Estados largos en la API**: un `long_policy` de 15 000 caracteres debe llegar entero a `render_state` y codificarse hasta 1 024 tokens sin error 422 ni corte silencioso → test en Tarea 2 (`test_render_state_keeps_long_text`, `test_decider_long_max_len_from_config`).
2. **`noul` en forma de pregunta** ("Is this return eligible…?"): `clean_questions` no debe descartarla al importar casos difíciles → test en Tarea 7 (`test_import_keeps_question_form_noul`).
3. **Procedural con `noul` sin criterios y `choice` con `criteria: {label: null}`**: `parse_questions` exige descripciones de texto; el loader debe rellenar `null` con la propia clave → test en Tarea 5 (`test_procedural_null_criteria_become_keys`).
4. **Consistencia por permutación en `score`**: las opciones de `score` no se barajan; la pérdida debe seguir definida (KL entre dos pasadas con dropout distinto, mismo orden) y no romper por `order` idéntico → test en Tarea 6 (`test_consistency_loss_score_same_order`).
5. **Banda de abstención con `p = 0.80` exacto**: cuenta como decidido (≥ 0,80 es "sí"); `0.7999` no → test en Tarea 1 (`test_noul_band_edges`).

---

### Tarea 1: Puntuación de JevBench como funciones puras + harness local

**Files:**
- Create: `temis/jevbench.py`
- Create: `scripts/eval_jevbench.py`
- Create: `tests/test_jevbench.py`
- Modify: `.gitignore` (nada: `external/` ya está ignorado)

**Interfaces:**
- Produces:
  - `PUBLIC_FILES = ("easy", "original", "hard")`; `RAW = "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/datasets/public/{name}.jsonl"`; `PUBLIC_DIR = Path("external/jevbench/public")`
  - `fetch_public(dir: Path = PUBLIC_DIR) -> list[Path]` descarga los tres archivos si faltan (urllib, sin dependencias).
  - `load_public(dir: Path = PUBLIC_DIR) -> list[dict]` lee los tres y añade a cada ítem `tier` = `"easy" | "standard" | "hard"` (de `easy` / `original` / `hard`).
  - `to_request(item: dict) -> tuple[Any, dict]`: `(state, {"decision": question})`; la question es la del ítem tal cual (TypeSafe: `choice` criteria dict, `score` criteria list, `noul` criteria `{"true","false"}`).
  - `probs_for_labels(item: dict, answer: Answer) -> dict[str, float]`: mapea la `Answer` de Temis a los `labels` del ítem (noul: `{"yes": p_true, "no": 1-p_true}`; score: claves `"0".."k"`; choice: tal cual), renormalizado.
  - `score_item(item: dict, probs: dict[str, float]) -> dict` con claves: `correct` (argmax == expected; para score, argmax == str(expected)), `v15_correct` (noul: además `p_yes <= 0.20 or p_yes >= 0.80`; score: igual que `correct`; choice: igual), `conf` (max prob), `chance` (1/len(labels)), `nmae` (solo score: `|EV - expected| / (k-1)`), `nmae_chance` (solo score: `|(k-1)/2 - expected| / (k-1)`), `tvd` (solo si `provenance.gold_probs`: ½·Σ|p - gold|).
  - `summarize(rows: list[dict]) -> dict` donde cada row es `{**score_item(...), "tier", "family", "type", "id", "ms"}`. Devuelve: `n`, `accuracy` (v1.2), `by_tier`, `by_family` (clave `"{tier}/{family}"`), `by_type` — cada uno `{"n", "correct", "accuracy"}` —, `ece` (10 bins, top-label), `latency_ms` `{p50, p95}`, y `v15` = `{"accuracy", "noul_in_band", "cc_by_tier": {tier: {"choice", "noul", "score", "tier"}}, "intelligence"}` con:
    - `cc_choice = 100·(acc − mean chance)/(1 − mean chance)`, idem noul (usando `v15_correct`), recortado a ≥ 0;
    - `cc_score = 100·(1 − mean nmae / mean nmae_chance)`, recortado a ≥ 0;
    - `cc_tier = (50·cc_choice + 25·cc_noul + 25·cc_score) / suma de pesos de los tipos presentes`;
    - `intelligence = Σ w_tier·cc_tier / Σ w_tier` con `w = {"easy": 10, "standard": 20, "hard": 40}` (judge no es público). Documentar en el docstring: **estimación local, no es la puntuación oficial**.

- [ ] **Step 1: Escribir los tests que fallan** en `tests/test_jevbench.py`

```python
from temis.jevbench import probs_for_labels, score_item, summarize, to_request
from temis.schema import Answer

def item(kind, labels, expected, **extra):
    q = {"choice": {"type": "choice", "instructions": "Pick.", "criteria": {l: l for l in labels}},
         "noul": {"type": "noul", "instructions": "Is it?", "criteria": {"true": "yes", "false": "no"}},
         "score": {"type": "score", "instructions": "Rate.", "criteria": [f"level {i}" for i in labels]}}[kind]
    return {"id": f"hard-x-{kind}-01", "family": "x", "state": "s", "labels": labels, "expected": expected,
            "question": q, "provenance": extra, "tier": "hard"}

def test_to_request_wraps_question_as_decision():
    st, qs = to_request(item("choice", ["a", "b"], "a"))
    assert st == "s" and list(qs) == ["decision"] and qs["decision"]["type"] == "choice"

def test_noul_band_edges():
    it = item("noul", ["no", "yes"], "yes")
    assert score_item(it, {"yes": 0.80, "no": 0.20})["v15_correct"] is True
    r = score_item(it, {"yes": 0.7999, "no": 0.2001})
    assert r["correct"] is True and r["v15_correct"] is False

def test_score_uses_argmax_for_v12_and_nmae_for_v15():
    it = item("score", ["0", "1", "2", "3"], 1)
    r = score_item(it, {"0": 0.1, "1": 0.6, "2": 0.2, "3": 0.1})
    assert r["correct"] and abs(r["nmae"] - abs(1.3 - 1) / 3) < 1e-9 and abs(r["nmae_chance"] - 0.5 / 3) < 1e-9

def test_probs_for_labels_maps_noul_and_score():
    a = Answer("noul", "true", {"false": 0.3, "true": 0.7}, 0.7)
    assert probs_for_labels(item("noul", ["no", "yes"], "yes"), a) == {"no": 0.3, "yes": 0.7}
    a = Answer("score", "2", {"0": 0.1, "1": 0.2, "2": 0.7}, 0.7, expected=1.6)
    assert probs_for_labels(item("score", ["0", "1", "2"], 2), a) == {"0": 0.1, "1": 0.2, "2": 0.7}

def test_summarize_tier_weights_and_intelligence():
    rows = [{**score_item(item("choice", ["a", "b", "c", "d"], "a"), {"a": 1.0, "b": 0, "c": 0, "d": 0}),
             "tier": "hard", "family": "x", "type": "choice", "id": "1", "ms": 10.0}]
    s = summarize(rows)
    assert s["accuracy"] == 1.0 and s["by_tier"]["hard"]["accuracy"] == 1.0
    assert s["v15"]["cc_by_tier"]["hard"]["choice"] == 100.0 and s["v15"]["intelligence"] == 100.0
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `./.venv/bin/pytest tests/test_jevbench.py -q`
Expected: FAIL con `ModuleNotFoundError: temis.jevbench`

- [ ] **Step 3: Implementar `temis/jevbench.py`** con las firmas del bloque Interfaces. `summarize` calcula ECE como `scripts/evaluate.py::ece` (cópialo o impórtalo de ahí; no duplicar: mover `ece` a `temis/jevbench.py` y que `scripts/evaluate.py` lo importe).

- [ ] **Step 4: Correr los tests**

Run: `./.venv/bin/pytest tests/test_jevbench.py tests/test_core.py -q`
Expected: PASS

- [ ] **Step 5: Implementar `scripts/eval_jevbench.py`**

```
python -m scripts.eval_jevbench --model runs/<run>/onnx [--fetch] [--limit N] [--calibration soft|accuracy|jevbench]
```
Usa `Decider(model_dir, calibration=...)` directamente (sin servidor), mide `ms` por ítem (`time.perf_counter`), escribe `runs/<run>/eval_jevbench.json` = `summarize(rows)` + `"rows"` (id, tier, family, predicted, correct, v15_correct, conf, ms) y `"calibration"`, e imprime la tabla por tier/familia y las dos cifras: `accuracy` (v1.2) e `intelligence` (v1.5 est.). Si no existen los archivos públicos y no se pasa `--fetch`, aborta con el comando a ejecutar.

- [ ] **Step 6: Verificar contra el resultado publicado**

Run: `./.venv/bin/python -m scripts.eval_jevbench --model runs/v1108-2307/onnx --fetch --calibration accuracy`
Expected: `accuracy` entre 0,56 y 0,58 (el run oficial dio 0,571 con la misma calibración vía API), `by_tier.easy.accuracy == 1.0`, `v15.noul_in_band` ≈ 43/74. Si difiere en más de 2 puntos, revisar `probs_for_labels` (el adaptador de JevBench renormaliza y redondea a 2 decimales; aquí no se redondea).

- [ ] **Step 7: Commit**

```bash
git add temis/jevbench.py scripts/eval_jevbench.py tests/test_jevbench.py scripts/evaluate.py
git commit -m "JevBench local: puntuación v1.2 y v1.5 estimada, harness sin servidor"
```

---

### Tarea 2: Quitar los cortes de entrada (B1, B2)

**Files:**
- Modify: `temis/schema.py:113` (`render_state`)
- Modify: `temis/runtime.py:45-52` (`Decider.__init__`, `Decider.logits`)
- Modify: `temis/api.py` (límite de tamaño del `state`: buscar `MAX_STATE` / el `Field(max_length=...)` del estado cerca de la línea 160)
- Modify: `scripts/export_onnx.py:70-73` (config.json)
- Test: `tests/test_core.py`

**Interfaces:**
- Produces: `render_state(state, max_chars: int = 40_000)`; `Decider.long_max_len: int` (de `config["long_max_len"]`, default 1024); `Decider.logits(qs, state_text, context=None)` codifica con `max_len = 2048 if context else self.long_max_len`; `config.json` del export lleva `"long_max_len": meta.get("long_max_len", 1024)`.

- [ ] **Step 1: Tests que fallan** (añadir a `tests/test_core.py`)

```python
def test_render_state_keeps_long_text():
    assert len(render_state("x" * 20_000)) == 20_000

def test_encoder_long_max_len_keeps_state_up_to_budget():
    enc = Encoder(tok_path(), 640)
    q = parse_questions(Q)[0]
    short = enc.encode(q, "palabra " * 3000, max_len=640)
    long = enc.encode(q, "palabra " * 3000, max_len=1024)
    assert len(short.ids) == 640 and len(long.ids) == 1024
```

- [ ] **Step 2: Correr y ver que falla el primero**

Run: `./.venv/bin/pytest tests/test_core.py -q -k "long"`
Expected: `test_render_state_keeps_long_text` FAIL (devuelve 6000); el segundo pasa ya (documenta que el encoder respeta el presupuesto).

- [ ] **Step 3: Implementar**: `render_state` default 40 000; en `Decider.__init__` leer `self.long_max_len = int(self.config.get("long_max_len", 1024))` y el `Encoder` con `max(self.config.get("max_len", 1024), self.long_max_len)`; en `logits` usar `max_len=2048 if context else self.long_max_len`; en `api.py` subir el límite del estado a 60 000 caracteres (y el docstring de la línea ~89 que lo documenta); en `export_onnx.py` escribir `long_max_len` en `config.json`.

- [ ] **Step 4: Tests**

Run: `./.venv/bin/pytest tests -q`
Expected: PASS

- [ ] **Step 5: Medir el efecto sin reentrenar**

Run: `./.venv/bin/python -m scripts.eval_jevbench --model runs/v1108-2307/onnx --calibration accuracy` (añadir a mano `"long_max_len": 1024` al `config.json` del run activo, ya que no se reexporta)
Expected: hard `long_policy` y `multi_hop` ya no por debajo del azar; `latency_ms.p95` ≤ 400 en el Mac. Anotar las cifras en el mensaje del commit.

- [ ] **Step 6: Commit**

```bash
git add temis/schema.py temis/runtime.py temis/api.py scripts/export_onnx.py tests/test_core.py runs/registry.json
git commit -m "Estados largos: render_state hasta 40k caracteres y long_max_len 1024 en inferencia"
```

---

### Tarea 3: Pruebas de sesgo (A3) en el harness

**Files:**
- Modify: `temis/jevbench.py` (añadir `probes`)
- Modify: `scripts/eval_jevbench.py` (`--probes`)
- Test: `tests/test_jevbench.py`

**Interfaces:**
- Produces: `rotate_question(q: dict, k: int) -> dict` (rota `criteria` de un `choice` k posiciones; `score`/`noul` se devuelven iguales); `options_only_request(item) -> tuple[str, dict]` (estado vacío `""` para medir cuánto adivina sin leer); `flip_rate(pred_a: list[str], pred_b: list[str]) -> float`.
- `summarize` acepta rows con claves opcionales `pred_rot1` y `pred_blank` y añade `probes = {"rotation_flip_rate", "options_only_accuracy", "first_option_rate": {"pred", "gold"}}`.

- [ ] **Step 1: Tests**

```python
from temis.jevbench import flip_rate, rotate_question

def test_rotate_question_keeps_descriptions_and_cycles():
    q = {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B", "c": "C"}}
    assert list(rotate_question(q, 1)["criteria"]) == ["b", "c", "a"] and rotate_question(q, 1)["criteria"]["a"] == "A"
    s = {"type": "score", "instructions": "i", "criteria": ["0", "1"]}
    assert rotate_question(s, 1) == s

def test_flip_rate():
    assert flip_rate(["a", "b", "c"], ["a", "c", "c"]) == 1 / 3
```

- [ ] **Step 2: Correr, ver fallar, implementar, correr** (`./.venv/bin/pytest tests/test_jevbench.py -q` → PASS). En `scripts/eval_jevbench.py`, con `--probes`, cada ítem `choice` con ≥ 3 opciones se decide dos veces más (rotación 1 y estado vacío); el coste es 3× solo en esa pasada.

- [ ] **Step 3: Medir línea base**

Run: `./.venv/bin/python -m scripts.eval_jevbench --model runs/v1108-2307/onnx --calibration accuracy --probes`
Expected: `probes.rotation_flip_rate` reportado (esperado > 10 % hoy); `first_option_rate.pred` > `first_option_rate.gold`.

- [ ] **Step 4: Commit** `git commit -am "JevBench: pruebas de sesgo (rotación, solo opciones, primera opción)"`

---

### Tarea 4: Calibración por tipo × nº de opciones orientada a v1.5 (B3)

**Files:**
- Create: `temis/calibration.py`
- Modify: `scripts/calibrate.py` (usar el módulo; escribir `temperatures_jevbench`)
- Modify: `temis/runtime.py` (`calibration="jevbench"`)
- Test: `tests/test_calibration.py`

**Interfaces:**
- Produces en `temis/calibration.py`:
  - `softmax(z: np.ndarray, t: float) -> np.ndarray`
  - `fit_nll(rows: list[tuple[np.ndarray, int]], grid=np.arange(0.2, 2.51, 0.05)) -> float` (lo que hoy hace `calibrate.py`)
  - `fit_noul_band(rows, lo=0.20, hi=0.80, grid=...) -> float`: temperatura que maximiza `#(argmax correcto y p_true fuera de (lo, hi))`; empate → la más cercana a 1,0.
  - `fit_score_nmae(rows: list[tuple[np.ndarray, int]], grid=...) -> float`: minimiza la media de `|EV − y| / (k−1)`.
  - `fit_jevbench(by_kind_n: dict[tuple[str, int], list[tuple[np.ndarray, int]]]) -> dict[str, dict[str, float]]`: `{"choice": {"3": T, "4": T, ...}, "noul": {"2": T}, "score": {"3": T, ...}}`; grupos con < 30 filas heredan la T del tipo (todas las n juntas).
  - `pick_temperature(temps: dict, kind: str, n: int) -> float`: `temps[kind][str(n)]` si existe, si no `temps[kind]["*"]`, si no 1,0.
- `Decider(calibration="jevbench")` usa `config["temperatures_jevbench"]` vía `pick_temperature`; si no existe cae a `"accuracy"`.
- `scripts/calibrate.py` escribe `temperatures_jevbench` en `config.json` y `meta.json`, además de lo actual; imprime por tipo×n: NLL-T, band-T, `% fuera de banda y correcto` antes/después.

- [ ] **Step 1: Tests**

```python
import numpy as np
from temis.calibration import fit_noul_band, fit_score_nmae, pick_temperature

def test_fit_noul_band_sharpens_correct_hesitant_answers():
    rows = [(np.array([0.0, 0.6]), 1)] * 10 + [(np.array([0.6, 0.0]), 0)] * 10  # p_true ≈ 0.65 / 0.35: en banda
    t = fit_noul_band(rows)
    assert t < 0.5                      # con T=1 ambos quedan en la banda; afilar los saca sin cambiar el argmax
    p = np.exp(rows[0][0] / t); assert p[1] / p.sum() >= 0.8

def test_fit_score_nmae_prefers_sharp_when_argmax_right():
    rows = [(np.array([0.0, 1.0, 0.0, 0.0]), 1)] * 5
    assert fit_score_nmae(rows) <= 1.0

def test_pick_temperature_falls_back():
    temps = {"choice": {"4": 0.7, "*": 0.9}}
    assert pick_temperature(temps, "choice", 4) == 0.7 and pick_temperature(temps, "choice", 7) == 0.9 and pick_temperature(temps, "noul", 2) == 1.0
```

- [ ] **Step 2: Correr y ver fallar** (`ModuleNotFoundError`), **implementar**, **correr**: `./.venv/bin/pytest tests/test_calibration.py -q` → PASS.

- [ ] **Step 3: Recablear `scripts/calibrate.py`** para usar `temis/calibration.py` (quitar su `softmax`/`ece` propios; `ece` viene de `temis/jevbench.py`), agrupar `val` por `(kind, q.n)` y escribir las tres familias de temperaturas.

- [ ] **Step 4: Aplicar al activo y medir**

Run: `./.venv/bin/python -m scripts.calibrate --name v1108-2307 && ./.venv/bin/python -m scripts.eval_jevbench --model runs/v1108-2307/onnx --calibration jevbench`
Expected: `v15.noul_in_band` baja de ~43 a < 20 y `v15.intelligence` sube; `accuracy` (v1.2) no baja más de 1 punto; `ece` no sube más de 0,03. Si sube más, bajar el techo de la grid a 0,4.

- [ ] **Step 5: Commit** `git add temis/calibration.py scripts/calibrate.py temis/runtime.py tests/test_calibration.py && git commit -m "Calibración por tipo y nº de opciones orientada a JevBench v1.5"`

---

### Tarea 5: `procedural-typed-decisions` como fuente, conjunto interno hard-like y topes de mezcla (C1, A2, C6)

**Files:**
- Create: `temis/data/procedural.py`
- Create: `scripts/build_hardlike.py`
- Create: `scripts/eval_hardlike.py`
- Modify: `temis/data/sources.py` (registrar la fuente)
- Modify: `scripts/build_targets.py` (emitir targets `procedural`, aplicar topes, imprimir cuotas)
- Modify: `temis/data/balance.py` (`cap_share`)
- Modify: `scripts/train.py:41-44` (`load_examples`: clave de validación para `procedural`)
- Test: `tests/test_procedural.py`

**Interfaces:**
- Produces en `temis/data/procedural.py`:
  - `DATASET = "tasksource/procedural-typed-decisions"`; `HOLDOUT_TASKS = ("multi_view_adjudication", "policy_under_uncertainty")` (nunca entrenan: miden transferencia); `PER_TASK = 2000`; `LEVEL_WEIGHT = {0: 0.5, 1: 1, 2: 1, 3: 1.5, 4: 1.5}`.
  - `to_case(row: dict) -> dict`: `{"id": f"proc-{row['id']}", "workflow": row["task"], "level": row["level"], "state": json.loads(row["state"]), "questions": <criteria null → la clave; noul sin criteria → se deja sin>, "gold": json.loads(row["answers"]), "source": "procedural", "language": "en"}`.
  - `gold_targets(case: dict) -> list[dict]`: por pregunta, `{"qid", "kind", "target"}` con target one-hot suavizado 0,9 (noul: `[1-p, p]` suavizado igual si p∈{0,1}; si `noul` viene fraccional, se usa tal cual).
  - `load_procedural(split: str = "train", per_task: int = PER_TASK, seed: int = 0, exclude: tuple = HOLDOUT_TASKS) -> list[dict]`: muestrea por tarea con `LEVEL_WEIGHT` (sin reemplazo, determinista), usando `datasets.load_dataset(DATASET, "all", split=split)`.
  - `load_hardlike(n: int = 2000, seed: int = 0) -> list[dict]`: del split `test`: la mitad de `HOLDOUT_TASKS` (todos los niveles) y la mitad de niveles 3–4 del resto.
- `scripts/build_hardlike.py` escribe `data/hardlike.jsonl` (un caso por línea, con `gold`); `scripts/eval_hardlike.py --model runs/<run>/onnx` evalúa como `scripts/evaluate.py` (reutiliza `score_one`; mover `score_one` y `evaluate(cases, ...)` a una función común en `scripts/evaluate.py` que acepte una lista de casos) y escribe `runs/<run>/eval_hardlike.json` con `accuracy`, `by_workflow`, `by_level`.
- `temis/data/balance.py`: `cap_share(rows: list[dict], source: str, max_share: float, seed: int = 0) -> list[dict]` deja a lo sumo `max_share` de las filas con `row["source"] == source` (muestreo determinista) y conserva el orden del resto.
- `scripts/build_targets.py`: después de typed-decisions y synth, añade `procedural` (`source: "procedural"`, `schema_id: f"proc-{task}-{level}"`, `teachers: ["gold"]`), aplica `cap_share(rows, "procedural", 0.35)` y al final imprime `[mix] source → n (share %)` para `td-train`, `synth`, `procedural` y, dentro de synth, la cuota de `agent:*` (familias de Clio) con aviso si < 35 %.
- `scripts/train.py::load_examples`: filas `procedural` usan `schema_id` como clave de validación (ya es el default para no-`td-train`); los casos se cargan vía `load_procedural()` en el dict `cases` solo si hay filas `procedural` en los targets (para no descargar 396k filas sin necesidad: el loader usa `per_task` del `meta`… **decisión:** `build_targets` guarda el caso completo en la fila, `"case": {...}`, como ya hace `user_targets`; así `train.py` no vuelve a cargar el dataset).

- [ ] **Step 1: Tests** (`tests/test_procedural.py`, sin red: filas construidas a mano)

```python
import json
from temis.data.procedural import gold_targets, to_case
from temis.data.balance import cap_share
from temis.schema import parse_questions

ROW = {"id": "arithmetic:train:0", "task": "arithmetic", "level": 2,
       "state": json.dumps({"opening_balance": 35}),
       "questions": json.dumps({"final": {"type": "choice", "instructions": "Closing balance?", "criteria": {"£0": None, "£20": None}},
                                "neg": {"type": "noul", "instructions": "Does it go negative?"},
                                "cnt": {"type": "score", "instructions": "Count.", "criteria": ["0", "1", "2"]}}),
       "answers": json.dumps({"final": {"type": "choice", "choice": "£20", "probabilities": {"£0": 0.0, "£20": 1.0}},
                              "neg": {"type": "noul", "noul": 0.0},
                              "cnt": {"type": "score", "score": 1.0, "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}}})}

def test_procedural_null_criteria_become_keys():
    c = to_case(ROW)
    qs = {q.qid: q for q in parse_questions(c["questions"])}
    assert qs["final"].descriptions == ("£0", "£20") and c["id"] == "proc-arithmetic:train:0" and c["source"] == "procedural"

def test_gold_targets_are_smoothed_one_hot():
    t = {x["qid"]: x for x in gold_targets(to_case(ROW))}
    assert t["final"]["target"] == [0.1, 0.9] and t["neg"]["target"] == [0.9, 0.1] and t["cnt"]["target"] == [0.05, 0.9, 0.05]

def test_cap_share_limits_one_source():
    rows = [{"source": "procedural"}] * 80 + [{"source": "synth"}] * 20
    out = cap_share(rows, "procedural", 0.35)
    n = sum(r["source"] == "procedural" for r in out)
    assert n <= round(0.35 * len(out)) + 1 and sum(r["source"] == "synth" for r in out) == 20
```

- [ ] **Step 2: Correr y ver fallar; implementar; correr** → `./.venv/bin/pytest tests/test_procedural.py -q` PASS.

- [ ] **Step 3: Construir el conjunto interno y comprobar la mezcla**

Run: `./.venv/bin/python -m scripts.build_hardlike && wc -l data/hardlike.jsonl && ./.venv/bin/python -m scripts.build_targets`
Expected: 2000 líneas; `[mix]` muestra procedural ≤ 35 %, y el total de targets ≈ 27 600 + procedural. Nota: la primera descarga del dataset (396k filas) tarda; `datasets` lo cachea en `~/.cache/huggingface`.

- [ ] **Step 4: Línea base del activo en hard-like**

Run: `./.venv/bin/python -m scripts.eval_hardlike --model runs/v1108-2307/onnx`
Expected: imprime `accuracy` y `by_workflow`; se anota en el commit como línea base (esperado: bajo, ~30–45 %).

- [ ] **Step 5: Commit** (`data/hardlike.jsonl` sí se versiona, es pequeño y es el conjunto de promoción; `data/targets.jsonl` también, como hasta ahora)

```bash
git add temis/data/procedural.py temis/data/balance.py temis/data/sources.py scripts/build_hardlike.py scripts/eval_hardlike.py scripts/build_targets.py scripts/evaluate.py scripts/train.py tests/test_procedural.py data/hardlike.jsonl data/targets.jsonl
git commit -m "procedural-typed-decisions como fuente con tope del 35 %, conjunto interno hard-like"
```

---

### Tarea 6: Entrenamiento: consistencia por permutación + R-Drop, entradas largas, CUDA (D1, D4, H)

**Files:**
- Modify: `temis/model.py` (`consistency_loss`, `Decisor.__init__` attn)
- Modify: `temis/encoding.py` (nada; `Encoded.order` ya existe)
- Modify: `scripts/train.py` (`encode_batch` gemelo, `--twin`, `--consistency`, `--long-max-len`, `--long-bs`, bf16, sdpa, `empty_cache` solo en MPS)
- Test: `tests/test_train_loss.py`

**Interfaces:**
- Produces en `temis/model.py`:
  - `align_to_options(p: Tensor, order: list[list[int]], n: list[int]) -> Tensor`: `(B, K)` en orden de marcadores → `(B, K)` en orden original de opciones (posiciones ≥ n quedan 0).
  - `consistency_loss(logits_a, logits_b, mask_a, mask_b, order_a, order_b, n) -> Tensor`: KL simétrica media entre las dos distribuciones alineadas al orden de opciones (`0.5·(KL(a‖b)+KL(b‖a))`).
  - `Decisor(base, attn=None)`: sin cambio de firma; `scripts/train.py` pasa `attn="sdpa"` cuando `device == "cuda"`.
- `scripts/train.py`:
  - `encode_batch(enc, exs, rng, ctx_pool, max_len, long_max_len)`: devuelve además `"order"` y `"n"` por ejemplo; un ejemplo cuyo `len(ids)` a `max_len` queda truncado (`len(ids) == max_len`) se recodifica con `long_max_len`.
  - `batches(exs, bs, rng, long_bs, enc, max_len)`: agrupa por longitud estimada (ya lo hace); los grupos cuyo ejemplo más largo supera `max_len` se parten en lotes de `long_bs`.
  - Flags: `--twin` (default on; `--no-twin` apaga), `--consistency 0.5` (peso λ), `--long-max-len 1536`, `--long-bs 4`, `--bf16` (default on en cuda; off en mps/cpu).
  - Paso de entrenamiento con `--twin`: el lote se codifica dos veces (dos `rng` derivados: barajado y contexto distintos), se concatena en un solo `collate` de `2B` secuencias, `loss = decision_loss(todo) + λ·consistency_loss(mitad a, mitad b)`. Dropout queda activo (R-Drop viene gratis).
  - `meta.json` añade `long_max_len`, `twin`, `consistency`, `bf16`, `attn`.

- [ ] **Step 1: Tests** (`tests/test_train_loss.py`)

```python
import torch
from temis.model import align_to_options, consistency_loss

def test_align_to_options_inverts_shuffle():
    p = torch.tensor([[0.2, 0.5, 0.3, 0.0]])
    assert torch.allclose(align_to_options(p, [[2, 0, 1]], [3]), torch.tensor([[0.5, 0.3, 0.2, 0.0]]))

def test_consistency_loss_zero_when_same_distribution_in_option_order():
    la = torch.tensor([[1.0, 2.0, 3.0, -1e4]]); lb = torch.tensor([[3.0, 1.0, 2.0, -1e4]])
    m = torch.tensor([[True, True, True, False]])
    assert consistency_loss(la, lb, m, m, [[0, 1, 2]], [[2, 0, 1]], [3]).item() < 1e-6
    assert consistency_loss(la, la, m, m, [[0, 1, 2]], [[2, 0, 1]], [3]).item() > 0.1

def test_consistency_loss_score_same_order():
    la = torch.tensor([[1.0, 2.0, 3.0]]); lb = torch.tensor([[1.1, 2.0, 2.9]])
    m = torch.ones(1, 3, dtype=torch.bool)
    assert 0 < consistency_loss(la, lb, m, m, [[0, 1, 2]], [[0, 1, 2]], [3]).item() < 0.05
```

- [ ] **Step 2: Correr y ver fallar; implementar `align_to_options` y `consistency_loss`; correr** → PASS.

- [ ] **Step 3: Modificar `scripts/train.py`** según Interfaces. `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` alrededor del forward cuando `--bf16`; `torch.mps.empty_cache()` solo si `device == "mps"`; `Decisor(a.base, attn="sdpa" if device == "cuda" else None)`.

- [ ] **Step 4: Humo en CPU con 20 pasos** (no entrena de verdad; comprueba que el ciclo corre con lotes gemelos y un ejemplo largo)

Run: `./.venv/bin/python -m scripts.train --name smoke --epochs 0.002 --bs 4 --long-bs 2 --max-len 256 --long-max-len 512 --base jhu-clsp/mmBERT-small --threads 4`
Expected: imprime `[train] step …` y termina escribiendo `runs/smoke/meta.json` con `"twin": true`; borrar `runs/smoke` después.

- [ ] **Step 5: Tests completos y commit**

Run: `./.venv/bin/pytest tests -q` → PASS
```bash
git add temis/model.py scripts/train.py tests/test_train_loss.py
git commit -m "Entrenamiento: lotes gemelos con consistencia por permutación, entradas largas, bf16/SDPA en CUDA"
```

---

### Tarea 7: Importar casos difíciles con `rationale` y familias nuevas (C2/C3, importador)

**Files:**
- Modify: `scripts/import_cases.py` (`FAMILIES`, campos opcionales)
- Modify: `temis/data/families.py` (`HARD_FAMILIES`, pesos de cola)
- Test: `tests/test_import.py`

**Interfaces:**
- Produces: `HARD_FAMILIES = ("long_policy", "multi_hop", "trap", "ambiguous", "adversarial", "tradeoff", "routing_hard", "judge_hard", "temporal_numeric", "probability", "state_tracking", "rubric_short")` en `temis/data/families.py`; `import_cases.FAMILIES = AGENT_FAMILIES | set(HARD_FAMILIES) | {"other"}`.
- Función extraída y testeable: `import_cases.parse_line(line: str, source: str) -> dict | None` (devuelve el caso normalizado o `None` con el motivo impreso). El caso conserva, si vienen, `rationale: str`, `surface_answer: dict[qid, option]`, `contrast_of: str` (id del par de contraste); `domain = f"hard:{fam}"` para familias difíciles; `workflow = fam`.
- Validación extra para familias difíciles: `rationale` obligatorio (≥ 40 caracteres); `surface_answer` opcional pero sus claves deben ser qids y sus valores opciones válidas.

- [ ] **Step 1: Tests** (`tests/test_import.py`)

```python
import json
from scripts.import_cases import parse_line

def line(**kw):
    base = {"family": "trap", "language": "en", "state": "x" * 200,
            "questions": {"ok": {"type": "noul", "instructions": "Is this return eligible for the instant refund?",
                                 "criteria": {"true": "Eligible.", "false": "Not eligible."}}},
            "rationale": "The seal is split, so the item is not unopened under the stated definition.",
            "surface_answer": {"ok": "true"}}
    return json.dumps({**base, **kw})

def test_import_keeps_question_form_noul():
    c = parse_line(line(), "gpt")
    assert c and c["workflow"] == "trap" and c["domain"] == "hard:trap" and "ok" in c["questions"]
    assert c["rationale"].startswith("The seal") and c["surface_answer"] == {"ok": "true"}

def test_import_hard_family_requires_rationale_and_valid_surface():
    assert parse_line(line(rationale=""), "gpt") is None
    assert parse_line(line(surface_answer={"ok": "maybe"}), "gpt") is None

def test_import_agent_families_unchanged():
    assert parse_line(line(family="memory_op", rationale=None, surface_answer=None), "gpt")["domain"] == "agent:memory_op"
```

- [ ] **Step 2: Correr y ver fallar (`ImportError: parse_line`); refactorizar `main` sobre `parse_line`; correr** → PASS. Verificar que `clean_questions` conserva "Is this return eligible…?" (lo hace: no empieza por palabra WH ni imperativo); si no, ajustar `quality.WH` para no vetar `is/does/was/can/should` — no están en la lista, no hace falta tocarla.

- [ ] **Step 3: Commit** `git add scripts/import_cases.py temis/data/families.py tests/test_import.py && git commit -m "Importador: familias difíciles con rationale, respuesta superficial y pares de contraste"`

---

### Tarea 8: Prompts para el modelo privado y segunda opinión (C2, C3, C4)

**Files:**
- Create: `docs/prompt_casos_dificiles.md`
- Create: `docs/prompt_rubrica_corta.md`
- Create: `docs/prompt_segunda_opinion.md`
- Create: `scripts/second_opinion.py`
- Modify: `scripts/build_targets.py` (filtrar por acuerdo)
- Test: `tests/test_import.py` (añadir)

**Interfaces:**
- `docs/prompt_casos_dificiles.md`: mismo formato de entrega que `docs/prompt_generar_casos.md` (JSONL, una línea por caso, claves `family, language, workflow, state, questions, rationale, surface_answer, contrast_of?`), con las 10 familias descritas **con nuestras propias palabras a partir de §C2 del spec** (no copiar texto de JevBench): longitudes objetivo (long_policy / multi_hop 1 500–4 000 tokens con relleno "parecido pero no"; el resto 150–600), `noul` **en forma de pregunta y de afirmación a partes iguales**, `cannot_determine`/`insufficient_information` como gold la mitad de las veces en `ambiguous`, pares de contraste (`contrast_of` = `workflow` del par), 50 % español, encargo por lote: "40 casos: 8 long_policy, 8 multi_hop, 4 trap, 4 ambiguous, 4 judge_hard, 4 temporal_numeric, 2 routing_hard, 2 tradeoff, 2 adversarial, 2 probability".
- `docs/prompt_rubrica_corta.md`: familia `rubric_short`, estados de 1–3 frases, las seis plantillas de §C3 (mención sin petición, negación, permiso con excepción, umbral inclusivo, adecuación escueta-correcta / fluida-con-error, especialista con pista engañosa), gold balanceado por plantilla, `rationale` de una frase, encargo por lote de 60.
- `docs/prompt_segunda_opinion.md`: entrada = JSONL de `scripts/second_opinion.py export` (`{id, state, questions, rationale}`); salida = JSONL `{id, answers: {qid: option}}` **sin ver las etiquetas de Jev**; instrucción: "si el rationale no justifica la opción que eliges, responde `null`".
- `scripts/second_opinion.py`:
  - `export out.jsonl [--family long_policy,...] [--unlabeled-only]`: casos de `load_synth()` con `domain.startswith("hard:")` que aún no tienen fila en `data/second_opinion.jsonl`.
  - `import answers.jsonl --by <model>`: añade filas `{id, qid, label, by}` a `data/second_opinion.jsonl` (valida qid y opción).
- `scripts/build_targets.py`: para casos `hard:*`, si existe segunda opinión y **no coincide** con el argmax de Jev, la fila no se emite y se cuenta en `[targets] … (N hard cases held back: teachers disagree)`; sin segunda opinión, el caso entra normal (Jev solo). Función pura `agree(jev_dist: list[float], options: list[str], label: str | None) -> bool` en `temis/data/balance.py`.

- [ ] **Step 1: Test**

```python
from temis.data.balance import agree

def test_agree_with_second_opinion():
    assert agree([0.2, 0.7, 0.1], ["a", "b", "c"], "b") and not agree([0.2, 0.7, 0.1], ["a", "b", "c"], "a")
    assert agree([0.2, 0.7, 0.1], ["a", "b", "c"], None)  # sin segunda opinión: entra
```

- [ ] **Step 2: Correr, ver fallar, implementar `agree`, `scripts/second_opinion.py` y el filtro en `build_targets`; correr** → PASS.

- [ ] **Step 3: Escribir los tres prompts** (en español, como el existente; con un ejemplo de línea válida cada uno). Comprobar que una línea de ejemplo de cada prompt pasa `parse_line` (ejecutar `python -c` con la línea pegada).

- [ ] **Step 4: Commit** `git add docs/prompt_*.md scripts/second_opinion.py scripts/build_targets.py temis/data/balance.py tests/test_import.py && git commit -m "Prompts para casos difíciles, rúbrica corta y segunda opinión; filtro por acuerdo"`

---

### Tarea 9: Integrar en `retrain.py`, puertas y README

**Files:**
- Modify: `scripts/retrain.py:47-76`
- Modify: `README.md` (sección Pipeline y Results: nota sobre `eval_jevbench` local)
- Modify: `docs/specs/2026-11-09-plan-jevbench.md` (marcar fase 0 como implementada, con fecha)

**Interfaces:**
- `retrain.py` ejecuta además `scripts.eval_hardlike` y `scripts.eval_jevbench --calibration jevbench` tras `eval_user`; la promoción añade `dhard = hardlike_new − hardlike_old ≥ −0.01` y `dflip ≤ 0.05` (de `eval_jevbench.json["probes"]` solo si existe); JevBench `accuracy` se imprime pero **no decide**. Los argumentos de `scripts.train` pasan `--long-max-len 1536 --long-bs 4` (y mantienen `--max-len 640`).

- [ ] **Step 1: Modificar `retrain.py`**, correr `./.venv/bin/pytest tests -q` (PASS) y un ensayo en seco: `./.venv/bin/python -c "import scripts.retrain"`.
- [ ] **Step 2: README**: en Pipeline añadir `python -m scripts.eval_jevbench --model runs/v1/onnx --fetch` y `scripts.build_hardlike` / `scripts.eval_hardlike`; en JevBench aclarar que la cifra local usa la misma puntuación y que se reporta pero no se usa para promover.
- [ ] **Step 3: Commit** `git commit -am "retrain: hard-like y JevBench local en la evaluación; README"`

---

## Al terminar (no es parte del plan de código)

Lo que sigue, en orden, ya con datos:
1. Pedir al modelo privado 2–3 lotes de `docs/prompt_rubrica_corta.md` (~150 casos) y 5–10 lotes de `docs/prompt_casos_dificiles.md` (~300 casos); importar con `scripts.import_cases --source <modelo>`.
2. `scripts.label` (Jev) → `scripts.second_opinion export` → modelo privado → `import` → `scripts.build_targets`.
3. Corrida limpia (fase 1): `python -m scripts.train --name v<fecha> --base jhu-clsp/mmBERT-base --epochs 4 --bs 16 --long-bs 4 --max-len 640 --long-max-len 1536 --extra data/user_targets.jsonl --extra data/reviewer_targets.jsonl` (en la GPU: `--bf16` por defecto), luego `export_onnx`, `calibrate`, `evaluate`, `eval_clio`, `eval_user`, `eval_hardlike`, `eval_jevbench`.
