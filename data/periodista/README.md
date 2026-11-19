# Temis periodista: base de entidades

Datos para entrenar **Temis periodista**, un Temis aparte (no el de Clio) que decide, para cada entidad que
nombra una nota de La Silla Vacía: si es una entidad, de qué tipo, cuál perfil o etiqueta de La Silla es, qué
oración la describe, qué pasajes del archivo le dan contexto, qué papel cumple en la historia y qué es
(funcionaria, congresista, partido…). Diseño y preguntas: [docs/specs/2026-11-19-entidades-periodista.md](../../docs/specs/2026-11-19-entidades-periodista.md).

La fuente es solo La Silla Vacía: notas de Nacional y sus regionales, perfiles del «Quién es quién» y etiquetas
del WordPress, bajados de su API pública.

## Qué hay

| Archivo | Qué es |
|---|---|
| `cases.jsonl` | 2.512 casos de entrenamiento (300 notas de 2019 a 2026, ~37 por año), 24.503 preguntas. Un caso = una entidad de una nota. |
| `bench_cases.jsonl` | 111 casos del benchmark (20 notas de 2025–2026), 1.122 preguntas. **Nunca se entrenan**: son otras notas. |
| `labels/jev.jsonl` | Respuestas de Jev a todos los casos (`{id, teacher, dists}`, como `data/labels/jev.jsonl`). |
| `bench_gold.jsonl` | Respuesta de referencia del benchmark: la opción más probable de Jev y su confianza. `revisado: false` hasta que alguien de la redacción la confirme o la cambie. |
| `targets.jsonl` | Objetivos de entrenamiento (sale de `scripts.periodista_targets`). |
| `raw/` | Lo bajado del WordPress (no se sube al repo; `scripts.periodista_fetch` lo regenera). |

Cada caso tiene la forma de `data/synth/cases.jsonl` (`id`, `schema_id`, `family`, `workflow`, `language`,
`state`, `questions`) más `meta`, que no va al modelo: la nota (`post_id`, `link`, `fecha`), la pista de tipo
del código (`tipo_codigo`), de dónde salió cada candidato (perfil o etiqueta, con su id y link) y de qué nota
sale cada pasaje de la fuente.

### Preguntas por caso

Siempre: `ent_es_entidad`, `ent_tipo`, `ent_rol_documento`, `ent_centralidad`, `ent_oracion_clave`. Según el
caso: `ent_candidato` (77 % de los casos), `ent_fuente_fN` (82 %, 1 a 3 pasajes), `ent_misma_mencion` (35 %),
`ent_rol_persona` + `ent_cargo_explicito` (cuando el código cree que es persona, 29 %),
`ent_rol_organizacion` (21 %), `ent_postura` (persona u organización, 50 %). Entre 5 y 13 preguntas por
caso. El estado tiene una mediana de 2.259 caracteres (máx. 3.592).

## Cómo se arma (reproducible)

```sh
python -m scripts.periodista_fetch                          # WordPress de La Silla -> data/periodista/raw/ (~3 min)
python -m scripts.periodista_build --documentos 300 --bench 20   # casos (determinista con --seed)
python -m scripts.periodista_label --workers 6              # Jev responde (TYPESAFE_API_KEY en .env); se reanuda
python -m scripts.periodista_targets                        # objetivos calibrados + 20 % de repaso typed-decisions
```

El código encuentra los candidatos (nombres propios, siglas, etiquetas de La Silla, perfiles del «Quién es
quién»), junta las menciones («Antonio Sanguino» y «Sanguino»), elige hasta 8 entidades por nota (más, a
veces, una dudosa para que `ent_es_entidad` tenga casos de «no») y trae hasta 3 pasajes de notas
**anteriores** del archivo que la nombran (uno con el nombre en el título si hay). Los ids de candidatos
(`c1`…) y pasajes (`f0`…) son neutros y en orden aleatorio.

## Entrenar

```sh
# desde Temis 2.4.0 (ya sabe leer peticiones tipadas)
TEMIS_TARGETS=data/periodista/targets.jsonl python -m scripts.train --name per-v1 --init v1112-mrbert-cortos --epochs 3
TEMIS_TARGETS=data/periodista/targets.jsonl python -m scripts.calibrate --name per-v1
TEMIS_TARGETS=data/periodista/targets.jsonl python -m scripts.export_onnx --name per-v1
```

`TEMIS_TARGETS` cambia el archivo de objetivos de `scripts.train`, `scripts.calibrate` y `scripts.export_onnx`
(sin la variable, todo sigue igual que antes). La validación de `scripts.train` aparta el 10 % de las notas (el
`schema_id` de cada fila es la nota). Vale la pena comparar con un entrenamiento desde MrBERT-es en blanco.

El benchmark de Clio no cuenta para esta línea: la versión que queda activa la decide `scripts.periodista_bench`.

## Benchmark

```sh
export TEMIS_TOKEN=...       # nunca en el repo
python -m scripts.periodista_bench --model temis-latest --nombre temis-2.4.0        # API alojada
python -m scripts.periodista_bench --url http://127.0.0.1:8765/v1/systemone --token-env NONE --nombre per-v1   # local
```

Califica choice y noul por acierto, score por acierto del nivel y error absoluto medio; `ent_fuente_fN` se
juntan en `ent_fuente`. Los resultados quedan en `benchmarks/periodista/`. Mide el **acuerdo con Jev** hasta que
la redacción revise `bench_gold.jsonl`.

## Limitaciones conocidas

- La referencia es Jev, no una persona. `senalado` y `afectado` son delicados: conviene que la redacción revise
  primero esos casos.
- La pista de tipo del código decide qué preguntas de rol lleva el caso: un 41 % de los casos no tiene pista y
  queda sin `ent_rol_persona` / `ent_rol_organizacion`.
- Las respuestas se van a cargar hacia `protagonista`, `mencion`, `funcionario` y `electo`. Faltan las capas
  de variaciones y casos difíciles de la spec (pares de contraste, homónimos, cargos vencidos).
- Solo notas de Nacional y regionales. No hay Opinión, Red de Expertos ni Detector.
- Agrupación de nombres con «De La»: en `per-431338-1` la entidad quedó como «Espriella» (el código quita «De»
  y «La» al inicio) y entre los candidatos no está la etiqueta «Abelardo de la Espriella», así que Jev responde
  bien `ninguno` a una pregunta mal armada. Hay que unir «De La Espriella» con la forma completa del título.
- Persona con perfil y etiqueta: en `per-431338-0` (Gustavo Petro) `c1` es su etiqueta y `c2` su perfil; las dos
  son correctas y Jev eligió una. Para la próxima versión: la etiqueta de una persona con perfil va unida al
  perfil (como en el verificador), no como otra opción.
- `ent_misma_mencion` está casi siempre en «no» (47 de 49 en el benchmark): el código elige un cargo de la nota
  al azar y casi siempre es de otra persona. Para la próxima versión: elegir el cargo más cercano a la
  entidad la mitad de las veces.
