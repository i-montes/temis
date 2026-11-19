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
| `set2/cases.jsonl` | **Set 2**: 8.316 casos, 77.557 preguntas, de 900 notas de 7 secciones, con casos raros y difíciles (abajo). |
| `set2/bench_estilos_cases.jsonl` | Benchmark de estilos: 207 casos de 36 notas de 2025–2026 (6 por sección nueva). **Nunca se entrenan**. |
| `set2/bench_estilos_gold.jsonl` | Referencia de Jev del benchmark de estilos (como `bench_gold.jsonl`). |
| `trazas/cases.jsonl` | Casos de las trazas del verificador (cuando se exporten: `scripts.periodista_trazas`). |
| `revision/` | Revisión a mano de Jev en los casos difíciles (`scripts.periodista_revision`). |
| `targets.jsonl` | Objetivos de entrenamiento de todos los lotes. **No se sube** (88 MB): `python -m scripts.periodista_targets` lo arma en segundos. |
| `held_back.jsonl` | Preguntas difíciles donde Jev duda (confianza < 0,7): no se entrenan, van a revisión. |
| `raw/` | Lo bajado del WordPress (no se sube al repo; `scripts.periodista_fetch` lo regenera). |

## Set 2: más estilos, casos raros y difíciles

```sh
python -m scripts.periodista_fetch --solo set2        # 4.747 notas nuevas de 7 secciones -> raw/notas_set2.jsonl
python -m scripts.periodista_build2                    # set2/cases.jsonl y set2/bench_estilos_cases.jsonl
python -m scripts.periodista_label --casos data/periodista/set2/cases.jsonl \
    --bench data/periodista/set2/bench_estilos_cases.jsonl --oro data/periodista/set2/bench_estilos_gold.jsonl
python -m scripts.periodista_revision                  # ¿Jev resuelve los difíciles? (antes de entrenar con ellos)
python -m scripts.periodista_targets                   # junta set 1, set 2 y trazas, con balance
```

| Sección (`meta.seccion`) | Notas | Casos | `documento.tipo` |
|---|---|---|---|
| Nacional (otras que las del set 1) | 300 | 3.025 | nota |
| En Vivo | 150 | 1.254 | nota corta de En Vivo |
| Opinión | 120 | 1.129 | columna de opinión |
| Red de Expertos | 120 | 1.067 | columna de un experto |
| Detector de Mentiras | 120 | 1.107 | chequeo del Detector |
| Silla Académica | 45 | 421 | resumen de una investigación |
| Amazonía | 45 | 385 | nota de Silla Amazonía |

- **Casos raros**: al elegir entidades pesan más las de oraciones con roles poco frecuentes (acusados, víctimas,
  fuerza pública, grupos armados, jueces, empresarios, líderes sociales); `ent_misma_mencion` usa la mitad de las
  veces el cargo más cercano a la entidad; más candidatos dudosos para `ent_es_entidad`.
- **Casos difíciles** (`meta.dificil`):
  - `sin_correcto` (246): se quita el perfil con el nombre exacto y quedan homónimos; la respuesta es «ninguno».
  - `apellido_solo` (401): la entidad solo por su apellido, con todos los perfiles que lo comparten.
  - `vigencia` (2.760 casos): pasajes de hace 2+ años con `ent_fuente_vigente_fN` (¿sigue vigente su cargo?).
- **Sin fugas**: ningún pasaje sale de una nota de un benchmark; `scripts.periodista_targets` además saca del
  set 1 los 20 casos que traían un pasaje de una nota del benchmark.
- **Reproducible**: los armados son deterministas desde el 2026-11-19 (antes, el orden de los cargos dependía de
  `PYTHONHASHSEED`). Los archivos del repo son los que respondió Jev; rearmar hoy cambia el cargo de
  `ent_misma_mencion` en 154 casos del set 2. Cada respuesta de Jev guarda la huella de sus preguntas (`qhash`):
  `scripts.periodista_label` vuelve a preguntar los casos que cambiaron y `scripts.periodista_targets` descarta las
  respuestas que no coinciden.
- **Balance por copias** en `scripts.periodista_targets` (como `temis/data/balance.py`: `scripts.train` no usa
  `weight`): la respuesta que pase del 35 % (60 % en noul) entra solo esa parte de las veces y la que no llegue
  al 8 % entra dos veces. En `ent_candidato` y `ent_oracion_clave` se balancea «ninguno» contra «alguno».

## ¿Sirve Jev de profesor en los difíciles? (revisión del 2026-11-19)

Si ni Jev resuelve un caso difícil, entrenar con su respuesta le enseña a Temis el error. `scripts.periodista_revision`
lo mide:

- **Automático**, donde la respuesta se sabe por construcción:

  | Caso difícil | n | Jev acierta | con confianza ≥ 0,7 | con confianza < 0,7 |
  |---|---|---|---|---|
  | `sin_correcto` (respuesta: ninguno) | 246 | 100 % | 238 de 238 | 7 de 8 |
  | `apellido_solo` (el perfil del nombre completo) | 90 | 90 % | 97 % (62 de 64) | 73 % (19 de 26) |

  Al revisar los fallos salieron 72 casos **mal armados** (no eran errores de Jev): en 66 `sin_correcto` quedaba
  otro perfil de la misma persona con el nombre más largo («Ana Teresa Bernal» → «Ana Teresa Bernal Montañez») y
  en 6 `apellido_solo` el apellido quedaba fuera del trozo visible. `scripts.periodista_build2` ya los quita.
- **Regla para entrenar**: en un caso difícil, la pregunta difícil solo entra si Jev la responde con confianza
  ≥ 0,7; si no, va a `held_back.jsonl` (2.404 preguntas, casi todas de vigencia).
- **A mano**: `revision/revision_jev.csv`, 130 filas (20 por categoría: vigencia, «misma mención» sí y no, roles
  raros del documento y de la persona, entidades descartadas, y los 10 fallos que quedan de la parte automática).
  Se llenan `jev_acierta` (si / no / dudoso), `respuesta_correcta` y `comentario`, y
  `python -m scripts.periodista_revision --calificar` da el acierto de Jev por categoría. Una categoría donde Jev
  acierte poco se saca del entrenamiento o se entrena solo con lo revisado.

## Trazas del verificador

En el verificador: opciones → «Modo depuración», navegar (redes, notas de otros medios, PDF), abrir la página de
depuración y «Descargar todo». Después:

```sh
python -m scripts.periodista_trazas ~/Downloads/depuracion-verificador-*.json   # -> trazas/cases.jsonl
python -m scripts.periodista_label --casos data/periodista/trazas/cases.jsonl --bench ''
```

Cada candidato de una petición de entidades a Jev se vuelve un caso con las preguntas de Temis periodista. Son
otros estilos (posts, notas de otros medios, páginas). Las respuestas originales de Jev al verificador quedan en
`meta.jev_verificador`.

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
python -m scripts.periodista_targets          # primero: arma data/periodista/targets.jsonl (no está en el repo)
export TEMIS_TARGETS=data/periodista/targets.jsonl

# A) desde Temis 2.4.0 (ya sabe leer peticiones tipadas). Necesita runs/v1112-mrbert-cortos/model.pt.
#    Receta de afinado de scripts/retrain.py, con más épocas porque las preguntas son nuevas.
python -m scripts.train --name per-v1 --init v1112-mrbert-cortos --base BSC-LT/MrBERT-es \
  --epochs 3 --lr 2.5e-5 --head-lr 2e-4 --bs 8 --accum 2 --max-len 1024 --long-max-len 1536 --long-bs 4

# B) para comparar: desde MrBERT-es en blanco (el README de Temis: con MrBERT-es, --head-lr 5e-5)
python -m scripts.train --name per-v1-cero --base BSC-LT/MrBERT-es \
  --epochs 5 --lr 3e-5 --head-lr 5e-5 --bs 8 --accum 2 --max-len 1024 --long-max-len 1536 --long-bs 4

# después, en este orden (como scripts/retrain.py): exportar y luego calibrar
python -m scripts.export_onnx --name per-v1
python -m scripts.calibrate --name per-v1
```

- `--base BSC-LT/MrBERT-es` es obligatorio con `--init`: `--init` solo carga los pesos, y el `--base` por
  defecto de `scripts.train` es mmBERT-small (no cargarían).
- `--max-len 1024`: medido con el tokenizador de MrBERT-es, cada pregunta ocupa una mediana de 745 tokens
  (p90 919, máx. 1.173). Con 640, el 75 % iría al camino de estados largos (más lento); con 1.536 no se corta
  ninguna.
- `TEMIS_TARGETS` cambia el archivo de objetivos de `scripts.train`, `scripts.calibrate` y `scripts.export_onnx`
  (sin la variable, todo sigue igual que antes). Las filas td-train del repaso necesitan el dataset
  `LocalLLaMA/typed-decisions` (lo baja `load_td`, como siempre).
- La validación de `scripts.train` aparta 2.246 preguntas de 27 notas enteras (el `schema_id` de cada fila es
  la nota); el entrenamiento queda con 22.257 preguntas propias más el repaso.
- En Windows, `PYTHONUTF8=1`: el código del repo abre archivos sin decir la codificación.
- La API alojada lee hasta 1.024 tokens por pregunta (`long_max_len`): unas pocas preguntas largas se cortan
  ahí. Para servir este modelo conviene subirlo a 1.536.

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

### Línea base: Temis 2.4.0 (el de Clio, sin entrenar en esto)

API alojada, 2026-11-19, 111 casos. Acuerdo con Jev frente a responder siempre lo más común:

| Pregunta | n | Temis 2.4.0 | Siempre lo más común |
|---|---|---|---|
| `ent_es_entidad` | 111 | 0,59 | 0,89 |
| `ent_tipo` | 111 | 0,23 | 0,53 |
| `ent_candidato` | 90 | **0,78** | 0,62 |
| `ent_oracion_clave` | 111 | **0,39** | 0,14 |
| `ent_fuente` | 293 | 0,35 | 0,64 |
| `ent_misma_mencion` | 49 | 0,94 | 0,96 |
| `ent_rol_documento` | 111 | 0,14 | 0,23 |
| `ent_centralidad` | 111 | 0,41 (error medio 0,71 niveles) | 0,49 |
| `ent_postura` | 51 | 0,33 | 0,43 |
| `ent_rol_persona` | 33 | 0,30 | 0,42 |
| `ent_rol_organizacion` | 18 | 0,22 | 0,28 |
| `ent_cargo_explicito` | 33 | 0,30 | 0,70 |
| **Total** | 1.122 | **0,40** | |

Solo supera a la respuesta fija en `ent_candidato` y `ent_oracion_clave`. En el resto cae en respuestas
casi fijas: dice «organización» en 105 de 111 casos de `ent_tipo` (también a personas) y «no» en 289 de 293
pasajes de `ent_fuente`. Lee una mediana de 7.926 tokens por caso (unos 800 por pregunta) y tarda una
mediana de 19 s por caso en la API alojada. Resultados por caso: `benchmarks/periodista/`.

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
