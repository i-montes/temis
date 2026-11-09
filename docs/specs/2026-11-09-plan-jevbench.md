# Temis: plan de mejora en JevBench sin dejar de ser liviano

Fecha: 2026-11-09. Estado: aprobado por Iván el 2026-11-09. **Fase 0 y el código de las fases 1–2 están
implementados** (`docs/plans/2026-11-09-antes-de-entrenar.md`); faltan los datos (C2–C4) y la corrida.
Parte de Temis 1.0.11 (mmBERT-base, int8 309 MB) con 57,1 % en los 231 ítems públicos de JevBench
(easy 100 %, standard 65 %, hard 33 %).

## 0. Regla de diseño (no negociable)

Temis se mejora con **datos, objetivo de entrenamiento y forma de leer**, no con tamaño. Límites:

| Límite | Valor | Por qué |
|---|---|---|
| Tamaño en disco (int8) | ≤ 400 MB (ideal ≤ 310 MB, el actual) | Debe caber en cualquier VPS o portátil |
| Parámetros no-embedding | ~110 M (los de mmBERT-base) | Un ModernBERT-large (343 M no-emb) cuesta ~3× en CPU: varios cientos de ms a 700 tokens |
| Latencia CPU (p50, 4 hilos, portátil) | ≤ 50 ms en casos cortos (hoy ~23 ms); ≤ 400 ms en documentos largos | JevBench tapa la latencia mediana ajustada en 1,23 s; Clio necesita decenas de ms |
| GPU | Opcional, nunca requerida | Regla del proyecto |
| Datos públicos de JevBench | **Solo evaluación, nunca entrenamiento** | JevBench exige declararlo y la mitad sellada lo castigaría |

Lo que queda **fuera**: decodificadores de 0,5–1 B (4–5× el cómputo, int8 frágil, y no ganan a un
encoder de 300 M en clasificación con razonamiento: Ettin 400 M-enc 91,3 vs 1 B-dec 88,2 en MNLI),
ModernBERT-large (solo inglés), EuroBERT-610m (610 MB int8), mdeberta/xlm-r (512 tokens).

## 1. Diagnóstico (ítem por ítem, 231 públicos)

### 1.1 Dónde falla

| Tier / familia | Temis | Referencias | Qué pasa |
|---|---|---|---|
| **standard** (72 ítems de 18–160 caracteres) | 47/72 | Laya 73 %, Verdict 68 %, SemIf 98 %, Jev 99 % | **No es longitud ni razonamiento: es leer la rúbrica.** |
| · policy (noul) | 7/12 | | Los 5 fallos son gold=sí → Temis "no" (0,61–0,75). "Trial users may export CSV; trial user asks for CSV" → no. |
| · intent | 5/12 | | "Where is the parcel? Keep the address as it is" → `other`; "I cancelled yesterday. Was it delivered?" → `cancel`. No distingue **mención** de **petición**. |
| · adequacy (noul) | 5/12 | | "Return only the sum of 17 and 25. Response: 42" → **no** (0,76). Respuestas correctas y escuetas juzgadas mal; "exactly two words"/"All done now" → sí. |
| · routing | 8/12 | | Default `general` ("Find LCM(12,18)" → general 0,37). |
| **hard** (111) | 37/111 | Laya 34 %, Verdict 38 %, GLiNER2 36 %, SemIf 60 %, Jev 74 % | |
| · long_policy (19, ~12 000 caracteres, 3 000–3 700 tokens) | **1/19 (5 %, azar 22 %)** | encoders 10–14/38; Jev 23/38 | El estado se corta a **6 000 caracteres en `render_state`** y luego a **640 tokens**: el modelo ve ~15 % del documento y responde por el sesgo del tipo de opción. |
| · multi_hop (18, ~10 000 caracteres) | 4/18 | encoders 6–13/35; Jev 30/35 | Igual: tabla → nota al pie → alias → regla; nada de eso entra en 640 tokens. |
| · ambiguous (7) | 1/7 | Verdict 6/14, Jev 11/14 | Nunca elige `cannot_determine`; cae en la respuesta superficial. |
| · trap (8) | 2/8 | Jev 16/16, kev-0.6B 9/16 | Negación lejana, petición citada, condición revocada. |
| · routing_hard (5) | 1/5 | Jev 10/10 | Palabras clave de la opción equivocada. |
| · judge_hard (17, noul) | 7/17 | Jev 26/33 | Sesgo a "sí" (fluido ⇒ correcto): en hard predice sí 28/38 con gold sí 17/38. |
| · temporal_numeric (15) | 7/15 | Jev 8/30, media del campo 30 % | Aritmética de fechas/umbrales: **nadie** lo resuelve bien; techo bajo para un encoder. |
| · probability (10) | 4/10 | Jev 16/20 | Calibración contra `gold_probs` exactos; cuenta doble (acierto + calibración). |
| · tradeoff, adversarial (6+6) | 5/6, 5/6 | | Bien. |

Señales transversales:

- **Respuesta superficial**: 37 de los 71 fallos en hard son exactamente el `surface_answer` que el
  autor del ítem puso como trampa. Temis lee léxicamente.
- **Sesgo de posición**: predice la opción 1.ª 37 veces frente a 26 veces que es la correcta.
- **Sesgo sí/no por plantilla**: en standard predice "no" 20/24 (gold 12/12); en hard "sí" 28/38.
- **Confianza**: ECE 0,125 en los 231; los fallos salen con 0,60 de confianza media.

### 1.2 Lo que castiga la puntuación actual de JevBench (v1.5+)

El tablero vivo ya no puntúa como v1.2. Lo que cambia para Temis:

- **noul con banda de abstención**: P(sí) entre 0,20 y 0,80 cuenta como **error**. De las 74 noul
  públicas, 43 caen en la banda; Temis pasaría de 41 aciertos a **17**. Hay que decidir con nitidez
  cuando se acierta (calibración por tipo pensada para esto, no solo para ECE).
- **score por valor esperado** (nMAE de la posición esperada, no argmax): hay que dar
  distribuciones centradas y afiladas, y el entrenamiento actual (ordinal 0,25) ya ayuda.
- **Pesos de tier 10/20/30/40**: hard vale 40 % de Inteligencia; **tipos** choice 50 / noul 25 / score 25.
- **Mitad sellada** (720 ítems, se renueva por versión): penaliza la brecha público–sellado > 8
  puntos. Entrenar contra los públicos se nota.
- Capability = media de Inteligencia y Calibración bajo topes de coste (≤ 0,0646 $/1 000) y latencia
  (≤ 1,23 s mediana ajustada: ×2 + 0,15 s para autoalojados). Temis cumple ambos de sobra.

### 1.3 Por qué los datos de entrenamiento no cubren nada de esto

| Hecho | Dato |
|---|---|
| Longitud de los estados sintéticos | mediana 312 caracteres, máximo 1 619; typed-decisions mediana 833, máximo 2 159. **Ningún ejemplo pasa de ~600 tokens.** |
| Dominios | ~80, todos "situación → decisión"; **cero** documentos normativos con excepciones, cero aritmética/fechas, cero "mención sin petición", cero opción `insufficient_information`. |
| Teacher en lo difícil | Jev acierta 74 % en hard: sus etiquetas blandas en casos difíciles llevan ~25 % de ruido. |
| max_len | 640 en entrenamiento e inferencia (2 048 solo si hay `context`). |
| Cuantización | En este Mac int8 ≈ fp32 (72 vs 75 ms a 320 tokens; 182 vs 212 a 680). El int8 solo gana tamaño aquí; gana velocidad en x86 con VNNI (el VPS). |
| Escalado de latencia (int8, 4 hilos, máquina cargada) | 320 tok 73 ms · 704 tok 193 ms · 2 112 tok 1,05 s · 4 160 tok 3,6 s. Leer documentos enteros es inviable: hay que **seleccionar evidencia**. |

## 2. Qué dice la literatura y los otros participantes (resumen útil)

- **Lectura (read-out) importa más que el tamaño del backbone** (mithril-lang/typed-decisions): una
  cabeza que empareja tokens de la opción con tokens de la pregunta superó a los marcadores
  simples; ModernBERT-base es inestable entre corridas; la aumentación con negaciones ayuda.
- **Permutación de opciones** (Verdict, PriDe, EMBER): KL simétrica entre dos barajados del mismo
  ítem baja la tasa de cambio de respuesta a <5 %; sin esto un modelo pequeño pasó de 72 % a 21 %
  al invertir sí/no.
- **Destilar el razonamiento del teacher sin generar texto** (arXiv 2510.11056, 2607.18693): el
  mismo encoder entrena en dos vías, una ve `estado + razonamiento`, otra solo `estado`; se alinean
  con KL + InfoNCE y se infiere con la segunda. +1–3 puntos en clasificación con razonamiento;
  razonamientos aleatorios empeoran (la ganancia es el contenido).
- **Selección de evidencia antes de clasificar** (IRIS ACL 2025, selección de oraciones): ~5
  oraciones relevantes alcanzan casi la exactitud del documento completo; los ModernBERT necesitan
  **fine-tuning con entradas largas** para rendir en largo.
- **Reglas + hechos**: encoders base resuelven RuleTaker al 98–99 %, pero **solo en el formato que
  vieron** (RoBERTa-large: 99 % dentro, 40 % en FOLIO). Hay que variar la forma de las reglas.
- **Calibración/distilación**: R-Drop +1,2 GLUE en BERT-base; promediar varias muestras del
  teacher reduce ruido; reponderar los ejemplos donde alumno y teacher discrepan.
- **Fallos típicos tras afinar** (paper LLM-as-Jev): sesgo a "sí", supresión de "ninguna de las
  anteriores", coincidencias léxicas, sobreconfianza. Son exactamente los de Temis.
- **Datos públicos en el mismo formato** (verificados): `tasksource/procedural-typed-decisions`
  (Apache-2.0, **396 000** casos deterministas, 18 tareas × niveles 0–4: aritmética de saldos,
  plazos con horas, reconstrucción de estado por eventos, suficiencia de evidencia, adjudicación
  entre versiones, navegación de grafos, ejecución de instrucciones…); `helmo/synthetic-typed-decisions`
  (MIT, 9 879, 207 dominios); `tasksource/tasksource-jev-typed-decisions` (1–2,6 M, licencias por fila,
  variantes de permutación/paráfrasis).

## 3. Líneas de trabajo

Cada línea dice qué, dónde (archivos), cómo se mide y qué se espera. El orden de ejecución está en §4.

### A. Medir bien (antes de tocar nada)

A1. `scripts/eval_jevbench.py`: corre los 231 públicos **directo contra `Decider`** (sin servidor),
informa por tier, familia y tipo, y calcula las dos puntuaciones: v1.2 (argmax) y **v1.5**
(banda de abstención noul, nMAE para score, pesos 10/20/30/40 y 50/25/25, ECE). Guarda
`runs/<run>/eval_jevbench.json`. El clon del repo de JevBench se baja a `external/jevbench`
(solo `datasets/public/*.jsonl`; sus licencias son MIT).

A2. **Conjunto interno "hard-like"**: 2 000 casos apartados de `procedural-typed-decisions`
(tareas y niveles no vistos) + 300 sintéticos propios de las 10 familias (§C2) etiquetados y
revisados. Es el que entra en la **puerta de promoción**; JevBench público solo se reporta, para no
sobreajustar la mitad que sí vemos.

A3. Pruebas de sesgo automáticas sobre validación: exactitud con *solo opciones*, con
*pregunta + opciones* (sin estado), tasa de cambio al rotar opciones, distribución sí/no predicha
vs gold por plantilla. Son los termómetros de "lee o adivina".

### B. Arreglos inmediatos (sin reentrenar)

B1. `render_state(max_chars=6000)` → 40 000. Hoy tira la mitad de cada `long_policy` antes de tokenizar.

B2. `max_len` dinámico en inferencia: 640 si el estado cabe; si no, hasta el presupuesto de §E1
(1 024–1 536) **después** de seleccionar evidencia. Sin selección, subir a 2 048 cuesta 1 s y
apenas cambia el resultado (33 → 34 % hard, ya medido): el cuello es que el modelo nunca entrenó
en largo, no solo el corte.

B3. Calibración por **tipo × número de opciones** y ajustada a la métrica de JevBench v1.5: para
noul, elegir la temperatura que maximiza el acierto *fuera de la banda* sin disparar el ECE; para
score, la que minimiza nMAE del valor esperado. Guardar como `temperatures_jevbench` y usarla en la
API cuando `model` lo pida (la de "acierto" sigue para Clio).

B4. Exportación ONNX: probar `optimum` con fusión de grafo (`--optimize`) y cuantización estática
con 500 casos de calibración; medir int8 vs fp32 en el VPS (x86) además del Mac. Probar
**int4 solo en embeddings** (197 de los 309 MB son el vocabulario de 256 k; no cuestan cómputo):
mmBERT-small perdió 0,002 de F1 con eso. Objetivo: ≤ 170 MB sin perder exactitud.

### C. Datos (la palanca principal)

C1. **procedural-typed-decisions** (gold exacto, cero coste de teacher): muestrear 30 000–40 000
preguntas balanceadas por tarea y nivel, con tope del **35 %** de la mezcla para que no desplace lo
de Clio. Cubre temporal_numeric, multi_hop, trap (adjudicación entre versiones), ambiguous
(suficiencia de evidencia) y state tracking (familia nueva de v1.5). Objetivo one-hot suavizado
(0,9). Es inglés: traducir un 20 % al español con Qwen local para no perder bilingüismo.
Apartar tareas/niveles enteros para A2.

C2. **Generador de familias difíciles propias** (`scripts/generate_hard.py`, prompt en
`docs/prompt_hard.md`): escenarios originales siguiendo la *especificación* pública de JevBench
(no sus ítems): documentos normativos de 1 500–4 000 tokens con 3–6 condiciones que interactúan,
enmiendas que anulan cláusulas, definiciones que cambian el sentido, relleno "parecido pero no"
(revisiones superadas, registros homónimos); cadenas de 3+ saltos (tabla → nota → alias → regla);
trampas (negación lejana, petición citada, condición revocada); rúbricas con
`insufficient_information` como gold la **mitad** de las veces; routing con palabras clave de la
opción equivocada; juicios de respuestas correctas-pero-escuetas y fluidas-pero-con-un-error.
Cada caso lleva `rationale` y `surface_answer` escritos por el generador (los usa §D2 y §E1).
**Pares de contraste**: por cada caso, una copia donde cambia un solo hecho y cambia la respuesta
(ataca directamente la lectura léxica; +18–20 % de robustez contrafactual en la literatura).
Español e inglés a partes iguales. Meta: 3 000 casos, ~6 000 preguntas.

C3. **Seguir la rúbrica en casos cortos** (para standard): 2 000 casos de una o dos frases:
mención sin petición ("Where is the parcel? Keep the address as it is"), negación ("Do not cancel;
refund the duplicate"), permiso bajo política con excepción ("unless registered contractor"),
umbrales inclusivos ("including exactly 24 hours"), adecuación de respuestas correctas y escuetas
("42"), especialista con pista engañosa. Balancear sí/no al 50 % **por plantilla de pregunta**.
Esto es barato y vale 28 % (v1.2) / 20 % (v1.5) de la Inteligencia.

C4. **Teacher para lo difícil**: Jev se equivoca ~25 % en hard. Para C2: etiqueta Jev + segunda
opinión de un **modelo privado** (decisión de Iván, 2026-11-09: el mismo que escribe los casos u
otro, vía prompt exportable como `docs/prompt_generar_casos.md`, con el `rationale` del autor a la
vista como verificador, no como respondedor); se conserva el caso si coinciden, si no va a la cola
de dudosos. Para C1 no hace falta (gold determinista). Promediar 3 muestras de Jev en los que se
conservan baja el ruido de la etiqueta blanda.

C5. `helmo/synthetic-typed-decisions` entero (9 879, MIT) y un subconjunto filtrado por licencia de
`tasksource-jev-typed-decisions` con las variantes `criteria_permutation` e `instruction_paraphrase`
(~20 000) como regularizador de generalidad. Prioridad baja; medir si aporta o solo diluye.

C6. Reglas de mezcla (en `temis/data/balance.py`): Clio (memoria, herramientas, seguridad, notas)
nunca por debajo del **35 %** de los targets; procedural ≤ 35 %; familias difíciles propias ~15 %;
rúbricas cortas ~5 %; resto general. La puerta de Clio (`eval_clio` ≥ actual − 0,01) sigue vigente.

### D. Entrenamiento

D1. **Consistencia ante permutaciones + R-Drop** (`decision_loss`): cada lote se codifica dos
veces con barajados distintos (y dropout distinto); se añade KL simétrica entre las dos
distribuciones (peso 0,5–1,0). Quita el sesgo de posición y estabiliza ModernBERT. Coste: 2×
cómputo por paso (o la mitad de lote). Métrica: tasa de cambio al rotar < 5 % (A3).

D2. **Destilación de razonamiento en doble vía** (solo entrenamiento): para los casos con
`rationale` (C2, y los sintéticos de Clio si se les pide al generar), una segunda codificación
`estado <ctx> razonamiento` con el mismo encoder; pérdida = CE(vía alumno) + 0,01·CE(vía teacher)
+ KL(alumno ‖ teacher) + InfoNCE entre los estados de `<cls>`/`<opt>` de ambas vías. La inferencia
no cambia. Reutiliza el marcador `<ctx>` ya existente.

D3. **Lectura**: ablación de la cabeza. Hoy: estado final del marcador `<opt>` concatenado con
`<opt>·<cls>`. Probar (a) media de los tokens de la opción en vez del marcador, (b) atención
cruzada opción → tokens de la pregunta + estado (la "mean-span matching head" que a mithril le
destrabó el aprendizaje). Mantener el que gane en validación interna; la exportación ONNX debe
seguir siendo trivial.

D4. **Entrenar en largo**: `max_len` 1 536 para el 15–20 % de ejemplos largos (C1 nivel alto, C2),
agrupados por longitud para que los lotes cortos sigan siendo baratos; `accum` ajustado. Currículo:
época 1 corto, épocas 2–4 mezcla. Barrido de LR (ModernBERT es sensible: 2e-5 … 6e-5, warmup
10 %), decaimiento por capa 0,9, **promedio de los 2–3 últimos checkpoints**. Embeddings siguen
congelados (memoria y multilingüismo), salvo prueba con `--train-emb` en MrBERT-es (vocabulario
pequeño: 39 M de parámetros, asequible).

D5. **Reponderación**: peso 1,5 a los ejemplos donde alumno y teacher discrepan al final de cada
época (estilo LGTM); peso por nivel de dificultad en C1 (niveles 3–4 ×1,5).

D6. Entrenamiento **desde la base**, no encadenar más fine-tunes: la cadena v1c → … → 1.0.11 arrastra
los sesgos. Una corrida limpia de 4–5 épocas con la mezcla nueva (como la nocturna que ya existe).

### E. Inferencia liviana para documentos largos

E1. **Selector de evidencia por líneas** (`temis/evidence.py`): el estado ya se renderiza en
líneas `clave: valor` o párrafos. Paso 1 (sin entrenar): puntuación léxica (BM25 sobre pregunta +
opciones, con bonus a líneas con números, fechas, "unless/except/salvo/supersede") y se conservan
encabezados/estructura + las mejores líneas hasta 1 024 tokens, en orden original y marcando los
cortes con «…». Paso 2: entrenar un selector pequeño (mmBERT-small, bi-encoder) con las líneas que
cita el `rationale` como positivas (datos de C2 y de los `rationale` de los ítems… **no**: solo de
C2, nunca de JevBench). Medir en el conjunto interno largo: exactitud con documento completo (lento)
vs seleccionado (rápido) vs cortado (hoy).

E2. Presupuesto de latencia por longitud: ≤ 50 ms corto, ≤ 400 ms largo en portátil (4 hilos);
registrar `ms` por ítem en A1 y en la API (`usage`).

E3. **TTA de permutación** opcional: 2 rotaciones circulares promediadas en `choice` con ≥ 3
opciones (2× latencia solo en esas). Activar solo si en validación gana ≥ 1 punto; para Clio queda
apagado por defecto.

E4. Sesión ONNX: hilos = núcleos físicos (en M-series 8 > 10 por defecto), fusiones de grafo,
providers como hoy (CPU, CoreML, CUDA).

### F. Modelo base

F1. **mmBERT-base sigue siendo la base**. Nada en el presupuesto la supera claramente en español +
inglés + 8 K de contexto + ONNX probado.

F2. **MrBERT-es** (corriendo ahora, `v1109-1028`): a falta de la última época va en
val acc 0,7615 / KL 0,157 frente a 0,7727 / 0,143 del 1.0.11 (que además venía afinado en cadena).
Decisión al terminar: evaluar en typed-decisions test, Clio, test personal y JevBench (A1). Se
adopta **solo** si iguala o supera en todo; su ventaja real es el archivo (~150 MB int8, vocabulario
de 51 k) y poder entrenar embeddings. Si queda 1 punto por debajo pero pesa la mitad, se guarda
como "Temis ligero" (major 2) sin cambiar el activo.

F3. Candidato secundario: `embeddinggemma-2` texto (Apache-2.0, 130 M en el transformer,
0,761–0,764 en typed-decisions) **solo** si se resuelve su exportación ONNX (diverge pasados ~257
tokens por la ventana deslizante) — no gastar más de medio día.

F4. Descartados (ver §0): ModernBERT-large, EuroBERT-610m, mdeberta, xlm-r-large, decodificadores.

### G. Apuesta de investigación (al final, si sobra tiempo)

G1. **Capas en bucle**: repetir las 4–6 capas superiores de mmBERT 2× con pesos compartidos y un
LoRA por iteración; +20–40 % de latencia, cero parámetros nuevos (fuera del LoRA). La evidencia es
en decodificadores y tareas sintéticas de razonamiento (12⊗2 > 12 capas); tratar como experimento
con exportación ONNX propia.

### H. Entrenar en la GPU de Iván (RTX 40xx, 8 GB)

La inferencia sigue en CPU; la GPU solo acorta el ciclo. `train.py` ya elige `cuda`; falta:
autocast **bf16**, `attn_implementation="sdpa"` al cargar en CUDA (eager se queda para la
exportación ONNX), lotes agrupados por longitud (los cortos no pagan el padding de los largos) y
`torch.mps.empty_cache` solo en MPS. Memoria: 110 M entrenables + Adam ≈ 1,3 GB; con checkpointing
y bf16, bs 16 a 640 tokens entra; en 1 536 tokens o con lotes gemelos (D1/D2), bs 4–8 con
`--accum` 2–4. Esperado: 8–15× más rápido que MPS (corrida limpia en 15–30 min en vez de 3,5 h),
lo que convierte barridos y ablaciones en cosa de una tarde. Qwen3.5-9B Q4 cabe en 8 GB para
generar; los modelos mayores no: para eso está el modelo privado (C4). `data/` y `runs/registry.json`
viajan por git; los pesos no.

## 4. Orden, esfuerzo y metas

| Fase | Contenido | Esfuerzo | Meta medible (231 públicos; v1.2 → también v1.5) |
|---|---|---|---|
| **0. Medir y arreglar** | A1, A3, B1, B2 (sin selector: 1 024 tokens), B3, B4 (medición) | 1 día | Mismo modelo: hard +3–6 ítems por el corte; **noul fuera de la banda** ≥ 70 % de las que acierta; informe por familia que sirve de línea base |
| **1. Rúbrica y sesgos** | C3, C1 (muestra de 15 k), D1, D6 (corrida limpia) | 2–3 días (una corrida de ~5 h en MPS) | **standard ≥ 85 %** (hoy 65), hard ≥ 40 %, Clio ≥ 0,85, td test ≥ 0,765, cambio al rotar < 5 % |
| **2. Largo y difícil** | C2 (3 000 casos + contrastes), C4, D4, E1 (léxico → entrenado), A2 | 1–1,5 semanas (generación local es lo lento) | hard ≥ 48 %: long_policy ≥ 30 %, multi_hop ≥ 40 %, trap/ambiguous ≥ 50 %; latencia larga ≤ 400 ms |
| **3. Afinar la lectura** | D2, D3, D5, E3, B4 (int4 emb, estática) | 3–4 días | +2–4 puntos totales; archivo ≤ 170 MB si int4-emb no cuesta exactitud |
| **4. Base y apuestas** | F2 decisión, F3 (½ día máx.), G1 | abierto | Solo se adopta lo que iguale todo y mejore algo |

**Meta global realista:** 57 % → **70–73 %** en los 231 públicos (standard ~90, hard ~48–52),
por encima de todos los encoders medidos (Laya 58,4 %) y en la zona de los LLM de 4 B en standard.
Jev (86,6 %) queda fuera de alcance para un encoder por `temporal_numeric` y `probability`: eso
requiere cálculo, no lectura. No se promete; se mide.

## 5. Puertas de promoción (amplían las de `scripts/retrain.py`)

Una versión nueva se activa solo si, frente a la activa:
val KL ≤ +0,01 · typed-decisions test acc ≥ −0,005 · Clio mean ≥ −0,01 · test personal ≥ −0,02 ·
**hard-like interno (A2) ≥ −0,01** · latencia p50 corta ≤ +20 % · tasa de cambio al rotar ≤ 5 %.
JevBench público se reporta siempre, pero **no decide** la promoción (se declara en el README como
hasta ahora: resultados propios sobre los públicos, no puntuación oficial).

## 6. Riesgos y cómo se controlan

| Riesgo | Control |
|---|---|
| Olvidar lo de Clio al meter 40 k casos nuevos | Tope del 35 % para procedural; Clio ≥ 35 %; puerta de Clio; replay siempre |
| Etiquetas de Jev erróneas en lo difícil enseñan el error | C4: doble opinión + cola de dudosos; gold determinista en C1 |
| Sobreajustar a los 231 públicos | Nunca se entrena con ellos; la promoción usa A2; se vigila la brecha si llega una puntuación sellada |
| Latencia que crece sin darnos cuenta | E2: presupuesto por longitud medido en cada evaluación y en `usage` |
| Procedural es inglés y aritmético: Temis se vuelve "calculadora en inglés" | 20 % traducido; mezcla acotada; medir español aparte en validación |
| ModernBERT inestable al entrenar | Barrido corto de LR, warmup 10 %, promedio de checkpoints, semilla fija |
| El generador local (Qwen3.5-9B) escribe documentos largos flojos | Validador automático (longitud, nº de condiciones, que la respuesta no esté nombrada en el estado); muestreo revisado a mano por lote; opción de pedir los lotes largos a un modelo mayor |
| int4 en embeddings o cuantización estática pierden exactitud | Se compara en los 4 conjuntos antes de adoptar; el fp32 queda disponible |

## 7. Qué no vamos a hacer

- Crecer de modelo "porque sube el benchmark".
- Depender de GPU en inferencia.
- Entrenar con ítems públicos de JevBench ni derivados directos (paráfrasis de sus ítems).
- Encadenar más fine-tunes sobre 1.0.11: la siguiente versión mayor sale de una corrida limpia.
