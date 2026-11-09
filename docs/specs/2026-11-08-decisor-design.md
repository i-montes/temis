# Temis (proyecto decisor): diseño (v1)

Fecha: 2026-11-08. Decisiones tomadas de forma autónoma a pedido de Iván ("toma todas las decisiones tú").

## Objetivo

Un modelo **pequeño y general** de decisiones tipadas con el mismo formato que TypeSafe/Jev
(`choice`, `score`, `noul`). Tiene que:

- correr en cualquier equipo, en CPU o GPU;
- servir para Clio (herramientas, seguridad, memoria), La silla vacía (etiquetas, enlaces, notas de
  contexto) y usos que aún no existen (por ejemplo, un cohete, siempre fuera del lazo de control de vuelo);
- mejorar con el uso, mientras Iván solo etiqueta una cola de casos dudosos desde una web.

**Éxito de la v1:**

- supera a `mrm8488/embeddinggemma-2-ft-typed-decision` (0,761) en el `test` de typed-decisions sin
  haberlo usado para entrenar;
- responde en menos de ~50 ms por pregunta en CPU con ONNX int8;
- etiquetar un caso en la web cuesta una o dos teclas.

## Decisiones

| Tema | Decisión | Por qué |
|---|---|---|
| Alumno | `jhu-clsp/mmBERT-small` (~140M, ModernBERT multilingüe, 8K de contexto) | Pequeño, multilingüe (español), rápido, exportable a ONNX. |
| Arquitectura | Cross-encoder empaquetado: `[tipo] pregunta <opt> opción… [SEP] estado`. Cada `<opt>` se puntúa con una MLP y se aplica softmax. | Lee la pregunta junto con el estado (lo que le falta al bi-encoder); una pasada por pregunta; esquema como datos. |
| Tipos | `noul` = 2 opciones (false/true); `score` = niveles ordenados con pérdida ordinal; `choice` = opciones libres (hasta 12). | Una sola cabeza para todo. |
| Teacher | **Solo Jev** (API, tope de 10 USD). Qwen3.5-9B local solo genera los casos sintéticos. | Qwen como teacher era flojo (ver el registro de cambios); decisión de Iván del 2026-11-08: quitarlo definitivamente. |
| Calibrar el teacher | Una temperatura de Jev por tipo, ajustada contra el gold de typed-decisions `train` (choice 2,6, noul 1,2, score 3,7). | Jev está mal calibrado (KL 1,44 en el leaderboard). |
| Datos | typed-decisions `train` (gold propio) + casos sintéticos generados por Qwen3.5-9B en ~80 dominios, en español e inglés. | La generalidad sale de la variedad de esquemas. |
| Validación | 10 % de los esquemas sintéticos (por `schema_id`) + 10 % de typed-decisions `train`, apartados. `test` solo al final. | Mide generalización a esquemas no vistos. |
| Pérdida | Entropía cruzada con etiquetas suaves + 0,25 Brier + 0,25 ordinal (solo `score`). | Igual que la referencia; apunta a buena calibración. |
| Calibración del alumno | Dos temperaturas por tipo, guardadas con el modelo: **por acierto** (la usan la API y la web: un 70 % acierta ~70 %) y **suave** (imita a los teachers; la usan los benchmarks para que el KL sea comparable). | Las etiquetas suaves dejaban al modelo infraconfiado (ECE sí/no 0,093 → 0,018 con la de acierto). |
| Aumentos | Barajar opciones (salvo `score`), y a veces añadir contexto irrelevante. | Que no dependa del orden y que aprenda a ignorar contexto que no aporta. |
| Runtime | ONNX + `tokenizers` + numpy (sin torch), con int8 opcional; proveedores CPU, CoreML o CUDA según el equipo. | Que corra en cualquier lado. |
| API | `POST /v1/systemone`, compatible con TypeSafe, más un campo opcional `context`. | Clio puede usarlo cambiando solo la URL. |
| Memoria de decisiones | SQLite `data/decisor.db`: decisiones, etiquetas, versiones de modelo. | Local, un archivo. |
| Cola de dudosos | Prioridad = máximo entre la duda del modelo, su desacuerdo con Jev y la duda de Jev; ordenada de mayor a menor. | Iván etiqueta solo lo que más enseña. |
| Precedentes | Las respuestas etiquetadas de la misma pregunta (misma firma: tipo + instrucciones + opciones) se mezclan con el modelo: P = λ·P_prec + (1−λ)·P_modelo, con λ creciente según el número de precedentes. | Aprende al instante sin reentrenar. |
| RAG / evidencia | Interfaz `Retriever`. Backend 1: **Clio Memory** (`clio-memory recall --json`), usando el binario si ya está instalado o descargándolo del release con verificación sha256, sin tocar la configuración de los agentes. Backend 2: SQLite FTS5 propio. Los pasajes recuperados entran como contexto marcado. | Lo pidió Iván: si existe un RAG, se usa; si no, Clio Memory. |
| Reentrenamiento | Botón en la web: reentrena con lo etiquetado más un repaso de los datos viejos; solo se promueve si mejora en validación. | Sin olvido catastrófico ni regresiones. |
| Recursos | `nice`, 6 hilos de CPU, un único LLM cargado (~6 GB), ventana de 24K tokens. | "No uses todo el computador". |

## Cambios durante la noche (registro de decisiones)

- **Qwen3.5-9B como teacher es flojo** frente al gold de typed-decisions: `choice` 0,51, `noul` 0,60, `score` 0,45, contra 0,72, 0,80 y 0,73 de Jev. Calibrados, la mezcla le da poco peso a Qwen (0,1 en `choice`, 0 en `score`). Decisión: Jev, calibrado, es el teacher principal; Qwen pasa a ser **generador** de esquemas y estados; sus etiquetas ya hechas se conservan para medir desacuerdo.
- **Generación con `llama-server`**, no con Ollama: Ollama atiende una petición a la vez y la generación bloqueaba el etiquetado. El GGUF de Ollama no carga en llama.cpp, así que se usa `unsloth/Qwen3.5-9B-GGUF` Q4_K_M (Apache-2.0) con 3 slots y caché KV en q8_0 (~5 GB).
- **Memoria al entrenar:** embeddings de tokens congelados, gradient checkpointing (de 7 GB a 2 GB en MPS, misma velocidad), padding a múltiplos de 64 tokens y tope de memoria MPS. Sin esto el sistema entraba en swap.
- **Cola de dudosos** sin un segundo teacher: la duda sale de la incertidumbre del propio teacher y del desacuerdo entre el modelo y el teacher.
- **Clio Memory:** su CLI no puede crear una memoria nueva (se crea con la primera escritura MCP). Reportado en i-montes/clio-memory#9 con una propuesta de `clio-memory init`.

- **Qwen fuera como teacher (definitivo, decisión de Iván):** confirmado también por las etiquetas de Iván (Qwen coincide ~70 % y Jev ~100 % en las primeras 32 respuestas). El código del teacher local se eliminó; Qwen sigue solo como generador.
- **Calibración por acierto:** el modelo era infraconfiado (cuando decía 80–90 % acertaba 96 %). Se añadió `scripts/calibrate.py` y la cola se recalcula al promover una versión.

- **Filtro de calidad de casos** (`temis/data/quality.py`), tras revisar la cola: una pregunta `noul` debe ser una afirmación verdadera o falsa. Las condicionales («Determine if X») se reescriben como «X»; las vagas («Evalúa el peso emocional»), las disyuntivas y los estados casi vacíos se descartan. Se aplica al generar, al cargar los datos (sin tocar el archivo) y a la cola. Sobre los datos existentes: 1.509 preguntas reescritas y 334 descartadas.

- **Estancamiento y plan de mejora** (13:30). Con la versión v1108-1022, el acuerdo con Jev en validación es más bajo justo donde Jev duda: operación de memoria 62 % con Jev al 50 % de confianza, elegir herramienta 61 % con Jev al 61 %. En cambio, etiquetas de notas 100 % y go/no-go 94 %. Las familias débiles tenían pocos datos y la medición es ruidosa (±17 puntos). Cambios:
  1. `temis/data/families.py`: pesos por familia. El generador saca hasta 3 veces más casos de las débiles y pide que 3 de cada 4 estados sean claros.
  2. La cola prioriza las familias débiles (columna `priority` = duda + bonus).
  3. Test personal: 1 de cada 5 casos etiquetados por Iván queda reservado y nunca se entrena (`scripts/eval_user.py`). Cuenta para promover versiones desde 150 preguntas.
  4. Una vez por noche (desde la 1:00), el ciclo entrena desde cero 5 épocas en lugar de un ajuste corto.

- **Nombre:** el modelo se llama **Temis** (decisión de Iván), con versiones `MAYOR.MENOR.PARCHE` (`temis/naming.py`, `runs/registry.json`). La API responde `"model": "temis-1.0.5"` y mantiene el identificador del entrenamiento en `"run"`.

## Fuera de alcance (v1)

`box` (imágenes) y `span` (ubicar un fragmento de texto); el control de vuelo del cohete; aprendizaje
online de pesos completos.
- 2026-11-08 · El ciclo automático (generación con Qwen y entrenamiento cada 800 casos o cada noche) queda apagado por decisión de Iván: sin datos nuevos no aporta. Se entrena a mano con `scripts.retrain`, que ahora también hace que Jev etiquete los casos nuevos, reconstruye los objetivos, ajusta 2 épocas, refresca la cola y libera espacio.
- 2026-11-08 · Rebalanceo de memory_op (temis/data/balance.py): Temis llamaba "corrige" a toda propuesta ya guardada (0 de 6 en el benchmark de Clio) porque en los casos de Qwen Jev elegía corregir en el 51 % y "repetida" en el 4 %. Ahora las correcciones de Qwen entran el 30 % de las veces y los casos externos de memory_op cuentan doble: la mezcla queda nueva 31 %, sospechosa 28 %, corrige 27 %, repetida 13 %. El prompt para otros modelos pide "repetidas" disfrazadas, pares de contraste y casos límite de herramientas.
- 2026-11-08 · Revisión fuera de la web (scripts/review_sheet.py): la cola se exporta a CSV (máx. 20 casos por familia) y otro modelo la responde. Esas respuestas entran como sugerencias y además como segunda opinión en el entrenamiento (data/reviewer_targets.jsonl): objetivo suave de 0,7 en la opción elegida, peso de un objetivo de Jev, fuera del test personal de Iván, de la validación y de lo que Iván ya etiquetó.
