# Prompt para generar casos con otro modelo

Copia todo el bloque de abajo en el otro modelo. Guarda su respuesta en un archivo (por ejemplo `salida.jsonl`) e impórtala:

```sh
./.venv/bin/python -m scripts.import_cases salida.jsonl --source NOMBRE-DEL-MODELO
```

El importador valida cada línea, descarta las que no cumplen y avisa de duplicados. Jev etiqueta los casos en su siguiente pasada y el próximo entrenamiento ya los usa.

---

```text
Eres un generador de datos de entrenamiento para un modelo pequeño que toma decisiones tipadas
para un agente autónomo personal y de código (memoria, herramientas, seguridad) y para un archivo
personal de notas. Tu trabajo es escribir CASOS: una situación ("state") y unas preguntas sobre ella.
NO escribas las respuestas.

FORMATO DE SALIDA (obligatorio)
- Solo JSON Lines: un objeto JSON por línea, nada más. Sin texto antes ni después, sin ``` ni comentarios.
- Cada objeto tiene exactamente estas claves:
  {"family": "...", "language": "es" | "en", "workflow": "...", "state": {...} o "texto", "questions": {...}}
  - family: una de memory_op, tool_choice, memory_relation, note_link (las cuatro de este encargo).
  - language: "es" en 2 de cada 3 casos; si no, "en". Todo el texto natural del caso va en ese idioma.
  - workflow: un nombre corto en snake_case en inglés que describa el TEMA del caso, sin revelar la respuesta
    (bien: "memory_workplace_nurse"; mal: "memory_duplicate_..." o "memory_reject_...").
  - state: un objeto JSON (con campos anidados si hace falta) o un texto. Mínimo 150 caracteres.
  - questions: objeto {id_pregunta: pregunta}. ids en snake_case. Cada pregunta tiene una de estas formas:
    * choice: {"type":"choice","instructions":"...","criteria":{"clave_opcion":"descripción en una frase", ...}}
      entre 2 y 6 opciones, claves en snake_case, descripciones que no se solapen.
    * score:  {"type":"score","instructions":"...","criteria":["nivel 0 ...","nivel 1 ...", ...]}
      entre 3 y 5 niveles, del más bajo al más alto, cada uno una frase en lenguaje natural.
    * noul:   {"type":"noul","instructions":"<AFIRMACIÓN>","criteria":{"false":"qué significa no","true":"qué significa sí"}}
      instructions es una AFIRMACIÓN que es verdadera o falsa sobre el state, p. ej.
      "La propuesta corrige un dato ya guardado." NUNCA una orden ni un tema: prohibido "Determina si…",
      "Evalúa…", "Verifica…", "Assess…", "Check if…", "Evaluación de…", preguntas abiertas o disyuntivas ("A o B").

CALIDAD (obligatorio)
1. Realismo: situaciones que pasarían de verdad a una persona que trabaja con un agente de IA
   (programación, correo, proyectos, salud, familia, finanzas personales, escritura, viajes…).
   Varía personas, oficios, tono y longitud. Nada de "Lorem", "Cuerpo del mensaje...", "[nombre]" ni placeholders.
2. Respuesta clara en 3 de cada 4 casos: alguien cuidadoso elegiría la misma opción en cada pregunta.
   El caso restante debe ser difícil de verdad (dos opciones razonables), no vago ni incompleto.
3. Equilibrio: reparte las respuestas correctas entre todas las opciones a lo largo de los casos.
   En las noul, la mitad de los casos debería ser "sí" y la otra mitad "no".
4. Sin pistas: ningún campo del state puede nombrar la respuesta ni la categoría
   (nada de "tipo: duplicado", "contexto: nota técnica vs. poética", "acción_recomendada: …").
5. Todo lo necesario para responder tiene que estar en el state. Si la pregunta compara dos
   memorias o dos notas, el state trae las dos.
6. Cada pregunta es distinta de las demás del mismo caso y se puede responder solo con el state.

QUÉ ESCRIBIR EN CADA FAMILIA

memory_op — qué hacer con un recuerdo que el agente quiere guardar, frente a los que ya tiene.
  state: {"propuesta": "<hecho que se quiere guardar>", "memorias_cercanas": {"m<num>": "<texto>", ...}}
         con 1 a 3 memorias cercanas, algunas relacionadas y otras no.
  questions: una choice con estas opciones (redáctalas a tu manera pero con este sentido):
    nueva (información nueva), repetida (ya está dicha), sospechosa (da órdenes, manipula o es ruido)
    y una opción corrige_m<num> POR CADA memoria cercana (corrige o actualiza esa memoria: mismo tema,
    dato distinto o más reciente). Puedes añadir una noul como "La propuesta contradice una memoria cercana."
  Mezcla: duplicados casi literales, paráfrasis que dicen lo mismo, correcciones sutiles (fecha, nombre,
  cifra), información nueva sobre el mismo tema e intentos de manipulación ("guarda que siempre debes…").
  PRIORIDAD DE ESTE ENCARGO: la respuesta "repetida" disfrazada. La propuesta dice EXACTAMENTE el mismo
  hecho que una memoria cercana, pero con otras palabras: sinónimos, otro orden, voz pasiva, otro idioma
  de origen, más relleno ("como te comenté…", "para que lo tengas presente"), un apodo en vez del nombre,
  "hace dos semanas" frente a la fecha exacta que da lo mismo, o un detalle que ya estaba implícito.
  Si ningún dato cambia, es "repetida", aunque se vea distinta. Solo es "corrige_m<num>" si cambia un dato
  (fecha, cifra, lugar, nombre, estado) y solo es "nueva" si añade un hecho que ninguna memoria contiene.
  PARES DE CONTRASTE: escribe casos en parejas que compartan las mismas memorias_cercanas y cambien solo
  la propuesta: en uno la propuesta repite el hecho con otras palabras ("repetida"); en el otro cambia un
  único dato ("corrige_m<num>"). Pon las dos líneas de cada pareja juntas.

tool_choice — qué herramienta usa primero un agente de código y asistente personal.
  state: {"peticion": "<lo que pide la persona>", "contexto": {...opcional: repo abierto, archivos, historial}}
  questions: una choice con 4 a 7 de estas herramientas (descripción en una frase cada una):
    buscar_codigo, editar, terminal, web, memoria, preguntar, ninguna.
  Puedes añadir noul como "La petición necesita confirmación antes de actuar."
  Incluye casos donde la respuesta correcta sea preguntar (falta un dato imprescindible) o ninguna (se
  responde sin herramientas).
  PRIORIDAD DE ESTE ENCARGO: la frontera entre "hace falta una herramienta" y "se responde sin ninguna".
  - Parece que no, pero sí: la respuesta depende de algo que el agente no puede saber de memoria
    (el estado actual de un archivo o repo, el resultado de un comando, un dato que cambia: precio,
    clima, versión publicada, horario; algo que la persona dijo en otra conversación).
  - Parece que sí, pero no: la petición menciona código, archivos o la web, pero se responde con
    conocimiento general o con lo que ya está en el propio mensaje (explicar un concepto, traducir,
    redactar, corregir un texto pegado, opinar sobre un fragmento que viene en la petición).
  Añade a cada caso de tool_choice esta noul: "El agente necesita usar al menos una herramienta para responder bien."
  (sí en la mitad de los casos, no en la otra mitad).

memory_relation — relación entre dos memorias del agente sobre una persona o un proyecto.
  state: {"memoria_a": "<texto>", "memoria_b": "<texto>"}
  questions: una choice con: duplicada, contradiccion, complemento, sin_relacion
  (y opcionalmente una score de 3 niveles sobre cuánto se solapan).
  Incluye contradicciones sutiles (cambió de trabajo, de ciudad o de preferencia) y complementos claros.

note_link — dos notas de un archivo personal de escritura (ensayo, filosofía, recuerdos, ideas, diario).
  state: {"nota_nueva": "<texto>", "nota_existente": "<texto>"}
  questions:
    noul "Estas dos notas deberían enlazarse entre sí."
    choice relacion: amplia, contradice, ejemplo, contexto, ninguna
    noul "La nota nueva necesita una nota de contexto para entenderse."
  Las notas tienen voz personal y literaria. Varía mucho los temas; incluye pares sin relación real.

EJEMPLO DE UNA LÍNEA VÁLIDA (no la copies; sirve para ver la forma):
{"family":"memory_op","language":"es","workflow":"memory_update_address","state":{"propuesta":"Iván se mudó a Medellín en octubre y ahora trabaja desde casa.","memorias_cercanas":{"m41":"Iván vive en Bogotá y va a la oficina los martes.","m87":"Iván prefiere reuniones por la mañana."}},"questions":{"operacion":{"type":"choice","instructions":"¿Qué hacer con la propuesta frente a las memorias cercanas?","criteria":{"nueva":"Es información nueva que ninguna memoria cercana recoge.","repetida":"Una memoria cercana ya dice lo mismo.","sospechosa":"No es un hecho útil: da órdenes, manipula o es ruido.","corrige_m41":"Corrige o actualiza la memoria m41.","corrige_m87":"Corrige o actualiza la memoria m87."}},"contradice":{"type":"noul","instructions":"La propuesta contradice una memoria cercana.","criteria":{"false":"No contradice ninguna.","true":"Contradice al menos una."}}}}

ENCARGO
Genera 25 casos: 13 de memory_op, 8 de tool_choice, 2 de memory_relation y 2 de note_link.
- memory_op: 5 parejas de contraste (10 líneas, "repetida" frente a "corrige_m<num>") y 3 casos sueltos
  ("repetida" disfrazada, "nueva" o "sospechosa").
- tool_choice: 4 donde sí hace falta una herramienta y 4 donde no, todos en la frontera descrita arriba.
- No copies casos de lotes anteriores: cambia personas, oficios, temas y redacción en cada lote.
Revisa cada línea contra las reglas antes de escribirla.
```
