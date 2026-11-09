# Prompt para generar casos cortos de rúbrica con otro modelo

Casos de una a tres frases donde lo difícil es seguir la rúbrica al pie de la letra: distinguir una
mención de una petición, respetar una negación, aplicar un permiso con excepción, un umbral
inclusivo, juzgar una respuesta escueta pero correcta, elegir el especialista pese a una palabra
engañosa. Es donde Temis más falla con menos texto.

```sh
./.venv/bin/python -m scripts.import_cases rubrica.jsonl --source NOMBRE-DEL-MODELO
```

---

```text
Eres un generador de datos de entrenamiento para un modelo pequeño que toma decisiones tipadas:
lee una situación corta ("state") y responde una pregunta cerrada siguiendo una rúbrica. Tu trabajo
es escribir CASOS CORTOS en los que la respuesta correcta sale de aplicar la rúbrica literalmente,
y una lectura por palabras clave da otra respuesta. NO escribas la respuesta en el state.

FORMATO DE SALIDA (obligatorio)
- Solo JSON Lines: un objeto por línea, sin texto antes ni después, sin ``` ni comentarios.
- Claves: {"family": "rubric_short", "language": "es" | "en", "workflow": "...", "state": "texto",
           "questions": {...}, "expected": {...}, "rationale": "...", "surface_answer": {...}}
  - language: "es" en la mitad de los casos y "en" en la otra mitad.
  - workflow: nombre corto en snake_case en inglés con el TEMA (sin revelar la respuesta).
  - state: texto de 1 a 3 frases (mínimo 120 caracteres: añade contexto natural si hace falta).
  - questions: UNA pregunta, con una de estas formas:
    * choice: {"type":"choice","instructions":"<la rúbrica>","criteria":{"clave":"descripción", ...}} (3 a 6 opciones)
    * noul:   {"type":"noul","instructions":"<pregunta cerrada o afirmación>","criteria":{"false":"...","true":"..."}}
    * score:  {"type":"score","instructions":"<la rúbrica>","criteria":["nivel 0","nivel 1",...]} (3 a 5 niveles)
    La rúbrica va dentro de instructions, en una o dos frases, y dice cómo decidir en los casos dudosos.
  - expected: {id_pregunta: opción correcta} (noul: "true"/"false"; score: índice del nivel como texto).
  - rationale: una frase que cita la regla de la rúbrica y el hecho del state que la activa.
  - surface_answer: {id_pregunta: opción} con la respuesta tentadora equivocada (para noul, "true"/"false").

PLANTILLAS (escribe casos de las seis, repartidos)
1. Mención sin petición. Rúbrica: "Elige la acción principal que se pide. Mencionar algo no es pedirlo."
   State: un mensaje que menciona una acción (cancelar, reembolso, dirección) pero pide otra, o no pide
   ninguna ("Gracias, ya entendí la política de reembolsos"). Opciones: acciones + "other"/"ninguna".
2. Negación. "No canceles la suscripción; devuélveme el cobro duplicado" → la acción es el reembolso.
   Negaciones lejos del verbo, dobles negaciones naturales, "ya no hace falta".
3. Permiso con excepción. Rúbrica: "Según la política, ¿se permite la acción? Las condiciones no
   demostradas cuentan como no cumplidas." State: política de una frase con excepción ("necesitan
   escolta salvo contratistas registrados") y un caso que la cumple, no la cumple o no lo demuestra.
   Mitad permitido, mitad no.
4. Umbral inclusivo. "Cancelación gratis hasta 24 horas antes, incluidas exactamente 24" frente a
   "antes de"; ≥ frente a >; "a partir de"; "hasta el día 15 inclusive". Mitad dentro, mitad fuera.
5. Adecuación de una respuesta. state: "Petición: ... Referencia: ... Respuesta: ...". Rúbrica:
   "¿La respuesta satisface por completo la petición usando la referencia?" Respuestas CORRECTAS y
   escuetas ("42", "No, cierra los domingos") que deben ser sí, y respuestas fluidas que incumplen un
   requisito explícito ("exactamente dos palabras", "solo el número") que deben ser no. Mitad y mitad.
6. Especialista con pista engañosa. Rúbrica con 4–6 especialistas (matemáticas, código, documentos,
   herramientas/calendario, general, ...) y descripciones que se solapan; la petición usa una palabra
   de la opción equivocada ("calcula" en una tarea de documentos, "archivo" en una de cálculo).

CALIDAD
- Una sola respuesta defendible. Equilibra las respuestas entre las opciones y, en noul, sí/no al 50 %
  DENTRO de cada plantilla (no todas las de permiso "no", no todas las de adecuación "sí").
- Varía personas, servicios, productos y redacción. Nada de placeholders. Mitad español, mitad inglés.
- Escribe también las versiones "fáciles" de cada plantilla (sin trampa) en 1 de cada 4 casos, para
  que el modelo no aprenda que siempre hay trampa.

EJEMPLO DE UNA LÍNEA VÁLIDA (no la copies):
{"family":"rubric_short","language":"en","workflow":"locker_pickup_code_question","state":"Message from a customer to the pickup-locker helpdesk: My pickup code has not arrived yet, can you tell me when it will be sent? And please do not change the locker location, the one near the station suits me fine.","questions":{"accion":{"type":"choice","instructions":"Which action does the customer actually ask for? Something that is only mentioned, or explicitly asked NOT to be done, is not a request.","criteria":{"reenviar_codigo":"Send the pickup code again or say when it comes","cambiar_casillero":"Move the parcel to another locker","cancelar_recogida":"Cancel the pickup","consulta_general":"Answer a general question with no action","ninguna":"Nothing is being asked"}}},"expected":{"accion":"reenviar_codigo"},"rationale":"The only thing asked is when the code arrives; the locker location is mentioned to say it must stay as it is.","surface_answer":{"accion":"cambiar_casillero"}}

ENCARGO
Genera 120 casos: 20 por plantilla, mitad en español, mitad en inglés, respuestas equilibradas dentro
de cada plantilla. No repitas escenarios de lotes anteriores.
```
