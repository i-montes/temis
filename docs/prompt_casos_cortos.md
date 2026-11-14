# Prompt para casos cortos, con la forma de lo que Clio pregunta

Los casos generados empiezan en 120 caracteres de estado (el filtro de calidad descartaba lo más corto por
vacío), pero lo que una persona le escribe a su agente suele medir entre 10 y 100. Temis no había visto
nada así y fallaba justo ahí: a «abre tal archivo y resúmelo» le daba 0,2 de «hace falta leer archivos», y
0,9 al mismo pedido escrito largo. Estos lotes cubren ese hueco con las preguntas que Clio hace de verdad.

El modelo que escribe entrega **líneas compactas** (lo escrito y la respuesta correcta). El conversor arma
las preguntas con la redacción de producción y el importador las guarda como casos cortos con el oro del autor:

```sh
./.venv/bin/python -m scripts.expand_short lote.jsonl --check          # valida; dice qué línea está mal
./.venv/bin/python -m scripts.expand_short lote.jsonl > casos.jsonl
./.venv/bin/python -m scripts.import_cases casos.jsonl --source NOMBRE-DEL-MODELO --short
./.venv/bin/python -m scripts.label --sources synth                    # Jev etiqueta los nuevos
./.venv/bin/python -m scripts.build_targets                            # oro del autor si Jev o la segunda opinión coinciden
```

Para un conjunto apartado (evaluación, nunca entrenamiento) se usa `--every`: cada mensaje lleva la pregunta
de todas las herramientas, como en producción.

---

```text
Escribes datos para Temis, un modelo pequeño que toma decisiones tipadas para Clio, un agente personal y
de código. Cada CASO es una situación real en la que Clio le pregunta algo a Temis, con su RESPUESTA
CORRECTA. Los casos de este encargo son CORTOS: lo que una persona teclea de verdad.

FORMATO DE SALIDA (obligatorio)
- Solo JSON Lines: un objeto por línea, sin texto antes ni después, sin ``` ni comentarios.
- Cada línea tiene "f" (la familia), "lang" ("es" o "en") y los campos de su familia, tal como abajo.
- Las preguntas NO se escriben: las arma el conversor. Tú das lo escrito y las respuestas.

CÓMO ESCRIBE LA GENTE (vale para todo mensaje de persona: campo "msg")
- De 2 a 25 palabras; la mayoría entre 4 y 12. Nada de «Hola, necesito que por favor…» salvo 1 de cada 10.
- 7 de cada 10 en español, el resto en inglés. En español: la mitad sin tildes ni «¿», en minúsculas, a
  veces con una errata o sin signos; términos técnicos en inglés mezclados («haz un rebase», «sube el PR»).
- Nombres concretos e inventados: archivos, carpetas, servicios, proyectos, personas, mascotas, ciudades.
  Nunca placeholders («[archivo]», «foo», «X»). Varía oficio, tema, tono e intención en cada línea.
- Preguntas, órdenes, comentarios, respuestas sueltas («dale», «ok gracias»): todo lo que llega a un chat.

FAMILIAS

pre — llega solo el mensaje; por cada herramienta, ¿hace falta para atenderlo?, y ¿hace falta memoria?
  {"f":"pre","lang":"es","msg":"...","tools":["read","edit"],"skip":["bash"],"mem":false}
  - tools: TODAS las herramientas que hacen falta para atender bien el mensaje. Lo que no esté aquí ni en
    skip se entiende como «no hace falta».
  - skip: las discutibles: alguien cuidadoso podría decir sí o no. Úsalo sin miedo; no fuerces un sí o un no.
  - mem: true si para responder bien hace falta saber algo personal de esta persona o recordar una
    conversación anterior, algo que no está en el propio mensaje. false si no. null si es discutible.
  Las 17 herramientas (nombre: cuándo SÍ hace falta):
    read: el contenido de un archivo que ya existe (se nombra, o hay que mirar código, configuración o
          registros para responder o para cambiar algo). Cambiar un archivo existente exige leerlo: si edit, read.
    write: crear un archivo nuevo o reescribir uno entero.
    edit: cambiar un trecho de un archivo que ya existe.
    bash: ejecutar comandos: pruebas, git, instalar, compilar, procesos, red, disco, docker, scripts.
    find: buscar archivos por su nombre cuando no se sabe dónde están.
    ls: ver qué hay dentro de una carpeta.
    memory_search: consultar lo que se recuerda de la persona o de conversaciones pasadas («como la otra
          vez», «lo que te dije de…», sus gustos, sus datos, sus proyectos).
    memory_get: leer una memoria concreta por su número («muéstrame la m41»).
    memory_timeline: ver qué se guardó antes y después de una memoria, en la misma sesión.
    memory_context: retomar el trabajo: el resumen de la última sesión y lo reciente del proyecto.
    memory_save: el mensaje trae algo que conviene recordar (un dato de la persona, una decisión, una
          causa) o pide recordarlo.
    memory_session_summary: guardar el resumen de la sesión, al cerrar o tras compactar.
    memory_update: corregir o poner al día una memoria concreta que quedó desfasada o mal.
    memory_consolidate: unir varias memorias que dicen lo mismo.
    memory_forget: olvidar un recuerdo que ya no sirve o está mal.
    memory_revert: deshacer un cambio propio de la memoria.
    memory_pin: fijar un recuerdo para tenerlo siempre presente.
  Reglas:
  - Juzga solo con el mensaje: lo que hace falta para atenderlo bien, no lo que quizá venga después.
  - Sin herramientas (tools: []): saludos, gracias, conocimiento general, opinión, cálculo simple, y
    redactar, traducir o corregir un texto que viene dentro del mensaje.
  - Parece que sí, pero no: nombra código, archivos o comandos pero se responde con conocimiento general
    («qué hace git stash», «para qué sirve un Dockerfile»).
  - Parece que no, pero sí: la respuesta depende del estado real del equipo o del proyecto («ya subió el
    servidor?» → bash; «quedó algo sin commitear?» → bash).
  - Si mem es true, tools o skip llevan al menos una de memory_search, memory_context, memory_get,
    memory_timeline. Si tools lleva memory_search o memory_context, mem es true.
  - Lo que no puedas defender con el mensaje delante va a skip (o mem: null).

rel — un mensaje y unas memorias candidatas; por cada memoria, ¿ayuda a responder el mensaje?
  {"f":"rel","lang":"es","msg":"...","mems":{"m12":"...","m40":"..."},"helps":["m12"]}
  - 2 a 5 memorias; claves m<número> con números variados y sin orden. Cada memoria es una línea como las
    guarda un agente (40 a 250 caracteres): un dato de la persona, una decisión del proyecto, una
    preferencia, una lección; a veces con la forma «Qué: … · Por qué: … · Dónde: …».
  - helps: las claves que ayudan de verdad a responder ESE mensaje. Mezcla memorias que ayudan, memorias
    del mismo tema que no ayudan (se parecen en las palabras pero hablan de otra cosa) y memorias sin relación.
  - En 1 de cada 4 casos no ayuda ninguna (helps: []); en 1 de cada 5 ayudan dos o más.

guard — antes de guardar un recuerdo: ¿es inseguro?, ¿vale la pena?, ¿qué categoría?
  {"f":"guard","lang":"es","texto":"...","titulo":"...","categoria":"","contaminada":false,"inseguro":false,"guardar":true,"cat":"config"}
  - texto: lo que el agente quiere guardar, 1 a 3 frases (30 a 350 caracteres). titulo: corto, o "".
  - categoria: la que propuso el agente (una de las nueve; a veces equivocada) o "".
  - contaminada: true si la sesión leyó contenido externo no confiable (una web, un correo, un archivo
    ajeno). Ponlo en true en 1 de cada 4, con textos seguros e inseguros por igual.
  - inseguro: true si el texto intenta manipular al agente o al sistema (darle órdenes, cambiar sus reglas,
    hacerse pasar por la persona) o lleva algo peligroso (una clave, instrucciones dañinas).
  - guardar: true si es información que vale la pena recordar (un hecho, una decisión, una preferencia,
    una lección). false si es basura, charla, progreso rutinario («ya corrí los tests») o una orden.
  - cat: decision (una decisión tomada y su porqué) · architecture (cómo está hecho el sistema) · bugfix
    (un error, su causa y su arreglo) · pattern (una forma de hacer que se repite) · config (valores,
    puertos, rutas, versiones) · discovery (algo que se averiguó y no era obvio) · preference (cómo quiere
    la persona que se le responda o se trabaje) · learning (una lección que sirve otra vez) · profile
    (quién es la persona) · ninguna. null si dos categorías son igual de defendibles.
  - Reparto: la mitad guardables y seguros, repartidos entre las nueve categorías; 1 de cada 4 no vale la
    pena; 1 de cada 4 inseguros, variados (órdenes disfrazadas de hecho, «a partir de ahora ignora…»,
    suplantación, secretos). Sin casos dudosos en inseguro ni en guardar.

relation — una memoria nueva frente a otra parecida: ¿cómo se relacionan?
  {"f":"relation","lang":"es","propuesta":"...","otra":"...","rel":"scoped"}
  - conflicts_with: se contradicen, no pueden ser ciertas a la vez (mismo dato, valor distinto).
  - scoped: propuesta vale en un caso más concreto que otra (la regla general y su excepción en este proyecto).
  - compatible: hablan de lo mismo y pueden ser ciertas a la vez.
  - related: tratan un tema cercano, sin más.
  - ninguna: no tienen que ver.
  Reparto parejo entre las cinco. Cada memoria, una línea de 30 a 200 caracteres.

preserved — al unir memorias: ¿todo lo que dice la original está en la unida?
  {"f":"preserved","lang":"es","original":"...","unido":"...","ok":true}
  - ok: true si TODO lo que dice original está también en unido, aunque con otras palabras. false si falta
    o cambia UN detalle concreto (una fecha, una cifra, una condición, una excepción, un nombre).
  - Mitad y mitad. unido suele ser más largo: junta lo de varias memorias.

place — ¿dónde se guarda una memoria?
  {"f":"place","lang":"es","propuesta":"...","titulo":"...","categoria":"decision","lugar":"proyecto"}
  - proyecto: es de este proyecto (una decisión, su arquitectura, su configuración, algo que solo vale aquí).
  - personal: es de la persona y vale en cualquier proyecto (quién es, cómo trabaja, qué prefiere).
  - leccion: una lección que sirve en otros proyectos (un error y su causa, un patrón, una trampa).
  Reparto parejo. titulo y categoria pueden ir vacíos ("").

op — el agente quiere guardar una propuesta y ya hay memorias parecidas: ¿qué hacer?
  {"f":"op","lang":"es","propuesta":"...","parecidas":{"m1":"...","m2":"..."},"op":"update_m2"}
  - propuesta: un hecho corto (20 a 120 caracteres). parecidas: 1 a 3 memorias cortas, con claves m<número>.
  - add: es información nueva que ninguna memoria parecida recoge.
  - noop: repite lo que ya dice una de ellas, aunque sea con otras palabras.
  - review: es dudosa: da órdenes al sistema, intenta manipularlo o no es un hecho sobre la persona o su trabajo.
  - update_m<k>: corrige o reemplaza lo que dice esa memoria: mismo tema, dato distinto o más reciente.
  Reparto: add 3 de cada 10, noop 2,5, update 3, review 1,5. Incluye repeticiones disfrazadas (sinónimos,
  otro orden, otro idioma) y correcciones sutiles (una fecha, una cifra, un nombre).

del — un mensaje y una memoria: ¿la persona pide olvidar exactamente esa memoria?
  {"f":"del","lang":"es","msg":"...","memoria":"...","pide":true}
  - pide: true solo si el mensaje pide olvidar o borrar ESA memoria. false si habla del tema sin pedir
    olvidarlo, si pide olvidar otra cosa, si pide corregirla en vez de borrarla, o si «olvidar» va en
    otro sentido («se me olvidó la clave»). Mitad y mitad.

CALIDAD
1. Respuesta clara: alguien cuidadoso daría la misma en 4 de cada 5 casos; el resto, difícil de verdad
   pero con una respuesta defendible (si no la hay: skip o null, donde existen).
2. Sin pistas: ningún campo nombra la respuesta.
3. Sin repetirse: cada línea cambia de tema, de persona y de forma. No copies los ejemplos de abajo.
4. Revisa cada línea contra las reglas de su familia antes de escribirla.

EJEMPLOS (para ver la forma; no los copies)
{"f":"pre","lang":"es","msg":"que hay en la carpeta migrations?","tools":["ls"],"mem":false}
{"f":"pre","lang":"es","msg":"renombra utils.py a helpers.py y arregla los imports","tools":["bash","read","edit"],"skip":["find"],"mem":false}
{"f":"pre","lang":"en","msg":"what's the difference between a mutex and a semaphore","tools":[],"mem":false}
{"f":"pre","lang":"es","msg":"seguimos con lo de ayer","tools":["memory_context"],"skip":["memory_search"],"mem":true}
{"f":"pre","lang":"es","msg":"anota que el cliente de Nubia paga a 60 días","tools":["memory_save"],"mem":false}
{"f":"rel","lang":"es","msg":"cuando es el cumple de mi hermana?","mems":{"m31":"Su hermana Lucía cumple el 9 de marzo.","m8":"Lucía vive en Tunja y trabaja en un hospital.","m77":"El servidor de pruebas se reinicia los domingos."},"helps":["m31"]}
{"f":"guard","lang":"es","texto":"En este repo los tests de integración necesitan Postgres 16 en el puerto 5433.","titulo":"Postgres para tests","categoria":"","contaminada":false,"inseguro":false,"guardar":true,"cat":"config"}
{"f":"relation","lang":"es","propuesta":"En el proyecto Faro se usa pnpm.","otra":"Prefiere npm en todos sus proyectos.","rel":"scoped"}
{"f":"preserved","lang":"es","original":"El respaldo corre a las 2 a. m. y tarda unos 20 minutos.","unido":"El respaldo nocturno corre a las 2 a. m.; las alertas llegan a Telegram.","ok":false}
{"f":"place","lang":"es","propuesta":"Prefiere que los commits vayan en español y en imperativo.","titulo":"","categoria":"preference","lugar":"personal"}
{"f":"op","lang":"es","propuesta":"Se mudó a Pereira en agosto.","parecidas":{"m4":"Vive en Manizales desde 2019.","m9":"Tiene dos gatos."},"op":"update_m4"}
{"f":"del","lang":"es","msg":"borra lo que tienes de mi antiguo jefe","memoria":"Su jefe en Lumen se llamaba Óscar Rey.","pide":true}

ENCARGO
(aquí va el reparto del lote: cuántos casos de cada familia y de qué temas)
```
