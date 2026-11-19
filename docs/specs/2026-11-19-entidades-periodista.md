# Temis periodista: entidades, su contexto y su rol (propuesta v0)

Fecha: 2026-11-19. Estado: estructura aprobada; todavía no hay código ni datos.

**Decisiones (2026-11-19):**

- La fuente es **solo La Silla Vacía**: perfiles del «Quién es quién», etiquetas del WordPress y notas
  del archivo. Nada de Croma ni de otras fuentes.
- Los roles de abajo (en el documento y de la entidad) quedan como están.
- Es un **modelo aparte, especializado** («Temis periodista»), no una familia más del Temis de Clio.

## Qué se quiere

Hacer con Temis lo que hoy hace Jev en el verificador de La Silla (`extension-verificador-x`,
`lib/entidades.js`, `lib/perfiles.js`, `lib/etiquetas.js`), pero solo para **entidades** y con más
información sobre cada una:

1. **Identificar** las entidades de un documento (persona, organización, lugar, tema).
2. **Resolverlas** contra una fuente (perfiles del «Quién es quién», etiquetas del WordPress, notas del
   archivo, o cualquier otra fuente que se le pase).
3. **Contexto**: qué dice el documento de la entidad y qué pasajes de la fuente lo explican.
4. **Rol en el documento**: qué papel cumple en *esta* historia (protagonista, fuente, señalado…).
5. **Rol de la entidad**: qué es en el mundo según el documento (funcionaria, congresista, contratista…).

## Lo que no cambia: Jev y Temis no escriben texto

Temis, como Jev, solo responde `noul`, `choice` y `score`. No hay `span` (está fuera de alcance en el
diseño de Temis). Por eso el reparto es el mismo del verificador:

| Paso | Lo hace el código | Lo decide Temis |
|---|---|---|
| Encontrar candidatos | nombres propios, siglas, alias, diccionario de etiquetas, cargos («el ministro») | — |
| ¿Es una entidad? ¿De qué tipo? | — | `noul` + `choice` |
| ¿Cuál de la fuente es? | buscar perfiles/etiquetas/notas parecidos | `choice` con «ninguno» |
| ¿Esta mención es la misma entidad? | juntar apellidos, cargos y siglas del texto | `noul` |
| Contexto | partir el documento en oraciones; traer pasajes de la fuente | `choice` de oración + `noul` por pasaje |
| Roles | — | `choice` + `score` |

El «contexto» que devuelve el sistema no es un resumen redactado: son **referencias** (las oraciones del
documento y los pasajes de la fuente que Temis elige). Si después se quiere una frase, la arma una
plantilla con el rol y el cargo, o un LLM fuera de este lazo.

## Una petición por entidad (no por documento)

El verificador manda a Jev todas las entidades juntas (`candidatos[k]`, `etiquetas[k]`). Para Temis eso no
sirve: el codificador lee el estado entero **en cada pregunta** y entrena con 640 tokens, así que un
documento largo se trunca y las preguntas de la entidad 7 no ven sus oraciones.

Propuesta: **un caso = una entidad**, con un estado corto que lleve solo lo que esa entidad necesita
(cabe en ~450 tokens). Las peticiones de un documento se mandan en paralelo; con 25 ms por pregunta en
CPU, 10 entidades × 6 preguntas son ~1,5 s sin GPU.

Para seguir usando a Jev como teacher en este formato basta con mandarle las mismas peticiones por
entidad (es más caro en llamadas que el formato agrupado, pero entrenamos exactamente con lo que Temis va
a ver).

## El estado

```jsonc
{
  "tarea": "Una entidad que nombra un documento periodístico, lo que el documento dice de ella y pasajes de una fuente que podrían ser sobre ella.",
  "documento": {
    "tipo": "nota",                       // nota | pagina | post | comunicado | pdf | transcripcion
    "titulo": "El Gobierno radica otra vez la reforma laboral",
    "fecha": "2026-10-03",
    "medio": "La Silla Vacía",            // opcional
    "bajada": "…"                         // opcional, 1–2 frases
  },
  "entidad": {
    "mencion": "Sanguino",                // como aparece la primera vez
    "nombre": "Antonio Sanguino",          // la forma más completa que hay en el texto
    "otras_menciones": ["el ministro del Trabajo", "Sanguino"]
  },
  "oraciones": {                          // SOLO las oraciones donde aparece (máx. 6), con id estable
    "o0": "El ministro del Trabajo, Antonio Sanguino, radicó el proyecto en la Cámara.",
    "o3": "Sanguino dijo que esta vez el Gobierno sí tiene los votos.",
    "o7": "La oposición acusó a Sanguino de presionar a los congresistas con puestos."
  },
  "candidatos": {                         // de la fuente: perfiles o etiquetas con nombre parecido (0–8)
    "perfil_antonio_sanguino": "Antonio Sanguino Páez. Ministro del Trabajo desde 2025; antes senador de la Alianza Verde.",
    "perfil_antonio_sanguino_2": "Antonio Sanguino. Concejal de Cúcuta (2020–2023)."
  },
  "fuente": {                             // pasajes del archivo / RAG (0–5), con id estable
    "f0": {"titulo": "Sanguino, el verde que llegó al Ministerio del Trabajo", "fecha": "2025-03-04", "texto": "…trozo donde aparece…"},
    "f1": {"titulo": "Así quedó la votación de la reforma laboral", "fecha": "2025-06-17", "texto": "…"}
  }
}
```

Reglas del estado (las mismas que ya pide `docs/prompt_generar_casos.md`):

- Ningún campo nombra la respuesta: nada de `"rol": "protagonista"` ni `"tipo": "persona"`.
- Las claves de `candidatos` y `fuente` son ids neutros (no `perfil_correcto`).
- `candidatos` y `fuente` pueden venir vacíos; las preguntas que dependen de ellos se quitan.
- Si se usa el campo `context` de la API de Temis (RAG), los pasajes de la fuente pueden ir ahí en vez de
  en `fuente`; para entrenar conviene dejarlos en el estado, porque así las preguntas los nombran por id.

## Las preguntas

Cada `qid` es una familia nueva (prefijo `ent_`). Los `noul` van como **afirmaciones** (el filtro de
`temis/data/quality.py` descarta «¿Qué…?», «Determina si…»). Instrucciones de máx. ~160 tokens y cada
opción de máx. ~80 (límites de `temis/encoding.py`); `choice` hasta 12 opciones.

### 1. Identificar

```jsonc
"ent_es_entidad": {
  "type": "noul",
  "instructions": "`entidad.mencion` nombra a una persona, organización, lugar o tema concreto, no un cargo genérico, una palabra con mayúscula por empezar la frase ni un título de obra.",
  "criteria": {"false": "Es un cargo genérico, una palabra común o un error de detección.", "true": "Nombra algo concreto que un lector podría buscar."}
},
"ent_tipo": {
  "type": "choice",
  "instructions": "Qué es `entidad` en las `oraciones` del documento.",
  "criteria": {
    "persona": "Una persona.",
    "organizacion": "Institución, entidad pública, empresa, partido, grupo armado, organización social o medio.",
    "lugar": "País, departamento, municipio, región, río, sitio.",
    "tema": "Un hecho, proceso, política pública, ley, evento o periodo.",
    "otro": "Una obra, producto, cuenta de red social u otra cosa."
  }
}
```

### 2. Resolver contra la fuente

```jsonc
"ent_candidato": {
  "type": "choice",
  "instructions": "Cuál de los `candidatos` es la misma entidad que nombra el documento.",
  "criteria": {
    "perfil_antonio_sanguino": "Antonio Sanguino Páez. Ministro del Trabajo desde 2025…",
    "perfil_antonio_sanguino_2": "Antonio Sanguino. Concejal de Cúcuta (2020–2023).",
    "ninguno": "Ninguno: es otra entidad, o el documento no da para saber cuál es."
  }
},
"ent_misma_mencion": {               // una por cada mención dudosa (cargo, apellido, sigla)
  "type": "noul",
  "instructions": "En el documento, «el ministro del Trabajo» se refiere a la misma entidad que `entidad.nombre`."
}
```

### 3. Contexto

```jsonc
"ent_oracion_clave": {
  "type": "choice",
  "instructions": "Cuál de las `oraciones` dice mejor qué hace o qué le pasa a `entidad` en este documento.",
  "criteria": {"o0": "<texto de o0, cortado>", "o3": "…", "o7": "…", "ninguna": "Ninguna dice nada de la entidad; solo la nombran."}
},
"ent_fuente_f0": {                    // una por pasaje
  "type": "noul",
  "instructions": "El pasaje `fuente.f0` es sobre la misma `entidad` y ayuda a entender el papel que tiene en este documento (no solo la menciona de pasada ni es otra con nombre parecido)."
},
"ent_fuente_vigente_f0": {            // opcional: el pasaje sigue siendo cierto a la fecha del documento
  "type": "noul",
  "instructions": "Lo que `fuente.f0` dice del cargo o la situación de `entidad` sigue vigente en `documento.fecha`."
}
```

### 4. Rol en el documento

```jsonc
"ent_rol_documento": {
  "type": "choice",
  "instructions": "Qué papel cumple `entidad` en la historia que cuenta el documento.",
  "criteria": {
    "protagonista": "La historia es sobre lo que hace, decide o le pasa.",
    "fuente": "Aparece sobre todo porque declara, opina o entrega información.",
    "senalado": "Se le acusa, investiga, critica o cuestiona.",
    "afectado": "Sufre las consecuencias de lo que se cuenta (víctima, beneficiario, perjudicado).",
    "autoridad": "Decide, juzga, regula o investiga el asunto sin ser su protagonista.",
    "aliado_o_rival": "Aparece como apoyo u oposición de quien protagoniza la historia.",
    "escenario": "Es el lugar, marco o tema donde pasa la historia.",
    "mencion": "Solo se menciona de pasada; quitarla no cambiaría la historia."
  }
},
"ent_centralidad": {
  "type": "score",
  "instructions": "Cuánto depende la historia del documento de `entidad`.",
  "criteria": [
    "Nada: se menciona de pasada.",
    "Poco: aporta un detalle o una cita.",
    "Bastante: es una de las partes principales.",
    "Total: la historia es sobre ella."
  ]
},
"ent_postura": {                      // solo para personas y organizaciones
  "type": "choice",
  "instructions": "Qué posición tiene `entidad` frente al hecho central del documento, según lo que el documento le atribuye.",
  "criteria": {"a_favor": "Lo impulsa, defiende o apoya.", "en_contra": "Se opone, lo critica o lo denuncia.", "neutral": "Informa o decide sin tomar partido.", "no_se_sabe": "El documento no le atribuye ninguna posición."}
}
```

`senalado` y `afectado` son delicados: conviene que el código no los muestre con menos de ~0,7 de
probabilidad, y que la cola de revisión les dé prioridad.

### 5. Rol de la entidad (qué es, según el documento)

Las opciones dependen del tipo, así que se pregunta después de `ent_tipo` (en el caso de entrenamiento
se escribe ya con el tipo correcto).

```jsonc
"ent_rol_persona": {
  "type": "choice",
  "instructions": "Qué es `entidad` según lo que dice el documento en `documento.fecha`.",
  "criteria": {
    "funcionario": "Funcionario o funcionaria del Ejecutivo (ministerio, alcaldía, gobernación, entidad pública).",
    "electo": "Congresista, concejal, diputado u otro cargo de elección popular en ejercicio.",
    "candidato": "Aspirante o precandidato a un cargo.",
    "justicia_control": "Juez, fiscal, magistrado, procurador, contralor, defensor.",
    "fuerza_publica": "Militar o policía.",
    "armado_ilegal": "Integrante o jefe de un grupo armado ilegal o red criminal.",
    "empresario": "Empresario, contratista, gremio o directivo de empresa.",
    "sociedad_civil": "Líder social, activista, sindicalista, ONG, comunidad.",
    "experto": "Académico, analista, consultor.",
    "periodista": "Periodista, columnista o medio.",
    "ciudadano": "Persona sin cargo público que aparece por lo que le pasó o hizo.",
    "otro": "Ninguna de las anteriores o no se puede saber."
  }
},
"ent_rol_organizacion": {
  "type": "choice",
  "instructions": "Qué clase de organización es `entidad` según el documento.",
  "criteria": {
    "ejecutivo": "Ministerio, departamento administrativo, alcaldía, gobernación o entidad del Ejecutivo.",
    "legislativo": "Congreso, una de sus cámaras o comisiones, asamblea, concejo.",
    "justicia_control": "Corte, tribunal, Fiscalía, Procuraduría, Contraloría, Registraduría.",
    "partido": "Partido, movimiento o coalición política.",
    "empresa": "Empresa privada o mixta, gremio.",
    "armado_ilegal": "Grupo armado ilegal o red criminal.",
    "sociedad_civil": "ONG, sindicato, organización social, comunidad, iglesia.",
    "medio": "Medio de comunicación.",
    "internacional": "Gobierno extranjero u organismo internacional.",
    "otro": "Otra clase o no se puede saber."
  }
},
"ent_cargo_explicito": {
  "type": "noul",
  "instructions": "El documento dice de forma explícita el cargo o la ocupación de `entidad` (no hay que deducirlo de la fuente)."
}
```

### Qué preguntas lleva cada caso

| Tipo de la entidad | Preguntas |
|---|---|
| todas | `ent_es_entidad`, `ent_tipo`, `ent_rol_documento`, `ent_centralidad`, `ent_oracion_clave` |
| con candidatos | + `ent_candidato` |
| con pasajes de la fuente | + `ent_fuente_fN` (y opcional `ent_fuente_vigente_fN`) |
| con menciones dudosas | + `ent_misma_mencion` (una por mención) |
| persona | + `ent_rol_persona`, `ent_postura`, `ent_cargo_explicito` |
| organización | + `ent_rol_organizacion`, `ent_postura` |

Entre 5 y 10 preguntas por caso, como los casos de Clio.

## La respuesta (la misma de Jev / TypeSafe)

```jsonc
{
  "model": "temis-2.x",
  "answers": {
    "ent_tipo":          {"type": "choice", "choice": "persona", "confidence": 0.97, "probabilities": {"persona": 0.97, "organizacion": 0.01, …}},
    "ent_candidato":     {"type": "choice", "choice": "perfil_antonio_sanguino", "confidence": 0.91, …},
    "ent_rol_documento": {"type": "choice", "choice": "protagonista", "confidence": 0.74, …},
    "ent_centralidad":   {"type": "score", "score": 2.6, "confidence": 0.62, …},
    "ent_fuente_f0":     {"type": "noul", "noul": 0.88, …}
  }
}
```

Y lo que arma el código con eso (salida de la herramienta, no de Temis):

```jsonc
{"entidad": "Antonio Sanguino", "tipo": "persona", "perfil": "perfil_antonio_sanguino",
 "rol_en_documento": "protagonista", "centralidad": 2.6, "rol": "funcionario", "postura": "a_favor",
 "contexto": {"oracion": "o0", "fuente": ["f0"]}}
```

## Temis periodista: un modelo aparte

Mismo código, mismo formato y misma API que Temis; lo que cambia es con qué se entrena y cómo se mide.

- **Arranque**: desde los pesos de Temis 2.4.0 (`scripts/train.py --init v1112-mrbert-cortos`), que ya
  sabe leer peticiones tipadas, y no desde MrBERT-es en blanco. Si en la medición propia sale igual
  desde cero, se prefiere desde cero (menos dependencias).
- **Datos**: solo familias `ent_*`, más un repaso de ~20 % de typed-decisions `train` para que no olvide
  el formato general (preguntas nuevas que la redacción invente después).
- **Versiones**: registro propio (`runs/registry_periodista.json`) y nombre `temis-periodista-X.Y.Z`,
  para no mezclarse con la línea de Clio. El benchmark de Clio ya no decide nada aquí.
- **Medición propia, nunca entrenada**: notas reales anotadas a mano por la redacción
  (`data/entidades_heldout.jsonl`), separadas **por nota**, no por entidad. Es lo único que decide qué
  versión queda activa. Se reportan acierto por pregunta y, aparte, `senalado` y `afectado`.
- **Teacher**: Jev, con la calibración por temperatura que ya existe. Las respuestas de la redacción en
  la cola de `/m` pesan más, como hoy las de Temis.
- Casos con el formato de `data/synth/cases.jsonl` (`family`, `workflow` en snake_case que no revele la
  respuesta, `language: "es"`, `state`, `questions`).

## La base de entrenamiento (siguiente paso)

Todo sale de La Silla. Cuatro capas, de la más real a la más sintética:

1. **Notas reales** (la base). Unas 2.000 notas del archivo, repartidas por sección y año. Por cada una:
   - candidatos por **el mismo código del verificador** (`lib/entidades.js preparar`, que ya busca
     perfiles del «Quién es quién» y etiquetas del WordPress), corrido desde Node, para que Temis vea en
     entrenamiento los mismos candidatos que verá en producción;
   - un estado por entidad (oraciones donde aparece, candidatos, 3–5 pasajes de otras notas del
     archivo);
   - Jev responde las preguntas `ent_*`.
   Se aparta antes el ~10 % de las notas para la medición propia.
2. **Trazas del verificador**: el modo depuración guarda peticiones y respuestas reales de Jev sobre
   entidades (`perfil.*`, `es.*`, `tipo_etiqueta.*`, `trata_persona.*`). Se pasan al formato por
   entidad: dan casos de desambiguación y de pasajes que ya pasaron por producción.
3. **Variaciones con Qwen local** sobre notas de la capa 1, para lo que las notas reales traen poco:
   - la misma persona como `fuente` en una nota y como `senalado` en otra;
   - homónimos (un ministro y un concejal con el mismo nombre) donde la fecha o el cargo los separa;
   - un pasaje que la menciona de pasada frente a uno que es sobre ella;
   - un cargo vencido («exministro») frente a uno vigente.
   Qwen reescribe las oraciones; no responde. Jev etiqueta como en la capa 1.
4. **Casos difíciles escritos por otro modelo** con un prompt propio (como
   `docs/prompt_casos_dificiles.md`): correferencias largas, siglas, entidades que cambian de rol
   dentro de la misma nota.

**Equilibrio**: en notas reales dominarán `protagonista`, `mencion`, `funcionario` y `electo`. Se
recorta hasta que ninguna opción pase del ~35 % de su pregunta y las capas 3 y 4 se usan para llenar
`senalado`, `afectado`, `aliado_o_rival` y `ninguno`.

**Antes de escalar**: un piloto de ~200 notas para medir cuántas entidades salen por nota, cuánto cuesta
Jev y si la estructura del estado se queda bajo ~450 tokens.

### Base de ejemplo (2026-11-19)

Hecha la capa 1 con 300 notas de entrenamiento y 20 de benchmark (2.512 + 111 casos), con Jev como
referencia. Todavía no se entrena: el branch `data/entidades-periodista` deja datos, scripts y la medición de
Temis 2.4.0 para que entrene quien tenga el compute. Detalle, cifras y comandos:
[data/periodista/README.md](../../data/periodista/README.md). Diferencias con lo de arriba:

- Los estados quedaron más largos de lo previsto (mediana 2.259 caracteres): caben en `--max-len 768` solo
  con pocas preguntas; los largos los relee `--long-max-len`.
- Los ids de candidatos son neutros (`c1`, `c2`…) en vez de `perfil_…`, y cada candidato dice si es un perfil
  del «Quién es quién» o una etiqueta.
- Las preguntas de rol por tipo (`ent_rol_persona`, `ent_rol_organizacion`) dependen de una pista del
  código; donde no hay pista, el caso no las lleva.
