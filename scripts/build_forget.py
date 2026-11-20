"""Training cases for Clio's `pide_olvidar` question, in the shape Clio sends: the person's message in the
first person and one stored memory in the third person ("m5: Ana trabaja…").

  python -m scripts.build_forget            # data/forget_targets.jsonl and data/forget_blind.jsonl

Yes: the message asks to delete that memory. No: it asks about it, asks to keep it, corrects it, asks to
delete another memory, or says "forget it" about the conversation. The blind set uses facts and phrasings
the training set never sees. None of the seven topics of Clio's own test (external/clio/testing/decisiones)
is used here, so that test stays a measure of new topics.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from scripts.build_review import NO, YES

QID = "pide_olvidar"
QUESTION = "En `mensaje_de_la_persona`, la persona pide olvidar o borrar exactamente la memoria `memoria`."
OUT, BLIND = Path("data/forget_targets.jsonl"), Path("data/forget_blind.jsonl")
NAMES = ["Ana", "Julián", "Camila", "Iván", "Sofía", "Andrés", "Valentina", "Mateo", "La persona", "El usuario"]

# (memory with {n} = who, first-person clause or None, topic, correction)
FACTS = [
    ("{n} trabaja como enfermera en una clínica.", "trabajo como enfermera", "mi trabajo", "ahora trabajo como docente"),
    ("{n} es alérgica a los mariscos.", "soy alérgica a los mariscos", "mi alergia", "en realidad es a las fresas"),
    ("{n} tiene dos hijos, Samuel y Lucía.", "tengo dos hijos", "mis hijos", "son tres, también está Emilio"),
    ("{n} entrena natación los martes y jueves.", "entreno natación los martes y jueves", "mis entrenamientos", "ahora entreno los lunes"),
    ("{n} es vegetariana desde 2021.", "soy vegetariana", "mi dieta", "ahora como pescado"),
    ("{n} estudió ingeniería civil en Manizales.", "estudié ingeniería civil", "mis estudios", "fue ingeniería industrial"),
    ("El número de teléfono de {n} termina en 4471.", "mi teléfono termina en 4471", "mi número de teléfono", "ahora termina en 9030"),
    ("{n} prefiere las reuniones después de las 10 de la mañana.", "prefiero las reuniones después de las 10", "mi horario de reuniones", "ahora las prefiero antes de las 9"),
    ("{n} toma un medicamento para la tiroides cada mañana.", "tomo un medicamento para la tiroides", "mi medicamento", "me lo cambiaron por otro"),
    ("La pareja de {n} se llama Andrea.", "mi pareja se llama Andrea", "mi pareja", "se llama Adriana, no Andrea"),
    ("{n} va a viajar a Lima en diciembre.", "voy a viajar a Lima en diciembre", "mi viaje a Lima", "el viaje es en enero"),
    ("{n} le debe 200 mil pesos a su hermano.", "le debo 200 mil pesos a mi hermano", "la deuda con mi hermano", "ya son 150 mil"),
    ("{n} usa talla 38 de zapatos.", "uso talla 38 de zapatos", "mi talla de zapatos", "es talla 39"),
    ("{n} quiere aprender portugués este año.", "quiero aprender portugués", "mi meta de aprender portugués", "mejor italiano"),
    ("{n} tiene cita con el odontólogo el día 9.", "tengo cita con el odontólogo el día 9", "mi cita con el odontólogo", "la pasaron para el 16"),
    ("{n} conduce un Renault Logan gris.", "conduzco un Renault Logan gris", "mi carro", "lo vendí, ahora tengo un Mazda 2"),
    ("{n} prefiere respuestas cortas y sin tablas.", "prefiero respuestas cortas", "mi preferencia de respuestas cortas", "ahora las quiero detalladas"),
    ("{n} toca el bajo en una banda los sábados.", "toco el bajo en una banda", "mi banda", "ahora toco la batería"),
    ("{n} paga el arriendo el día 5 de cada mes.", "pago el arriendo el día 5", "el pago de mi arriendo", "ahora lo pago el 15"),
    ("{n} es hincha del Once Caldas.", "soy hincha del Once Caldas", "mi equipo de fútbol", "soy del Junior"),
    ("{n} tiene miedo a volar.", "tengo miedo a volar", "mi miedo a volar", "ya lo superé"),
    ("El jefe de {n} se llama Ricardo Peña.", "mi jefe se llama Ricardo Peña", "mi jefe", "ahora es Laura Gómez"),
    ("{n} lee novela negra antes de dormir.", "leo novela negra antes de dormir", "mis lecturas", "ahora leo ensayo"),
    ("El proyecto Tucán despliega con GitHub Actions.", None, "el despliegue de Tucán", "ahora despliega con Jenkins"),
    ("La base de datos de Colibrí es MariaDB 11.", None, "la base de datos de Colibrí", "ahora es PostgreSQL 17"),
    ("El dominio del blog es notas.acme.dev.", None, "el dominio del blog", "ahora es blog.acme.dev"),
    ("La rama principal del repositorio de Faro se llama trunk.", None, "la rama principal de Faro", "ahora se llama main"),
    ("Las copias de seguridad corren a las 3 de la mañana.", None, "las copias de seguridad", "ahora corren a la medianoche"),
    ("El equipo usa Linear para las tareas.", None, "la herramienta de tareas", "ahora usamos Jira"),
    ("La API de pagos tiene un tiempo de espera de 30 segundos.", None, "el tiempo de espera de la API de pagos", "ahora es de 45 segundos"),
    ("El cliente Andina paga a 60 días.", None, "el plazo de pago de Andina", "ahora paga a 30 días"),
    ("La reunión semanal del equipo es los lunes a las 9.", None, "la reunión semanal", "se movió a los martes"),
    ("El puerto del servicio de colas es el 5672.", None, "el puerto del servicio de colas", "ahora es el 5673"),
    ("Los tests de integración de Faro tardan 12 minutos.", None, "los tests de integración de Faro", "ahora tardan 6"),
    ("El contrato con el proveedor de hosting vence en marzo.", None, "el contrato de hosting", "se renovó hasta septiembre"),
    ("La versión de Python del proyecto Tucán es la 3.12.", None, "la versión de Python de Tucán", "ya subimos a la 3.13"),
    ("El boletín sale los jueves a las 7 de la mañana.", None, "el horario del boletín", "ahora sale los viernes"),
]
BLIND_FACTS = [
    ("{n} canta en el coro de la universidad.", "canto en el coro de la universidad", "el coro", "lo dejé, ahora hago teatro"),
    ("{n} tiene una hermana que vive en Quito.", "tengo una hermana en Quito", "mi hermana", "se fue a Guayaquil"),
    ("{n} cobra por hora en sus proyectos de diseño.", "cobro por hora", "mi forma de cobrar", "ahora cobro por proyecto"),
    ("El proyecto Garza guarda los archivos en un bucket llamado garza-media.", None, "el bucket de Garza", "ahora es garza-assets"),
    ("La contadora de la empresa es Marcela Ortiz.", None, "la contadora", "ahora es Diana Ruiz"),
    ("El informe mensual se entrega el último viernes.", None, "la entrega del informe mensual", "pasó al primer lunes"),
]

# {t} = topic, {y} = first-person clause, {c} = correction
ASK = {
    "yes": ["Olvida {t}.", "Borra {t} de tu memoria.", "Elimina lo que guardaste sobre {t}.", "Quita lo de {t}.",
            "Ya no necesito que recuerdes {t}, elimínalo.", "Deja de recordar {t}.", "No quiero que sigas guardando {t}.",
            "Saca {t} de lo que recuerdas.", "Lo de {t} ya no vale: elimínalo.", "Bórrame lo que sabes de {t}.",
            "Olvida eso de que {y}.", "Borra que {y}.", "Deja de recordar que {y}.", "Ya no guardes que {y}.",
            "Elimina de tu memoria que {y}.", "Quita que {y}, ya no es así."],
    "ask": ["¿Qué recuerdas de {t}?", "Dime lo que tienes guardado sobre {t}.", "Repíteme {t}.", "¿Tienes anotado {t}?"],
    "keep": ["No vayas a olvidar {t}.", "Recuerda bien {t}.", "Guarda {t}, es importante.", "Ten presente que {y}.",
             "Nunca olvides que {y}.", "Que no se te pase que {y}."],
    "change": ["Actualiza {t}: {c}.", "Corrige {t}, {c}.", "Modifica lo de {t}: {c}.", "Sobre {t}: {c}. Cámbialo."],
    "talk": ["Olvídalo, no importa.", "Olvida lo que dije, sigamos con otra cosa.", "Olvida esa idea, probemos otra.",
             "Olvida la pregunta anterior.", "Olvidemos el asunto por hoy.", "Deja eso, luego lo vemos.",
             "Olvida lo que te pedí hace un momento sobre el informe.", "Olvida lo que dije recién de la canción."],
}
BLIND_ASK = {
    "yes": ["Puedes borrar {t}; ya no hace falta.", "Descarta {t}, no lo guardes.", "Suprime el recuerdo de {t}.",
            "No conserves {t}.", "Haz de cuenta que nunca te conté {t}: sácalo.", "Descarta eso de que {y}."],
    "ask": ["¿Me recuerdas {t}?"],
    "keep": ["Anota bien {t}, que no se pierda.", "Conserva que {y}."],
    "change": ["Hay que ajustar {t}: {c}."],
    "talk": ["Olvida mi último mensaje, me equivoqué de chat."],
}
NO_KINDS = ["other"] * 3 + ["ask"] * 2 + ["keep"] * 2 + ["change"] * 2 + ["talk"]


def _say(template: str, fact: tuple) -> str | None:
    if "{y}" in template and fact[1] is None:
        return None
    text = template.format(t=fact[2], y=fact[1], c=fact[3])
    return text.replace(" de el ", " del ").replace(" de El ", " del ")


def one(r: random.Random, facts: list, ask: dict, plain: bool = False) -> tuple[str, str, bool] | None:
    """(message, memory, yes). plain: no casing or courtesy noise (the blind set)."""
    fact, yes = r.choice(facts), r.random() < 0.5
    kind = "yes" if yes else r.choice(NO_KINDS)
    if kind == "other":  # a real deletion request, about another memory
        about = r.choice([f for f in facts if f is not fact])
        msg = _say(r.choice(ask["yes"]), about)
    else:
        msg = _say(r.choice(ask[kind]), fact)
    if msg is None:
        return None
    if not plain:
        if r.random() < 0.2:
            msg = r.choice(["Por favor, ", "Oye, ", "Mira, "]) + msg[0].lower() + msg[1:]
        if r.random() < 0.3:
            msg = msg.lower().rstrip(".")
    mem = fact[0].format(n=r.choice(NAMES))
    if r.random() < 0.7:
        mem = f"m{r.randint(1, 40)}: {mem}"
    return msg, mem, yes


def build(n: int, seed: int = 0) -> list[dict]:
    r = random.Random(seed)
    rows, seen, count = [], set(), {True: 0, False: 0}
    while len(rows) < n:
        got = one(r, FACTS, ASK)
        if got is None or got[:2] in seen or count[got[2]] >= (n + 1) // 2:
            continue
        msg, mem, yes = got
        seen.add((msg, mem))
        count[yes] += 1
        cid = f"forget-{seed}-{len(rows):05d}"
        rows.append({"id": cid, "qid": QID, "kind": "noul", "source": "forget", "schema_id": cid,
                     "target": YES if yes else NO,
                     "case": {"id": cid, "state": {"mensaje_de_la_persona": msg, "memoria": mem},
                              "questions": {QID: {"type": "noul", "instructions": QUESTION}}}})
    return rows


def blind(n: int = 80, seed: int = 1) -> list[dict]:
    """Cases for scripts.eval_cliolike: facts and phrasings that build() never uses."""
    r = random.Random(seed)
    rows, seen = [], set()
    while len(rows) < n:
        got = one(r, BLIND_FACTS, BLIND_ASK, plain=True)
        if got is None or got[:2] in seen:
            continue
        seen.add(got[:2])
        rows.append({"id": f"forget-blind-{len(rows):02d}", "family": "asks_to_forget",
                     "state": {"mensaje_de_la_persona": got[0], "memoria": got[1]},
                     "questions": {QID: {"type": "noul", "instructions": QUESTION}},
                     "gold": {QID: "true" if got[2] else "false"}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for path, rows in ((OUT, build(a.n, a.seed)), (BLIND, blind())):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        print(f"[forget] wrote {len(rows)} cases to {path}")


if __name__ == "__main__":
    main()
