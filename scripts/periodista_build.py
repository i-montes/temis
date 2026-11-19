"""Temis periodista: arma los casos de entidades (un caso = una entidad de una nota) a partir de lo que bajó
scripts.periodista_fetch. Diseño: docs/specs/2026-11-19-entidades-periodista.md.

  python -m scripts.periodista_build --documentos 300 --bench 20

Escribe:
  data/periodista/cases.jsonl        casos para entrenar, SIN respuestas (las pone Jev: scripts.label --sources periodista)
  data/periodista/bench_cases.jsonl  casos del benchmark (notas apartadas: nunca se entrenan), sin respuestas

El código hace lo que no es decisión: partir en oraciones, encontrar candidatos (nombres propios, siglas, etiquetas
de La Silla, perfiles del «Quién es quién»), agrupar menciones y traer pasajes del archivo. Todo lo demás lo
pregunta el caso. Determinista con --seed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

RAW = Path("data/periodista/raw")
OUT = Path("data/periodista")

# --- límites del estado (caracteres): que un caso quepa en ~700 tokens con sus preguntas ---
MAX_ORACIONES, LARGO_ORACION = 4, 260
MAX_CANDIDATOS, LARGO_CANDIDATO = 3, 200
MAX_FUENTE, LARGO_FUENTE = 3, 280
LARGO_BAJADA = 220
MAX_ENTIDADES = 8          # por nota
MAX_ENTIDADES_BENCH = 5

TAREA = ("Una entidad que nombra una nota de La Silla Vacía, las oraciones donde aparece, perfiles y etiquetas "
         "de La Silla con un nombre parecido (`candidatos`) y pasajes de notas anteriores del archivo que la nombran "
         "(`fuente`).")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return " ".join("".join(c for c in s if unicodedata.category(c) != "Mn").replace("-", " ").split())


def cortar(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


# --- oraciones -----------------------------------------------------------------------------------------------
ABREV = r"(?<!\bSr)(?<!\bSra)(?<!\bDr)(?<!\bDra)(?<!\bEE\.UU)(?<!\bp)(?<!\bart)(?<!\bNo)"
CORTE = re.compile(ABREV + r"(?<=[.!?…»”\"])\s+(?=[«“\"¿¡(]?[A-ZÁÉÍÓÚÑ0-9])")


def oraciones(parrafos: list[str]) -> list[str]:
    sal = []
    for p in parrafos:
        sal += [o.strip() for o in CORTE.split(p) if len(o.strip()) >= 25]
    return sal


# --- candidatos ----------------------------------------------------------------------------------------------
MAY = r"[A-ZÁÉÍÓÚÑ][\wáéíóúñü'’.-]*"
CONECT = r"(?:de|del|la|las|los)"
PROPIOS = {"silla vacia", "la silla vacia", "silla", "la silla"}   # el medio hablando de sí mismo
SECUENCIA = re.compile(rf"(?<![\wáéíóúñ]){MAY}(?:\s+(?:{CONECT}\s+){{0,2}}{MAY})*")
SIGLA = re.compile(r"\b[A-ZÁÉÍÓÚÑ]{2,7}\b")
# Palabras que con mayúscula al empezar la frase no nombran nada.
VACIAS = set(norm(w) for w in """el la los las un una unos unas de del y e o u en a al por para con sin sobre entre
según aunque pero mientras cuando como donde desde hasta tras ante este esta estos estas ese esa eso esto aquí allí
ya también además luego después antes hoy ayer mañana sí no ni que qué quien quién cual cuál lo le les se su sus
mi tu nos más menos muy todo todos toda todas otro otra otros otras cada ahora entonces así incluso solo sólo
es son fue fueron será ha han había hay tiene tienen dijo dice explicó agregó aseguró señaló contó
foto video lea vea escuche enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre
lunes martes miércoles jueves viernes sábado domingo""".split())
ORG = re.compile(r"^(ministerio|partido|corte|fiscal[ií]a|consejo|congreso|senado|c[aá]mara|gobierno|polic[ií]a|"
                 r"ej[eé]rcito|armada|universidad|banco|fondo|agencia|comisi[oó]n|superintendencia|registradur[ií]a|"
                 r"procuradur[ií]a|contralor[ií]a|defensor[ií]a|alcald[ií]a|gobernaci[oó]n|secretar[ií]a|tribunal|"
                 r"departamento|instituto|unidad|movimiento|alianza|pacto|centro|cambio|liga|asociaci[oó]n|"
                 r"federaci[oó]n|sindicato|empresa|grupo|clan|frente|ecopetrol|dian|eln|farc|onu|oea)\b", re.I)
LUGAR = re.compile(r"^(bogot[aá]|medell[ií]n|cali|barranquilla|cartagena|c[uú]cuta|bucaramanga|antioquia|cauca|"
                   r"valle|nari[nñ]o|choc[oó]|santander|caribe|colombia|venezuela|estados unidos|ecuador|per[uú]|"
                   r"catatumbo|magdalena|atl[aá]ntico|bol[ií]var|c[oó]rdoba|sucre|cesar|huila|tolima|meta|"
                   r"cundinamarca|boyac[aá]|caldas|risaralda|quind[ií]o|putumayo|caquet[aá]|guaviare|arauca|"
                   r"casanare|la guajira|amazonas|vichada|guain[ií]a|vaup[eé]s|san andr[eé]s)\b", re.I)
CARGO = re.compile(r"\b(?:el|la)\s+(?:ex)?(?:ministr[oa]|presidente|presidenta|vicepresident[ea]|alcalde|alcaldesa|"
                   r"gobernador(?:a)?|senador(?:a)?|representante|fiscal general|fiscal|procurador(?:a)?|"
                   r"contralor(?:a)?|registrador(?:a)?|magistrad[oa]|director(?:a)?|superintendente|candidat[oa]|"
                   r"precandidat[oa]|congresista|concejal|gerente|jefe|defensor(?:a)? del pueblo|canciller|"
                   r"secretari[oa]|embajador(?:a)?|comandante|general)(?:\s+(?:de|del)\s+(?:la\s+)?"
                   r"[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+)?)?", re.I)


def nombre_perfil(p: dict) -> str:
    """El nombre de la persona: el título sin «Quién es…» y sin lo que va después de la coma («X, el nuevo
    ministro…»); si el título es un titular largo, el «Perfil de X,» de la descripción."""
    t = re.sub(r"^(qui[eé]n es|perfil de|el perfil de|este es|esta es)\s+", "", p["titulo"].strip(), flags=re.I)
    t = re.split(r"[,:;(]|\s+[-–]\s+", t)[0].rstrip("?¿ ").strip()
    if len(t.split()) > 5:
        m = re.match(r"^Perfil de ([^,.]+?)(?:,|\.|\s+(?:senador|representante|congresista|candidat|concejal))",
                     p["descripcion"])
        if m:
            t = m.group(1).strip()
    return t


class Indices:
    def __init__(self, perfiles: list[dict], etiquetas: list[dict]):
        self.perfiles = [dict(p, nombre=nombre_perfil(p)) for p in perfiles if len(nombre_perfil(p).split()) <= 6]
        # nombres de pila que se ven en los perfiles: pista de que una secuencia es una persona
        cuenta = defaultdict(int)
        for p in self.perfiles:
            toks = norm(p["nombre"]).split()
            if len(toks) >= 2:
                cuenta[toks[0]] += 1
        self.nombres_pila = {t for t, c in cuenta.items() if c >= 1 and len(t) >= 3 and t not in VACIAS}
        self.por_token = defaultdict(list)          # apellido / token -> perfiles
        for p in self.perfiles:
            for t in norm(p["nombre"]).split():
                if len(t) >= 4 and t not in VACIAS:
                    self.por_token[t].append(p)
        self.etiquetas = [e for e in etiquetas if e["total"] >= 3 and len(e["nombre"]) >= 3]
        self.etq_por_norm = {norm(e["nombre"]): e for e in self.etiquetas}
        self.etq_por_id = {e["id"]: e for e in etiquetas}

    def perfiles_parecidos(self, nombre: str) -> list[dict]:
        """Perfiles que comparten un token largo con el nombre; primero los que comparten más."""
        toks = [t for t in norm(nombre).split() if len(t) >= 4 and t not in VACIAS]
        cuenta = defaultdict(int)
        for t in toks:
            for p in self.por_token.get(t, []):
                cuenta[p["id"]] += 1
        por_id = {p["id"]: p for p in self.perfiles}
        # Con un nombre de 2+ tokens, exigir que coincida el último (apellido) o 2 tokens.
        sal = []
        for pid, c in sorted(cuenta.items(), key=lambda x: -x[1]):
            p = por_id[pid]
            pn = norm(p["nombre"]).split()
            if len(toks) >= 2 and c < 2 and toks[-1] not in pn:
                continue
            sal.append(p)
        return sal


def candidatos_de_nota(nota: dict, idx: Indices) -> list[dict]:
    """Entidades candidatas de una nota, con todas sus menciones, ordenadas por importancia."""
    ors = oraciones(nota["parrafos"])
    texto_n = norm(" ".join(nota["parrafos"]))
    etiquetas_nota = {idx.etq_por_id[i]["id"]: idx.etq_por_id[i] for i in nota["etiquetas"] if i in idx.etq_por_id}
    brutos: dict[str, dict] = {}

    def anotar(sup: str, k: int, inicio: bool) -> None:
        sup = sup.strip(" .,;:-’'")
        # quitar palabras vacías al inicio ("De la Espriella" -> "Espriella", "El Gobierno" -> "Gobierno")
        while sup and norm(sup.split(" ", 1)[0]) in VACIAS:
            sup = sup.split(" ", 1)[1] if " " in sup else ""
        if len(sup) < 3:
            return
        n = norm(sup)
        if n in PROPIOS or (len(n.split()) == 1 and inicio and len(sup) < 4):
            return
        b = brutos.setdefault(n, {"sup": sup, "oraciones": [], "inicio": 0, "n": 0})
        b["n"] += 1
        b["inicio"] += int(inicio)
        if k not in b["oraciones"]:
            b["oraciones"].append(k)

    for k, o in enumerate(ors):
        for m in SECUENCIA.finditer(o):
            inicio = m.start() <= 2
            anotar(m.group(0), k, inicio)
        for m in SIGLA.finditer(o):
            anotar(m.group(0), k, False)

    # Agrupar: "Sanguino" se une a "Antonio Sanguino" si es el último token de una sola forma larga.
    largas = [n for n in brutos if len(n.split()) >= 2]
    grupos: dict[str, dict] = {}
    for n, b in sorted(brutos.items(), key=lambda x: -len(x[0])):
        if len(n.split()) == 1:
            # apellido ("Sanguino") o nombre de pila ("Abelardo") de una sola forma larga
            # de una persona: no se une «Antioquia» a «Universidad de Antioquia» ni «Medellín» a «Alcaldía de Medellín»
            duenos = [l for l in largas if not ORG.match(l) and not LUGAR.match(l)
                      and (n in l.split()[1:] or (n == l.split()[0] and n in idx.nombres_pila))]
            if len(set(duenos)) == 1 and duenos[0] in grupos:
                g = grupos[duenos[0]]
                g["menciones"].append(b["sup"])
                g["oraciones"] = sorted(set(g["oraciones"]) | set(b["oraciones"]))
                g["n"] += b["n"]
                continue
        grupos[n] = {"nombre": b["sup"], "menciones": [b["sup"]], "oraciones": list(b["oraciones"]), "n": b["n"],
                     "solo_inicio": b["inicio"] == b["n"]}

    sal = []
    for n, g in grupos.items():
        etq = idx.etq_por_norm.get(n)
        parecidos = idx.perfiles_parecidos(g["nombre"]) if len(n.split()) >= 1 else []
        exacto = any(norm(p["nombre"]) == n for p in parecidos)
        de_nota = bool(etq and etq["id"] in etiquetas_nota)
        puntos = g["n"] + 4 * de_nota + 3 * exacto + (2 if etq else 0) + (1 if len(n.split()) >= 2 else 0)
        if g["solo_inicio"] and not (etq or parecidos):
            puntos -= 3
        g.update(norm=n, etiqueta=etq, de_nota=de_nota, parecidos=parecidos, exacto=exacto, puntos=puntos)
        sal.append(g)
    sal.sort(key=lambda g: (-g["puntos"], g["oraciones"][0]))
    return sal, ors, texto_n


def tipo_codigo(g: dict, idx: Indices) -> str | None:
    """Pista del código para decidir qué preguntas de rol lleva el caso (la respuesta la da el teacher)."""
    n = g["nombre"]
    if g["exacto"]:
        return "persona"
    if ORG.match(n) or (n.isupper() and 2 <= len(n) <= 7):
        return "organizacion"
    if LUGAR.match(n):
        return "lugar"
    toks = norm(n).split()
    if 2 <= len(toks) <= 5 and toks[0] in idx.nombres_pila:
        return "persona"
    return None


# --- pasajes del archivo -------------------------------------------------------------------------------------
class Archivo:
    def __init__(self, notas: list[dict]):
        self.notas = notas
        self.texto = [norm(" ".join([n["titulo"]] + n["parrafos"])) for n in notas]

    def pasajes(self, g: dict, nota: dict, rng: random.Random) -> list[dict]:
        """Hasta MAX_FUENTE pasajes de notas ANTERIORES que nombran a la entidad: si hay, al menos uno con el
        nombre en el título (puede ser sobre ella) y otros que la nombran en el cuerpo (pueden ser de pasada)."""
        clave = g["norm"]
        if len(clave) < 4:
            return []
        patron = re.compile(rf"(?<![\w]){re.escape(clave)}(?![\w])")
        en_titulo, en_cuerpo = [], []
        for n, t in zip(self.notas, self.texto):
            if n["id"] == nota["id"] or n["fecha"] >= nota["fecha"] or not patron.search(t):
                continue
            (en_titulo if patron.search(norm(n["titulo"])) else en_cuerpo).append(n)
        rng.shuffle(en_cuerpo)
        en_titulo.sort(key=lambda n: n["fecha"], reverse=True)
        elegidas = en_titulo[:1] + en_cuerpo[: MAX_FUENTE - min(1, len(en_titulo))]
        sal = []
        for n in elegidas:
            trozo = next((p for p in n["parrafos"] if patron.search(norm(p))), n["parrafos"][0])
            sal.append({"titulo": n["titulo"], "fecha": n["fecha"], "texto": trozo_con(trozo, g["menciones"], LARGO_FUENTE),
                        "_id": n["id"], "_en_titulo": n in en_titulo})
        rng.shuffle(sal)
        return sal


def trozo_con(texto: str, nombres: list[str], n: int) -> str:
    t = " ".join(texto.split())
    if len(t) <= n:
        return t
    pos = min((t.find(x) for x in nombres if t.find(x) >= 0), default=0)
    ini = max(0, pos - n // 3)
    ini = t.rfind(" ", 0, ini) + 1 if ini else 0
    s = t[ini: ini + n]
    return ("…" if ini else "") + s.rsplit(" ", 1)[0] + "…"


# --- preguntas -----------------------------------------------------------------------------------------------
def noul(instr: str, falso: str | None = None, cierto: str | None = None) -> dict:
    q = {"type": "noul", "instructions": instr}
    if falso:
        q["criteria"] = {"false": falso, "true": cierto}
    return q


ROL_DOCUMENTO = {
    "protagonista": "La historia es sobre lo que hace, decide o le pasa.",
    "fuente": "Aparece sobre todo porque declara, opina o entrega información.",
    "senalado": "Se le acusa, investiga, critica o cuestiona.",
    "afectado": "Sufre las consecuencias de lo que se cuenta (víctima, beneficiario, perjudicado).",
    "autoridad": "Decide, juzga, regula o investiga el asunto sin ser su protagonista.",
    "aliado_o_rival": "Aparece como apoyo u oposición de quien protagoniza la historia.",
    "escenario": "Es el lugar, marco o tema donde pasa la historia.",
    "mencion": "Solo se menciona de pasada; quitarla no cambiaría la historia.",
}
ROL_PERSONA = {
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
    "otro": "Ninguna de las anteriores, no es una persona o no se puede saber.",
}
ROL_ORGANIZACION = {
    "ejecutivo": "Ministerio, departamento administrativo, alcaldía, gobernación o entidad del Ejecutivo.",
    "legislativo": "Congreso, una de sus cámaras o comisiones, asamblea, concejo.",
    "justicia_control": "Corte, tribunal, Fiscalía, Procuraduría, Contraloría, Registraduría.",
    "partido": "Partido, movimiento o coalición política.",
    "empresa": "Empresa privada o mixta, gremio.",
    "armado_ilegal": "Grupo armado ilegal o red criminal.",
    "sociedad_civil": "ONG, sindicato, organización social, comunidad, iglesia.",
    "medio": "Medio de comunicación.",
    "internacional": "Gobierno extranjero u organismo internacional.",
    "otro": "Otra clase, no es una organización o no se puede saber.",
}
POSTURA = {
    "a_favor": "Lo impulsa, defiende o apoya.",
    "en_contra": "Se opone, lo critica o lo denuncia.",
    "neutral": "Informa o decide sin tomar partido.",
    "no_se_sabe": "El documento no le atribuye ninguna posición.",
}


def preguntas(estado: dict, tipo: str | None, cargo: str | None) -> dict:
    q = {
        "ent_es_entidad": noul(
            "`entidad.mencion` nombra a una persona, organización, lugar o tema concreto, no un cargo genérico, una "
            "palabra con mayúscula por empezar la frase ni un título de obra.",
            "Es un cargo genérico, una palabra común o un error de detección.",
            "Nombra algo concreto que un lector podría buscar."),
        "ent_tipo": {"type": "choice", "instructions": "Qué es `entidad` en las `oraciones` de la nota.", "criteria": {
            "persona": "Una persona.",
            "organizacion": "Institución, entidad pública, empresa, partido, grupo armado, organización social o medio.",
            "lugar": "País, departamento, municipio, región, río, sitio.",
            "tema": "Un hecho, proceso, política pública, ley, evento o periodo.",
            "otro": "Una obra, producto, cuenta de red social u otra cosa."}},
        "ent_rol_documento": {"type": "choice", "instructions": "Qué papel cumple `entidad` en la historia que cuenta la nota.",
                              "criteria": ROL_DOCUMENTO},
        "ent_centralidad": {"type": "score", "instructions": "Cuánto depende la historia de la nota de `entidad`.", "criteria": [
            "Nada: se menciona de pasada.", "Poco: aporta un detalle o una cita.",
            "Bastante: es una de las partes principales.", "Total: la historia es sobre ella."]},
        "ent_oracion_clave": {"type": "choice",
                              "instructions": "Cuál de las `oraciones` dice mejor qué hace o qué le pasa a `entidad` en esta nota.",
                              "criteria": {**{k: cortar(v, 200) for k, v in estado["oraciones"].items()},
                                           "ninguna": "Ninguna dice nada de la entidad; solo la nombran."}},
    }
    if estado.get("candidatos"):
        q["ent_candidato"] = {"type": "choice",
                              "instructions": "Cuál de los `candidatos` es la misma entidad que nombra la nota.",
                              "criteria": {**{k: cortar(v, 200) for k, v in estado["candidatos"].items()},
                                           "ninguno": "Ninguno: es otra entidad, o la nota no da para saber cuál es."}}
    for k in estado.get("fuente", {}):
        q[f"ent_fuente_{k}"] = noul(
            f"El pasaje `fuente.{k}` es sobre la misma `entidad` y ayuda a entender el papel que tiene en esta nota "
            "(no solo la menciona de pasada ni es otra con nombre parecido).")
    if cargo:
        q["ent_misma_mencion"] = noul(f"En la nota, «{cargo}» se refiere a la misma entidad que `entidad.nombre`.")
    if tipo == "persona":
        q["ent_rol_persona"] = {"type": "choice", "instructions": "Qué es `entidad` según lo que dice la nota en `documento.fecha`.",
                                "criteria": ROL_PERSONA}
        q["ent_cargo_explicito"] = noul("La nota dice de forma explícita el cargo o la ocupación de `entidad` (no hay que "
                                        "deducirlo de la fuente).")
    if tipo == "organizacion":
        q["ent_rol_organizacion"] = {"type": "choice", "instructions": "Qué clase de organización es `entidad` según la nota.",
                                     "criteria": ROL_ORGANIZACION}
    if tipo in ("persona", "organizacion"):
        q["ent_postura"] = {"type": "choice", "instructions": "Qué posición tiene `entidad` frente al hecho central de la nota, "
                            "según lo que la nota le atribuye.", "criteria": POSTURA}
    return q


# --- casos ---------------------------------------------------------------------------------------------------
def casos_de_nota(nota: dict, idx: Indices, archivo: Archivo, rng: random.Random, max_ent: int) -> list[dict]:
    grupos, ors, _ = candidatos_de_nota(nota, idx)
    if not grupos:
        return []
    elegidos = grupos[:max_ent]
    # Uno dudoso (solo al inicio de frase, sin etiqueta ni perfil) para que ent_es_entidad tenga casos de "no".
    dudosos = [g for g in grupos[max_ent:] if g["puntos"] <= 0]
    if dudosos and rng.random() < 0.5:
        elegidos.append(rng.choice(dudosos))
    cargos = sorted({m.group(0) for o in ors for m in CARGO.finditer(o)}, key=len, reverse=True)
    sal = []
    for j, g in enumerate(elegidos):
        ks = g["oraciones"][:MAX_ORACIONES]
        estado = {
            "tarea": TAREA,
            "documento": {"tipo": "nota", "titulo": nota["titulo"], "fecha": nota["fecha"], "medio": "La Silla Vacía",
                          "bajada": cortar(nota["bajada"], LARGO_BAJADA)},
            "entidad": {"mencion": g["menciones"][-1] if len(g["menciones"]) > 1 and g["oraciones"] else g["nombre"],
                        "nombre": g["nombre"],
                        "otras_menciones": [m for m in dict.fromkeys(g["menciones"]) if m != g["nombre"]][:3]},
            "oraciones": {f"o{k}": cortar(ors[k], LARGO_ORACION) for k in ks},
        }
        # la mención es como aparece primero
        primera = ors[g["oraciones"][0]]
        estado["entidad"]["mencion"] = next((m for m in g["menciones"] if m in primera), g["nombre"])
        # candidatos: perfiles parecidos (+ la etiqueta si existe), ids neutros y en orden aleatorio
        cands = [("perfil", p) for p in g["parecidos"][:MAX_CANDIDATOS]]
        if g["etiqueta"] and len(cands) < MAX_CANDIDATOS + 1:
            cands.append(("etiqueta", g["etiqueta"]))
        rng.shuffle(cands)
        estado["candidatos"] = {}
        meta_cands = {}
        for n, (clase, c) in enumerate(cands, 1):
            if clase == "perfil":
                txt = f"Perfil del «Quién es quién»: {c['nombre']}. {c['descripcion']}"
            else:
                txt = f"Etiqueta de La Silla «{c['nombre']}» ({c['total']} notas). {c['descripcion']}"
            estado["candidatos"][f"c{n}"] = cortar(txt, LARGO_CANDIDATO)
            meta_cands[f"c{n}"] = {"clase": clase, "id": c["id"], "link": c["link"]}
        if not estado["candidatos"]:
            del estado["candidatos"]
        pas = archivo.pasajes(g, nota, rng)
        estado["fuente"] = {f"f{n}": {k: v for k, v in p.items() if not k.startswith("_")} for n, p in enumerate(pas)}
        if not estado["fuente"]:
            del estado["fuente"]
        tipo = tipo_codigo(g, idx)
        cargo = None
        if tipo in ("persona", None) and cargos and rng.random() < 0.6:
            cargo = rng.choice(cargos[:4])
        caso = {
            "id": f"per-{nota['id']}-{j}",
            "schema_id": "periodista_entidad",
            "family": "periodista_entidad",
            "workflow": "entidad_nota_nacional",
            "domain": "periodismo: entidades de una nota de La Silla Vacía, su contexto y su rol",
            "language": "es",
            "state": estado,
            "questions": preguntas(estado, tipo, cargo),
            "meta": {"post_id": nota["id"], "link": nota["link"], "fecha": nota["fecha"], "tipo_codigo": tipo,
                     "etiqueta_de_la_nota": g["de_nota"], "candidatos": meta_cands,
                     "fuente": {f"f{n}": {"post_id": p["_id"], "nombre_en_titulo": p["_en_titulo"]} for n, p in enumerate(pas)}},
        }
        sal.append(caso)
    return sal


def leer_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--documentos", type=int, default=300, help="notas que se vuelven casos de entrenamiento")
    ap.add_argument("--bench", type=int, default=20, help="notas apartadas para el benchmark")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    idx = Indices(json.loads((RAW / "perfiles.json").read_text(encoding="utf-8")),
                  json.loads((RAW / "etiquetas.json").read_text(encoding="utf-8")))
    notas = leer_jsonl(RAW / "notas.jsonl")
    archivo = Archivo(notas)
    # Documentos: notas de largo normal, repartidas por año; el benchmark sale de las más recientes (2025-2026),
    # que son las que más se parecen a lo que verá en producción.
    utiles = [n for n in notas if 5 <= len(n["parrafos"]) <= 60]
    por_anio = defaultdict(list)
    for n in utiles:
        por_anio[n["fecha"][:4]].append(n)
    for v in por_anio.values():
        rng.shuffle(v)
    recientes = [n for y in ("2026", "2025") for n in por_anio.get(y, [])]
    bench = recientes[: a.bench]
    usadas = {n["id"] for n in bench}
    resto = [n for y in sorted(por_anio) for n in por_anio[y] if n["id"] not in usadas]
    rng.shuffle(resto)
    # repartir por año: tomar por turnos
    anios = sorted({n["fecha"][:4] for n in resto})
    colas = {y: [n for n in resto if n["fecha"][:4] == y] for y in anios}
    docs = []
    while len(docs) < a.documentos and any(colas.values()):
        for y in anios:
            if colas[y] and len(docs) < a.documentos:
                docs.append(colas[y].pop())
    train = [c for n in docs for c in casos_de_nota(n, idx, archivo, rng, MAX_ENTIDADES)]
    bcases = [c for n in bench for c in casos_de_nota(n, idx, archivo, rng, MAX_ENTIDADES_BENCH)]
    for nombre, cs in (("cases.jsonl", train), ("bench_cases.jsonl", bcases)):
        with (OUT / nombre).open("w", encoding="utf-8") as f:
            for c in cs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    nq = sum(len(c["questions"]) for c in train)
    print(f"entrenamiento: {len(docs)} notas, {len(train)} casos, {nq} preguntas")
    print(f"benchmark: {len(bench)} notas, {len(bcases)} casos, {sum(len(c['questions']) for c in bcases)} preguntas")


if __name__ == "__main__":
    main()
