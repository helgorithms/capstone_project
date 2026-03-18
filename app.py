"""
ChatBundestag — Argument Mining for Bundestag Debates (LP 19)
Streamlit MVP
"""

import os
import re
import json
import time
import streamlit as st
import pandas as pd
import base64

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from pydantic import BaseModel, ValidationError
from typing import Optional, Literal, List

from dotenv import load_dotenv
load_dotenv()


# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatBundestag",
    page_icon="🏛️",
    layout="centered",
)


# ── Constants ────────────────────────────────────────────────────
VECTOR_DB_PATH = "vector_databases/vector_db_debates_lp19"
DATA_PATH = "data/debates_lp19.csv"

GOV_STATUS_MAP = {1: "Regierungspartei", 0: "Opposition"}
PARTY_DISPLAY_MAP = {"Cabinet": "Bundesregierung"}


# ── Cached resources (load once) ─────────────────────────────────
@st.cache_resource
def load_embedding():
    """Load embedding model (cached across sessions)."""
    base = HuggingFaceEmbeddings(
        model_name='intfloat/multilingual-e5-small',
        model_kwargs={'device': 'cpu'},  # cloud: CUDA
        encode_kwargs={
            "normalize_embeddings": True,
            "prompt": "passage: "
        }
    )

    class E5QueryWrapper(Embeddings):
        def __init__(self, base_embedding):
            self.base = base_embedding
        def embed_query(self, text: str) -> List[float]:
            return self.base.embed_query(f"query: {text}")
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return self.base.embed_documents(texts)

    return E5QueryWrapper(base)


@st.cache_resource
def load_vectorstore(_embedding):
    """Load FAISS index (cached across sessions)."""
    return FAISS.load_local(
        folder_path=VECTOR_DB_PATH,
        embeddings=_embedding,
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.COSINE
    )


@st.cache_data
def load_known_speakers():
    """Load unique speaker names from dataset."""
    df = pd.read_csv(DATA_PATH)
    return {
        name.strip()
        for name in df['speech_identification_ent'].dropna().unique()
        if len(name.strip()) > 1
    }


@st.cache_resource
def load_llm():
    """Initialize LLM. Uses st.secrets on Streamlit Cloud, .env locally."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (FileNotFoundError, KeyError):
        api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY nicht gefunden. Bitte in .env oder Streamlit Secrets hinterlegen.")
        st.stop()
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=4096,
        timeout=None,
        max_retries=2,
        api_key=api_key,
    )


# ── FILTER_MAP ───────────────────────────────────────────────────
FILTER_MAP = {
    # parties
    "spd":       ("party", "SPD"),
    "cdu":       ("party", "CDU"),
    "csu":       ("party", "CSU"),
    "cdu/csu":   ("party", "CDU/CSU"),
    "grüne":     ("party", "GRÜNE"),
    "grünen":    ("party", "GRÜNE"),
    "fdp":       ("party", "FDP"),
    "linke":     ("party", "LINKE"),
    "afd":       ("party", "AfD"),
    "fraktionslos": ("party", "fraktionslos"),
    "parteilos": ("party", "parteilos"),
    # government status
    "kabinett":         ("party", "Cabinet"),
    "regierungspartei": ("government_status", 1),
    "bundesregierung":  ("government_status", 1),
    "regierung":        ("government_status", 1),
    "opposition":       ("government_status", 0),
    # roles
    "kanzler":         ("role", "Bundeskanzler"),
    "bundeskanzler":   ("role", "Bundeskanzler"),
    "kanzlerin":       ("role", "Bundeskanzler"),
    "bundeskanzlerin": ("role", "Bundeskanzler"),
    "minister":        ("role", "Bundesminister"),
    "bundesminister":  ("role", "Bundesminister"),
    "staatssekretär":  ("role", "Staatssekretär"),
    "staatsminister":  ("role", "Staatsminister"),
    "abgeordnete":     ("role", "MdB"),
    "mdb":             ("role", "MdB"),
    "mitglied des bundestags": ("role", "MdB"),
    # legislative periods
    "19. wahlperiode": ("legislative_period", 19),
    "19 wahlperiode":  ("legislative_period", 19),
    "wp19":            ("legislative_period", 19),
    # time frame
    "2021": ("year", "2021"),
    "2020": ("year", "2020"),
    "2019": ("year", "2019"),
    "2018": ("year", "2018"),
    "2017": ("year", "2017"),
}


# ── Pipeline functions ───────────────────────────────────────────
def parse_query_filters(user_input: str, known_speakers: set) -> tuple[str, dict]:
    """Parse user query into semantic search string and metadata filters."""
    filters = {}
    semantic = user_input.lower()

    # FILTER_MAP lookup — word boundaries, longest first
    for term, (key, value) in sorted(FILTER_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r'(?<!\w)' + re.escape(term) + r'(?!\w)'
        if re.search(pattern, semantic):
            filters[key] = value
            semantic = re.sub(pattern, '', semantic)

    # Party takes precedence over conflicting government_status
    if "party" in filters and "government_status" in filters:
        del filters["government_status"]

    # Session
    session_match = re.search(r'(\d+)\.\s*sitzung|sitzung\s*(\d+)', semantic)
    if session_match:
        session_num = session_match.group(1) or session_match.group(2)
        filters["session"] = int(session_num)
        semantic = re.sub(r'(\d+)\.\s*sitzung|sitzung\s*(\d+)', '', semantic)

    # Date
    date_match = re.search(r'\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4}', semantic)
    if date_match:
        filters["date"] = date_match.group()
        semantic = semantic.replace(date_match.group(), "")

    # Speaker — match against known speakers
    semantic_for_speaker = re.sub(r'(\w)s\b', r'\1', semantic)
    speaker_found = None
    for name in sorted(known_speakers, key=len, reverse=True):
        if name.lower() in semantic_for_speaker:
            speaker_found = name
            break
    if speaker_found:
        filters["speaker_name"] = speaker_found
        semantic = re.sub(re.escape(speaker_found.lower()) + r's?\b', '', semantic)

    semantic = re.sub(r'\s+', ' ', semantic).strip()
    return semantic, filters


def get_filtered_retriever(vectorstore, filters: dict, k: int = 15):
    """Return retriever with metadata filters and appropriate fetch_k."""
    search_kwargs = {"k": k}
    if filters:
        search_kwargs["filter"] = filters
        if "speaker_name" in filters:
            search_kwargs["k"] = 30
            search_kwargs["fetch_k"] = 500
        elif "party" in filters:
            search_kwargs["k"] = 50
            search_kwargs["fetch_k"] = 300
        elif "government_status" in filters:
            search_kwargs["k"] = 30
            search_kwargs["fetch_k"] = 200
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def format_context_with_metadata(docs: list[Document], max_chunks: int = 8) -> str:
    """Prepend metadata header to each retrieved chunk."""
    formatted = []
    for doc in docs[:max_chunks]:
        m = doc.metadata
        header_parts = []
        if m.get("speaker_name"):
            header_parts.append(f"Redner: {m['speaker_name']}")
        if m.get("party"):
            display_party = PARTY_DISPLAY_MAP.get(m["party"], m["party"])
            header_parts.append(f"Partei: {display_party}")
        if "government_status" in m:
            header_parts.append(f"Regierungsstatus: {GOV_STATUS_MAP.get(m['government_status'], m['government_status'])}")
        if m.get("role"):
            header_parts.append(f"Rolle: {m['role']}")
        if m.get("date"):
            header_parts.append(f"Datum: {m['date']}")
        if m.get("year"):
            header_parts.append(f"Jahr: {m['year']}")
        if m.get("session"):
            header_parts.append(f"Sitzung: {m['session']}")
        if m.get("legislative_period"):
            header_parts.append(f"Wahlperiode: {m['legislative_period']}")

        header = f"[{' | '.join(header_parts)}]" if header_parts else ""
        formatted.append(f"{header}\n{doc.page_content}".strip())

    return "\n\n---\n\n".join(formatted)


# ── Pydantic schema ──────────────────────────────────────────────
class ArgumentStructure(BaseModel):
    claim:               Optional[str] = None
    grounds:             Optional[list[str]] = []
    rebuttal:            Optional[list[str]] = []
    attack:              Optional[list[str]] = []
    speaker:             Optional[str] = None
    party:               Optional[str] = None
    government_status:   Optional[str] = None
    role:                Optional[str] = None
    date:                Optional[str] = None
    session:             Optional[str] = None
    legislative_period:  Optional[str] = None
    confidence:          Literal["high", "medium", "low"] = "low"
    reasoning:           Optional[str] = None
    note:                Optional[str] = None


def parse_llm_output(raw_output: str) -> dict:
    """Clean, parse, and validate LLM output."""
    try:
        cleaned = re.sub(r"```json|```", "", raw_output).strip()
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not json_match:
            if cleaned.strip().startswith("{"):
                cleaned_repaired = cleaned.strip() + "}"
                json_match = re.search(r"\{.*\}", cleaned_repaired, re.DOTALL)
            if not json_match:
                return {"status": "json_error", "raw": raw_output, "error": "No JSON block found"}

        parsed = json.loads(json_match.group())
        validated = ArgumentStructure(**parsed)
        return {"status": "ok", "data": validated.model_dump()}

    except json.JSONDecodeError as e:
        return {"status": "json_error", "raw": raw_output, "error": str(e)}
    except ValidationError as e:
        return {"status": "validation_error", "raw": raw_output, "error": str(e)}


# ── Prompt ───────────────────────────────────────────────────────
PROMPT_TEMPLATE = """

    Du extrahierst Argumentationsstrukturen aus Bundestagsdebatten (Kontext) anhand der Nutzerfrage als thematischem Fokus. Verwende ausschließlich den gegebenen Kontext.

    Definitionen:
    - claim: zentrale politische Position, für oder gegen die argumentiert wird
    - grounds: faktische oder normative Belege für den claim
    - rebuttal: antizipierter Einwand der Gegenseite + Widerlegung durch den Sprecher
    - attack: offensive Kritik an der Gegenposition


    ### Beispiel 2:
    Frage 2: "Welche Position vertritt Gregor Gysi zum Atomwaffenverbotsvertrag?"

    Kontext 2: [Redner: Gregor Gysi | Partei: DIE LINKE | Rolle: MdB | Datum: 2021-01-29 | Sitzung: 207 | Wahlperiode: 19 | Regierungsstatus: Opposition]

    122 Staaten der UNO haben den Atomwaffenverbotsvertrag beschlossen. Deutschland muss alle Maßnahmen zum Abbau des Atomwaffenarsenals unterstützen. Doch die Bundesregierung steht abseits. Das Hauptargument der Regierung ist ein angeblicher Widerspruch zum Atomwaffensperrvertrag. Der Wissenschaftliche Dienst des Bundestages hat widerlegt, dass der Nichtverbreitungsvertrag durch den Verbotsvertrag angegriffen wird. Nachdem alle Argumente der Regierung widerlegt sind, muss ich feststellen: CDU/CSU und SPD wollen kein Verbot von Atomwaffen.

    Output 2:
    {{
    "claim": "Deutschland muss dem Atomwaffenverbotsvertrag beitreten – aus historischer Verantwortung und weil alle Gegenargumente der Bundesregierung sachlich widerlegt sind.",
    "grounds": ["122 Staaten haben den Vertrag beschlossen; der Wissenschaftliche Dienst des Bundestages bestätigt, dass er den Nichtverbreitungsvertrag ergänzt, ohne ihn zu untergraben."],
    "rebuttal": ["Die Bundesregierung behauptet einen Widerspruch zum Nichtverbreitungsvertrag. Der Wissenschaftliche Dienst hat diesen Einwand explizit widerlegt."],
    "attack": ["CDU/CSU und SPD wollen kein Verbot von Atomwaffen – sie ignorieren die Faktenlage und die historische Verantwortung Deutschlands."],
    "speaker": "Gregor Gysi",
    "party": "DIE LINKE",
    "government_status": "Opposition",
    "role": "MdB",
    "date": "2021-01-29",
    "session": "207",
    "legislative_period": "19",
    "confidence": "high",
    "note": null
    }}

    ### Beispiel 3:
    Frage 3: "Wie argumentiert die CDU/CSU für die Rolle der Landwirtschaft im Klimaschutz?"

    Kontext 3: [Redner: Gitta Connemann | Partei: CDU/CSU | Rolle: MdB | Datum: 2020-09-17 | Sitzung: 176 | Wahlperiode: 19 | Regierungsstatus: Regierungspartei]

    Die größten Klimaschützer in Deutschland sind unsere Waldbauern. Jeder Hektar Wald bindet 8 Tonnen CO2 pro Jahr. Selbsternannte Experten fordern den Verzicht auf die Waldbewirtschaftung. Das ist verantwortungslos, denn Holz aus Ländern mit niedrigeren Standards müsste importiert werden. Grünland ohne Bewirtschaftung verbuscht – Artenvielfalt adieu. Ohne Nutzung keine Nachhaltigkeit.

    Output 3:
    {{
    "claim": "Nachhaltigkeit erfordert die aktive Bewirtschaftung durch Landwirte und Waldbauern – ohne sie gibt es weder Klima- noch Artenschutz.",
    "grounds": ["Jeder Hektar Wald bindet 8 Tonnen CO2 pro Jahr; artenreiches Grünland ist durch landwirtschaftliche Nutzung über Generationen entstanden."],
    "rebuttal": ["Der Verzicht auf Waldbewirtschaftung ist verantwortungslos, denn Holz aus Ländern mit niedrigeren Standards müsste importiert werden – das bricht die Nachhaltigkeit."],
    "attack": ["Grünlandnutzung zu verbieten bewirkt das Gegenteil: Ohne Bewirtschaftung verbuscht Grünland und die Artenvielfalt geht verloren."],
    "speaker": "Gitta Connemann",
    "party": "CDU/CSU",
    "government_status": "Regierungspartei",
    "role": "MdB",
    "date": "2020-09-17",
    "session": "176",
    "legislative_period": "19",
    "confidence": "high",
    "note": null
    }}

    ### Beispiel 4:
    Frage 4: "Wie kritisiert die FDP die Mehrwertsteuersenkung im Zweiten Corona-Steuerhilfegesetz?"

    Kontext 4: [Redner: Christian Dürr | Partei: FDP | Rolle: MdB | Datum: 2020-06-29 | Sitzung: 168 | Wahlperiode: 19 | Regierungsstatus: Opposition]

    Die vorübergehende Mehrwertsteuersenkung ist nicht die richtige Antwort auf diese Krise. Sie ist verbunden mit absurdem bürokratischem Aufwand. Selbst im optimistischsten Fall sind es 30 Euro für einen durchschnittlichen Haushalt pro Monat. Die Profiteure werden Onlineversandhändler wie Amazon sein. Wir legen konkrete Alternativen vor: Abschaffung des Solidaritätszuschlages, Abschaffung des Mittelstandsbauchs und eine negative Gewinnsteuer in Höhe von 25 Milliarden Euro. Ihre eigenen Experten sagen: Die Mehrwertsteuersenkung ist ineffizient und bürokratisch.

    Output 4:
    {{
    "claim": "Die vorübergehende Mehrwertsteuersenkung ist die falsche Krisenmaßnahme – sie belastet den Mittelstand bürokratisch und kommt bei den Bürgern kaum an.",
    "grounds": ["Die Mehrwertsteuersenkung bringt selbst im optimistischsten Fall nur 30 Euro pro Haushalt und Monat; selbst die von der Koalition benannten Experten bewerten sie als ineffizient."],
    "rebuttal": ["Drei konkrete Alternativen: vollständige Abschaffung des Solidaritätszuschlages, Abschaffung des Mittelstandsbauchs und eine negative Gewinnsteuer über steuerliche Verlustverrechnung in Höhe von 25 Milliarden Euro."],
    "attack": ["Die Profiteure der Mehrwertsteuersenkung sind nicht die Krisenbetroffenen, sondern Onlineversandhändler wie Amazon – der Mittelstand wird mit Bürokratie belastet, Amazon wird entlastet."],
    "speaker": "Christian Dürr",
    "party": "FDP",
    "government_status": "Opposition",
    "role": "MdB",
    "date": "2020-06-29",
    "session": "168",
    "legislative_period": "19",
    "confidence": "high",
    "note": null
    }}


    ### Beispiel 6:
    Frage 6: "Wie argumentieren die Grünen für eine Verschärfung der Mietpreisbremse?"

    Kontext 6: [Redner: Daniela Wagner | Partei: GRÜNE | Rolle: MdB | Datum: 2018-03-01 | Sitzung: 17 | Wahlperiode: 19 | Regierungsstatus: Opposition]

    Die Mietpreisbremse hat viele Ausnahmen und Umgehungsmöglichkeiten. Die Folge waren noch schneller steigende Preise. Allein von 2016 auf 2017 sind die Mietpreise um bis zu 10 Prozent gestiegen. Armutsgefährdete Haushalte müssen bis knapp die Hälfte ihres Einkommens für die Miete aufwenden. Die Losung „Bauen, bauen, bauen" hat nichts geändert. Wir brauchen die Pflicht zur Angabe der Vormiete, die Abschaffung der Rügepflicht und eine Absenkung der Modernisierungsumlage.

    Output 6:
    {{
    "claim": "Die Mietpreisbremse muss zu einem wirksamen Instrument gemacht werden – in der aktuellen Form ist sie wirkungslos, weil zu viele Ausnahmen bestehen.",
    "grounds": ["Allein von 2016 auf 2017 sind die Mietpreise um bis zu 10 Prozent gestiegen; armutsgefährdete Haushalte müssen bis knapp die Hälfte ihres Einkommens für die Miete aufwenden."],
    "rebuttal": ["Die Wohnungswirtschaft fordert die Abschaffung der Mietpreisbremse. Das Gegenteil ist nötig: Pflicht zur Angabe der Vormiete, Abschaffung der Rügepflicht und Absenkung der Modernisierungsumlage."],
    "attack": ["Die Losung 'Bauen, bauen, bauen' hat nichts geändert – steuerliche Anreize allein produzieren Mitnahmeeffekte, schaffen aber keinen bezahlbaren Wohnraum."],
    "speaker": "Daniela Wagner",
    "party": "GRÜNE",
    "government_status": "Opposition",
    "role": "MdB",
    "date": "2018-03-01",
    "session": "17",
    "legislative_period": "19",
    "confidence": "high",
    "note": null
    }}

    Frage:
    {question}

    Regeln:
    1. Verwende NUR den Kontext. Keine externen Kenntnisse, nichts erfinden.
    2. Entnimm Metadaten (Redner, Partei, Rolle, Datum, Sitzung, Wahlperiode, Regierungsstatus) aus den Kontext-Kopfzeilen. Fehlende Felder = null. Keine erklärenden Texte in Metadatenfeldern.
    3. Grounds, rebuttal, attack eng am Text paraphrasieren. Claim darf synthetisiert werden.
    4. Fokussiere auf die Nutzerfrage — ignoriere thematisch Irrelevantes.
    5. Fehlende Argumentteile = leere Liste [].
    6. Antwort ist ausschließlich valides JSON. Kein Text davor oder danach.
    7. Wenn der Kontext keine relevanten Informationen enthält:
       {{"claim": null, "grounds": [], "rebuttal": [], "attack": [], "confidence": "low", "note": "Keine relevanten Informationen gefunden."}}

    Kontext:
    {context}
    
    """

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


# ── RAG chain ────────────────────────────────────────────────────
def run_query(user_input: str, vectorstore, known_speakers: set, llm) -> dict:
    """Execute the full RAG pipeline for a single query."""
    semantic_query, filters = parse_query_filters(user_input, known_speakers)
    filtered_retriever = get_filtered_retriever(vectorstore, filters)
    docs = filtered_retriever.invoke(semantic_query)
    context = format_context_with_metadata(docs)

    prompt_input = {"context": context, "question": user_input}
    msg = llm.invoke(prompt.format(**prompt_input))
    result = parse_llm_output(msg.content)
    result["_filters"] = filters
    result["_num_docs"] = len(docs)
    return result


# ── UI ───────────────────────────────────────────────────────────
def render_result(data: dict):
    """Render a structured argument result."""
    speaker = data.get("speaker") or "unbekannt"
    party = data.get("party") or "?"
    gov_status = data.get("government_status") or "?"
    role = data.get("role") or "?"
    date = data.get("date") or "?"
    period = data.get("legislative_period") or "?"
    session = data.get("session") or "?"

    # Header card
    st.markdown(f"### 👤 {speaker}")
    cols = st.columns(3)
    cols[0].markdown(f"**Partei:** {party}")
    cols[1].markdown(f"**Rolle:** {role}")
    cols[2].markdown(f"**Status:** {gov_status}")

    cols2 = st.columns(3)
    cols2[0].markdown(f"**Datum:** {date}")
    cols2[1].markdown(f"**Sitzung:** {session}")
    cols2[2].markdown(f"**Wahlperiode:** {period}")

    st.divider()

    # Claim
    if data.get("claim"):
        st.markdown(f"**📌 Standpunkt**")
        st.markdown(f"> {data['claim']}")

    # Arguments
    grounds = data.get("grounds") or []
    rebuttal = data.get("rebuttal") or []
    attack = data.get("attack") or []

    if grounds:
        st.markdown("**💬 Begründung (Grounds)**")
        for g in grounds:
            st.markdown(f"- {g}")

    if rebuttal:
        st.markdown("**🔄 Widerlegung (Rebuttal)**")
        for r in rebuttal:
            st.markdown(f"- {r}")

    if attack:
        st.markdown("**⚔️ Angriff (Attack)**")
        for a in attack:
            st.markdown(f"- {a}")


def set_background(image_path):
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    [data-testid="stApp"] {{
        background-image: url("data:image/png;base64,{data}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stApp"] > div {{
        background-color: rgba(255, 255, 255, 0.85);
    }}
    </style>
    """, unsafe_allow_html=True)

# ── MAIN ───────────────────────────────────────────────────────────
def main():
    set_background("images/reichstagsgebaeude_wiese.png")
    st.title("ChatBundestag 🏛️")
    st.markdown(
        "Argumentationsanalyse für Bundestagsdebatten der 19. Legislaturperiode (2017–2021)."
    )

    # Load resources
    with st.spinner("Lade Modelle und Daten..."):
        embedding = load_embedding()
        vectorstore = load_vectorstore(embedding)
        known_speakers = load_known_speakers()
        llm = load_llm()

    # Instructions 
    # Instructions (after known_speakers is loaded)
    with st.expander("🔎 Anleitung & Beispiele"):
        st.markdown(
"ChatBundestag analysiert Argumentationsstrukturen in Bundestagsdebatten der 19. Legislaturperiode (2017–2021).\n\n"
"**Was kann abgefragt werden?**\n"
"- **Parteien:** CDU/CSU, SPD (Koalition) · Grüne, Linke, AfD, FDP (Opposition)\n"
"- **Personen:** Alle Mitglieder des Deutschen Bundestages während der 19. Legislaturperiode (s. Redner/innenauswahl)\n"
"- **Themen:** Alle Themen, die debattiert wurden (s. Themenbeispiele)\n\n"
"**Beispielabfragen:**\n"
"- *Wie steht die FDP zum Bürokratieabbau? *\n"
"- *Welche Position vertritt Angela Merkel zur Europäischen Integration?*\n"
"- *Wie argumentiert die Linke zum Mindestlohn?*\n"
"- *Wie stehen die Grünen zur Energiewende?*\n\n"
"**Output:**\n"
"- Zentrale Position (Standpunkt)\n"
"- Faktische Begründung (Grounds)\n"
"- Entkräftung von Gegenargumenten (Rebuttal)\n"
"- Angriff der Gegenposition (Attack)"
        )

        st.selectbox(
            "📋 Themenbeispiele",
            ["(Thema auswählen)"] + [
                "Mietpreisbremse", "Energiewende", "Klimaschutzgesetz", "Mindestlohn",
                "Europäische Integration", "Corona-Steuerhilfsgesetz", "Atomwaffenverbotsvertrag", "Sicherheitspolitik",
                "digitale Bildung", "Steuerpolitik", "Bürokratieabbau",
            ],
            key="topic_browse",
        )

        st.selectbox(
            f"👥 Redner/innen ({len(known_speakers)})",
            ["(Person auswählen)"] + sorted(known_speakers),
            key="speaker_browse",
        )

    # Query input
    user_input = st.text_input(
        "**Deine Frage:**",
        placeholder="z.B. Wie steht die Linke zum Atomwaffenverbotsvertrag?",
    )

    if user_input:
        with st.spinner("Suche und analysiere..."):
            start = time.time()
            result = run_query(user_input, vectorstore, known_speakers, llm)
            duration = round(time.time() - start, 1)

        # Status bar
        st.caption(
            f"🔍 {result.get('_num_docs', 0)} Dokumente gefunden · "
            f"⏱️ {duration}s · "
            f"Filter: {result.get('_filters', {}) or 'keine'}"
        )

        if result["status"] != "ok":
            st.error(f"Parsing fehlgeschlagen: {result.get('error', 'Unbekannter Fehler')}")
            with st.expander("Rohausgabe"):
                st.code(result.get("raw", ""))
            return

        data = result["data"]

        if data.get("note"):
            st.warning(data["note"])
            return

        if not data.get("claim"):
            st.warning(
                "Keine relevanten Informationen gefunden. "
                "Versuche die Anfrage zu vereinfachen oder weniger Filter zu verwenden."
            )
            return

        render_result(data)

        # Debug expander
        with st.expander("🔧 Debug-Info"):
            st.json(data)


if __name__ == "__main__":
    main()