<p align="center">
  <img src="images/chatbundestag_logo.png" width="320" alt="ChatBundestag">
</p>

<h1 align="center">ChatBundestag</h1>

<p align="center">
  <strong>Argument mining in German parliamentary debates</strong><br>
  Ask a question in plain German — get a structured argument with full source attribution.
</p>

<p align="center">
  <a href="https://chatbundestag.streamlit.app"><img src="https://img.shields.io/badge/demo-live-brightgreen" alt="Live demo"></a>
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11">
  <img src="https://img.shields.io/badge/RAG-LangChain%20%2B%20FAISS-orange" alt="Stack">
</p>

---

## What it does

Political positions in the Bundestag are public, but effectively unsearchable. The protocols exist as tens of thousands of speeches; finding *what a specific party actually argued about a specific policy* means reading them.

ChatBundestag turns that into a question. It retrieves the relevant speech passages and extracts the **argumentative structure** behind them — not a summary, but a typed object you can verify against the source:

| Field | Meaning |
|---|---|
| `claim` | The central political position being argued for or against |
| `grounds` | Factual or normative evidence offered for the claim |
| `rebuttal` | An anticipated counter-argument, and the speaker's response to it |
| `attack` | Offensive criticism directed at the opposing position |

Every result carries speaker, party, role, government status, date, session and legislative period, plus a self-reported `confidence` level.

**Live demo:** [chatbundestag.streamlit.app](https://chatbundestag.streamlit.app)

### Example

> **Frage:** *Welche Position vertritt Gregor Gysi zum Atomwaffenverbotsvertrag?*

```json
{
  "claim": "Deutschland muss dem Atomwaffenverbotsvertrag beitreten – aus historischer
            Verantwortung und weil alle Gegenargumente der Bundesregierung sachlich
            widerlegt sind.",
  "grounds": ["122 Staaten haben den Vertrag beschlossen; der Wissenschaftliche Dienst
               des Bundestages bestätigt, dass er den Nichtverbreitungsvertrag ergänzt,
               ohne ihn zu untergraben."],
  "rebuttal": ["Die Bundesregierung behauptet einen Widerspruch zum Nichtverbreitungs-
                vertrag. Der Wissenschaftliche Dienst hat diesen Einwand widerlegt."],
  "attack": ["CDU/CSU und SPD wollen kein Verbot von Atomwaffen – sie ignorieren die
              Faktenlage und die historische Verantwortung Deutschlands."],
  "speaker": "Gregor Gysi",
  "party": "DIE LINKE",
  "government_status": "Opposition",
  "role": "MdB",
  "date": "2021-01-29",
  "session": "207",
  "legislative_period": "19",
  "confidence": "high"
}
```

---

## Architecture

```
User query (German, natural language)
        │
        ▼
┌───────────────────────────────────────────┐
│ 1. Query parsing          parse_query_filters()
│    Rule-based extraction of party, role,  │
│    speaker, session, date, period →       │
│    metadata filters + a clean semantic    │
│    search string                          │
└───────────────────────────────────────────┘
        │  semantic string          │  filters
        ▼                           ▼
┌───────────────────────────────────────────┐
│ 2. Retrieval              FAISS + MMR
│    multilingual-e5-small embeddings,      │
│    metadata-filtered, adaptive fetch_k:   │
│      speaker → k=30,  fetch_k=500         │
│      party   → k=50,  fetch_k=300         │
│      gov.    → k=30,  fetch_k=200         │
└───────────────────────────────────────────┘
        │  top-k chunks
        ▼
┌───────────────────────────────────────────┐
│ 3. Context assembly       format_context_with_metadata()
│    Each chunk prefixed with a metadata    │
│    header so the LLM can attribute        │
│    claims to the correct speaker          │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│ 4. Extraction             Groq · llama-3.1-8b-instant
│    Few-shot German prompt with four       │
│    worked examples → JSON                 │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│ 5. Validation             pydantic ArgumentStructure
│    Typed, nullable fields; malformed or   │
│    unsupported output degrades to a       │
│    low-confidence result rather than a    │
│    confident hallucination                │
└───────────────────────────────────────────┘
        │
        ▼
   Streamlit UI
```

---

## Data

| | |
|---|---|
| **Source** | [Offene Parlamentsdaten](https://www.bundestag.de/services/opendata) — German Bundestag plenary protocols |
| **Scope** | 19th legislative period (2017–2021) |
| **Volume** | 26,902 speeches → 219,116 text chunks |
| **Parties** | CDU/CSU, SPD (coalition) · Grüne, Linke, AfD, FDP (opposition) · Cabinet |
| **Metadata per chunk** | speaker, party, role, government status, date, session, legislative period |

Datasets and vector indexes are not stored in this repository. See [Setup](#setup) for placement.

---

## Design decisions

The interesting parts of this project are the choices, not the pipeline.

| Decision | Choice | Reasoning |
|---|---|---|
| **Embedding model** | `intfloat/multilingual-e5-small` | Started with `all-MiniLM-L6-v2`. It is trained predominantly on English, and German retrieval quality was visibly worse on the same queries. E5 requires `query:` / `passage:` prefixes, so retrieval is wrapped in `E5QueryWrapper` to apply them asymmetrically. |
| **Chunk size** | 400 chars, 50 overlap | Tested 500/50 in the baseline. Parliamentary speeches pack a full argument into a short span; smaller chunks improved the precision of retrieved passages without fragmenting claims. |
| **Structured output** | pydantic schema, not free text | A prose answer about politics is unfalsifiable. A typed object with mandatory attribution can be checked against the protocol — which is the entire point of the project. |
| **Filter before search** | Rule-based parser, not LLM-based | Party and speaker names are a closed, known vocabulary. A regex map is deterministic, free and instant; an LLM call here would add latency and a second failure mode. |
| **Adaptive `fetch_k`** | Scaled by filter narrowness | Narrower filters need a larger candidate pool to survive post-filtering (see [Limitations](#known-limitations)). |
| **LLM** | Groq `llama-3.1-8b-instant` | Fast enough for interactive use, cheap enough to leave running, and adequate for extraction given a strongly constrained few-shot prompt. |
| **Argument model** | Toulmin-derived | `claim` / `grounds` / `rebuttal` / `attack` maps cleanly onto how parliamentary rhetoric is actually structured — a position, its support, a pre-empted objection, and an attack on the other side. |

---

## Evaluation

**Current status: qualitative, not quantitative.** This is stated plainly because it is the most important open item in the project.

Evaluation to date consists of roughly 30 manually logged queries across three categories:

- **`expected`** — queries that return correct, well-attributed arguments
- **`erroneous`** — queries that return an answer that is wrong, misattributed, or hallucinated
- **`no-info-found`** — queries that should return an answer but return nothing

These logs live in `AdvancedModel_ChatBundestag.ipynb` and drove the design decisions above. They are not a benchmark: there is no gold-standard annotated set, no retrieval metrics, and no measured precision. **Building that harness is roadmap item #1** — without it, "the system got better" is not a claim this project can currently support.

---

## Known limitations

Documented from the error logs, with diagnosed causes.

**1. Cabinet / party metadata collision** *(highest impact)*
Ministers are labelled `party: "Cabinet"` rather than their actual party. A query filtered on `{"party": "CDU/CSU"}` therefore silently excludes Merkel, Altmaier, Scheuer and every other CDU cabinet member. They surface only via role queries (`"Bundeskanzlerin"`) or government-status queries (`"Bundesregierung"`). This accounts for most entries in the no-info-found log.

**2. FAISS post-filters rather than pre-filters**
LangChain's FAISS integration retrieves first and applies metadata filters afterwards. With a narrow filter, most retrieved candidates are discarded and results thin out — which is why `fetch_k` is inflated up to 500. This is a workaround, not a fix. True filter-then-search requires a vector store that supports it natively (Qdrant, Weaviate).

**3. Single legislative period**
Only LP19 (2017–2021) is indexed. Questions about earlier periods return nothing, with no indication that the period is simply out of scope.

**4. Topic vocabulary barrier**
Retrieval requires the user to know the parliamentary term. Users search for *"Atomwaffen"*; the debate says *"Atomwaffenverbotsvertrag"*. They search *"Finanzpolitik"*; the debate says *"Finanztransaktionssteuer"*. There is currently no path from an everyday policy field to the specific terminology used in the chamber.

**5. Ambiguous or ill-formed queries**
The system attempts an answer rather than declining. A malformed query can return an argument drawn from the wrong speaker.

---

## Roadmap

1. **Evaluation harness** — a gold set of annotated queries with retrieval and attribution metrics, so subsequent changes can be measured rather than asserted
2. **Fix the Cabinet/party mapping** — resolve ministers to their actual party while retaining government status
3. **Migrate to Qdrant** — native pre-filtering, and a vector store that scales past a single period
4. **Expand to all legislative periods** — 1949 to present, via [Open Discourse](https://opendiscourse.de) / CPP-BT rather than re-scraping
5. **Hybrid retrieval + reranking** — BM25 alongside dense retrieval, with a cross-encoder reranker over the candidate set
6. **Topic navigation** — a precomputed topic hierarchy (policy field → theme → actual parliamentary terminology) so users can browse in from *"Energiepolitik"* without knowing what to type

---

## Setup

Requires Python 3.11 and a [Groq API key](https://console.groq.com).

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt    # requirements_notebook.txt for the notebooks
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Place the dataset and prebuilt index:

```
data/debates_lp19.csv
vector_databases/vector_db_debates_lp19/{index.faiss,index.pkl}
```

Run:

```bash
streamlit run app.py
```

---

## Repository structure

```
app.py                              Streamlit application — full pipeline
EDA_ChatBundestag.ipynb             Corpus exploration: speech distributions,
                                    speaker roles, length analysis, word clouds
BasicRAG_ChatBundestag.ipynb        First pipeline: MiniLM, 500-char chunks
BaselineModel_ChatBundestag.ipynb   Baseline: e5-small, chunking comparison
AdvancedModel_ChatBundestag.ipynb   Current model + error analysis and query logs
requirements.txt                    Application dependencies
requirements_notebook.txt           Notebook dependencies
images/                             Plots and assets
```

---

## Context

Built as the capstone project of the Data Science & AI bootcamp at SPICED / neue fische Academy, Berlin (2026). Solo project.

Data: [Offene Parlamentsdaten](https://www.bundestag.de/services/opendata), German Bundestag. Plenary protocols are public documents.
