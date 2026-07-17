# EHR / Wearables → FHIR R4 Transformation Pipeline

A pipeline that is designed to transforms raw EHR and wearable device data into validated FHIR R4 JSON bundles. Built with LangGraph, ChromaDB and Claude.

---

## Pipeline Overview

```
Dictionary CSVs  →  SchemaParser  →  CodeMapper  →  FHIRBuilder  →  Validator  →  JSON Bundles
                     (Node 1)        (Node 2)         (Node 3)       (Node 4)
                     Deterministic   ChromaDB +       Pydantic       Structural
                     CSV Parsing     Claude RAG       validated      validation
```
---

## Project Structure

```
fhir_mvp/
├── main.py                        # Entry point (--dataset selects the study)
├── config.py                      # Paths, model settings, DATASETS + cluster registries
├── data/
│   ├── sample/                    # MFU/ComfortAge source CSVs (comma-delimited)
│   ├── sample_ACE/                # ACE study CSVs (semicolon-delimited) + Excel dictionary
│   └── terminology/               # LOINC, SNOMED, RxNorm seed CSVs
├── src/
│   ├── graph.py                   # LangGraph pipeline definition
│   ├── state.py                   # Shared pipeline state (carries the active dataset)
│   ├── dictionary.py              # Unified dictionary loader (MFU CSV + ACE xlsx)
│   ├── agents/
│   │   ├── schema_parser.py       # Node 1
│   │   ├── code_mapper.py         # Node 2 (with on-disk mapping cache)
│   │   ├── fhir_builder.py        # Node 3 (MFU + ACE strategies)
│   │   └── validator.py           # Node 4
│   ├── fhir/
│   │   └── resources.py           # FHIR R4 resource factory
│   └── vector_store/
│       └── store.py               # ChromaDB terminology store
└── output/                        # Generated FHIR bundles (git-ignored)
```

---

## Datasets

The pipeline is dataset-driven. A single `DATASETS` registry in `config.py`
describes each study i.e. its data directory, CSV delimiter, subject-ID column,
dictionary format, cluster registry, and output sub-folder. Then the same four
agent nodes process any of them. Two dataset types:

| Dataset | Data dir | Delimiter | ID column | Dictionary | Output |
|---|---|---|---|---|---|
| `mfu` (default) | `data/sample/` | `,` | `subject_id` | per-cluster CSVs | `output/` |
| `ace` | `data/sample_ACE/` | `;` | `faceid` | one Excel workbook | `output/ace/` |


Add a new study by adding one entry to `DATASETS` (and a cluster registry); no
agent code changes are required for standard wide/long tables.

---

## Setup

**Required:** Python 3.10+

**1. Clone the repo and navigate to the project**

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Copy .env.example and set up your Anthropic API key:**
```bash
cp .env.example .env
```
Open `.env` and replace the placeholder with your key:
```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running the Pipeline

For a quick demo, i.e. first 3 subjects only:
```bash
python main.py
```

For example, ACE dataset — all clusters, all subjects → `output/ace/`:
```bash
python main.py --dataset ace --clusters all --max-subjects 0
```
A capped ACE demo (subjects shared across clusters):
```bash
python main.py --dataset ace --clusters all --max-subjects 25
```
Reuse cached code mappings on a rerun (skips the LLM mapping step):
```bash
python main.py --dataset ace --clusters all --max-subjects 0 --skip-mapping
```

All subjects, all clusters:
```bash
python main.py --max-subjects 0
```

Skip LLM mapping (uses cached results from a previous run, very fast):
```bash
python main.py --skip-mapping
```

Rebuild the ChromaDB vector store from scratch:
```bash
python main.py --rebuild-store
```

Mapping report only (parse + map, write `mapping_report.json`, no bundles):
```bash
python main.py --dataset ace --clusters all --mapping-only
```

Choose the mapping model (Anthropic by default; others need their key — see below):
```bash
python main.py --dataset ace --clusters all --model gpt-4o
```

Verbose logging:
```bash
python main.py --verbose
```

---

## Dashboard (run + view in the browser)

A local server turns the viewer into a small web app where you can upload data,
choose options, run the pipeline, and inspect the results on your machine
(the server binds to `localhost`).

```bash
python server.py        # then open http://127.0.0.1:8000
```

In the dashboard, click **▶ Run pipeline** and:

1. upload the data files (or a whole folder),
2. pick a model, dataset (auto-detected from the files), output type
   (mapping report or full bundles), and patient count,
3. hit Run after which results load straight into the viewer.

Use the View selector to switch between the FHIR bundle view (Patient,
Observations, Conditions, DiagnosticReports, Procedures) and the mapping-report
view (coverage summary + filterable variable→code table). You can also open
existing `*_fhir_bundle.json` / `mapping_report.json` files via **Open file(s)…**
without the server.

### Editing / correcting codes (overrides)

In the **Mapping report** view (when run via `server.py`), every row is editable:
a technician can fix a wrong code, change its system, edit the display, or assign
a code to an unmapped variable. Two buttons:

- **Save overrides** — persists the edits to `overrides/<dataset>_overrides.json`.
- **Save & rebuild** — saves, then rebuilds the FHIR bundles with the corrected
  codes (reusing the last run's data and cached mappings — no re-mapping/LLM cost).

Overrides are **sticky**: the pipeline re-applies them on top of every run, so
manual corrections survive re-mapping, model changes, and new runs until changed.
Clearing a row's code resets that variable to its local fallback. Overrides also
apply to command-line runs (they're keyed per dataset).

### Model providers / API keys

Anthropic (Claude) models use the native SDK. OpenAI, Google Gemini, and Meta
Llama models route through [LiteLLM](https://github.com/BerriAI/litellm) and need
that provider's key in your `.env`:

| Provider | Example model id | Env var |
|---|---|---|
| Anthropic | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| OpenAI | `gpt-4o`, `gpt-4o-mini` | `OPENAI_API_KEY` |
| Google | `gemini/gemini-1.5-pro` | `GEMINI_API_KEY` |
| Meta (via Groq) | `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY` |

Model ids for non-Anthropic providers follow LiteLLM's naming and may need
updating as providers release new versions (edit `MODEL_CHOICES` in `config.py`).

---

## Output

All outputs are written to `output/`:

- `{subject_id}_fhir_bundle.json` — one FHIR R4 Bundle per subject containing their Patient, Observations, and DiagnosticReports
- `mapping_report.json` — summary of every variable → LOINC/SNOMED mapping with confidence levels and validation results

---

## First Run vs Subsequent Runs

On the **first run**, the sentence-transformer model (`all-MiniLM-L6-v2`, ~90MB) is downloaded and LOINC/SNOMED terms are embedded into a local ChromaDB vector store. This takes a few seconds.

On **subsequent runs**, the vector store is loaded instantly from disk only the Claude API calls take time.

---


## Dependencies

| Package | Purpose |
|---|---|
| `anthropic` | Claude API for LLM code selection |
| `langgraph` | Multi-agent pipeline orchestration |
| `chromadb` | Local vector store for terminology search |
| `sentence-transformers` | Embedding model for semantic search |
| `fhir.resources` | Pydantic-validated FHIR R4 resource models |
| `pandas` | CSV parsing |
| `rich` | Console output formatting |
