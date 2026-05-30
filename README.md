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
├── main.py                        # Entry point
├── config.py                      # Paths, model settings, cluster registry
├── data/
│   ├── sample/                    # Source CSVs (EHR and wearables)
│   └── terminology/               # LOINC and SNOMED seed CSVs
├── src/
│   ├── graph.py                   # LangGraph pipeline definition
│   ├── state.py                   # Shared pipeline states
│   ├── agents/
│   │   ├── schema_parser.py       # Node 1
│   │   ├── code_mapper.py         # Node 2
│   │   ├── fhir_builder.py        # Node 3
│   │   └── validator.py           # Node 4
│   ├── fhir/
│   │   └── resources.py           # FHIR R4 resource factory
│   └── vector_store/
│       └── store.py               # ChromaDB terminology store
└── output/                        # Generated FHIR bundles (git-ignored)
```

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

**For a quick demo** - first 3 subjects only:
```bash
python main.py
```

**All subjects, all clusters:**
```bash
python main.py --max-subjects 0
```

**Skip LLM mapping** (uses cached results from a previous run — much faster):
```bash
python main.py --skip-mapping
```

**Rebuild the ChromaDB vector store from scratch:**
```bash
python main.py --rebuild-store
```

**Verbose logging:**
```bash
python main.py --verbose
```

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
