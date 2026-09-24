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
├── server.py                      # Flask dashboard: run the pipeline, view and correct results
├── config.py                      # Paths, model settings, DATASETS + cluster registries
├── requirements.txt
├── Dockerfile, docker-compose.yml # Java + validator jar pre-installed
├── .env.example                   # Copy to .env and add your API key
├── src/
│   ├── graph.py                   # LangGraph pipeline definition
│   ├── state.py                   # Shared pipeline state (carries the active dataset)
│   ├── dictionary.py              # Unified dictionary loader (MFU CSV + ACE xlsx)
│   ├── llm.py                     # Model access: Anthropic native, others via LiteLLM
│   ├── overrides.py               # Persisted manual code corrections, re-applied each run
│   ├── nodes/
│   │   ├── schema_parser.py       # Node 1 — variable population from the dictionaries
│   │   ├── code_mapper.py         # Node 2 — retrieval + selection (on-disk mapping cache)
│   │   ├── fhir_builder.py        # Node 3 — MFU + ACE resource strategies
│   │   └── validator.py           # Node 4 — conformance + terminology checks
│   ├── fhir/
│   │   └── resources.py           # FHIR R4 resource factory
│   ├── validation/
│   │   ├── conformance.py         # Validation 1 — official HL7 validator_cli.jar
│   │   └── terminology.py         # Validation 2 — $validate-code against a tx server
│   └── vector_store/
│       └── store.py               # ChromaDB terminology store
├── tools/
│   ├── umls_seed_builder.py       # Build / extend the terminology seeds from UMLS
│   ├── build_annotation_set.py    # Build the evaluation annotation worksheet
│   ├── evaluate_mapping.py        # Score mappings against the reference standard
│   └── validator_cli.jar          # Downloaded, not committed (~150 MB)
├── viewer/
│   └── fhir_viewer.html           # Dashboard front-end, served by server.py
├── docs/
│   ├── annotation_protocol.md     # How the reference standard was produced
│   └── README_umls_seed.md        # Seed-building notes
│
│   ### Not in version control — see .gitignore and "Data provenance and access"
├── data/
│   ├── sample_mfu/                # MFU source CSVs (comma-delimited)
│   ├── sample_ACE/                # ACE CSVs (semicolon-delimited) + Excel dictionary
│   └── terminology/               # LOINC, SNOMED CT, RxNorm seed CSVs
├── output/                        # Generated FHIR bundles
│   ├── mfu/                       # MFU bundles + mapping_report.json
│   └── ace/                       # ACE bundles + mapping_report.json
├── runs/                          # Archived experiment runs
├── _archive/                      # Superseded outputs
├── evaluation/                    # Annotation worksheets, key, evaluation results
├── overrides/                     # Manual code corrections, written by the dashboard
└── .chromadb/                     # Built vector store
```

---

## Datasets

The pipeline is dataset-driven. A single `DATASETS` registry in `config.py`
describes each study i.e. its data directory, CSV delimiter, subject-ID column,
dictionary format, cluster registry, and output sub-folder. Then the same four
processing nodes handle any of them. Two dataset types:

| Dataset | Data dir | Delimiter | ID column | Dictionary | Output |
|---|---|---|---|---|---|
| `mfu` (default) | `data/sample_mfu/` | `,` | `subject_id` | per-cluster CSVs | `output/mfu/` |
| `ace` | `data/sample_ACE/` | `;` | `faceid` | one Excel workbook | `output/ace/` |


Add a new study by adding one entry to `DATASETS` (and a cluster registry); no
node code changes are required for standard wide/long tables.

### Data provenance and access

**No study data is included in this repository.** Both datasets are contributed by pilot studies of the COMFORTage Horizon
Europe project and are restricted-access: MFU comes from Pilot 7 (Faculty of
Medicine, University of Ljubljana), ACE from Pilot 3 (Ace Alzheimer Center
Barcelona). They contain pseudonymised participant records, including clinical,
genetic and biomarker measurements. Obtaining them requires authorisation from
the contributing pilot; they cannot be redistributed here.

`.gitignore` and `.dockerignore` exclude `data/`, generated bundles (`output/`,
`runs/`, `_archive/`), the vector store (`.chromadb/`), the correction files
(`overrides/`) and the evaluation worksheets (`evaluation/`). If you add a new
data directory or output location, add it to both files before your first commit.

The pipeline itself sends no subject-level data to any external service. The
code mapper receives only variable-level metadata i.e. column name, label, declared
type, units and cluster. Plus, for the two long-format clusters, the set of
distinct metric names and each numeric metric's cohort-level minimum and
maximum. Individual measurements are read only during local resource
construction. The terminology server (Validation 2) receives codes only, never
the values.

Terminology seeds are also excluded, for licensing rather than privacy reasons:
LOINC and SNOMED CT content carries its own terms of use, and SNOMED CT requires
an affiliate licence in most territories. Please do supply the seeds yourself or
regenerate them from UMLS with `--enrich-seed` (see step 4 below).

---

## Getting started (fresh clone)

The repository contains **code only**. Study data and terminology seeds are
excluded by `.gitignore`, so a clone cannot run the pipeline until you supply
them. What you need, and where to get it:

| Prerequisite | In the repo? | How to obtain |
|---|---|---|
| Python deps | via `requirements.txt` | `pip install -r requirements.txt` |
| `.env` with API key | no | `cp .env.example .env`, then edit |
| **Terminology seeds** (`data/terminology/*_seed.csv`) | **no** | Regenerate from UMLS (below), or request from the author |
| **Study data** (`data/sample_mfu/`, `data/sample_ACE/`) | **no** | Must be obtained separately; contains study data |
| FHIR validator jar | no | One `curl` (below); optional, pipeline degrades gracefully |

Terminology seeds are excluded deliberately: SNOMED CT redistribution requires
an affiliate licence, so each user should obtain terminology content under their
own licence rather than receiving a copy.

```bash
# 1. environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. configuration
cp .env.example .env          # then edit: ANTHROPIC_API_KEY (required)

# 3. FHIR validator for conformance checking (optional but recommended)
mkdir -p tools && curl -sSL -o tools/validator_cli.jar \
  https://github.com/hapifhir/org.hl7.fhir.core/releases/latest/download/validator_cli.jar

# 4. terminology seeds — either place the CSVs in data/terminology/,
#    or regenerate them from UMLS (needs a free UTS account + UMLS_API_KEY):
python main.py --enrich-seed          # fetch candidates for review
#    review data/terminology/seed_candidates/umls_seed_candidates.csv (set accept=Y)
python main.py --apply-seed           # merge accepted rows into the seeds

# 5. place study data under data/sample_mfu/ and/or data/sample_ACE/
#    (see Project Structure and the Datasets table for the expected layout)
#    NOTE: study data is restricted-access and is NOT shipped with this repo —
#    see "Data provenance and access" above.

# 6. run
python main.py --rebuild-store --dataset mfu --clusters all --max-subjects 20
```

Alternatively use Docker (Java and the validator jar are pre-installed) — see
the Docker section. Data and seeds are still required.

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

### Configuration

`.env` holds credentials (`ANTHROPIC_API_KEY`, optional `UMLS_API_KEY` and other
provider keys). Behaviour is configured in `config.py`, and every setting below
can be overridden by an environment variable of the same name:

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Model used by the code mapper |
| `FHIR_BASE_URL` | `https://comfortage.eu/fhir` | Identifier namespace for generated resources |
| `PORT` / `HOST` | `8000` / `127.0.0.1` | Dashboard server bind address |
| `VECTOR_TOP_K` | `5` | Candidates retrieved per query before the model re-ranks |
| `NO_RAG` | `0` | Retrieval ablation — see below. `1` disables vector search entirely |

Validation settings (`USE_OFFICIAL_VALIDATOR`, `VALIDATOR_MAX_BUNDLES`,
`TX_SERVER_URL`, `VALIDATOR_TIMEOUT_SEC`, `TX_TIMEOUT_SEC`, …) are documented in
the [Validation](#validation) section.

> **`NO_RAG=1` is an experiment, not an operating mode.** It makes the code
> mapper skip the vector store and ask the model to recall a code from its own
> parametric knowledge, with no shortlist supplied. It exists to measure what
> retrieval contributes, and it removes the guarantee that an emitted code
> exists. Leave it unset for any real run:
> ```bash
> NO_RAG=1 python main.py --dataset mfu --clusters all --mapping-only
> ```

> **`FHIR_BASE_URL` is an identifier, not a link.** Nothing dereferences it — in
> the same way `http://loinc.org` names LOINC without serving anything. It must
> be globally unique and owned by your project, and must **not** be an
> `example.org`/`example.com` address: the official HL7 validator rejects
> reserved example domains, since anyone could claim them.

**4. Copy .env.example and set up your Anthropic API key:**
```bash
cp .env.example .env
```
Open `.env` and replace the placeholder with your key:
```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Docker

The project ships with a `Dockerfile` and `docker-compose.yml`. The image bundles
the small terminology seeds (`data/terminology/`) so the vector store builds on
first run; the bulky sample datasets and your API key are **not** baked in.

**1. Set your API key** (compose reads it from `.env`):
```bash
cp .env.example .env      # then edit ANTHROPIC_API_KEY=sk-ant-...
```

**2. Build and start the dashboard:**
```bash
docker compose up --build   # dashboard → http://127.0.0.1:8000
```
The container binds to `0.0.0.0` internally (via `HOST`) and publishes port 8000
to your machine. `.chromadb/`, `output/`, `overrides/`, and `data/` are mounted
as volumes, so the built vector store and generated bundles persist on the host.

**Run the CLI instead of the dashboard** (override the default command):
```bash
docker compose run --rm dashboard python main.py --dataset ace --clusters all --max-subjects 25
```

**Plain `docker` (without compose):**
```bash
docker build -t fhir-pipeline .
docker run --rm -p 8000:8000 --env-file .env \
  -v "$PWD/.chromadb:/app/.chromadb" -v "$PWD/output:/app/output" \
  fhir-pipeline
```

> **Note on data:** study data and terminology seeds are git-ignored, so a fresh
> clone won't contain them. The Docker build copies whatever is in your local
> `data/terminology/` at build time. If those seed CSVs are missing, regenerate
> them with the UMLS seed builder (see `tools/README_umls_seed.md`) before
> building.
>
> On **Linux hosts**, bind-mounted volumes are written as the container's
> `appuser` (uid 1000); if you hit permission errors on `output/` or `.chromadb/`,
> either `chown` those dirs to uid 1000 or switch them to named volumes. On
> macOS/Windows (Docker Desktop) this is handled automatically.

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

## Validation

Quality is assessed at three levels, which answer different questions and are
not substitutes for one another. The first two are **validation** — automated
checks against an external specification, run as Node 4 on every pipeline run.
The third is **evaluation** — an offline research study measuring how well the
mapping performs, which requires human judgement and therefore cannot run
inside the pipeline.

| | Question | How | Where |
|---|---|---|---|
| **Validation 1** — Conformance | Is this legal FHIR R4? | Official HL7 `validator_cli.jar` + profiles declared in `meta.profile` | `src/validation/conformance.py` (automated) |
| **Validation 2** — Terminology | Do these codes exist? | `$validate-code` against a terminology server | `src/validation/terminology.py` (automated) |
| **Evaluation** — Semantic accuracy | Is this the *right* code for the variable? | Comparison against a reference standard | `tools/evaluate_mapping.py` (offline study) |

The distinction matters: a resource can be perfectly conformant (Validation 1)
and carry a real code (Validation 2) that is nonetheless the wrong concept for
the variable. Only the evaluation can detect that.

Validation results are written into `mapping_report.json` under `conformance`
and `terminology`, so conformance rate and code-resolution rate can be reported
directly. Evaluation results are written separately to
`evaluation/evaluation_results.json`.

**Validation 1 — conformance.** Uses the HL7 reference implementation rather than
hand-written checks, so the conformance rate is authoritative. Profiles a
resource declares in `meta.profile` (e.g. `vitalsigns`) are verified
automatically. Requires Java 11+ and the validator jar:

```bash
mkdir -p tools && curl -sSL -o tools/validator_cli.jar \
  https://github.com/hapifhir/org.hl7.fhir.core/releases/latest/download/validator_cli.jar
```

If Java or the jar is missing, the node logs a warning and falls back to the
built-in structural checks — the pipeline never hard-fails, and the report
records which validator produced the result. In Docker both are pre-installed.

To validate against **HL7 Europe base** profiles (EHDS context) as a gap
analysis, set:
```bash
VALIDATOR_IG_PACKAGES=hl7.fhir.eu.base#2.0.0
```

**Validation 2 — terminology.** Every distinct code is checked once against
`tx.fhir.org` and cached in `output/terminology_cache.json`. Project-local codes
are reported separately and excluded from the resolution rate (no server can
know them). If the server is unreachable, codes are marked `unchecked` — never
`invalid` — so network failures cannot understate coverage.

**Relevant settings** (all overridable via `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `USE_OFFICIAL_VALIDATOR` | `true` | Use the HL7 validator for Validation 1 |
| `FHIR_VALIDATOR_JAR` | `tools/validator_cli.jar` | Path to the jar |
| `VALIDATOR_MAX_BUNDLES` | `25` | Bundles validated per run (`0` = all) |
| `VALIDATOR_IG_PACKAGES` | *(empty)* | Extra IG packages, comma-separated |
| `VALIDATOR_TIMEOUT_SEC` | `900` | Per-invocation timeout for the HL7 validator |
| `CHECK_TERMINOLOGY` | `true` | Enable Validation 2 |
| `TX_SERVER_URL` | `https://tx.fhir.org/r4` | Terminology server |
| `TX_TIMEOUT_SEC` | `15` | Per-request timeout for the terminology server |

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
| `langgraph` | Pipeline orchestration (state graph) |
| `chromadb` | Local vector store for terminology search |
| `sentence-transformers` | Embedding model for semantic search |
| `fhir.resources` | Pydantic-validated FHIR R4 resource models |
| `pandas` | CSV parsing |
| `rich` | Console output formatting |

---

## Licence

Source code is released under the MIT Licence (see `LICENSE`). The licence covers
the code only — not the study data, which is restricted-access and not
distributed here, and not LOINC / SNOMED CT / RxNorm content, which carries its
own terms of use. See "Data provenance and access" above.
