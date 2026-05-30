"""
Configuration for the EHR/Wearables to FHIR Transformation
Copy .env.example to .env and set ANTHROPIC_API_KEY before running.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TERMINOLOGY_DIR = DATA_DIR / "terminology"
SAMPLE_DIR = DATA_DIR / "sample"
OUTPUT_DIR = BASE_DIR / "output"
CHROMA_DIR = BASE_DIR / ".chromadb"   # persisted vector store

# Anthropic / Claude
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
# Model used by all agents — can be swapped to claude-opus-4-6 is underperformed
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# Vector store
# Sentence-transformers model for embedding terminology terms
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_LOINC: str = "loinc_terms"
CHROMA_COLLECTION_SNOMED: str = "snomed_terms"
# no.of candidates to retrieve per query before Claude re-ranks
VECTOR_TOP_K: int = 5

# FHIR
FHIR_VERSION: str = "4.0.1"
# Base URL used in FHIR resource identifiers (can be changed to actual while deploying)
FHIR_BASE_URL: str = "http://comfortage.example.org/fhir"

# Cluster registry
# Describes every supported data cluster.
# 'strategy' drives which FHIR builder to call:
#   'wide_lab'        → one-row-per-visit wide table → DiagnosticReport + Observations
#   'timeseries'      → long table with quantity_kind col → Observation per row
#   'patient'         → demographics → Patient resource
#   'condition'       → diagnoses → Condition resource
#   'vital_signs'     → vitals → Observation (vital-signs profile)
#   'questionnaire'   → questionnaire totals → QuestionnaireResponse
CLUSTER_REGISTRY = {
    "blood_labs": {
        "data_file": "blood_labs.csv",
        "dict_file": "blood_dictionary.csv",
        "strategy": "wide_lab",
        "fhir_category": "laboratory",
        "id_col": "subject_id",
        "visit_col": "visit_id",
        "date_col": "visit_date",
    },
    "blood_cytokines": {
        "data_file": "blood_cytokines.csv",
        "dict_file": "blood_dictionary.csv",
        "strategy": "wide_lab",
        "fhir_category": "laboratory",
        "id_col": "subject_id",
        "visit_col": "visit_id",
        "date_col": "visit_date",
    },
    "wearables": {
        "data_file": "wearables_observations.csv",
        "dict_file": "wearables_dictionary.csv",
        "strategy": "timeseries",
        "fhir_category": "activity",
        "id_col": "subject_id",
        "timestamp_col": "timestamp_utc",
        "kind_col": "quantity_kind",
        "value_col": "value",
        "unit_col": "unit",
        "device_col": "device_id",
        "source_col": "source",
    },
    # Future clusters to add later:
    # "demographics": {"strategy": "patient", ...},
    # "clinical_vitals": {"strategy": "vital_signs", ...},
    # "clinical_diagnoses": {"strategy": "condition", ...},
    # "clinical_medications": {"strategy": "medication_request", ...},
    # "questionnaires": {"strategy": "questionnaire", ...},
    # "dental": {"strategy": "wide_lab", ...},
    # "microbiota": {"strategy": "wide_lab", ...},
    # "eeg": {"strategy": "wide_lab", ...},
}
