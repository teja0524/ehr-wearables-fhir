"""
Configuration for the transformation pipeline, including paths, API keys, model names, and cluster registry.
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
# Model used by all agents. Haiku keeps the per-run cost low; bump to
# claude-sonnet-4-6 (or claude-opus-4-6) if mapping quality underperforms.
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

# Vector store
# Sentence-transformers model for embedding terminology terms
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_LOINC: str = "loinc_terms"
CHROMA_COLLECTION_SNOMED: str = "snomed_terms"
CHROMA_COLLECTION_RXNORM: str = "rxnorm_terms"
# no.of candidates to retrieve per query before Claude re-ranks
VECTOR_TOP_K: int = 5

# FHIR
FHIR_VERSION: str = "4.0.1"
# Base URL used in FHIR resource identifiers (can be changed to actual while deploying)
FHIR_BASE_URL: str = "http://comfortage.example.org/fhir"

# Code system URIs
SYSTEM_LOINC: str = "http://loinc.org"
SYSTEM_SNOMED: str = "http://snomed.info/sct"
SYSTEM_RXNORM: str = "http://www.nlm.nih.gov/research/umls/rxnorm"
SYSTEM_NCBI_TAXON: str = "http://www.ncbi.nlm.nih.gov/taxonomy"
# Local project code system for study-specific instruments and indices that
# have no standard LOINC/SNOMED/RxNorm representation (questionnaire totals,
# dental indices, microbiome metrics, EEG band powers).
SYSTEM_LOCAL: str = f"{FHIR_BASE_URL}/CodeSystem/comfortage-local"

# RxNorm seed used as a deterministic ingredient lookup for medications.
RXNORM_SEED: Path = TERMINOLOGY_DIR / "rxnorm_seed.csv"

# Cluster registry
# Describes every supported data cluster.
#
# 'strategy' drives which FHIR builder to call:
#   'wide_lab'             → one-row-per-visit wide table → DiagnosticReport + Observations
#   'timeseries'          → long table with quantity_kind col → Observation per row
#   'patient'             → demographics → Patient resource (+ AllergyIntolerance)
#   'cohort'              → study group → Group membership Observation
#   'encounter'          → visits → Encounter resource per row
#   'condition'          → diagnoses → Condition resource (SNOMED code in data)
#   'medication_request' → medications → MedicationRequest (RxNorm ingredient lookup)
#   'social_history'     → lifestyle → social-history Observations
#   'survey'             → questionnaire / dental score tables → survey Observations
#   'microbiota_diversity' → diversity indices → Observations
#   'microbiota_abundance' → per-taxon relative abundance → Observations (NCBI taxonomy)
#   'eeg'                → EEG PSD/connectivity → DiagnosticReport + band-power Observations
#
# 'mapping' drives how codes are obtained in the CodeMapper node:
#   'rag_loinc' → vector search + Claude RAG over the LOINC store
#   'in_data'   → standard code already present in the data file (no mapping)
#   'rxnorm'    → deterministic RxNorm ingredient lookup in the builder
#   'local'     → deterministic local/curated codes built in the builder
ALL_DATA_CLUSTERS = [
    "blood_labs", "blood_cytokines", "wearables",
    "demographics", "cohort", "visits",
    "clinical_vitals", "clinical_diagnoses", "clinical_medications", "lifestyle",
    "questionnaires", "dental_clinical", "dental_fdi", "dental_ohip",
    "microbiota_diversity", "microbiota_abundance", "eeg",
]

CLUSTER_REGISTRY = {
    "blood_labs": {
        "data_file": "blood/blood_labs.csv",
        "dict_file": "blood/blood_dictionary.csv",
        "strategy": "wide_lab", "mapping": "rag_loinc",
        "fhir_category": "laboratory",
        "id_col": "subject_id", "visit_col": "visit_id", "date_col": "visit_date",
    },
    "blood_cytokines": {
        "data_file": "blood/blood_cytokines.csv",
        "dict_file": "blood/blood_dictionary.csv",
        "strategy": "wide_lab", "mapping": "rag_loinc",
        "fhir_category": "laboratory",
        "id_col": "subject_id", "visit_col": "visit_id", "date_col": "visit_date",
    },
    "wearables": {
        "data_file": "wearables/wearables_observations.csv",
        "dict_file": "wearables/wearables_dictionary.csv",
        "strategy": "timeseries", "mapping": "rag_loinc",
        "fhir_category": "activity",
        "id_col": "subject_id", "timestamp_col": "timestamp_utc",
        "kind_col": "quantity_kind", "value_col": "value", "unit_col": "unit",
        "device_col": "device_id", "source_col": "source",
    },
    "clinical_vitals": {
        "data_file": "clinical/vitals.csv",
        "dict_file": "clinical/clinical_dictionary.csv",
        "strategy": "wide_lab", "mapping": "rag_loinc",
        "fhir_category": "vital-signs",
        "id_col": "subject_id", "visit_col": "visit_id", "date_col": "visit_date",
        "dict_form": "vitals", "no_report": True,
    },
    "demographics": {
        "data_file": "demographics/demographics.csv",
        "dict_file": "demographics/demographics_dictionary.csv",
        "strategy": "patient", "mapping": "local",
        "id_col": "subject_id", "date_col": "visit_date",
    },
    "cohort": {
        "data_file": "cohort/cohort.csv",
        "dict_file": "cohort/cohort_dictionary.csv",
        "strategy": "cohort", "mapping": "local",
        "id_col": "subject_id",
    },
    "visits": {
        "data_file": "cohort/visits.csv",
        "dict_file": "cohort/cohort_dictionary.csv",
        "strategy": "encounter", "mapping": "local",
        "id_col": "subject_id", "visit_col": "visit_id", "date_col": "visit_date",
    },
    "clinical_diagnoses": {
        "data_file": "clinical/diagnoses.csv",
        "dict_file": "clinical/clinical_dictionary.csv",
        "strategy": "condition", "mapping": "in_data",
        "id_col": "subject_id", "date_col": "visit_date",
        "code_col": "snomed_id", "display_col": "label_en",
    },
    "clinical_medications": {
        "data_file": "clinical/medications.csv",
        "dict_file": "clinical/clinical_dictionary.csv",
        "strategy": "medication_request", "mapping": "rag_rxnorm",
        "id_col": "subject_id", "date_col": "visit_date",
        "drug_col": "drug_name", "dose_col": "dose", "freq_col": "freq",
    },
    "lifestyle": {
        "data_file": "clinical/lifestyle.csv",
        "dict_file": "clinical/clinical_dictionary.csv",
        "strategy": "social_history", "mapping": "rag_clinical",
        "id_col": "subject_id", "date_col": "visit_date",
        "dict_form": "lifestyle", "local_fallback": True,
    },
    "questionnaires": {
        "data_file": "questionnaires/questionnaire_totals.csv",
        "dict_file": "questionnaires/questionnaire_dictionary.csv",
        "strategy": "wide_lab", "mapping": "rag_clinical",
        "id_col": "subject_id", "date_col": "visit_date",
        "fhir_category": "survey",
        "no_report": True, "local_fallback": True,
    },
    "dental_clinical": {
        "data_file": "dental/dental_clinical.csv",
        "dict_file": "dental/dental_dictionary.csv",
        "strategy": "wide_lab", "mapping": "rag_clinical",
        "id_col": "subject_id", "date_col": "visit_date",
        "fhir_category": "exam", "dict_form": "dental_clinical",
        "no_report": True, "local_fallback": True,
    },
    "dental_fdi": {
        "data_file": "dental/dental_fdi.csv",
        "dict_file": "dental/dental_dictionary.csv",
        "strategy": "wide_lab", "mapping": "rag_clinical",
        "id_col": "subject_id", "date_col": "visit_date",
        "fhir_category": "survey", "dict_form": "dental_fdi",
        "no_report": True, "local_fallback": True,
    },
    "dental_ohip": {
        "data_file": "dental/dental_ohip.csv",
        "dict_file": "dental/dental_dictionary.csv",
        "strategy": "wide_lab", "mapping": "rag_clinical",
        "id_col": "subject_id", "date_col": "visit_date",
        "fhir_category": "survey", "dict_form": "dental_ohip",
        "no_report": True, "local_fallback": True,
    },
    "microbiota_diversity": {
        "data_file": "microbiota/diversity.csv",
        "dict_file": "microbiota/microbiota_dictionary.csv",
        "strategy": "wide_lab", "mapping": "rag_clinical",
        "id_col": "subject_id", "date_col": "visit_date",
        "fhir_category": "laboratory", "dict_form": "diversity",
        "no_report": True, "local_fallback": True,
    },
    "microbiota_abundance": {
        "data_file": "microbiota/abundance.csv",
        "dict_file": "microbiota/microbiota_dictionary.csv",
        "strategy": "microbiota_abundance", "mapping": "local",
        "id_col": "subject_id", "date_col": "visit_date",
        "taxon_col": "taxon_id", "taxon_name_col": "taxon_name",
        "value_col": "rel_abund_pct",
    },
    "eeg": {
        "data_file": "eeg/eeg_psd.csv",
        "dict_file": "eeg/eeg_dictionary.csv",
        "strategy": "eeg", "mapping": "local",
        "id_col": "subject_id",
        "connectivity_file": "eeg/eeg_connectivity.csv",
        "metadata_file": "eeg/eeg_metadata.csv",
    },
}

# Mapping modes whose codes come from the LLM/RAG mapping node, and which
# terminologies each one searches.
#   rag_loinc   → LOINC only
#   rag_rxnorm  → RxNorm only (medications)
#   rag_clinical→ LOINC + SNOMED (instruments, findings, social history)
RAG_MAPPINGS = {"rag_loinc", "rag_rxnorm", "rag_clinical"}
MAPPING_VOCABS = {
    "rag_loinc": ["loinc"],
    "rag_rxnorm": ["rxnorm"],
    "rag_clinical": ["loinc", "snomed"],
}
