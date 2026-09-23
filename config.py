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
SAMPLE_MFU_DIR = DATA_DIR / "sample_mfu"    # MFU/ComfortAge sample data
SAMPLE_DIR = SAMPLE_MFU_DIR                 # backwards-compatible alias
SAMPLE_ACE_DIR = DATA_DIR / "sample_ACE"    # ACE study data
OUTPUT_DIR = BASE_DIR / "output"
OVERRIDES_DIR = BASE_DIR / "overrides"  # technician code overrides (per dataset)
CHROMA_DIR = BASE_DIR / ".chromadb"   # persisted vector store

# Anthropic / Claude
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
# Model used by the code-mapping node. Haiku keeps the per-run cost low; bump to
# claude-sonnet-4-6 (or claude-opus-4-8) if mapping quality underperforms.
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

# Models offered in the dashboard / --model flag. Anthropic uses the native SDK;
# all others route through LiteLLM (see src/llm.py) and need that provider's API
# key in .env. Model ids for non-Anthropic providers follow LiteLLM's naming and
# may need updating as providers release new versions.
#   {"label": shown in UI, "id": model id, "provider": ..., "key": env var needed}
MODEL_CHOICES = [
    {"label": "Claude Haiku (fast, cheap)", "id": "claude-haiku-4-5-20251001", "provider": "Anthropic", "key": "ANTHROPIC_API_KEY"},
    {"label": "Claude Sonnet", "id": "claude-sonnet-4-6", "provider": "Anthropic", "key": "ANTHROPIC_API_KEY"},
    {"label": "Claude Opus (best)", "id": "claude-opus-4-8", "provider": "Anthropic", "key": "ANTHROPIC_API_KEY"},
    {"label": "OpenAI GPT-4o", "id": "gpt-4o", "provider": "OpenAI", "key": "OPENAI_API_KEY"},
    {"label": "OpenAI GPT-4o mini", "id": "gpt-4o-mini", "provider": "OpenAI", "key": "OPENAI_API_KEY"},
    {"label": "Google Gemini 1.5 Pro", "id": "gemini/gemini-1.5-pro", "provider": "Google", "key": "GEMINI_API_KEY"},
    {"label": "Google Gemini 1.5 Flash", "id": "gemini/gemini-1.5-flash", "provider": "Google", "key": "GEMINI_API_KEY"},
    {"label": "Meta Llama 3.3 70B (Groq)", "id": "groq/llama-3.3-70b-versatile", "provider": "Meta/Groq", "key": "GROQ_API_KEY"},
]

# Vector store
# Sentence-transformers model for embedding terminology terms
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_LOINC: str = "loinc_terms"
CHROMA_COLLECTION_SNOMED: str = "snomed_terms"
CHROMA_COLLECTION_RXNORM: str = "rxnorm_terms"
# no.of candidates to retrieve per query before Claude re-ranks
VECTOR_TOP_K: int = int(os.getenv("VECTOR_TOP_K", "5"))

# Retrieval ablation. When NO_RAG=1 the code-mapping node skips vector search
# altogether and asks the model to recall a code from its own parametric
# knowledge, with no shortlist supplied. This exists to measure what retrieval
# contributes; it is not a supported operating mode. Defaults off, so an
# unset environment reproduces the standard pipeline exactly.
NO_RAG: bool = os.getenv("NO_RAG", "0") == "1"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
# Validation 1 — conformance via the official HL7 validator (the reference
# implementation). Falls back to the built-in structural checks when Java or the
# jar is missing, so the pipeline still runs on a bare machine.
#   Download: https://github.com/hapifhir/org.hl7.fhir.core/releases/latest/download/validator_cli.jar
USE_OFFICIAL_VALIDATOR: bool = os.getenv("USE_OFFICIAL_VALIDATOR", "true").lower() == "true"
FHIR_VALIDATOR_JAR: Path = Path(
    os.getenv("FHIR_VALIDATOR_JAR", str(BASE_DIR / "tools" / "validator_cli.jar"))
)
# Bundles validated per run (JVM cost is per-run, but large cohorts still add
# up). 0 = validate every bundle. The conformance rate is reported over exactly
# the bundles checked.
VALIDATOR_MAX_BUNDLES: int = int(os.getenv("VALIDATOR_MAX_BUNDLES", "25"))
VALIDATOR_TIMEOUT_SEC: int = int(os.getenv("VALIDATOR_TIMEOUT_SEC", "900"))
# Terminology is checked in Validation 2, so the validator's own tx lookups are off.
VALIDATOR_DISABLE_TX: bool = True
# Implementation Guide packages to validate against, in addition to base R4.
# Profiles declared in meta.profile (e.g. vitalsigns) are always checked without
# needing an entry here. For the European/EHDS context, add the HL7 Europe base
# package below to produce a gap analysis against EU profiles:
#   VALIDATOR_IG_PACKAGES = ["hl7.fhir.eu.base#2.0.0"]
# Kept empty by default: resources are not yet built to satisfy the EU profiles,
# so enabling it reports conformance gaps rather than passes.
VALIDATOR_IG_PACKAGES: list[str] = [
    ig for ig in os.getenv("VALIDATOR_IG_PACKAGES", "").split(",") if ig.strip()
]

# Validation 2 — terminology validation ($validate-code on a terminology server).
CHECK_TERMINOLOGY: bool = os.getenv("CHECK_TERMINOLOGY", "true").lower() == "true"
TX_SERVER_URL: str = os.getenv("TX_SERVER_URL", "https://tx.fhir.org/r4")
TX_TIMEOUT_SEC: int = int(os.getenv("TX_TIMEOUT_SEC", "15"))

# FHIR
FHIR_VERSION: str = "4.0.1"
# Namespace URI for identifiers, references and the local CodeSystem.
#
# This is an IDENTIFIER, not a link — nothing dereferences it (the same way
# "http://loinc.org" names LOINC without serving anything). It only has to be
# globally unique and owned by the project, so a path under the study's real
# domain is the conventional choice.
#
# It must NOT be example.org/example.com: the official HL7 validator rejects
# reserved example domains ("Example URLs are not allowed in this context")
# precisely because anyone could claim them, so they identify nothing.
FHIR_BASE_URL: str = os.getenv("FHIR_BASE_URL", "https://comfortage.eu/fhir")

# Code system URIs.
# The standard-vocabulary URIs (LOINC/SNOMED/RxNorm/NCBI taxonomy) live in
# src.fhir.resources (SYS_*); only the two systems referenced directly from
# config are defined here.
SYSTEM_ICD10: str = "http://hl7.org/fhir/sid/icd-10"
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


# ---------------------------------------------------------------------------
# ACE study dataset
# ---------------------------------------------------------------------------
# The ACE data differs structurally from the MFU/ComfortAge sample:
#   * files are semicolon-delimited
#   * the subject identifier column is 'faceid' (not 'subject_id')
#   * a single Excel workbook documents every variable, with an explicit
#     FHIR_RESOURCE column (Patient / Observation / Condition / DiagnosticReport
#     / Procedure) that drives how each column is transformed.
#
# Two ACE-specific builder strategies cover all clusters:
#   'ace_table'   → dictionary-driven; routes each column to an Observation,
#                   Condition, or DiagnosticReport-grouped Observation based on
#                   its FHIR_RESOURCE. Codes for measures come from the same RAG
#                   mapper as MFU (LOINC/SNOMED), with a local-code fallback.
#   'ace_patient' → demographics → Patient (gender, birthDate) plus
#                   education Observations.
#   'ace_procedure' → intervention flag → Procedure.
#
# 'dict_category' selects this cluster's rows from the shared ACE dictionary
# (its CATEGORY column). 'default_resource' is used when a column's
# FHIR_RESOURCE cell is blank in the dictionary.
ACE_DICT_FILE = "ace_v2.dictionary_FTSS.xlsx"

ACE_ALL_CLUSTERS = [
    "ace_demographic", "ace_anthropometric", "ace_csf", "ace_plasma",
    "ace_mri", "ace_genetic", "ace_neurology", "ace_neuropsychology",
    "ace_intervention",
]

ACE_CLUSTER_REGISTRY = {
    "ace_demographic": {
        "data_file": "demographic.csv",
        "strategy": "ace_patient", "mapping": "local",
        "dict_category": "demographic", "id_col": "faceid",
        "sex_col": "sex_0M1F", "birth_col": "date_of_birth",
    },
    "ace_anthropometric": {
        "data_file": "anthropometric.csv",
        "strategy": "ace_table", "mapping": "rag_clinical",
        "dict_category": "anthropometric", "id_col": "faceid",
        "date_col": "anthropometric_date",
        "fhir_category": "vital-signs",
        "default_resource": "Observation",
        "local_fallback": True, "no_report": True,
    },
    "ace_csf": {
        "data_file": "csf.csv",
        "strategy": "ace_table", "mapping": "rag_clinical",
        "dict_category": "csf", "id_col": "faceid", "date_col": "csf_date",
        "fhir_category": "laboratory",
        "default_resource": "Observation",
        "local_fallback": True, "make_report": True,
        "report_title": "CSF Biomarker Panel",
        "report_loinc": ("33717-0", "Cerebrospinal fluid panel"),
    },
    "ace_plasma": {
        "data_file": "plasma.csv",
        "strategy": "ace_table", "mapping": "rag_clinical",
        "dict_category": "plasma", "id_col": "faceid", "date_col": "plasma_date",
        "fhir_category": "laboratory",
        "default_resource": "Observation",
        "local_fallback": True, "make_report": True,
        "report_title": "Plasma Biomarker Panel",
        "report_loinc": ("11502-2", "Laboratory report"),
    },
    "ace_mri": {
        "data_file": "mri.csv",
        "strategy": "ace_table", "mapping": "rag_clinical",
        "dict_category": "mri", "id_col": "faceid", "date_col": "mri_date",
        "fhir_category": "imaging",
        "default_resource": "Observation",
        "local_fallback": True, "make_report": True,
        "report_title": "Brain MRI Volumetric Summary",
        "report_loinc": ("24590-2", "MR Brain"),
    },
    "ace_genetic": {
        "data_file": "genetic.csv",
        "strategy": "ace_table", "mapping": "rag_clinical",
        "dict_category": "genetic", "id_col": "faceid", "date_col": None,
        "fhir_category": "laboratory",
        "default_resource": "Observation",
        "local_fallback": True, "no_report": True,
    },
    "ace_neurology": {
        "data_file": "neurology.csv",
        "strategy": "ace_table", "mapping": "rag_clinical",
        "dict_category": "neurology", "id_col": "faceid",
        "date_col": "neurology_date",
        "fhir_category": "survey",
        "default_resource": "Observation",
        "local_fallback": True, "no_report": True,
    },
    "ace_neuropsychology": {
        "data_file": "neuropsychology.csv",
        "strategy": "ace_table", "mapping": "rag_clinical",
        "dict_category": "neuropsychology", "id_col": "faceid",
        "date_col": "neuropsychology_date",
        "fhir_category": "survey",
        "default_resource": "Observation",
        "local_fallback": True, "no_report": True,
    },
    "ace_intervention": {
        "data_file": "intervention.csv",
        "strategy": "ace_procedure", "mapping": "local",
        "dict_category": "intervention", "id_col": "faceid", "date_col": None,
        "flag_col": "intervention_0N1Y",
    },
}


# ---------------------------------------------------------------------------
# Dataset registry — the single switch the pipeline reads to know how to load
# and route a study's data. Add a new study by adding an entry here.
# ---------------------------------------------------------------------------
DATASETS = {
    "mfu": {
        "name": "mfu",
        "data_dir": SAMPLE_MFU_DIR,
        "delimiter": ",",
        "dict_format": "mfu_csv",   # per-cluster dictionary CSVs
        "dict_file": None,
        "id_col": "subject_id",
        "clusters": CLUSTER_REGISTRY,
        "all_clusters": ALL_DATA_CLUSTERS,
        "output_subdir": "mfu",     # MFU bundles → output/mfu/
    },
    "ace": {
        "name": "ace",
        "data_dir": SAMPLE_ACE_DIR,
        "delimiter": ";",
        "dict_format": "ace_xlsx",  # one shared Excel dictionary
        "dict_file": ACE_DICT_FILE,
        "id_col": "faceid",
        "clusters": ACE_CLUSTER_REGISTRY,
        "all_clusters": ACE_ALL_CLUSTERS,
        "output_subdir": "ace",     # ACE bundles → output/ace/
    },
}

DEFAULT_DATASET = "mfu"
