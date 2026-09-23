"""
Node 1 — SchemaParser

Reads each cluster's dictionary CSV and extracts VariableMeta records
for all measurable variables. Uses deterministic CSV parsing. No LLM.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src import dictionary
from src.state import STRUCTURAL_COLUMNS, PipelineState, VariableMeta

logger = logging.getLogger(__name__)

# Variable types we want to map (skip string ID columns, date/visit keys, etc.)
MAPPABLE_TYPES = dictionary.RAG_MAPPABLE_TYPES


def parse_schema(state: PipelineState, cfg: Any) -> PipelineState:
    """
    LangGraph node: parse all cluster dictionary CSVs into VariableMeta records.
    Timeseries clusters are expanded by unique quantity_kind from the data file.
    """
    dataset = state.get("dataset") or _default_dataset(cfg)
    registry = dataset["clusters"]
    data_dir = dataset["data_dir"]
    rag_mappings = getattr(cfg, "RAG_MAPPINGS", {"rag_loinc"})
    all_meta: list[VariableMeta] = []

    for cluster_name in state["clusters"]:
        if cluster_name not in registry:
            state["errors"].append(f"Cluster '{cluster_name}' not found in dataset registry")
            continue

        cluster_cfg = registry[cluster_name]

        # Only clusters that rely on LLM/RAG code mapping are parsed here.
        # Clusters with codes already in the data, RxNorm lookups, or local
        # study-specific codes are handled deterministically by their builders.
        if cluster_cfg.get("mapping", "rag_loinc") not in rag_mappings:
            logger.info("Skipping schema parse for '%s' (deterministic mapping)", cluster_name)
            continue

        strategy = cluster_cfg.get("strategy", "wide_lab")
        logger.info("Parsing schema for cluster '%s'", cluster_name)

        try:
            if strategy == "timeseries":
                dict_path = data_dir / cluster_cfg["dict_file"]
                meta_list = _parse_timeseries_cluster(cluster_name, cluster_cfg, dict_path, cfg, dataset)
            elif strategy == "medication_request":
                meta_list = _parse_medication_cluster(cluster_name, cluster_cfg, cfg, dataset)
            elif dataset.get("dict_format") == "ace_xlsx" or strategy == "ace_table":
                meta_list = _parse_dictionary_cluster(cluster_name, cluster_cfg, dataset)
            else:
                meta_list = _parse_wide_cluster(
                    cluster_name, data_dir / cluster_cfg["dict_file"], cluster_cfg.get("dict_form")
                )

            logger.info("  → %d mappable variables found", len(meta_list))
            all_meta.extend(meta_list)

        except Exception as exc:
            state["errors"].append(f"Schema parse failed for '{cluster_name}': {exc}")
            logger.exception("Schema parse error")

    state["variable_metadata"] = all_meta
    logger.info("SchemaParser: %d total variables across %d clusters",
                len(all_meta), len(state["clusters"]))
    return state


def _default_dataset(cfg: Any) -> dict:
    """Fallback dataset config (MFU) when state carries none — backward compat."""
    return cfg.DATASETS[cfg.DEFAULT_DATASET]


def _parse_dictionary_cluster(
    cluster_name: str, cluster_cfg: dict, dataset: dict
) -> list[VariableMeta]:
    """
    Parse a cluster whose variables come from a unified data dictionary
    (the ACE Excel workbook). Only variables that map to a measured FHIR
    resource (Observation / DiagnosticReport) of a mappable type are sent to
    the code mapper; Conditions, Procedures, Patient fields and dates are built
    deterministically by the FHIR builder and skipped here.
    """
    variables = dictionary.load_cluster_variables(dataset, cluster_cfg, cluster_name)
    meta_list: list[VariableMeta] = []
    for var, meta in variables.items():
        if var == dataset.get("id_col"):
            continue
        if not dictionary.is_rag_mappable(meta):
            continue
        meta_list.append(VariableMeta(
            variable=var,
            label=meta["label"],
            units=meta["units"],
            var_type=meta["type"],
            min_val="", max_val="",
            notes=meta["notes"][:200],
            cluster=cluster_name,
        ))
    return meta_list


def _parse_wide_cluster(
    cluster_name: str, dict_path: Path, dict_form: str | None = None
) -> list[VariableMeta]:
    """Parse a wide-format cluster dictionary. Returns one VariableMeta per measurable column."""
    df = pd.read_csv(dict_path)
    # Dictionary may contain rows for multiple forms (e.g. clinical_dictionary
    # holds vitals + lifestyle + diagnoses + medications). Filter to this
    # cluster's form: explicit 'dict_form' wins, else fall back to a name match.
    df = dictionary.select_form_rows(df, cluster_name, dict_form)

    meta_list: list[VariableMeta] = []
    for _, row in df.iterrows():
        var = str(row.get("variable", "")).strip()
        if not var or var in STRUCTURAL_COLUMNS:
            continue
        var_type = str(row.get("type", "")).strip().lower()
        if var_type not in MAPPABLE_TYPES:
            continue

        meta_list.append(VariableMeta(
            variable=var,
            label=str(row.get("label", var)),
            units=str(row.get("units_or_choices", "")),
            var_type=var_type,
            min_val=str(row.get("min", "")),
            max_val=str(row.get("max", "")),
            notes=str(row.get("notes", "")),
            cluster=cluster_name,
        ))

    return meta_list


def _parse_medication_cluster(
    cluster_name: str, cluster_cfg: dict, cfg: Any, dataset: dict
) -> list[VariableMeta]:
    """
    Expand a medications table into one VariableMeta per distinct drug name.
    Each drug is mapped to an RxNorm ingredient by the CodeMapper.
    """
    data_path = dataset["data_dir"] / cluster_cfg["data_file"]
    if not data_path.exists():
        return []
    df = dictionary.read_table(data_path, dataset)
    drug_col = cluster_cfg.get("drug_col", "drug_name")
    if drug_col not in df.columns:
        return []
    meta_list: list[VariableMeta] = []
    for drug in sorted({str(d).strip() for d in df[drug_col].dropna() if str(d).strip()}):
        meta_list.append(VariableMeta(
            variable=drug, label=drug, units="", var_type="medication",
            min_val="", max_val="",
            notes="Medication ingredient; map to the RxNorm ingredient (IN) code.",
            cluster=cluster_name,
        ))
    return meta_list


def _parse_timeseries_cluster(
    cluster_name: str,
    cluster_cfg: dict,
    dict_path: Path,
    cfg: Any,
    dataset: dict,
) -> list[VariableMeta]:
    """
    Parse a timeseries cluster by expanding unique quantity_kind values from the data file.
    Unit and value range are inferred from the data itself.
    """
    data_path = dataset["data_dir"] / cluster_cfg["data_file"]
    if not data_path.exists():
        logger.warning("Data file not found: %s — skipping timeseries expansion", data_path)
        return []

    data_df = dictionary.read_table(data_path, dataset)
    kind_col = cluster_cfg.get("kind_col", "quantity_kind")
    unit_col = cluster_cfg.get("unit_col", "unit")
    value_col = cluster_cfg.get("value_col", "value")

    if kind_col not in data_df.columns:
        return []

    # Load the unit-description mapping from the wearables dictionary
    dict_df = pd.read_csv(dict_path)
    unit_choices_row = dict_df[dict_df["variable"] == unit_col]
    unit_choices_text = ""
    if not unit_choices_row.empty:
        unit_choices_text = str(unit_choices_row.iloc[0].get("units_or_choices", ""))

    kind_notes_row = dict_df[dict_df["variable"] == kind_col]
    kind_notes_text = ""
    if not kind_notes_row.empty:
        kind_notes_text = str(kind_notes_row.iloc[0].get("notes", ""))

    # Curated human-readable descriptions per wearable quantity_kind, used to
    # enrich the vector-search query when the dictionary label is terse.
    KIND_LABELS = {
        "heart_rate": "Heart rate (continuous, hourly average)",
        "heart_rate_resting": "Resting heart rate (daily baseline)",
        "rr_interval": "RR interval (interbeat interval, per-beat)",
        "step_count": "Step count (daily)",
        "active_minutes": "Active minutes per day",
        "calories": "Energy expenditure / calories burned",
        "distance": "Distance walked/travelled",
        "daily_activity_index": "Daily activity index (0–1 dimensionless)",
        "sleep_total": "Total sleep duration",
        "sleep_deep": "Deep (slow-wave) sleep duration",
        "sleep_light": "Light sleep duration",
        "sleep_rem": "REM sleep duration",
        "sleep_awake": "Wakefulness during sleep period",
        "sleep_score": "Sleep quality score (0–100)",
        "rmssd": "RMSSD — overnight HRV (heart rate variability)",
        "breathing_rate": "Breathing / respiratory rate during sleep",
        "ans_charge": "Autonomic nervous system charge score (0–100)",
        "cardio_load": "Cardio load (training stress index)",
        "cardio_load_ratio": "Cardio load ratio (acute/chronic)",
    }

    meta_list: list[VariableMeta] = []
    for kind, group in data_df.groupby(kind_col):
        kind = str(kind)
        # Infer unit from most common value in unit column
        unit = ""
        if unit_col in group.columns:
            unit_counts = group[unit_col].value_counts()
            unit = str(unit_counts.index[0]) if not unit_counts.empty else ""

        # Infer value range from data
        val_min, val_max = "", ""
        if value_col in group.columns:
            numeric_vals = pd.to_numeric(group[value_col], errors="coerce").dropna()
            if not numeric_vals.empty:
                val_min = str(round(numeric_vals.min(), 2))
                val_max = str(round(numeric_vals.max(), 2))

        label = KIND_LABELS.get(kind, kind.replace("_", " ").title())

        meta_list.append(VariableMeta(
            variable=kind,
            label=label,
            units=unit,
            var_type="numeric",
            min_val=val_min,
            max_val=val_max,
            notes=f"Wearable timeseries quantity. {kind_notes_text[:200]}",
            cluster=cluster_name,
        ))

    return meta_list
