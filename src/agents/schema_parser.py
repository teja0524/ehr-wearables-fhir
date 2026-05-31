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

from src.state import PipelineState, VariableMeta

logger = logging.getLogger(__name__)

# Variable types we want to map (skip string ID columns, date/visit keys, etc.)
MAPPABLE_TYPES = {"numeric", "integer", "yesno", "categorical"}

# Columns that are structural keys, never clinical measurements
STRUCTURAL_VARS = {
    "subject_id", "visit_id", "visit_date", "timestamp_utc",
    "device_id", "source", "quantity_kind", "value", "unit",
}


def parse_schema(state: PipelineState, cfg: Any) -> PipelineState:
    """
    LangGraph node: parse all cluster dictionary CSVs into VariableMeta records.
    Timeseries clusters are expanded by unique quantity_kind from the data file.
    """
    registry = cfg.CLUSTER_REGISTRY
    all_meta: list[VariableMeta] = []

    for cluster_name in state["clusters"]:
        if cluster_name not in registry:
            state["errors"].append(f"Cluster '{cluster_name}' not found in CLUSTER_REGISTRY")
            continue

        cluster_cfg = registry[cluster_name]
        dict_path = cfg.SAMPLE_DIR / cluster_cfg["dict_file"]
        strategy = cluster_cfg.get("strategy", "wide_lab")

        if not dict_path.exists():
            state["errors"].append(f"Dictionary file not found: {dict_path}")
            continue

        logger.info("Parsing schema for cluster '%s' from %s", cluster_name, dict_path.name)

        try:
            if strategy == "timeseries":
                meta_list = _parse_timeseries_cluster(cluster_name, cluster_cfg, dict_path, cfg)
            else:
                meta_list = _parse_wide_cluster(cluster_name, dict_path)

            logger.info("  → %d mappable variables found", len(meta_list))
            all_meta.extend(meta_list)

        except Exception as exc:
            state["errors"].append(f"Schema parse failed for '{cluster_name}': {exc}")
            logger.exception("Schema parse error")

    state["variable_metadata"] = all_meta
    logger.info("SchemaParser: %d total variables across %d clusters",
                len(all_meta), len(state["clusters"]))
    return state


def _parse_wide_cluster(cluster_name: str, dict_path: Path) -> list[VariableMeta]:
    """Parse a wide-format cluster dictionary. Returns one VariableMeta per measurable column."""
    df = pd.read_csv(dict_path)
    # Dictionary may contain rows for multiple forms (blood_labs + blood_cytokines)
    # Filter to rows that belong to this cluster's form
    if "form" in df.columns:
        # Try to match on cluster_name (strip trailing _b suffixes etc.)
        forms_in_dict = df["form"].unique()
        # Find the best-matching form
        matching_forms = [f for f in forms_in_dict if cluster_name in f or f in cluster_name]
        if matching_forms:
            df = df[df["form"].isin(matching_forms)]

    meta_list: list[VariableMeta] = []
    for _, row in df.iterrows():
        var = str(row.get("variable", "")).strip()
        if not var or var in STRUCTURAL_VARS:
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


def _parse_timeseries_cluster(
    cluster_name: str,
    cluster_cfg: dict,
    dict_path: Path,
    cfg: Any,
) -> list[VariableMeta]:
    """
    Parse a timeseries cluster by expanding unique quantity_kind values from the data file.
    Unit and value range are inferred from the data itself.
    """
    data_path = cfg.SAMPLE_DIR / cluster_cfg["data_file"]
    if not data_path.exists():
        logger.warning("Data file not found: %s — skipping timeseries expansion", data_path)
        return []

    data_df = pd.read_csv(data_path)
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

    # Human-readable descriptions per quantity_kind (built from the domain knowledge of Claude)
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
