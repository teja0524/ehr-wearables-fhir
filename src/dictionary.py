"""
Unified data-dictionary loader.

Both supported study formats are reduced to a single per-variable metadata shape
so the rest of the pipeline never has to know whether a dataset's dictionary was
a set of MFU CSVs or the single ACE Excel workbook.

Common variable-meta shape (a plain dict):
    {
        "variable":      str,   # column name in the data file
        "label":         str,   # human-readable description
        "units":         str,   # units / value choices
        "type":          str,   # normalised: numeric | integer | categorical | yesno | date | string
        "fhir_resource": str,   # Observation | Condition | DiagnosticReport | Procedure | Patient | ""
        "measure":       str,   # raw MEASURE text (ACE) / "" (MFU)
        "notes":         str,
        "cluster":       str,
    }

Two entry points are used elsewhere:
    load_cluster_variables(dataset_cfg, cluster_cfg, cluster_name)
        → {variable: meta} for one cluster
    read_table(path, dataset_cfg)
        → pandas DataFrame read with the dataset's delimiter
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

# Variable types the RAG code mapper will attempt to map.
RAG_MAPPABLE_TYPES = {"numeric", "integer", "categorical", "yesno"}

# FHIR resource types whose columns carry a measured value (and therefore are
# candidates for terminology mapping). Conditions/Procedures/Patient are built
# deterministically and do not go through the mapper.
MEASURE_RESOURCES = {"Observation", "DiagnosticReport"}


def read_table(path: Path, dataset_cfg: dict) -> pd.DataFrame:
    """Read a data CSV using the dataset's delimiter (default type inference)."""
    return pd.read_csv(path, sep=dataset_cfg.get("delimiter", ","))


def select_form_rows(
    df: pd.DataFrame, cluster_name: str, dict_form: str | None
) -> pd.DataFrame:
    """
    Restrict a shared MFU dictionary to one cluster's rows.

    A single dictionary CSV (e.g. clinical_dictionary) may hold several forms.
    An explicit `dict_form` wins; otherwise fall back to a name match between the
    cluster name and the 'form' column. No 'form' column → the df is unchanged.
    """
    if "form" not in df.columns:
        return df
    if dict_form:
        return df[df["form"] == dict_form]
    match = [f for f in df["form"].unique()
             if cluster_name in str(f) or str(f) in cluster_name]
    return df[df["form"].isin(match)] if match else df


# -
# ACE: single Excel workbook keyed by CATEGORY
# -

def _ace_norm_type(type_str: str, measure: str) -> str:
    """Map ACE TYPE/MEASURE to the pipeline's normalised type vocabulary."""
    t = (type_str or "").strip().lower()
    m = (measure or "").strip().lower()
    if t == "date" or m == "date":
        return "date"
    if t in ("double", "float"):
        return "numeric"
    if t == "int":
        # int + a 0/1 nominal measure is really a yes/no flag
        if "0n1y" in m or m.startswith("qualitative-nominal"):
            return "yesno" if "nominal" in m else "integer"
        return "integer"
    if t == "string":
        return "categorical" if m.startswith("qualitative") else "string"
    return "string"


@functools.lru_cache(maxsize=8)
def _load_ace_workbook(dict_path: str) -> pd.DataFrame:
    """Load and tidy the ACE dictionary workbook once (CATEGORY forward-filled)."""
    df = pd.read_excel(dict_path)
    df["CATEGORY"] = df["CATEGORY"].ffill()
    return df


def _ace_cluster_variables(dataset_cfg: dict, cluster_cfg: dict,
                           cluster_name: str) -> dict[str, dict]:
    dict_path = str(Path(dataset_cfg["data_dir"]) / dataset_cfg["dict_file"])
    wb = _load_ace_workbook(dict_path)
    category = cluster_cfg.get("dict_category", "")
    rows = wb[wb["CATEGORY"].astype(str).str.strip() == category]

    default_resource = cluster_cfg.get("default_resource", "Observation")
    out: dict[str, dict] = {}
    for _, r in rows.iterrows():
        var = str(r.get("VARIABLE NAME", "")).strip()
        if not var:
            continue
        fhir_resource = str(r.get("FHIR_RESOURCE", "")).strip()
        if fhir_resource in ("", "nan", "NaN"):
            fhir_resource = default_resource
        measure = str(r.get("MEASURE", "")).strip()
        out[var] = {
            "variable": var,
            "label": str(r.get("LABEL", var)).strip() or var,
            "units": "" if str(r.get("VALUES", "")).strip().lower() in ("na", "nan", "") else str(r.get("VALUES", "")).strip(),
            "type": _ace_norm_type(str(r.get("TYPE", "")), measure),
            "fhir_resource": fhir_resource,
            "measure": measure,
            "notes": str(r.get("COMMENTS", "")).strip(),
            "cluster": cluster_name,
        }
    return out


# -
# MFU: per-cluster dictionary CSVs (variable/label/type/units_or_choices/...)
# -

def _mfu_cluster_variables(dataset_cfg: dict, cluster_cfg: dict,
                           cluster_name: str) -> dict[str, dict]:
    dict_path = Path(dataset_cfg["data_dir"]) / cluster_cfg["dict_file"]
    out: dict[str, dict] = {}
    if not dict_path.exists():
        return out
    df = pd.read_csv(dict_path)
    # A shared MFU dictionary (e.g. clinical_dictionary) may hold several forms.
    df = select_form_rows(df, cluster_name, cluster_cfg.get("dict_form"))
    for _, r in df.iterrows():
        var = str(r.get("variable", "")).strip()
        if not var:
            continue
        out[var] = {
            "variable": var,
            "label": str(r.get("label", var)),
            "units": str(r.get("units_or_choices", "")),
            "type": str(r.get("type", "")).strip().lower(),
            "fhir_resource": "Observation",
            "measure": "",
            "notes": str(r.get("notes", "")),
            "cluster": cluster_name,
        }
    return out


def load_cluster_variables(dataset_cfg: dict, cluster_cfg: dict,
                           cluster_name: str) -> dict[str, dict]:
    """Return {variable: meta} for one cluster, regardless of dictionary format."""
    if dataset_cfg.get("dict_format") == "ace_xlsx":
        return _ace_cluster_variables(dataset_cfg, cluster_cfg, cluster_name)
    return _mfu_cluster_variables(dataset_cfg, cluster_cfg, cluster_name)


def is_rag_mappable(meta: dict) -> bool:
    """True if a variable should be sent to the RAG code mapper."""
    return (
        meta.get("fhir_resource", "Observation") in MEASURE_RESOURCES
        and meta.get("type") in RAG_MAPPABLE_TYPES
    )
