"""
Node 3 — FHIRBuilder

Reads each cluster's data CSV, applies code_mappings from the CodeMapper
agent, and produces validated FHIR R4 resources using the fhir.resources
Pydantic library. Resources are validated at construction time, any
structural violation raises immediately rather than being caught later.

Strategy used:
  wide_lab   → DiagnosticReport + one Observation per column
  timeseries → one Observation per data row (with quantity_kind filter)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from fhir.resources.bundle import Bundle
from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.observation import Observation
from fhir.resources.patient import Patient

from src.fhir.resources import (
    build_bundle,
    build_diagnostic_report,
    build_patient,
    observations_from_timeseries_rows,
    observations_from_wide_row,
    to_dict,
)
from src.state import PipelineState

logger = logging.getLogger(__name__)

# Type alias for any fhir.resources model
FHIRModel = Patient | Observation | DiagnosticReport | Bundle


def build_fhir(state: PipelineState, cfg: Any) -> PipelineState:
    """
    LangGraph node: build validated FHIR R4 resources for all subjects.

    Resources are fhir.resources Pydantic models during construction
    then serialised to plain dicts via to_dict() for storage in state 
    (JSON-serialisable).

    Outputs:
        state["fhir_bundles"]   — {subject_id: Bundle dict (JSON-ready)}
        state["fhir_resources"] — flat list of all individual resource dicts
    """
    registry = cfg.CLUSTER_REGISTRY
    mapping_index = state.get("mapping_index", {})
    max_subjects = state.get("max_subjects")

    # Accumulate per-subject fhir.resources model lists (not dicts yet)
    subject_models: dict[str, list[FHIRModel]] = {}

    for cluster_name in state["clusters"]:
        if cluster_name not in registry:
            continue

        cluster_cfg = registry[cluster_name]
        data_path = cfg.SAMPLE_DIR / cluster_cfg["data_file"]
        if not data_path.exists():
            state["warnings"].append(f"Data file not found: {data_path}")
            continue

        strategy = cluster_cfg.get("strategy", "wide_lab")
        logger.info("FHIRBuilder: processing cluster '%s' (%s)", cluster_name, strategy)

        df = pd.read_csv(data_path)
        id_col = cluster_cfg.get("id_col", "subject_id")

        subjects_in_data = df[id_col].unique().tolist() if id_col in df.columns else []
        if max_subjects:
            subjects_in_data = subjects_in_data[:max_subjects]

        for sid in subjects_in_data:
            if sid not in state.get("subjects", []):
                state.setdefault("subjects", []).append(sid)

        if strategy == "wide_lab":
            _process_wide_lab(
                df=df, subjects=subjects_in_data,
                cluster_name=cluster_name, cluster_cfg=cluster_cfg,
                mapping_index=mapping_index, subject_models=subject_models, cfg=cfg,
            )
        elif strategy == "timeseries":
            _process_timeseries(
                df=df, subjects=subjects_in_data,
                cluster_name=cluster_name, cluster_cfg=cluster_cfg,
                mapping_index=mapping_index, subject_models=subject_models, cfg=cfg,
            )

    # Build Patient + Bundle per subject; serialise to dicts for state storage
    all_resource_dicts: list[dict] = []
    bundle_dicts: dict[str, dict] = {}

    for subject_id, models in subject_models.items():
        patient = build_patient(subject_id, cfg.FHIR_BASE_URL)
        all_models: list[FHIRModel] = [patient] + models

        # build_bundle() returns a validated Bundle model
        bundle = build_bundle(all_models, base_url=cfg.FHIR_BASE_URL)

        # Serialise: exclude_none removes optional fields that are unset
        bundle_dict = to_dict(bundle)
        bundle_dicts[subject_id] = bundle_dict
        all_resource_dicts.extend(to_dict(m) for m in all_models)

    state["fhir_resources"] = all_resource_dicts
    state["fhir_bundles"]   = bundle_dicts

    obs_count = sum(
        1 for r in all_resource_dicts if r.get("resourceType") == "Observation"
    )
    logger.info(
        "FHIRBuilder: %d subjects, %d total resources (%d Observations)",
        len(subject_models), len(all_resource_dicts), obs_count,
    )
    return state


# Strategy implementations

def _process_wide_lab(
    df: pd.DataFrame,
    subjects: list[str],
    cluster_name: str,
    cluster_cfg: dict,
    mapping_index: dict,
    subject_models: dict[str, list[FHIRModel]],
    cfg: Any,
) -> None:
    """Wide-format lab data → DiagnosticReport + Observation models per subject."""
    id_col       = cluster_cfg.get("id_col", "subject_id")
    date_col     = cluster_cfg.get("date_col", "visit_date")
    category_code = cluster_cfg.get("fhir_category", "laboratory")

    for subject_id in subjects:
        subject_rows = df[df[id_col] == subject_id]
        if subject_rows.empty:
            continue

        subject_models.setdefault(subject_id, [])
        all_obs: list[Observation] = []

        # Compute report_id before building observations so we can set partOf
        report_id     = f"dr-{subject_id}-{cluster_name}".replace("_", "-")
        report_ref    = f"{cfg.FHIR_BASE_URL}/DiagnosticReport/{report_id}"

        for _, row in subject_rows.iterrows():
            row_dict   = row.to_dict()
            visit_date = str(row_dict.get(date_col, ""))

            observations = observations_from_wide_row(
                row=row_dict,
                subject_id=subject_id,
                visit_date=visit_date,
                mapping_index=mapping_index,
                cluster=cluster_name,
                category_code=category_code,
                part_of_ref=report_ref,
                base_url=cfg.FHIR_BASE_URL,
            )
            all_obs.extend(observations)

        subject_models[subject_id].extend(all_obs)

        if all_obs:
            first_row      = subject_rows.iloc[0].to_dict()
            effective_date = str(first_row.get(date_col, ""))

            obs_refs = [
                f"{cfg.FHIR_BASE_URL}/Observation/{obs.id}"
                for obs in all_obs
            ]

            report = build_diagnostic_report(
                subject_id=subject_id,
                report_id=report_id,
                effective_date=f"{effective_date}T00:00:00Z",
                observation_refs=obs_refs,
                title=f"{cluster_name.replace('_', ' ').title()} Panel",
                base_url=cfg.FHIR_BASE_URL,
            )
            # Insert report before observations so bundle reads naturally
            subject_models[subject_id].insert(
                subject_models[subject_id].index(all_obs[0]), report
            )


def _process_timeseries(
    df: pd.DataFrame,
    subjects: list[str],
    cluster_name: str,
    cluster_cfg: dict,
    mapping_index: dict,
    subject_models: dict[str, list[FHIRModel]],
    cfg: Any,
) -> None:
    """Long-format timeseries data → Observation models per subject."""
    id_col        = cluster_cfg.get("id_col", "subject_id")
    kind_col      = cluster_cfg.get("kind_col", "quantity_kind")
    value_col     = cluster_cfg.get("value_col", "value")
    unit_col      = cluster_cfg.get("unit_col", "unit")
    timestamp_col = cluster_cfg.get("timestamp_col", "timestamp_utc")
    device_col    = cluster_cfg.get("device_col", "device_id")
    source_col    = cluster_cfg.get("source_col", "source")

    for subject_id in subjects:
        subject_df = df[df[id_col] == subject_id]
        if subject_df.empty:
            continue

        subject_models.setdefault(subject_id, [])

        # Returns list[Observation] — each Pydantic-validated
        observations = observations_from_timeseries_rows(
            rows=subject_df.to_dict("records"),
            subject_id=subject_id,
            mapping_index=mapping_index,
            cluster=cluster_name,
            kind_col=kind_col,
            value_col=value_col,
            unit_col=unit_col,
            timestamp_col=timestamp_col,
            device_col=device_col,
            source_col=source_col,
            base_url=cfg.FHIR_BASE_URL,
        )
        subject_models[subject_id].extend(observations)
