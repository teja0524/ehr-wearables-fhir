"""
Node 3 — FHIRBuilder

Reads each cluster's data CSV and produces validated FHIR R4 (R4B) resources
using the fhir.resources Pydantic library. Resources are validated at
construction time; any structural violation raises immediately.

Strategies:
  wide_lab             → DiagnosticReport + one Observation per column (labs, cytokines, vitals)
  timeseries           → one Observation per data row (wearables)
  patient              → enrich Patient (gender/birthDate) + AllergyIntolerance (demographics)
  cohort               → study-group Observation
  encounter            → one Encounter per visit row
  condition            → one Condition per diagnosis row (SNOMED code in data)
  medication_request   → one MedicationRequest per row (RxNorm ingredient lookup)
  social_history       → social-history Observations (lifestyle)
  survey               → survey/exam Observations (questionnaires, dental scores)
  microbiota_diversity → diversity-index Observations
  microbiota_abundance → per-taxon relative-abundance Observations (NCBI taxonomy)
  eeg                  → DiagnosticReport + band-power summary Observations
"""

from __future__ import annotations

import csv
import logging
from typing import Any

import pandas as pd

from src.fhir.resources import (
    SYS_LOINC, SYS_SNOMED, SYS_NCBI_TAXON,
    build_allergy,
    build_bundle,
    build_condition,
    build_diagnostic_report,
    build_encounter,
    build_medication_request,
    build_observation,
    build_patient,
    observations_from_timeseries_rows,
    observations_from_wide_row,
    to_dict,
)
from src.state import PipelineState

logger = logging.getLogger(__name__)

# Structural columns never emitted as their own Observation.
_STRUCTURAL = {"subject_id", "visit_id", "visit_date", "visit_type",
               "dx_index", "med_index", "snomed_id", "label_en", "taxon_id",
               "taxon_name", "group", "enrollment_date"}

# Curated standard LOINC codes for questionnaire totals that have one.
# Everything else falls back to the local project code system.
_QUESTIONNAIRE_LOINC = {
    "mmse_total": ("72172-0", "Total score [MMSE]"),
    "gds_total":  ("48545-8", "Total score [GDS]"),
}

# SNOMED CT value codes for tobacco smoking status (LOINC 72166-2).
_SMOKING_VALUE = {
    "no":         ("266919005", "Never smoked tobacco"),
    "previously": ("8517006",   "Ex-smoker"),
    "yes":        ("77176002",  "Smoker"),
}

# Drug-name aliases → the ingredient name as it appears in rxnorm_seed.csv.
_RXNORM_ALIASES = {
    "paracetamol": "acetaminophen",
    "salbutamol":  "albuterol",
    "macrogol":    "polyethylene glycol 3350",
}

# EEG frequency bands (Hz) used to aggregate the power spectral density.
_EEG_BANDS = {
    "delta": (3.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 50.0),
}


def build_fhir(state: PipelineState, cfg: Any) -> PipelineState:
    """LangGraph node: build validated FHIR R4 resources for all subjects."""
    registry = cfg.CLUSTER_REGISTRY
    mapping_index = state.get("mapping_index", {})
    max_subjects = state.get("max_subjects")
    local_system = cfg.SYSTEM_LOCAL

    subject_models: dict[str, list[Any]] = {}
    # demographics → Patient enrichment (gender / birthDate) per subject
    subject_demo: dict[str, dict] = {}
    rxnorm_lookup = _load_rxnorm(cfg)

    def register_subjects(df: pd.DataFrame, id_col: str) -> list[str]:
        subs = df[id_col].unique().tolist() if id_col in df.columns else []
        if max_subjects:
            subs = subs[:max_subjects]
        for sid in subs:
            if sid not in state.get("subjects", []):
                state.setdefault("subjects", []).append(sid)
            subject_models.setdefault(sid, [])
        return subs

    for cluster_name in state["clusters"]:
        if cluster_name not in registry:
            state["warnings"].append(f"Cluster '{cluster_name}' not in registry — skipped")
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
        subjects = register_subjects(df, id_col)

        try:
            if strategy == "wide_lab":
                _process_wide_lab(df, subjects, cluster_name, cluster_cfg, mapping_index, subject_models, cfg)
            elif strategy == "timeseries":
                _process_timeseries(df, subjects, cluster_name, cluster_cfg, mapping_index, subject_models, cfg)
            elif strategy == "patient":
                _process_demographics(df, subjects, cluster_cfg, subject_demo, subject_models, cfg)
            elif strategy == "cohort":
                _process_cohort(df, subjects, cluster_cfg, subject_models, cfg, local_system)
            elif strategy == "encounter":
                _process_encounter(df, subjects, cluster_cfg, subject_models, cfg)
            elif strategy == "condition":
                _process_condition(df, subjects, cluster_cfg, subject_models, cfg)
            elif strategy == "medication_request":
                _process_medication(df, subjects, cluster_name, cluster_cfg, mapping_index, subject_models, cfg, rxnorm_lookup)
            elif strategy == "social_history":
                _process_social_history(df, subjects, cluster_name, cluster_cfg, mapping_index, subject_models, cfg, local_system)
            elif strategy == "survey":
                _process_survey(df, subjects, cluster_name, cluster_cfg, subject_models, cfg, local_system)
            elif strategy == "microbiota_diversity":
                _process_microbiota_diversity(df, subjects, cluster_cfg, subject_models, cfg, local_system)
            elif strategy == "microbiota_abundance":
                _process_microbiota_abundance(df, subjects, cluster_cfg, subject_models, cfg)
            elif strategy == "eeg":
                _process_eeg(df, subjects, cluster_cfg, subject_models, cfg, local_system)
            else:
                state["warnings"].append(f"Unknown strategy '{strategy}' for {cluster_name}")
        except Exception as exc:  # keep building other clusters
            state["warnings"].append(f"Builder failed for '{cluster_name}': {exc}")
            logger.exception("Builder error in cluster %s", cluster_name)

    # Build Patient + Bundle per subject; serialise to dicts.
    all_resource_dicts: list[dict] = []
    bundle_dicts: dict[str, dict] = {}

    for subject_id, models in subject_models.items():
        demo = subject_demo.get(subject_id, {})
        patient = build_patient(
            subject_id, cfg.FHIR_BASE_URL,
            gender=demo.get("gender"), birth_date=demo.get("birth_date"),
        )
        all_models = [patient] + models
        bundle = build_bundle(all_models, base_url=cfg.FHIR_BASE_URL)
        bundle_dicts[subject_id] = to_dict(bundle)
        all_resource_dicts.extend(to_dict(m) for m in all_models)

    state["fhir_resources"] = all_resource_dicts
    state["fhir_bundles"] = bundle_dicts

    obs_count = sum(1 for r in all_resource_dicts if r.get("resourceType") == "Observation")
    logger.info("FHIRBuilder: %d subjects, %d resources (%d Observations)",
                len(subject_models), len(all_resource_dicts), obs_count)
    return state


# HELPERS

def _clean_id(s: str) -> str:
    return str(s).replace("_", "-").replace(".", "-").replace(":", "-").replace(" ", "-")


def _num(v) -> float | None:
    try:
        f = float(v)
        return f if pd.notna(f) else None
    except (TypeError, ValueError):
        return None


def _load_rxnorm(cfg: Any) -> dict[str, tuple[str, str]]:
    """ingredient_name(lower) → (rxnorm_code, ingredient_name)."""
    out: dict[str, tuple[str, str]] = {}
    path = getattr(cfg, "RXNORM_SEED", None)
    if not path or not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["ingredient_name"].strip().lower()] = (row["rxnorm_code"], row["ingredient_name"])
    return out


def _load_dict_labels(cfg: Any, dict_file: str) -> dict[str, dict]:
    """variable → {label, units, type} for every row of a dictionary CSV."""
    path = cfg.SAMPLE_DIR / dict_file
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        var = str(r.get("variable", "")).strip()
        if var:
            out[var] = {
                "label": str(r.get("label", var)),
                "units": str(r.get("units_or_choices", "")),
                "type": str(r.get("type", "")).strip().lower(),
            }
    return out


def _baseline_dates(cfg: Any) -> dict[str, str]:
    """subject_id → baseline visit_date (for clusters lacking their own date, e.g. EEG)."""
    out: dict[str, str] = {}
    visits = cfg.SAMPLE_DIR / "cohort/visits.csv"
    if visits.exists():
        df = pd.read_csv(visits)
        base = df[df.get("visit_type") == "baseline"] if "visit_type" in df.columns else df
        for _, r in base.iterrows():
            out.setdefault(str(r["subject_id"]), str(r["visit_date"]))
    cohort = cfg.SAMPLE_DIR / "cohort/cohort.csv"
    if cohort.exists():
        df = pd.read_csv(cohort)
        for _, r in df.iterrows():
            out.setdefault(str(r["subject_id"]), str(r.get("enrollment_date", "")))
    return out


# STRATEGIES

def _process_wide_lab(df, subjects, cluster_name, cluster_cfg, mapping_index, subject_models, cfg):
    """Wide-format data → (optional DiagnosticReport) + Observation models per subject."""
    id_col = cluster_cfg.get("id_col", "subject_id")
    date_col = cluster_cfg.get("date_col", "visit_date")
    category_code = cluster_cfg.get("fhir_category", "laboratory")
    make_report = not cluster_cfg.get("no_report", False)
    local_fallback = cluster_cfg.get("local_fallback", False)
    dict_units = {v: m.get("units", "") for v, m in _load_dict_labels(cfg, cluster_cfg["dict_file"]).items()}

    for subject_id in subjects:
        rows = df[df[id_col] == subject_id]
        if rows.empty:
            continue
        subject_models.setdefault(subject_id, [])
        all_obs = []
        effective_date = str(rows.iloc[0].to_dict().get(date_col, ""))
        report_id = f"dr-{subject_id}-{cluster_name}-{effective_date}".replace("_", "-")
        report_ref = f"{cfg.FHIR_BASE_URL}/DiagnosticReport/{report_id}"

        for _, row in rows.iterrows():
            row_dict = row.to_dict()
            visit_date = str(row_dict.get(date_col, ""))
            all_obs.extend(observations_from_wide_row(
                row=row_dict, subject_id=subject_id, visit_date=visit_date,
                mapping_index=mapping_index, cluster=cluster_name,
                category_code=category_code,
                part_of_ref=report_ref if make_report else None,
                local_fallback=local_fallback, local_system=cfg.SYSTEM_LOCAL,
                dict_units=dict_units,
                base_url=cfg.FHIR_BASE_URL,
            ))

        if not all_obs:
            continue
        subject_models[subject_id].extend(all_obs)

        if make_report:
            obs_refs = [f"{cfg.FHIR_BASE_URL}/Observation/{o.id}" for o in all_obs]
            report = build_diagnostic_report(
                subject_id=subject_id, report_id=report_id,
                effective_date=f"{effective_date}T00:00:00Z",
                observation_refs=obs_refs,
                title=f"{cluster_name.replace('_', ' ').title()} Panel",
                base_url=cfg.FHIR_BASE_URL,
            )
            subject_models[subject_id].insert(
                subject_models[subject_id].index(all_obs[0]), report
            )


def _process_timeseries(df, subjects, cluster_name, cluster_cfg, mapping_index, subject_models, cfg):
    id_col = cluster_cfg.get("id_col", "subject_id")
    for subject_id in subjects:
        sdf = df[df[id_col] == subject_id]
        if sdf.empty:
            continue
        subject_models.setdefault(subject_id, [])
        subject_models[subject_id].extend(observations_from_timeseries_rows(
            rows=sdf.to_dict("records"), subject_id=subject_id,
            mapping_index=mapping_index, cluster=cluster_name,
            kind_col=cluster_cfg.get("kind_col", "quantity_kind"),
            value_col=cluster_cfg.get("value_col", "value"),
            unit_col=cluster_cfg.get("unit_col", "unit"),
            timestamp_col=cluster_cfg.get("timestamp_col", "timestamp_utc"),
            device_col=cluster_cfg.get("device_col", "device_id"),
            source_col=cluster_cfg.get("source_col", "source"),
            base_url=cfg.FHIR_BASE_URL,
        ))


def _process_demographics(df, subjects, cluster_cfg, subject_demo, subject_models, cfg):
    id_col = cluster_cfg.get("id_col", "subject_id")
    for subject_id in subjects:
        rows = df[df[id_col] == subject_id]
        if rows.empty:
            continue
        r = rows.iloc[0].to_dict()
        subject_demo[subject_id] = {
            "gender": str(r.get("gender", "")) or None,
            "birth_date": str(r.get("date_of_birth", "")) or None,
        }
        # Allergies → AllergyIntolerance (only when explicitly recorded)
        if str(r.get("allergies", "")).strip().lower() == "yes":
            substance = str(r.get("specific_allergies", "")).strip() or "Unspecified allergen"
            subject_models[subject_id].append(build_allergy(
                subject_id=subject_id,
                allergy_id=f"allergy-{subject_id}-1".replace("_", "-"),
                substance_text=substance, base_url=cfg.FHIR_BASE_URL,
            ))


def _process_cohort(df, subjects, cluster_cfg, subject_models, cfg, local_system):
    id_col = cluster_cfg.get("id_col", "subject_id")
    for subject_id in subjects:
        rows = df[df[id_col] == subject_id]
        if rows.empty:
            continue
        r = rows.iloc[0].to_dict()
        group = str(r.get("group", "")).strip()
        enroll = str(r.get("enrollment_date", "")).strip() or "1900-01-01"
        if not group:
            continue
        subject_models[subject_id].append(build_observation(
            subject_id=subject_id, code="study_group", code_system=local_system,
            display="Study cohort group", value=None, unit="",
            value_string=group, effective_datetime=f"{enroll}T00:00:00Z",
            category_code="survey", base_url=cfg.FHIR_BASE_URL,
            obs_id=f"obs-{subject_id}-study-group".replace("_", "-"),
        ))


def _process_encounter(df, subjects, cluster_cfg, subject_models, cfg):
    id_col = cluster_cfg.get("id_col", "subject_id")
    date_col = cluster_cfg.get("date_col", "visit_date")
    visit_col = cluster_cfg.get("visit_col", "visit_id")
    for subject_id in subjects:
        rows = df[df[id_col] == subject_id]
        for _, row in rows.iterrows():
            r = row.to_dict()
            vid = str(r.get(visit_col, ""))
            subject_models[subject_id].append(build_encounter(
                subject_id=subject_id,
                encounter_id=f"enc-{_clean_id(vid)}",
                period_date=str(r.get(date_col, "")),
                type_text=str(r.get("visit_type", "")) or None,
                base_url=cfg.FHIR_BASE_URL,
            ))


def _process_condition(df, subjects, cluster_cfg, subject_models, cfg):
    id_col = cluster_cfg.get("id_col", "subject_id")
    date_col = cluster_cfg.get("date_col", "visit_date")
    code_col = cluster_cfg.get("code_col", "snomed_id")
    disp_col = cluster_cfg.get("display_col", "label_en")
    for subject_id in subjects:
        rows = df[df[id_col] == subject_id]
        for _, row in rows.iterrows():
            r = row.to_dict()
            code = str(r.get(code_col, "")).strip()
            if not code or code.lower() == "nan":
                continue
            idx = str(r.get("dx_index", "")) or "x"
            subject_models[subject_id].append(build_condition(
                subject_id=subject_id,
                condition_id=f"cond-{subject_id}-{idx}".replace("_", "-"),
                snomed_code=code, display=str(r.get(disp_col, "")).strip(),
                onset_date=str(r.get(date_col, "")), base_url=cfg.FHIR_BASE_URL,
            ))


def _process_medication(df, subjects, cluster_name, cluster_cfg, mapping_index, subject_models, cfg, rxnorm_lookup):
    """Drug name → RxNorm code from the agent (mapping_index); curated seed lookup as fallback."""
    id_col = cluster_cfg.get("id_col", "subject_id")
    date_col = cluster_cfg.get("date_col", "visit_date")
    drug_col = cluster_cfg.get("drug_col", "drug_name")
    dose_col = cluster_cfg.get("dose_col", "dose")
    freq_col = cluster_cfg.get("freq_col", "freq")
    for subject_id in subjects:
        rows = df[df[id_col] == subject_id]
        for _, row in rows.iterrows():
            r = row.to_dict()
            drug = str(r.get(drug_col, "")).strip()
            if not drug or drug.lower() == "nan":
                continue
            # 1) agent mapping (RxNorm RAG), 2) curated seed lookup, else text-only
            rx_code = None
            m = mapping_index.get(f"{cluster_name}::{drug}")
            if m and m.get("code") not in (None, "", "UNMAPPED"):
                rx_code = m["code"]
            if rx_code is None:
                key = _RXNORM_ALIASES.get(drug.lower(), drug.lower())
                rx = rxnorm_lookup.get(key)
                rx_code = rx[0] if rx else None
            dose = str(r.get(dose_col, "")).strip()
            freq = str(r.get(freq_col, "")).strip()
            dosage = " ".join(p for p in (dose, freq) if p and p.lower() != "nan") or None
            idx = str(r.get("med_index", "")) or "x"
            subject_models[subject_id].append(build_medication_request(
                subject_id=subject_id,
                mr_id=f"medreq-{subject_id}-{idx}".replace("_", "-"),
                display=drug, rxnorm_code=rx_code, dosage_text=dosage,
                authored_date=str(r.get(date_col, "")), base_url=cfg.FHIR_BASE_URL,
            ))


def _process_social_history(df, subjects, cluster_name, cluster_cfg, mapping_index, subject_models, cfg, local_system):
    """Codes come from the agent (mapping_index); the smoking *value* uses a fixed SNOMED map."""
    id_col = cluster_cfg.get("id_col", "subject_id")
    date_col = cluster_cfg.get("date_col", "visit_date")

    def code_for(var, fb_code, fb_disp, fb_system):
        m = mapping_index.get(f"{cluster_name}::{var}")
        if m and m.get("code") not in (None, "", "UNMAPPED"):
            return m["code"], m["code_system"], m["display"]
        return fb_code, fb_system, fb_disp

    for subject_id in subjects:
        rows = df[df[id_col] == subject_id]
        if rows.empty:
            continue
        r = rows.iloc[0].to_dict()
        eff = f"{str(r.get(date_col, '') or '1900-01-01')}T00:00:00Z"

        smoke = str(r.get("smoke_status", "")).strip().lower()
        if smoke in _SMOKING_VALUE:
            vcode, vdisp = _SMOKING_VALUE[smoke]
            code, system, disp = code_for("smoke_status", "72166-2", "Tobacco smoking status", SYS_LOINC)
            subject_models[subject_id].append(build_observation(
                subject_id=subject_id, code=code, code_system=system,
                display=disp, value=None, unit="",
                value_codeable=(SYS_SNOMED, vcode, vdisp),
                effective_datetime=eff, category_code="social-history",
                base_url=cfg.FHIR_BASE_URL,
                obs_id=f"obs-{subject_id}-smoking-status".replace("_", "-"),
            ))
        upw = _num(r.get("units_per_week"))
        if upw is not None:
            code, system, disp = code_for("units_per_week", "74013-4", "Alcoholic drinks per week", SYS_LOINC)
            subject_models[subject_id].append(build_observation(
                subject_id=subject_id, code=code, code_system=system,
                display=disp, value=upw, unit="{drinks}/wk", unit_code="{drinks}/wk",
                effective_datetime=eff, category_code="social-history",
                base_url=cfg.FHIR_BASE_URL,
                obs_id=f"obs-{subject_id}-alcohol-per-week".replace("_", "-"),
            ))
        yrs = _num(r.get("smoke_years"))
        if yrs is not None:
            code, system, disp = code_for("smoke_years", "smoke_years", "Number of years smoked", local_system)
            subject_models[subject_id].append(build_observation(
                subject_id=subject_id, code=code, code_system=system,
                display=disp, value=yrs, unit="a", unit_code="a",
                effective_datetime=eff, category_code="social-history",
                base_url=cfg.FHIR_BASE_URL,
                obs_id=f"obs-{subject_id}-smoke-years".replace("_", "-"),
            ))


def _process_survey(df, subjects, cluster_name, cluster_cfg, subject_models, cfg, local_system):
    """Questionnaire / dental score tables → one survey Observation per numeric field per visit."""
    id_col = cluster_cfg.get("id_col", "subject_id")
    date_col = cluster_cfg.get("date_col", "visit_date")
    category = cluster_cfg.get("fhir_category", "survey")
    labels = _load_dict_labels(cfg, cluster_cfg["dict_file"])

    for subject_id in subjects:
        rows = df[df[id_col] == subject_id]
        for _, row in rows.iterrows():
            r = row.to_dict()
            visit_date = str(r.get(date_col, "")) or "1900-01-01"
            vtag = str(r.get("visit_type", "")) or visit_date
            for col, raw in r.items():
                if col in _STRUCTURAL:
                    continue
                meta = labels.get(col, {})
                # Only emit numeric measures; skip string interpretation columns.
                if meta and meta.get("type") not in ("numeric", "integer"):
                    continue
                val = _num(raw)
                if val is None:
                    continue
                std = _QUESTIONNAIRE_LOINC.get(col)
                if std:
                    code, code_system, display = std[0], SYS_LOINC, std[1]
                else:
                    code, code_system = col, local_system
                    display = meta.get("label", col.replace("_", " ").title())
                units = meta.get("units", "")
                ucum = units if units and units not in ("nan",) and len(units) <= 12 else ""
                subject_models[subject_id].append(build_observation(
                    subject_id=subject_id, code=code, code_system=code_system,
                    display=display, value=val, unit=ucum, unit_code=ucum or None,
                    effective_datetime=f"{visit_date}T00:00:00Z",
                    category_code=category, base_url=cfg.FHIR_BASE_URL,
                    obs_id=f"obs-{subject_id}-{_clean_id(col)}-{_clean_id(vtag)}",
                ))


def _process_microbiota_diversity(df, subjects, cluster_cfg, subject_models, cfg, local_system):
    id_col = cluster_cfg.get("id_col", "subject_id")
    date_col = cluster_cfg.get("date_col", "visit_date")
    metrics = {"chao1": "Chao1 richness estimator", "shannon": "Shannon diversity index"}
    for subject_id in subjects:
        rows = df[df[id_col] == subject_id]
        for _, row in rows.iterrows():
            r = row.to_dict()
            visit_date = str(r.get(date_col, "")) or "1900-01-01"
            vtag = str(r.get("visit_type", "")) or visit_date
            for col, disp in metrics.items():
                val = _num(r.get(col))
                if val is None:
                    continue
                subject_models[subject_id].append(build_observation(
                    subject_id=subject_id, code=col, code_system=local_system,
                    display=disp, value=val, unit="{index}", unit_code="{index}",
                    effective_datetime=f"{visit_date}T00:00:00Z",
                    category_code="laboratory", base_url=cfg.FHIR_BASE_URL,
                    obs_id=f"obs-{subject_id}-{col}-{_clean_id(vtag)}",
                ))


def _process_microbiota_abundance(df, subjects, cluster_cfg, subject_models, cfg):
    id_col = cluster_cfg.get("id_col", "subject_id")
    date_col = cluster_cfg.get("date_col", "visit_date")
    taxon_col = cluster_cfg.get("taxon_col", "taxon_id")
    name_col = cluster_cfg.get("taxon_name_col", "taxon_name")
    value_col = cluster_cfg.get("value_col", "rel_abund_pct")
    for subject_id in subjects:
        rows = df[df[id_col] == subject_id]
        for _, row in rows.iterrows():
            r = row.to_dict()
            taxon = str(r.get(taxon_col, "")).strip()
            val = _num(r.get(value_col))
            if not taxon or val is None:
                continue
            visit_date = str(r.get(date_col, "")) or "1900-01-01"
            vtag = str(r.get("visit_type", "")) or visit_date
            subject_models[subject_id].append(build_observation(
                subject_id=subject_id, code=taxon, code_system=SYS_NCBI_TAXON,
                display=f"{str(r.get(name_col, '')).strip()} relative abundance",
                value=val, unit="%", unit_code="%",
                effective_datetime=f"{visit_date}T00:00:00Z",
                category_code="laboratory", base_url=cfg.FHIR_BASE_URL,
                obs_id=f"obs-{subject_id}-taxon-{taxon}-{_clean_id(vtag)}",
            ))


def _process_eeg(df, subjects, cluster_cfg, subject_models, cfg, local_system):
    """
    EEG PSD/connectivity → per-subject DiagnosticReport + band-power summary
    Observations (mean across channels per condition per band), plus connectivity
    summaries (mean |imaginary coherence| per condition per band).
    """
    id_col = cluster_cfg.get("id_col", "subject_id")
    dates = _baseline_dates(cfg)
    conn_path = cfg.SAMPLE_DIR / cluster_cfg.get("connectivity_file", "")
    conn_df = pd.read_csv(conn_path) if conn_path.exists() else None

    psd_freq_cols = [c for c in df.columns if c.startswith("psd_")]
    freqs = {c: float(c.split("_", 1)[1]) for c in psd_freq_cols}

    for subject_id in subjects:
        sdf = df[df[id_col] == subject_id]
        if sdf.empty:
            continue
        eeg_date = dates.get(subject_id, "1900-01-01") or "1900-01-01"
        eff = f"{eeg_date}T00:00:00Z"
        obs_list = []

        for condition, cdf in sdf.groupby("condition") if "condition" in sdf.columns else [("all", sdf)]:
            for band, (lo, hi) in _EEG_BANDS.items():
                band_cols = [c for c in psd_freq_cols if lo <= freqs[c] < hi]
                if not band_cols:
                    continue
                band_power = float(cdf[band_cols].to_numpy().mean())
                obs_list.append(build_observation(
                    subject_id=subject_id,
                    code=f"eeg_psd_{band}_{condition}", code_system=local_system,
                    display=f"EEG mean {band} band power ({condition})",
                    value=round(band_power, 6), unit="uV2/Hz", unit_code="uV2/Hz",
                    effective_datetime=eff, category_code="procedure",
                    base_url=cfg.FHIR_BASE_URL,
                    obs_id=f"obs-{subject_id}-eeg-psd-{band}-{_clean_id(condition)}",
                ))

        if conn_df is not None:
            csub = conn_df[conn_df[id_col] == subject_id]
            if not csub.empty and {"condition", "band", "value"} <= set(csub.columns):
                grouped = csub.groupby(["condition", "band"])["value"].apply(
                    lambda s: s.abs().mean()
                )
                for (condition, band), mval in grouped.items():
                    obs_list.append(build_observation(
                        subject_id=subject_id,
                        code=f"eeg_conn_{band}_{condition}", code_system=local_system,
                        display=f"EEG mean |imaginary coherence| {band} ({condition})",
                        value=round(float(mval), 6), unit="1", unit_code="1",
                        effective_datetime=eff, category_code="procedure",
                        base_url=cfg.FHIR_BASE_URL,
                        obs_id=f"obs-{subject_id}-eeg-conn-{band}-{_clean_id(condition)}",
                    ))

        if not obs_list:
            continue
        subject_models[subject_id].extend(obs_list)
        report_id = f"dr-{subject_id}-eeg-{eeg_date}".replace("_", "-")
        obs_refs = [f"{cfg.FHIR_BASE_URL}/Observation/{o.id}" for o in obs_list]
        report = build_diagnostic_report(
            subject_id=subject_id, report_id=report_id,
            effective_date=eff, observation_refs=obs_refs,
            title="Resting-state EEG quantitative summary",
            loinc_code="24708-6", loinc_display="EEG study",
            base_url=cfg.FHIR_BASE_URL,
        )
        subject_models[subject_id].insert(
            subject_models[subject_id].index(obs_list[0]), report
        )
