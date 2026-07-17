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
import re
from typing import Any

import pandas as pd

from src import dictionary
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
    build_procedure,
    observations_from_timeseries_rows,
    observations_from_wide_row,
    to_dict,
)
from src.state import STRUCTURAL_COLUMNS, PipelineState

logger = logging.getLogger(__name__)

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
    dataset = state.get("dataset") or cfg.DATASETS[cfg.DEFAULT_DATASET]
    registry = dataset["clusters"]
    data_dir = dataset["data_dir"]
    default_id_col = dataset.get("id_col", "subject_id")
    mapping_index = state.get("mapping_index", {})
    max_subjects = state.get("max_subjects")
    local_system = cfg.SYSTEM_LOCAL

    subject_models: dict[str, list[Any]] = {}
    # demographics → Patient enrichment (gender / birthDate) per subject
    subject_demo: dict[str, dict] = {}
    rxnorm_lookup = _load_rxnorm(cfg)

    # When capped (max_subjects), fix a single subject whitelist from the first
    # cluster processed so every cluster contributes to the same subjects and
    # bundles are coherent. Uncapped runs process every subject.
    selected: dict[str, set] = {"ids": None}

    def register_subjects(df: pd.DataFrame, id_col: str) -> list[str]:
        if id_col not in df.columns:
            return []
        all_subs = df[id_col].dropna().unique().tolist()
        if max_subjects:
            if selected["ids"] is None:
                selected["ids"] = set(all_subs[:max_subjects])
            subs = [s for s in all_subs if s in selected["ids"]]
        else:
            subs = all_subs
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
        data_path = data_dir / cluster_cfg["data_file"]
        if not data_path.exists():
            state["warnings"].append(f"Data file not found: {data_path}")
            continue

        strategy = cluster_cfg.get("strategy", "wide_lab")
        logger.info("FHIRBuilder: processing cluster '%s' (%s)", cluster_name, strategy)

        df = dictionary.read_table(data_path, dataset)
        id_col = cluster_cfg.get("id_col", default_id_col)
        subjects = register_subjects(df, id_col)

        try:
            if strategy == "ace_table":
                _process_ace_table(df, subjects, cluster_name, cluster_cfg, mapping_index,
                                   subject_models, cfg, dataset)
            elif strategy == "ace_patient":
                _process_ace_demographics(df, subjects, cluster_cfg, subject_demo,
                                          subject_models, cfg, dataset)
            elif strategy == "ace_procedure":
                _process_ace_procedure(df, subjects, cluster_cfg, subject_models, cfg)
            elif strategy == "wide_lab":
                _process_wide_lab(df, subjects, cluster_name, cluster_cfg, mapping_index, subject_models, cfg, dataset)
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
                _process_survey(df, subjects, cluster_name, cluster_cfg, subject_models, cfg, local_system, dataset)
            elif strategy == "microbiota_diversity":
                _process_microbiota_diversity(df, subjects, cluster_cfg, subject_models, cfg, local_system)
            elif strategy == "microbiota_abundance":
                _process_microbiota_abundance(df, subjects, cluster_cfg, subject_models, cfg)
            elif strategy == "eeg":
                _process_eeg(df, subjects, cluster_cfg, subject_models, cfg, local_system, dataset)
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


def _baseline_dates(dataset: dict) -> dict[str, str]:
    """subject_id → baseline visit_date (for clusters lacking their own date, e.g. EEG).

    Reads from the active dataset's data directory so uploaded data resolves
    against the uploaded cohort files, not the bundled sample.
    """
    data_dir = dataset["data_dir"]
    out: dict[str, str] = {}
    visits = data_dir / "cohort/visits.csv"
    if visits.exists():
        df = pd.read_csv(visits)
        base = df[df.get("visit_type") == "baseline"] if "visit_type" in df.columns else df
        for _, r in base.iterrows():
            out.setdefault(str(r["subject_id"]), str(r["visit_date"]))
    cohort = data_dir / "cohort/cohort.csv"
    if cohort.exists():
        df = pd.read_csv(cohort)
        for _, r in df.iterrows():
            out.setdefault(str(r["subject_id"]), str(r.get("enrollment_date", "")))
    return out


# STRATEGIES

def _process_wide_lab(df, subjects, cluster_name, cluster_cfg, mapping_index, subject_models, cfg, dataset):
    """Wide-format data → (optional DiagnosticReport) + Observation models per subject."""
    id_col = cluster_cfg.get("id_col", "subject_id")
    date_col = cluster_cfg.get("date_col", "visit_date")
    category_code = cluster_cfg.get("fhir_category", "laboratory")
    make_report = not cluster_cfg.get("no_report", False)
    local_fallback = cluster_cfg.get("local_fallback", False)
    dict_vars = dictionary.load_cluster_variables(dataset, cluster_cfg, cluster_name)
    dict_units = {v: m.get("units", "") for v, m in dict_vars.items()}

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


def _process_survey(df, subjects, cluster_name, cluster_cfg, subject_models, cfg, local_system, dataset):
    """Questionnaire / dental score tables → one survey Observation per numeric field per visit."""
    id_col = cluster_cfg.get("id_col", "subject_id")
    date_col = cluster_cfg.get("date_col", "visit_date")
    category = cluster_cfg.get("fhir_category", "survey")
    labels = dictionary.load_cluster_variables(dataset, cluster_cfg, cluster_name)

    for subject_id in subjects:
        rows = df[df[id_col] == subject_id]
        for _, row in rows.iterrows():
            r = row.to_dict()
            visit_date = str(r.get(date_col, "")) or "1900-01-01"
            vtag = str(r.get("visit_type", "")) or visit_date
            for col, raw in r.items():
                if col in STRUCTURAL_COLUMNS:
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


def _process_eeg(df, subjects, cluster_cfg, subject_models, cfg, local_system, dataset):
    """
    EEG PSD/connectivity → per-subject DiagnosticReport + band-power summary
    Observations (mean across channels per condition per band), plus connectivity
    summaries (mean |imaginary coherence| per condition per band).
    """
    id_col = cluster_cfg.get("id_col", "subject_id")
    dates = _baseline_dates(dataset)
    conn_path = dataset["data_dir"] / cluster_cfg.get("connectivity_file", "")
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


# ===========================================================================
# ACE strategies
#
# The ACE study is dictionary-driven: a single Excel workbook tags every
# variable with a target FHIR resource. _process_ace_table reads those tags and
# routes each column to the right resource, so one function covers anthropometric,
# CSF, plasma, MRI, genetic, neurology and neuropsychology clusters. Subject
# grouping is done once with groupby() so the builder scales to all ~38k subjects.
# ===========================================================================

# Curated SNOMED CT codes for ACE comorbidity-history flags (value 1 = present).
_ACE_COMORBIDITY_SNOMED = {
    "comorb_arthrosis_history": ("396275006", "Osteoarthritis"),
    "comorb_ischemic_stroke_history": ("422504002", "Ischemic stroke"),
    "comorb_cardiopathy_history": ("56265001", "Heart disease"),
    "comorb_dyslipidemia_history": ("370992007", "Dyslipidemia"),
    "comorb_diabetes_history": ("73211009", "Diabetes mellitus"),
    "comorb_hypertension_history": ("38341003", "Hypertensive disorder"),
    "comorb_copd_history": ("13645005", "Chronic obstructive pulmonary disease"),
    "comorb_chronic_renal_insufficiency_history": ("709044004", "Chronic kidney disease"),
    "comorb_depression_history": ("35489007", "Depressive disorder"),
    "comorb_schizophrenia_history": ("58214004", "Schizophrenia"),
    "comorb_epilepsy_history": ("84757009", "Epilepsy"),
}

# ICD-10 code embedded in ACE neurology diagnosis text, e.g.
# "Alzheimer's disease - probable (G30.9)".
_ICD10_IN_TEXT = re.compile(r"\(([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?)\)")


def _eff(date_val: str | None) -> str | None:
    """Format a YYYY-MM-DD source date as a FHIR dateTime, or None if absent."""
    if not date_val:
        return None
    s = str(date_val).strip()
    if s in ("", "nan", "NaT"):
        return None
    return f"{s}T00:00:00Z"


def _ace_unit_hint(col: str, mapped_unit: str) -> str:
    """Prefer the terminology UCUM unit; otherwise infer one from the column name."""
    if mapped_unit and mapped_unit not in ("nan",):
        return mapped_unit
    suffix_units = {
        "_kg": "kg", "_m": "m", "_cm": "cm", "_mm": "mm",
        "_pct": "%", "_circum": "cm", "_volume": "mm3",
    }
    for suf, unit in suffix_units.items():
        if col.endswith(suf):
            return unit
    return ""


def _process_ace_table(df, subjects, cluster_name, cluster_cfg, mapping_index,
                       subject_models, cfg, dataset):
    """
    Dictionary-driven ACE builder. For every (subject, assessment-date) row, each
    column is emitted as the FHIR resource its dictionary entry specifies:
      Observation / DiagnosticReport member → Observation (coded via the RAG
        mapper, with a local-code fallback); report members are grouped under a
        per-row DiagnosticReport when the cluster sets make_report.
      Condition → comorbidity flag (curated SNOMED, only when present) or a
        free-text diagnosis (ICD-10 extracted from the text when available).
    """
    variables = dictionary.load_cluster_variables(dataset, cluster_cfg, cluster_name)
    id_col = cluster_cfg.get("id_col", dataset.get("id_col", "faceid"))
    date_col = cluster_cfg.get("date_col")
    category = cluster_cfg.get("fhir_category", "laboratory")
    default_resource = cluster_cfg.get("default_resource", "Observation")
    local_fallback = cluster_cfg.get("local_fallback", True)
    make_report = cluster_cfg.get("make_report", False)
    local_system = cfg.SYSTEM_LOCAL
    subject_set = set(subjects)

    for subject_id, sdf in df.groupby(id_col):
        if subject_id not in subject_set:
            continue
        subject_models.setdefault(subject_id, [])
        models = subject_models[subject_id]

        for ridx, (_, row) in enumerate(sdf.iterrows()):
            r = row.to_dict()
            eff = _eff(r.get(date_col)) if date_col else None
            date_tag = (str(r.get(date_col, "")) if date_col else "") or str(ridx)

            # When grouping into a DiagnosticReport, the report ref is known up
            # front (it depends only on the row's date), so report-member
            # Observations are linked via partOf at construction time.
            rep_id = _clean_id(f"dr-{subject_id}-{cluster_name}-{date_tag}")
            rep_ref = f"{cfg.FHIR_BASE_URL}/DiagnosticReport/{rep_id}" if make_report else None
            report_members = []

            for col, raw in r.items():
                if col == id_col or col == date_col:
                    continue
                meta = variables.get(col)
                if meta is None or meta.get("type") == "date":
                    continue
                resource = meta.get("fhir_resource", default_resource) or default_resource

                sval = "" if raw is None else str(raw).strip()
                if sval in ("", "nan", "NaN", "NA"):
                    continue

                if resource == "Condition":
                    cond = _ace_condition(subject_id, col, sval, meta, ridx, eff,
                                          local_system, cfg)
                    if cond is not None:
                        models.append(cond)
                    continue
                if resource in ("Patient", "Procedure"):
                    continue  # handled by their own clusters

                is_member = make_report and resource in ("DiagnosticReport", "Observation")
                obs = _ace_observation(subject_id, col, raw, meta, cluster_name,
                                       mapping_index, local_fallback, local_system,
                                       category, eff, date_tag, ridx, cfg,
                                       part_of=rep_ref if is_member else None)
                if obs is None:
                    continue
                if is_member:
                    report_members.append(obs)
                else:
                    models.append(obs)

            if make_report and report_members:
                models.extend(report_members)
                loinc = cluster_cfg.get("report_loinc", ("11502-2", "Laboratory report"))
                report = build_diagnostic_report(
                    subject_id=subject_id, report_id=rep_id,
                    effective_date=eff or _now_placeholder(),
                    observation_refs=[f"{cfg.FHIR_BASE_URL}/Observation/{o.id}" for o in report_members],
                    title=cluster_cfg.get("report_title", cluster_name.replace("_", " ").title()),
                    loinc_code=loinc[0], loinc_display=loinc[1],
                    base_url=cfg.FHIR_BASE_URL,
                )
                models.insert(models.index(report_members[0]), report)


def _now_placeholder() -> str:
    # DiagnosticReport.effectiveDateTime is required by our validator; use a
    # stable sentinel when the source row has no date.
    return "1900-01-01T00:00:00Z"


def _ace_observation(subject_id, col, raw, meta, cluster_name, mapping_index,
                     local_fallback, local_system, category, eff, date_tag, ridx,
                     cfg, part_of):
    """Build one ACE Observation, coded via the RAG mapper or a local fallback."""
    mapping = mapping_index.get(f"{cluster_name}::{col}")
    if mapping and mapping.get("code") not in (None, "", "UNMAPPED"):
        code = mapping["code"]
        code_system = mapping["code_system"]
        display = mapping["display"]
        mapped_unit = mapping.get("ucum_unit", "")
    else:
        if not local_fallback:
            return None
        code = col
        code_system = local_system
        display = meta.get("label", col.replace("_", " ").title())
        mapped_unit = ""

    value = _num(raw)
    value_string = None
    unit = ""
    if value is not None:
        unit = _ace_unit_hint(col, mapped_unit)
    else:
        # Non-numeric (e.g. genetic apoe genotype) → carry as a string value.
        value_string = str(raw).strip()

    obs_id = f"obs-{subject_id}-{_clean_id(col)}-{_clean_id(date_tag)}-{ridx}"
    return build_observation(
        subject_id=subject_id, code=code, code_system=code_system, display=display,
        value=value, unit=unit, unit_code=unit or None,
        value_string=value_string,
        effective_datetime=eff, category_code=category,
        part_of_ref=part_of, base_url=cfg.FHIR_BASE_URL, obs_id=obs_id,
    )


def _ace_condition(subject_id, col, sval, meta, ridx, eff, local_system, cfg):
    """
    Build an ACE Condition from either a comorbidity flag (0/1) or a free-text
    diagnosis. Returns None when a comorbidity flag is absent/negative.
    """
    onset = (eff or "").replace("T00:00:00Z", "") or None

    # Comorbidity history flags: emit only when present (== 1).
    if col in _ACE_COMORBIDITY_SNOMED:
        flag = _num(sval)
        if flag != 1:
            return None
        snomed, disp = _ACE_COMORBIDITY_SNOMED[col]
        return build_condition(
            subject_id=subject_id,
            condition_id=f"cond-{subject_id}-{_clean_id(col)}-{ridx}",
            snomed_code=snomed, display=disp, code_system=SYS_SNOMED,
            onset_date=onset, base_url=cfg.FHIR_BASE_URL,
            category="problem-list-item", category_display="Problem List Item",
        )

    # Generic 0/1 flag columns tagged as Condition: present only when == 1.
    if meta.get("type") in ("yesno", "integer") and sval in ("0", "0.0", "1", "1.0"):
        if _num(sval) != 1:
            return None
        return build_condition(
            subject_id=subject_id,
            condition_id=f"cond-{subject_id}-{_clean_id(col)}-{ridx}",
            snomed_code=col, display=meta.get("label", col), code_system=local_system,
            onset_date=onset, base_url=cfg.FHIR_BASE_URL,
            category="problem-list-item", category_display="Problem List Item",
        )

    # Free-text diagnosis (e.g. neurology_diagnosis_primary): keep the text and
    # extract an ICD-10 code from it when present.
    m = _ICD10_IN_TEXT.search(sval)
    if m:
        return build_condition(
            subject_id=subject_id,
            condition_id=f"cond-{subject_id}-{_clean_id(col)}-{ridx}",
            snomed_code=m.group(1), display=sval, code_system=cfg.SYSTEM_ICD10,
            onset_date=onset, text=sval, base_url=cfg.FHIR_BASE_URL,
        )
    return build_condition(
        subject_id=subject_id,
        condition_id=f"cond-{subject_id}-{_clean_id(col)}-{ridx}",
        snomed_code=None, display=sval, text=sval,
        onset_date=onset, base_url=cfg.FHIR_BASE_URL,
    )


def _process_ace_demographics(df, subjects, cluster_cfg, subject_demo,
                              subject_models, cfg, dataset):
    """ACE demographics → Patient (sex, birthDate) + education Observations."""
    id_col = cluster_cfg.get("id_col", "faceid")
    sex_col = cluster_cfg.get("sex_col", "sex_0M1F")
    birth_col = cluster_cfg.get("birth_col", "date_of_birth")
    local_system = cfg.SYSTEM_LOCAL
    sex_map = {"0": "male", "0.0": "male", "1": "female", "1.0": "female"}
    subject_set = set(subjects)

    for subject_id, sdf in df.groupby(id_col):
        if subject_id not in subject_set:
            continue
        r = sdf.iloc[0].to_dict()
        gender = sex_map.get(str(r.get(sex_col, "")).strip())
        birth = str(r.get(birth_col, "")).strip()
        subject_demo[subject_id] = {
            "gender": gender,
            "birth_date": birth if birth not in ("", "nan") else None,
        }
        subject_models.setdefault(subject_id, [])

        # Years of formal education → social-history Observation
        yschool = _num(r.get("yschool"))
        if yschool is not None:
            subject_models[subject_id].append(build_observation(
                subject_id=subject_id, code="yschool", code_system=local_system,
                display="Years of formal education", value=yschool, unit="a",
                unit_code="a", category_code="social-history",
                base_url=cfg.FHIR_BASE_URL,
                obs_id=f"obs-{subject_id}-yschool",
            ))
        edu = str(r.get("education_level", "")).strip()
        if edu and edu.lower() not in ("nan", "na", ""):
            subject_models[subject_id].append(build_observation(
                subject_id=subject_id, code="education_level", code_system=local_system,
                display="Maximum level of education attained", value=None, unit="",
                value_string=edu, category_code="social-history",
                base_url=cfg.FHIR_BASE_URL,
                obs_id=f"obs-{subject_id}-education-level",
            ))


def _process_ace_procedure(df, subjects, cluster_cfg, subject_models, cfg):
    """ACE intervention flag → Procedure (1 = completed, 0 = not-done, 999 = skip)."""
    id_col = cluster_cfg.get("id_col", "faceid")
    flag_col = cluster_cfg.get("flag_col", "intervention_0N1Y")
    local_system = cfg.SYSTEM_LOCAL
    subject_set = set(subjects)

    for subject_id, sdf in df.groupby(id_col):
        if subject_id not in subject_set:
            continue
        flag = _num(sdf.iloc[0].to_dict().get(flag_col))
        if flag is None or flag == 999:
            continue
        status = "completed" if flag == 1 else "not-done"
        subject_models.setdefault(subject_id, [])
        subject_models[subject_id].append(build_procedure(
            subject_id=subject_id,
            procedure_id=f"proc-{subject_id}-intervention",
            code="intervention", display="ACE study intervention",
            code_system=local_system, status=status, base_url=cfg.FHIR_BASE_URL,
        ))
