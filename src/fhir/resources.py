"""
FHIR R4 resource factory using the `fhir.resources` Pydantic library.
Resources are validated at construction time — field errors surface immediately.

Spec: https://hl7.org/fhir/R4/
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

# fhir.resources imports
# Each class maps 1-to-1 with a FHIR R4 resource or data type.
from fhir.resources.bundle import Bundle, BundleEntry
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.identifier import Identifier
from fhir.resources.meta import Meta
from fhir.resources.observation import Observation
from fhir.resources.patient import Patient
from fhir.resources.quantity import Quantity
from fhir.resources.reference import Reference


def make_resource_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coding(system: str, code: str, display: str | None = None) -> Coding:
    return Coding(system=system, code=code, display=display)


def _codeable_concept(
    system: str, code: str, display: str | None = None, text: str | None = None
) -> CodeableConcept:
    return CodeableConcept(
        coding=[_coding(system, code, display)],
        text=text or display,
    )


def _subject_ref(subject_id: str, base_url: str) -> Reference:
    return Reference(
        reference=f"{base_url}/Patient/{subject_id}",
        display=subject_id,
    )


# FHIR profile URLs
# Base R4 profiles from HL7 International
PROFILE_PATIENT      = "http://hl7.org/fhir/StructureDefinition/Patient"
PROFILE_OBSERVATION  = "http://hl7.org/fhir/StructureDefinition/Observation"
PROFILE_VITALSIGNS   = "http://hl7.org/fhir/StructureDefinition/vitalsigns"
PROFILE_DIAG_REPORT  = "http://hl7.org/fhir/StructureDefinition/DiagnosticReport"

# HL7 Europe base profiles (v2.0.0, Apr 2026) — for EHDS-compliant deployments
# Switch meta.profile to these when targeting European health data exchange.
# Ref: https://hl7.eu/fhir/base
PROFILE_EU_PATIENT      = "https://hl7.eu/fhir/base/StructureDefinition/patient-eu-core"
PROFILE_EU_OBSERVATION  = "https://hl7.eu/fhir/base/StructureDefinition/observation-eu-core"

# Code systems
SYS_OBS_CATEGORY  = "http://terminology.hl7.org/CodeSystem/observation-category"
SYS_V2_0074       = "http://terminology.hl7.org/CodeSystem/v2-0074"
SYS_ABSENT_REASON = "http://terminology.hl7.org/CodeSystem/data-absent-reason"
SYS_UCUM          = "http://unitsofmeasure.org"
SYS_LOINC         = "http://loinc.org"


# Patient

def build_patient(subject_id: str, base_url: str) -> Patient:
    """FHIR R4 Patient resource for a pseudonymised study subject."""
    return Patient(
        id=subject_id,
        meta=Meta(profile=[PROFILE_PATIENT]),
        identifier=[
            Identifier(
                system=f"{base_url}/identifier/subject",
                value=subject_id,
            )
        ],
        active=True,
    )


# Observation

# Category codes and their display labels
_CATEGORY_DISPLAY = {
    "laboratory":   "Laboratory",
    "vital-signs":  "Vital Signs",
    "activity":     "Activity",
    "exam":         "Exam",
    "survey":       "Survey",
}

# Vital-signs Observations must carry the vitalsigns profile per FHIR spec
_VITAL_SIGNS_PROFILE = [PROFILE_VITALSIGNS]
_BASE_OBS_PROFILE    = [PROFILE_OBSERVATION]


def build_observation(
    *,
    subject_id: str,
    code: str,
    code_system: str,
    display: str,
    value: float | None,
    unit: str,
    unit_system: str = SYS_UCUM,
    unit_code: str | None = None,
    effective_datetime: str,
    status: str = "final",
    category_code: str = "laboratory",
    device_id: str | None = None,
    base_url: str,
    obs_id: str | None = None,
) -> Observation:
    """
    Build a validated FHIR R4 Observation.
    If value is None, dataAbsentReason is set instead of valueQuantity.
    device_id, if provided, populates Observation.device for wearables.
    """
    resource_id = obs_id or make_resource_id("obs-")
    cat_display = _CATEGORY_DISPLAY.get(category_code, category_code.title())

    # Vital-signs carry a different mandatory profile per FHIR spec
    profile = _VITAL_SIGNS_PROFILE if category_code == "vital-signs" else _BASE_OBS_PROFILE

    obs_data: dict[str, Any] = {
        "id": resource_id,
        "meta": Meta(profile=profile),
        "status": status,
        "category": [
            CodeableConcept(
                coding=[_coding(SYS_OBS_CATEGORY, category_code, cat_display)]
            )
        ],
        "code": _codeable_concept(code_system, code, display),
        "subject": _subject_ref(subject_id, base_url),
        "effectiveDateTime": effective_datetime,
    }

    if value is not None:
        qty_fields: dict[str, Any] = {"value": value, "system": unit_system}
        resolved_code = unit_code or unit
        if unit:
            qty_fields["unit"] = unit
        if resolved_code:
            qty_fields["code"] = resolved_code
        obs_data["valueQuantity"] = Quantity(**qty_fields)
    else:
        obs_data["dataAbsentReason"] = _codeable_concept(
            SYS_ABSENT_REASON, "unknown", "Unknown"
        )

    if device_id:
        obs_data["device"] = Reference(
            identifier=Identifier(
                system=f"{base_url}/identifier/device",
                value=device_id,
            )
        )

    return Observation(**obs_data)


# DiagnosticReport

def build_diagnostic_report(
    *,
    subject_id: str,
    report_id: str,
    effective_date: str,
    observation_refs: list[str],
    title: str = "Laboratory Report",
    loinc_code: str = "11502-2",
    loinc_display: str = "Laboratory report",
    base_url: str,
) -> DiagnosticReport:
    """FHIR R4 DiagnosticReport grouping a set of Observations for one visit."""
    return DiagnosticReport(
        id=report_id,
        meta=Meta(profile=[PROFILE_DIAG_REPORT]),
        status="final",
        category=[
            CodeableConcept(
                coding=[_coding(SYS_V2_0074, "LAB", "Laboratory")]
            )
        ],
        code=_codeable_concept(SYS_LOINC, loinc_code, loinc_display, title),
        subject=_subject_ref(subject_id, base_url),
        effectiveDateTime=effective_date,
        issued=_now_iso(),
        result=[Reference(reference=ref) for ref in observation_refs],
    )


# Bundle

def build_bundle(
    resources: list[Patient | Observation | DiagnosticReport],
    base_url: str,
    bundle_type: str = "collection",
) -> Bundle:
    """
    FHIR R4 Bundle containing all resources for one subject.
    Use bundle_type="transaction" when POSTing to a live FHIR server.
    """
    entries = []
    for resource in resources:
        resource_type = resource.__resource_type__
        resource_id   = resource.id
        entries.append(
            BundleEntry(
                fullUrl=f"{base_url}/{resource_type}/{resource_id}",
                resource=resource,
            )
        )

    return Bundle(
        id=make_resource_id("bundle-"),
        meta=Meta(lastUpdated=_now_iso()),
        type=bundle_type,
        timestamp=_now_iso(),
        total=len(entries),
        entry=entries,
    )


# Observation batch builders

def observations_from_wide_row(
    *,
    row: dict,
    subject_id: str,
    visit_date: str,
    mapping_index: dict,
    cluster: str,
    category_code: str = "laboratory",
    base_url: str,
) -> list[Observation]:
    """
    One wide CSV row (e.g. a blood_labs row) → list of Observation models.
    One Observation is produced per mapped, non-null column.
    """
    skip_cols = {"subject_id", "visit_id", "visit_date"}
    observations: list[Observation] = []

    for col, raw_val in row.items():
        if col in skip_cols:
            continue

        mapping_key = f"{cluster}::{col}"
        if mapping_key not in mapping_index:
            continue

        mapping = mapping_index[mapping_key]
        if mapping["code"] == "UNMAPPED":
            continue

        value: float | None = None
        try:
            value = float(raw_val)
        except (TypeError, ValueError):
            pass

        # FHIR id pattern: [A-Za-z0-9\-.] only — replace underscores and colons
        obs_id = f"obs-{subject_id}-{col}-{visit_date}".replace("_", "-").replace(".", "-").replace(":", "-")

        ucum = mapping.get("ucum_unit", "")
        obs = build_observation(
            subject_id=subject_id,
            code=mapping["code"],
            code_system=mapping["code_system"],
            display=mapping["display"],
            value=value,
            unit=ucum,
            unit_code=ucum or None,
            effective_datetime=f"{visit_date}T00:00:00Z",
            category_code=category_code,
            base_url=base_url,
            obs_id=obs_id,
        )
        observations.append(obs)

    return observations


def observations_from_timeseries_rows(
    *,
    rows: list[dict],
    subject_id: str,
    mapping_index: dict,
    cluster: str,
    kind_col: str = "quantity_kind",
    value_col: str = "value",
    unit_col: str = "unit",
    timestamp_col: str = "timestamp_utc",
    device_col: str = "device_id",
    source_col: str = "source",
    base_url: str,
) -> list[Observation]:
    """
    Long-format wearables rows → list of Observation models.
    One Observation per row that has a code mapping.
    """
    observations: list[Observation] = []

    for i, row in enumerate(rows):
        kind = str(row.get(kind_col, ""))
        mapping_key = f"{cluster}::{kind}"
        if mapping_key not in mapping_index:
            continue

        mapping = mapping_index[mapping_key]
        if mapping["code"] == "UNMAPPED":
            continue

        value: float | None = None
        try:
            value = float(row.get(value_col, ""))
        except (TypeError, ValueError):
            pass

        unit      = str(row.get(unit_col, ""))
        timestamp = str(row.get(timestamp_col, ""))
        device_id = str(row.get(device_col, "")) or None
        source    = str(row.get(source_col, ""))

        category_code, _ = _wearable_category(kind, source)
        # FHIR id pattern: [A-Za-z0-9\-.] only — underscores not allowed
        obs_id = f"obs-{subject_id}-{kind}-{i}".replace("_", "-").replace(".", "-").replace(":", "-")

        obs = build_observation(
            subject_id=subject_id,
            code=mapping["code"],
            code_system=mapping["code_system"],
            display=mapping["display"],
            value=value,
            unit=unit,
            unit_code=_ucum_for_unit(unit),
            effective_datetime=timestamp,
            category_code=category_code,
            device_id=device_id,
            base_url=base_url,
            obs_id=obs_id,
        )
        observations.append(obs)

    return observations


# Serialisation helper

def to_dict(resource: Patient | Observation | DiagnosticReport | Bundle) -> dict:
    """Serialise a fhir.resources model to a plain dict suitable for json.dumps."""
    try:
        return resource.model_dump(mode="json", exclude_none=True)
    except AttributeError:
        # Pydantic v1 fallback: .json() serialises then we parse back to dict
        import json as _json
        return _json.loads(resource.json(exclude_none=True))


# Internal helpers

def _ucum_for_unit(unit_label: str) -> str:
    """Map Polar device unit labels to standard UCUM codes."""
    _map = {
        "bpm":               "/min",
        "count":             "{steps}",
        "minutes":           "min",
        "kcal":              "kcal",
        "m":                 "m",
        "ms":                "ms",
        "breaths_per_minute":"/min",
        "dimensionless":     "1",
    }
    return _map.get(unit_label, unit_label)


def _wearable_category(kind: str, source: str) -> tuple[str, str]:
    """Map a wearable quantity_kind to a FHIR Observation category."""
    _sleep    = {"sleep_total", "sleep_deep", "sleep_light",
                 "sleep_rem", "sleep_awake", "sleep_score"}
    _activity = {"step_count", "active_minutes", "calories",
                 "distance", "daily_activity_index",
                 "cardio_load", "cardio_load_ratio"}
    _vital    = {"heart_rate", "heart_rate_resting", "rr_interval",
                 "rmssd", "breathing_rate", "ans_charge"}

    if kind in _sleep:    return "exam",        "Exam"
    if kind in _activity: return "activity",    "Activity"
    if kind in _vital:    return "vital-signs", "Vital Signs"
    return "survey", "Survey"
