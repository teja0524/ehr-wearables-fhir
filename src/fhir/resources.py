"""
FHIR R4 resource factory using the `fhir.resources` Pydantic library.
Resources are validated at construction time. So, field errors surface immediately.

Spec: https://hl7.org/fhir/R4/
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from src.state import STRUCTURAL_COLUMNS

# fhir.resources imports - R4B modules (FHIR 4.x), matching FHIR_VERSION 4.0.1.
# The package's top-level modules are R5; R4B is the closest match to R4.
# Each class maps 1-to-1 with a FHIR R4B resource or data type.
from fhir.resources.R4B.allergyintolerance import AllergyIntolerance
from fhir.resources.R4B.bundle import Bundle, BundleEntry
from fhir.resources.R4B.codeableconcept import CodeableConcept
from fhir.resources.R4B.coding import Coding
from fhir.resources.R4B.condition import Condition
from fhir.resources.R4B.diagnosticreport import DiagnosticReport
from fhir.resources.R4B.encounter import Encounter
from fhir.resources.R4B.identifier import Identifier
from fhir.resources.R4B.medicationrequest import MedicationRequest
from fhir.resources.R4B.meta import Meta
from fhir.resources.R4B.observation import Observation, ObservationComponent
from fhir.resources.R4B.patient import Patient
from fhir.resources.R4B.procedure import Procedure
from fhir.resources.R4B.quantity import Quantity
from fhir.resources.R4B.reference import Reference


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
# Can switch meta.profile to these when targeting European health data exchange.
# Ref: https://hl7.eu/fhir/base
PROFILE_EU_PATIENT      = "https://hl7.eu/fhir/base/StructureDefinition/patient-eu-core"
PROFILE_EU_OBSERVATION  = "https://hl7.eu/fhir/base/StructureDefinition/observation-eu-core"

# Code systems
SYS_OBS_CATEGORY  = "http://terminology.hl7.org/CodeSystem/observation-category"
SYS_V2_0074       = "http://terminology.hl7.org/CodeSystem/v2-0074"
SYS_ABSENT_REASON = "http://terminology.hl7.org/CodeSystem/data-absent-reason"
SYS_UCUM          = "http://unitsofmeasure.org"
SYS_LOINC         = "http://loinc.org"
SYS_SNOMED        = "http://snomed.info/sct"
SYS_RXNORM        = "http://www.nlm.nih.gov/research/umls/rxnorm"
SYS_NCBI_TAXON    = "http://www.ncbi.nlm.nih.gov/taxonomy"
SYS_CONDITION_CLINICAL = "http://terminology.hl7.org/CodeSystem/condition-clinical"
SYS_CONDITION_VER      = "http://terminology.hl7.org/CodeSystem/condition-ver-status"
SYS_ALLERGY_CLINICAL   = "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical"
SYS_ENCOUNTER_CLASS    = "http://terminology.hl7.org/CodeSystem/v3-ActCode"

PROFILE_ENCOUNTER    = "http://hl7.org/fhir/StructureDefinition/Encounter"
PROFILE_CONDITION    = "http://hl7.org/fhir/StructureDefinition/Condition"
PROFILE_MEDREQUEST   = "http://hl7.org/fhir/StructureDefinition/MedicationRequest"
PROFILE_ALLERGY      = "http://hl7.org/fhir/StructureDefinition/AllergyIntolerance"


# Patient

def build_patient(
    subject_id: str,
    base_url: str,
    *,
    gender: str | None = None,
    birth_date: str | None = None,
) -> Patient:
    """
    FHIR R4 Patient resource for a pseudonymised study subject.
    gender (administrative: male|female|other|unknown) and birth_date
    (YYYY-MM-DD) are populated from the demographics cluster when available.
    """
    data: dict[str, Any] = {
        "id": subject_id,
        "meta": Meta(profile=[PROFILE_PATIENT]),
        "identifier": [
            Identifier(system=f"{base_url}/identifier/subject", value=subject_id)
        ],
        "active": True,
    }
    if gender:
        g = gender.strip().lower()
        if g in {"male", "female", "other", "unknown"}:
            data["gender"] = g
    if birth_date and str(birth_date).strip() not in ("", "nan"):
        data["birthDate"] = str(birth_date).strip()
    return Patient(**data)


def build_allergy(
    *, subject_id: str, allergy_id: str, substance_text: str, base_url: str,
) -> AllergyIntolerance:
    """FHIR R4 AllergyIntolerance — substance recorded as free text (no fabricated code)."""
    return AllergyIntolerance(
        id=allergy_id,
        meta=Meta(profile=[PROFILE_ALLERGY]),
        clinicalStatus=_codeable_concept(SYS_ALLERGY_CLINICAL, "active", "Active"),
        code=CodeableConcept(text=substance_text),
        patient=_subject_ref(subject_id, base_url),
    )


def build_encounter(
    *, subject_id: str, encounter_id: str, period_date: str,
    class_code: str = "AMB", class_display: str = "ambulatory",
    type_text: str | None = None, status: str = "finished", base_url: str,
) -> Encounter:
    """FHIR R4 Encounter for one study visit."""
    data: dict[str, Any] = {
        "id": encounter_id,
        "meta": Meta(profile=[PROFILE_ENCOUNTER]),
        "status": status,
        "class_fhir": _coding(SYS_ENCOUNTER_CLASS, class_code, class_display),
        "subject": _subject_ref(subject_id, base_url),
    }
    if type_text:
        data["type"] = [CodeableConcept(text=type_text)]
    if period_date and str(period_date).strip() not in ("", "nan"):
        from fhir.resources.R4B.period import Period
        data["period"] = Period(start=f"{period_date}T00:00:00Z")
    return Encounter(**data)


def build_condition(
    *, subject_id: str, condition_id: str, snomed_code: str | None = None, display: str,
    onset_date: str | None = None, base_url: str,
    code_system: str = SYS_SNOMED, text: str | None = None,
    category: str = "encounter-diagnosis", category_display: str = "Encounter Diagnosis",
) -> Condition:
    """
    FHIR R4 Condition.

    A coded Condition is produced when `snomed_code` (in `code_system`) is given.
    When no code is available (e.g. ACE free-text diagnoses), a text-only
    CodeableConcept is emitted using `text` (or `display`) so the diagnosis is
    still represented. `category` distinguishes encounter diagnoses from
    problem-list-item history (comorbidities).
    """
    if snomed_code and str(snomed_code).strip() not in ("", "nan"):
        code_cc = _codeable_concept(code_system, str(snomed_code), display, text or display)
    else:
        code_cc = CodeableConcept(text=text or display)
    data: dict[str, Any] = {
        "id": condition_id,
        "meta": Meta(profile=[PROFILE_CONDITION]),
        "clinicalStatus": _codeable_concept(SYS_CONDITION_CLINICAL, "active", "Active"),
        "verificationStatus": _codeable_concept(SYS_CONDITION_VER, "confirmed", "Confirmed"),
        "category": [_codeable_concept(
            "http://terminology.hl7.org/CodeSystem/condition-category",
            category, category_display)],
        "code": code_cc,
        "subject": _subject_ref(subject_id, base_url),
    }
    if onset_date and str(onset_date).strip() not in ("", "nan"):
        data["onsetDateTime"] = f"{onset_date}T00:00:00Z"
    return Condition(**data)


def build_procedure(
    *, subject_id: str, procedure_id: str, code: str, display: str,
    code_system: str, status: str = "completed",
    performed_date: str | None = None, base_url: str,
) -> "Procedure":
    """FHIR R4 Procedure (e.g. ACE study intervention)."""
    data: dict[str, Any] = {
        "id": procedure_id,
        "status": status,
        "code": _codeable_concept(code_system, str(code), display, display),
        "subject": _subject_ref(subject_id, base_url),
    }
    if performed_date and str(performed_date).strip() not in ("", "nan"):
        data["performedDateTime"] = f"{performed_date}T00:00:00Z"
    return Procedure(**data)


def build_medication_request(
    *, subject_id: str, mr_id: str, display: str, rxnorm_code: str | None = None,
    dosage_text: str | None = None, authored_date: str | None = None, base_url: str,
) -> MedicationRequest:
    """FHIR R4 MedicationRequest. RxNorm ingredient code when resolved, else text-only."""
    if rxnorm_code:
        med_cc = _codeable_concept(SYS_RXNORM, str(rxnorm_code), display, display)
    else:
        med_cc = CodeableConcept(text=display)
    data: dict[str, Any] = {
        "id": mr_id,
        "meta": Meta(profile=[PROFILE_MEDREQUEST]),
        "status": "active",
        "intent": "order",
        "medicationCodeableConcept": med_cc,
        "subject": _subject_ref(subject_id, base_url),
    }
    if authored_date and str(authored_date).strip() not in ("", "nan"):
        data["authoredOn"] = f"{authored_date}T00:00:00Z"
    if dosage_text and str(dosage_text).strip() not in ("", "nan"):
        from fhir.resources.R4B.dosage import Dosage
        data["dosageInstruction"] = [Dosage(text=str(dosage_text))]
    return MedicationRequest(**data)


# Observation

# Category codes and their display labels
_CATEGORY_DISPLAY = {
    "laboratory":     "Laboratory",
    "vital-signs":    "Vital Signs",
    "activity":       "Activity",
    "exam":           "Exam",
    "survey":         "Survey",
    "social-history": "Social History",
    "procedure":      "Procedure",
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
    effective_datetime: str | None = None,
    status: str = "final",
    category_code: str = "laboratory",
    device_id: str | None = None,
    part_of_ref: str | None = None,
    value_string: str | None = None,
    value_codeable: tuple[str, str, str] | None = None,
    base_url: str,
    obs_id: str | None = None,
) -> Observation:
    """
    Build a validated FHIR R4 Observation.
    Value precedence: valueQuantity (numeric) → valueCodeableConcept → valueString.
    If none is present, dataAbsentReason is set.
    device_id populates Observation.device for wearable data.
    part_of_ref populates Observation.partOf for reverse linking to a DiagnosticReport.
    """
    resource_id = obs_id or make_resource_id("obs-")
    cat_display = _CATEGORY_DISPLAY.get(category_code, category_code.title())

    # Guard against NaN floats (empty CSV cells) — treat as absent.
    if value is not None and value != value:
        value = None

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
    }
    # effective[x] is optional in FHIR R4; omit it when the source has no date
    # (e.g. ACE genetic / intervention tables carry no assessment date).
    if effective_datetime and str(effective_datetime).strip() not in ("", "nan", "TnanT00:00:00Z"):
        obs_data["effectiveDateTime"] = effective_datetime

    if value is not None:
        qty_fields: dict[str, Any] = {"value": value, "system": unit_system}
        resolved_code = unit_code or unit
        if unit:
            qty_fields["unit"] = unit
        if resolved_code:
            qty_fields["code"] = resolved_code
        obs_data["valueQuantity"] = Quantity(**qty_fields)
    elif value_codeable is not None:
        sys_, code_, disp_ = value_codeable
        obs_data["valueCodeableConcept"] = _codeable_concept(sys_, code_, disp_, disp_)
    elif value_string is not None and str(value_string).strip() not in ("", "nan"):
        obs_data["valueString"] = str(value_string)
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

    if part_of_ref:
        obs_data["partOf"] = [Reference(reference=part_of_ref)]

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
    part_of_ref: str | None = None,
    local_fallback: bool = False,
    local_system: str = "",
    dict_units: dict | None = None,
    base_url: str,
) -> list[Observation]:
    """
    One wide CSV row (e.g. a blood_labs row) → list of Observation models.
    One Observation is produced per parsed, non-null column.

    A column is only considered if the CodeMapper parsed it (i.e. it is in
    mapping_index). If the agent returned UNMAPPED, the column is dropped unless
    local_fallback is set, in which case it is emitted under local_system using
    the variable name as the code — so coverage is preserved and traceable.
    part_of_ref, if provided, sets Observation.partOf for reverse linking.
    """
    observations: list[Observation] = []

    for col, raw_val in row.items():
        if col in STRUCTURAL_COLUMNS:
            continue

        mapping_key = f"{cluster}::{col}"
        if mapping_key not in mapping_index:
            continue

        mapping = mapping_index[mapping_key]
        if mapping["code"] == "UNMAPPED":
            if not local_fallback:
                continue
            code = mapping.get("variable", col)
            code_system = local_system
            display = mapping.get("display") or col.replace("_", " ").title()
        else:
            code = mapping["code"]
            code_system = mapping["code_system"]
            display = mapping["display"]

        value: float | None = None
        try:
            value = float(raw_val)
        except (TypeError, ValueError):
            pass

        obs_id = f"obs-{subject_id}-{col}-{visit_date}".replace("_", "-").replace(".", "-").replace(":", "-")
        # Prefer the UCUM unit from the terminology candidate; otherwise derive a
        # sensible UCUM unit from the data dictionary / variable so dimensionless
        # scores and counts still carry a valid unit.
        ucum = mapping.get("ucum_unit", "") or _ucum_normalize(
            (dict_units or {}).get(col, ""), col, category_code
        )
        observations.append(build_observation(
            subject_id=subject_id,
            code=code,
            code_system=code_system,
            display=display,
            value=value,
            unit=ucum,
            unit_code=ucum or None,
            effective_datetime=f"{visit_date}T00:00:00Z",
            category_code=category_code,
            part_of_ref=part_of_ref,
            base_url=base_url,
            obs_id=obs_id,
        ))

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
        # ID: subject + kind + date + counter
        # Date extracted from timestamp; counter handles multiple readings per day.
        date_part = timestamp[:10] if timestamp else "nodate"
        obs_id = f"obs-{subject_id}-{kind}-{date_part}-{i}".replace("_", "-")

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

def _ucum_normalize(raw: str, col: str, category: str) -> str:
    """
    Best-effort UCUM unit for wide-table columns whose terminology candidate had
    no unit. Percentages → '%', counts → '{#}', dimensionless survey scores →
    '{score}', otherwise the UCUM unity '1'. Clean units pass through normalised.
    """
    raw = (raw or "").strip()
    cl = col.lower()
    if cl.endswith("_pct") or raw.startswith("%"):
        return "%"
    _clean = {
        "cm": "cm", "kg": "kg", "kg/m^2": "kg/m2", "kg/m2": "kg/m2",
        "mmhg": "mm[Hg]", "mm[hg]": "mm[Hg]", "bpm": "/min",
        "min": "min", "m": "m", "ms": "ms", "a": "a", "years": "a",
    }
    if raw.lower() in _clean:
        return _clean[raw.lower()]
    if "count" in raw.lower() or "count" in cl or "caries" in cl or "teeth" in cl \
            or "number of taxa" in raw.lower():
        return "{#}"
    if category in ("survey", "exam"):
        return "{score}"
    return "1"  # dimensionless unity (e.g. diversity indices)


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
