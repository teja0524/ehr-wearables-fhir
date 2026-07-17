"""
Node 4 — Validator

Lightweight structural validation of the generated FHIR R4 resources.
Limited to checking required fields, code system URIs, and resource 
completeness without requiring an actual FHIR server.

For full profile validation, hapi-fhir-cli can be utilised later.
"""

from __future__ import annotations

import logging
from typing import Any

from src.state import PipelineState, ValidationIssue

logger = logging.getLogger(__name__)

# Required fields per resource type
REQUIRED_FIELDS: dict[str, list[str]] = {
    # effective[x] is optional in FHIR R4 (some sources carry no assessment date),
    # so it is checked as a warning in _validate_observation, not required here.
    "Observation": ["status", "code", "subject"],
    "DiagnosticReport": ["status", "code", "subject"],
    "Patient": ["id"],
    "Bundle": ["type", "entry"],
    "Encounter": ["status", "class", "subject"],
    "Condition": ["code", "subject"],
    "MedicationRequest": ["status", "intent", "subject"],
    "AllergyIntolerance": ["patient"],
    "Procedure": ["status", "subject"],
}

VALID_CODE_SYSTEMS = {
    "http://loinc.org",
    "http://snomed.info/sct",
    "http://www.nlm.nih.gov/research/umls/rxnorm",
    "http://www.ncbi.nlm.nih.gov/taxonomy",
    "http://hl7.org/fhir/sid/icd-10",
    "http://terminology.hl7.org/CodeSystem/observation-category",
    "http://terminology.hl7.org/CodeSystem/v2-0074",
    "http://terminology.hl7.org/CodeSystem/v3-ActCode",
    "http://terminology.hl7.org/CodeSystem/condition-clinical",
    "http://terminology.hl7.org/CodeSystem/condition-ver-status",
    "http://terminology.hl7.org/CodeSystem/condition-category",
    "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
    "http://terminology.hl7.org/CodeSystem/data-absent-reason",
    "http://unitsofmeasure.org",
    "http://comfortage.example.org/fhir/CodeSystem/comfortage-local",
}

VALID_OBS_STATUSES = {"registered", "preliminary", "final", "amended",
                      "corrected", "cancelled", "entered-in-error", "unknown"}


def validate_fhir(state: PipelineState, cfg: Any) -> PipelineState:
    """
    LangGraph node: validate all generated FHIR resources.
    Populates state["validation_issues"] with errors and warnings.
    """
    issues: list[ValidationIssue] = []
    resources = state.get("fhir_resources", [])

    if not resources:
        issues.append(ValidationIssue(
            resource_id="pipeline",
            severity="error",
            message="No FHIR resources were generated",
        ))
        state["validation_issues"] = issues
        return state

    logger.info("Validator: checking %d resources", len(resources))

    for resource in resources:
        resource_type = resource.get("resourceType", "Unknown")
        resource_id = resource.get("id", "unknown-id")
        resource_issues = _validate_resource(resource, resource_type, resource_id)
        issues.extend(resource_issues)

    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    infos = [i for i in issues if i["severity"] == "info"]

    logger.info(
        "Validator: %d errors, %d warnings, %d info",
        len(errors), len(warnings), len(infos),
    )

    if errors:
        for err in errors[:5]:
            logger.error("  FHIR error [%s]: %s", err["resource_id"], err["message"])

    # Append summary to state warnings
    if errors:
        state["errors"].append(f"FHIR validation: {len(errors)} error(s) found")

    state["validation_issues"] = issues
    return state


def _validate_resource(
    resource: dict, resource_type: str, resource_id: str
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    # 1. Required fields
    required = REQUIRED_FIELDS.get(resource_type, [])
    for field in required:
        if field not in resource or resource[field] is None:
            issues.append(ValidationIssue(
                resource_id=resource_id,
                severity="error",
                message=f"Missing required field '{field}' in {resource_type}",
            ))

    # 2. Observation-specific checks
    if resource_type == "Observation":
        issues.extend(_validate_observation(resource, resource_id))

    # 3. Bundle-specific checks
    if resource_type == "Bundle":
        issues.extend(_validate_bundle(resource, resource_id))

    # 4. Code system validity
    issues.extend(_check_code_systems(resource, resource_id))

    return issues


def _validate_observation(resource: dict, resource_id: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    # Status must be valid
    status = resource.get("status", "")
    if status not in VALID_OBS_STATUSES:
        issues.append(ValidationIssue(
            resource_id=resource_id,
            severity="error",
            message=f"Invalid Observation.status '{status}'",
        ))

    # Must carry a value[x] or an explicit dataAbsentReason. All FHIR R4
    # Observation.value[x] choice types are acceptable, not just valueQuantity.
    _VALUE_X = (
        "valueQuantity", "valueCodeableConcept", "valueString", "valueBoolean",
        "valueInteger", "valueRange", "valueRatio", "valueSampledData",
        "valueTime", "valueDateTime", "valuePeriod",
    )
    has_value = any(k in resource for k in _VALUE_X)
    has_absent = "dataAbsentReason" in resource
    if not has_value and not has_absent:
        issues.append(ValidationIssue(
            resource_id=resource_id,
            severity="warning",
            message="Observation has no value[x] and no dataAbsentReason",
        ))

    # valueQuantity should have value, unit, system
    if "valueQuantity" in resource:
        vq = resource["valueQuantity"]
        for field in ["value", "unit"]:
            if field not in vq:
                issues.append(ValidationIssue(
                    resource_id=resource_id,
                    severity="warning",
                    message=f"Observation.valueQuantity missing '{field}'",
                ))

    # code.coding should exist and have system + code
    code = resource.get("code", {})
    codings = code.get("coding", [])
    if not codings:
        issues.append(ValidationIssue(
            resource_id=resource_id,
            severity="error",
            message="Observation.code has no coding entries",
        ))
    else:
        for coding in codings:
            if not coding.get("code"):
                issues.append(ValidationIssue(
                    resource_id=resource_id,
                    severity="error",
                    message="Observation.code.coding has empty 'code'",
                ))

    return issues


def _validate_bundle(resource: dict, resource_id: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    entries = resource.get("entry", [])

    if len(entries) == 0:
        issues.append(ValidationIssue(
            resource_id=resource_id,
            severity="warning",
            message="Bundle has no entries",
        ))

    # total should match entry count
    total = resource.get("total")
    if total is not None and total != len(entries):
        issues.append(ValidationIssue(
            resource_id=resource_id,
            severity="warning",
            message=f"Bundle.total ({total}) != len(entry) ({len(entries)})",
        ))

    return issues


def _check_code_systems(resource: dict, resource_id: str) -> list[ValidationIssue]:
    """Recursively find all coding.system values and warn on unknowns."""
    issues: list[ValidationIssue] = []
    _walk_for_systems(resource, resource_id, issues)
    return issues


def _walk_for_systems(obj: Any, resource_id: str, issues: list) -> None:
    if isinstance(obj, dict):
        if "system" in obj and "code" in obj:
            system = obj["system"]
            if system and system not in VALID_CODE_SYSTEMS:
                # Only warn for unexpected systems (not errors — could be custom)
                issues.append(ValidationIssue(
                    resource_id=resource_id,
                    severity="info",
                    message=f"Unrecognised code system '{system}' — verify it is intentional",
                ))
        for v in obj.values():
            _walk_for_systems(v, resource_id, issues)
    elif isinstance(obj, list):
        for item in obj:
            _walk_for_systems(item, resource_id, issues)
