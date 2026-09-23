"""
Node 4 — Validator

Runs the pipeline's two automated validation checks over the generated
resources. Both are VERIFICATION in the engineering sense — they compare the
artifact against an external specification, are objective and repeatable, and
require no human judgement:

  Validation 1 — Conformance (src/validation/conformance.py)
      The official HL7 `validator_cli.jar` checks each Bundle against the FHIR
      R4 specification and any profile declared in `meta.profile`. This is the
      reference implementation, so the resulting conformance rate is
      authoritative rather than a self-defined subset of rules.
      If Java or the jar is unavailable the node falls back to the built-in
      structural checks below, and records which path was taken.

  Validation 2 — Terminology (src/validation/terminology.py)
      Confirms every emitted code actually exists in its code system via
      `$validate-code` on a terminology server.

The built-in `_validate_resource` checks are retained as the offline fallback
for Validation 1: they are fast and dependency-free, but they only cover required
fields, a status enum, and a code-system allow-list — they are NOT a substitute
for spec conformance, and results say which validator produced them.

Neither check can judge whether a code is the *correct* one for a given
variable. That is semantic accuracy, and it is not verifiable against any
specification — it requires human judgement against a reference standard. It is
therefore handled outside the pipeline as an offline EVALUATION study
(`tools/build_annotation_set.py`, `tools/evaluate_mapping.py`) rather than as a
third check here.
"""

from __future__ import annotations

import logging
from typing import Any

from src.state import PipelineState, ValidationIssue
from src.validation import conformance, terminology

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
}
# The project-local CodeSystem URI is derived from FHIR_BASE_URL, so it is added
# at call time rather than hardcoded here (a literal would go stale the moment
# the base URL changes, and silently flag every local code as unrecognised).

VALID_OBS_STATUSES = {"registered", "preliminary", "final", "amended",
                      "corrected", "cancelled", "entered-in-error", "unknown"}


def validate_fhir(state: PipelineState, cfg: Any) -> PipelineState:
    """
    LangGraph node: validate all generated FHIR resources.

    Populates state["validation_issues"], plus state["conformance_summary"] and
    state["terminology_summary"] with the quantitative results of each layer.
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

    # --- Validation 1: conformance -------------------------------------------------
    # The official validator is preferred, but it must be treated as failed
    # unless it actually returns a verdict. A run that could not execute has NOT
    # demonstrated conformance, so we fall back rather than report zero issues —
    # otherwise a missing Java runtime silently looks like a perfect result.
    available, reason = conformance.validator_available(cfg)
    conf_summary: dict = {}
    conf_issues: list[dict] = []
    if available:
        conf_issues, conf_summary = conformance.validate_bundles(
            state.get("fhir_bundles", {}), cfg
        )
        if not conf_summary.get("ok", False):
            reason = conf_summary.get("note", "validator did not return a verdict")
            available = False

    if available:
        issues.extend(ValidationIssue(**i) for i in conf_issues)
    else:
        logger.warning("Validator: official FHIR validator unavailable (%s) — "
                       "falling back to built-in structural checks", reason)
        for resource in resources:
            resource_type = resource.get("resourceType", "Unknown")
            resource_id = resource.get("id", "unknown-id")
            issues.extend(_validate_resource(resource, resource_type, resource_id,
                                             str(getattr(cfg, 'SYSTEM_LOCAL', ''))))
        conf_summary = {
            "ok": False,
            "validator": "built-in structural checks (fallback)",
            "resources_checked": len(resources),
            "conformance_rate_pct": None,
            "note": f"official validator not used: {reason}",
        }
    state["conformance_summary"] = conf_summary

    # --- Validation 2: terminology -------------------------------------------------
    try:
        term_issues, term_summary = terminology.check_resources(resources, cfg)
        issues.extend(ValidationIssue(**i) for i in term_issues)
    except Exception as exc:                    # noqa: BLE001 - never fail the run on a network hiccup
        logger.warning("Validator: terminology check failed: %s", exc)
        term_summary = {"note": f"terminology check failed: {exc}"}
    state["terminology_summary"] = term_summary

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
    resource: dict, resource_type: str, resource_id: str, local_system: str = ""
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
    issues.extend(_check_code_systems(resource, resource_id, local_system))

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


def _check_code_systems(resource: dict, resource_id: str,
                        local_system: str = "") -> list[ValidationIssue]:
    """Recursively find all coding.system values and warn on unknowns."""
    issues: list[ValidationIssue] = []
    known = VALID_CODE_SYSTEMS | ({local_system} if local_system else set())
    _walk_for_systems(resource, resource_id, issues, known)
    return issues


def _walk_for_systems(obj: Any, resource_id: str, issues: list, known: set) -> None:
    if isinstance(obj, dict):
        if "system" in obj and "code" in obj:
            system = obj["system"]
            if system and system not in known:
                # Only warn for unexpected systems (not errors — could be custom)
                issues.append(ValidationIssue(
                    resource_id=resource_id,
                    severity="info",
                    message=f"Unrecognised code system '{system}' — verify it is intentional",
                ))
        for v in obj.values():
            _walk_for_systems(v, resource_id, issues, known)
    elif isinstance(obj, list):
        for item in obj:
            _walk_for_systems(item, resource_id, issues, known)
