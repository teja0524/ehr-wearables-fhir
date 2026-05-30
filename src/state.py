"""
Shared pipeline state flowing through all LangGraph agent nodes.
Each node reads what it needs and writes its outputs back into the same dict.
"""

from __future__ import annotations

from typing import Any, TypedDict


class VariableMeta(TypedDict):
    """Metadata extracted by SchemaParser for a single variable."""
    variable: str
    label: str
    units: str
    var_type: str
    min_val: str
    max_val: str
    notes: str
    cluster: str


class CodeMapping(TypedDict):
    """LOINC/SNOMED code selected by CodeMapper for one variable."""
    variable: str
    cluster: str
    code_system: str        # "http://loinc.org" or "http://snomed.info/"
    code: str
    display: str
    confidence: str         # "high" | "medium" | "low"
    rationale: str
    candidates: list[dict]  # top-K raw vector search results
    ucum_unit: str          # UCUM unit from the winning candidate's metadata


class FHIRResource(TypedDict):
    """A single serialisable FHIR R4 resource dict."""
    resourceType: str
    id: str
    resource: dict[str, Any]


class ValidationIssue(TypedDict):
    resource_id: str
    severity: str       # "error" | "warning" | "info"
    message: str


class PipelineState(TypedDict):
    """Full shared state flowing through all LangGraph nodes."""
    clusters: list[str]
    subjects: list[str]
    max_subjects: int | None                # None = process all subjects
    variable_metadata: list[VariableMeta]
    code_mappings: list[CodeMapping]
    mapping_index: dict[str, CodeMapping]   # keyed as "cluster::variable"
    fhir_resources: list[FHIRResource]
    fhir_bundles: dict[str, dict]           # subject_id → Bundle JSON
    validation_issues: list[ValidationIssue]
    output_paths: list[str]                 # paths of written JSON files
    errors: list[str]
    warnings: list[str]
