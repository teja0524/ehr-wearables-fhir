"""
LangGraph orchestration graph for the EHR/Wearables → FHIR pipeline.

Five sequential nodes:
  parse_schema → map_codes → build_fhir → validate_fhir → export_output

Each node is a pure function (state: PipelineState) → PipelineState.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from src.agents.code_mapper import map_codes
from src.agents.fhir_builder import build_fhir
from src.agents.schema_parser import parse_schema
from src.agents.validator import validate_fhir
from src.state import PipelineState
from src.vector_store.store import TerminologyStore

logger = logging.getLogger(__name__)


def build_graph(cfg: Any, store: TerminologyStore) -> Any:
    """Construct and compile the LangGraph StateGraph."""

    def node_parse_schema(state: PipelineState) -> PipelineState:
        logger.info("━━ Node: parse_schema ━━━━━━━━━━━━━━━━━━━━━━━━")
        return parse_schema(state, cfg)

    def node_map_codes(state: PipelineState) -> PipelineState:
        logger.info("━━ Node: map_codes ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return map_codes(state, cfg, store)

    def node_build_fhir(state: PipelineState) -> PipelineState:
        logger.info("━━ Node: build_fhir ━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return build_fhir(state, cfg)

    def node_validate_fhir(state: PipelineState) -> PipelineState:
        logger.info("━━ Node: validate_fhir ━━━━━━━━━━━━━━━━━━━━━━━")
        return validate_fhir(state, cfg)

    def node_export_output(state: PipelineState) -> PipelineState:
        logger.info("━━ Node: export_output ━━━━━━━━━━━━━━━━━━━━━━━")
        return export_output(state, cfg)

    graph = StateGraph(PipelineState)

    graph.add_node("parse_schema", node_parse_schema)
    graph.add_node("map_codes", node_map_codes)
    graph.add_node("build_fhir", node_build_fhir)
    graph.add_node("validate_fhir", node_validate_fhir)
    graph.add_node("export_output", node_export_output)

    graph.add_edge(START, "parse_schema")
    graph.add_edge("parse_schema", "map_codes")
    graph.add_edge("map_codes", "build_fhir")
    graph.add_edge("build_fhir", "validate_fhir")
    graph.add_edge("validate_fhir", "export_output")
    graph.add_edge("export_output", END)

    return graph.compile()


def initial_state(
    clusters: list[str],
    max_subjects: int | None = None,
    dataset: dict | None = None,
    skip_mapping: bool = False,
    model: str = "",
    apply_overrides: bool = True,
) -> PipelineState:
    """Return a fresh PipelineState for a new pipeline run."""
    return PipelineState(
        dataset=dataset or {},
        model=model,
        skip_mapping=skip_mapping,
        apply_overrides=apply_overrides,
        clusters=clusters,
        subjects=[],
        max_subjects=max_subjects,
        variable_metadata=[],
        code_mappings=[],
        mapping_index={},
        fhir_resources=[],
        fhir_bundles={},
        validation_issues=[],
        output_paths=[],
        errors=[],
        warnings=[],
    )


# Export node

def export_output(state: PipelineState, cfg: Any) -> PipelineState:
    """Write one FHIR Bundle JSON per subject and a mapping_report.json.

    Output is namespaced per dataset via the dataset's output_subdir, so ACE
    bundles land in output/ace/ and never collide with the MFU bundles at
    output/ root.
    """
    subdir = state.get("dataset", {}).get("output_subdir", "")
    output_dir: Path = cfg.OUTPUT_DIR / subdir if subdir else cfg.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[str] = []

    # Write per-subject bundles
    bundles = state.get("fhir_bundles", {})
    for subject_id, bundle in bundles.items():
        out_path = output_dir / f"{subject_id}_fhir_bundle.json"
        out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False))
        output_paths.append(str(out_path))

    # Write code mapping report.
    # Confidence is only meaningful for variables that actually received a
    # standard code; a variable can be "high confidence UNMAPPED" (the model is
    # sure no good code exists), so confidence buckets are counted ONLY among
    # genuinely-mapped variables to avoid overstating coverage.
    all_mappings = state.get("code_mappings", [])
    mapped = [m for m in all_mappings if m.get("code") not in (None, "", "UNMAPPED")]
    unmapped_n = len(all_mappings) - len(mapped)
    mapping_report = {
        "summary": {
            "dataset": state.get("dataset", {}).get("name", ""),
            "total_variables": len(all_mappings),
            "mapped_to_standard_code": len(mapped),
            "mapped_high_confidence": sum(1 for m in mapped if m.get("confidence") == "high"),
            "mapped_medium_confidence": sum(1 for m in mapped if m.get("confidence") == "medium"),
            "mapped_low_confidence": sum(1 for m in mapped if m.get("confidence") == "low"),
            "unmapped": unmapped_n,
            "subjects": len(bundles),
            "total_resources": len(state.get("fhir_resources", [])),
        },
        "mappings": [
            {
                "cluster": m["cluster"],
                "variable": m["variable"],
                "code_system": m["code_system"],
                "code": m["code"],
                "display": m["display"],
                "confidence": m["confidence"],
                "rationale": m["rationale"],
            }
            for m in state.get("code_mappings", [])
        ],
        "validation": {
            "errors": [
                {"resource": v["resource_id"], "message": v["message"]}
                for v in state.get("validation_issues", [])
                if v["severity"] == "error"
            ],
            "warnings": [
                {"resource": v["resource_id"], "message": v["message"]}
                for v in state.get("validation_issues", [])
                if v["severity"] == "warning"
            ],
        },
    }

    report_path = output_dir / "mapping_report.json"
    report_path.write_text(json.dumps(mapping_report, indent=2, ensure_ascii=False))
    output_paths.append(str(report_path))

    state["output_paths"] = output_paths

    # Console summary
    summary = mapping_report["summary"]
    logger.info(
        "\n┌─────────────────────────────────────────────────┐\n"
        "│           FHIR Transformation Complete           │\n"
        "├─────────────────────────────────────────────────┤\n"
        "│  Subjects processed : %-5d                     │\n"
        "│  FHIR resources     : %-5d                     │\n"
        "│  Code mappings      : %-5d                     │\n"
        "│    ✓ high confidence: %-5d                     │\n"
        "│    ~ medium         : %-5d                     │\n"
        "│    ✗ low / unmapped : %-5d                     │\n"
        "│  Validation errors  : %-5d                     │\n"
        "│  Output files       : %-5d                     │\n"
        "└─────────────────────────────────────────────────┘",
        summary["subjects"],
        summary["total_resources"],
        summary["total_variables"],
        summary["mapped_high_confidence"],
        summary["mapped_medium_confidence"],
        summary["mapped_low_confidence"] + summary["unmapped"],
        len(mapping_report["validation"]["errors"]),
        len(output_paths),
    )

    return state
