"""
Node 2 — CodeMapper

Maps each VariableMeta to a LOINC/SNOMED code using a two-step RAG approach:
  1. ChromaDB vector search retrieves the top-K semantically similar terms.
  2. Claude selects the best code from the candidates and explains its reasoning.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from src.state import CodeMapping, PipelineState, VariableMeta
from src.vector_store.store import TerminologyStore

logger = logging.getLogger(__name__)

# FHIR code system URIs
LOINC_SYSTEM = "http://loinc.org"
SNOMED_SYSTEM = "http://snomed.info/"

# Clusters where SNOMED is preferred over LOINC (conditions, procedures)
SNOMED_PREFERRED_CLUSTERS = {"clinical_diagnoses", "demographics"}


def map_codes(
    state: PipelineState,
    cfg: Any,
    store: TerminologyStore,
) -> PipelineState:
    """LangGraph node: map every VariableMeta to a LOINC or SNOMED code."""
    client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
    mappings: list[CodeMapping] = []
    index: dict[str, CodeMapping] = {}

    variables = state.get("variable_metadata", [])
    logger.info("CodeMapper: mapping %d variables", len(variables))

    for var_meta in variables:
        try:
            mapping = _map_single_variable(var_meta, client, store, cfg)
            mappings.append(mapping)
            index[f"{var_meta['cluster']}::{var_meta['variable']}"] = mapping

            logger.info(
                "  [%s] %s → %s %s (%s)",
                var_meta["cluster"],
                var_meta["variable"],
                mapping["code"],
                mapping["display"][:50],
                mapping["confidence"],
            )
        except Exception as exc:
            state["warnings"].append(
                f"Code mapping failed for {var_meta['variable']}: {exc}"
            )
            logger.warning("Mapping failed for %s: %s", var_meta["variable"], exc)

    state["code_mappings"] = mappings
    state["mapping_index"] = index
    logger.info("CodeMapper: completed %d / %d mappings", len(mappings), len(variables))
    return state


def _map_single_variable(
    var_meta: VariableMeta,
    client: anthropic.Anthropic,
    store: TerminologyStore,
    cfg: Any,
) -> CodeMapping:
    """Map a single variable to the best LOINC or SNOMED code via vector search + Claude."""
    cluster = var_meta["cluster"]
    prefer_snomed = cluster in SNOMED_PREFERRED_CLUSTERS

    query = _build_search_query(var_meta)

    if prefer_snomed:
        candidates = store.search_snomed(query, top_k=cfg.VECTOR_TOP_K)
        code_key = "snomed_code"
        display_key = "preferred_term"
        system_uri = SNOMED_SYSTEM
    else:
        candidates = store.search_loinc(query, top_k=cfg.VECTOR_TOP_K)
        code_key = "loinc_code"
        display_key = "long_common_name"
        system_uri = LOINC_SYSTEM

    selected = _llm_select_best_code(var_meta, candidates, code_key, display_key, client, cfg)

    # Pull UCUM unit from the matching candidate; fall back to top result if unmatched.
    ucum_unit = ""
    for candidate in candidates:
        if candidate.get(code_key) == selected["code"]:
            ucum_unit = candidate.get("units", "")
            break
    if not ucum_unit and candidates:
        ucum_unit = candidates[0].get("units", "")

    return CodeMapping(
        variable=var_meta["variable"],
        cluster=cluster,
        code_system=system_uri,
        code=selected["code"],
        display=selected["display"],
        confidence=selected["confidence"],
        rationale=selected["rationale"],
        candidates=candidates,
        ucum_unit=ucum_unit,
    )


def _build_search_query(var_meta: VariableMeta) -> str:
    """Build a rich natural-language query from VariableMeta for vector search."""
    parts = [var_meta["label"]]
    if var_meta["units"] and var_meta["units"] not in ("", "nan"):
        parts.append(f"units: {var_meta['units']}")
    if var_meta["cluster"]:
        parts.append(f"cluster: {var_meta['cluster']}")
    if var_meta["notes"] and len(var_meta["notes"]) > 5:
        # Take first 150 chars of notes as additional context
        parts.append(var_meta["notes"][:150])
    return " | ".join(parts)


def _llm_select_best_code(
    var_meta: VariableMeta,
    candidates: list[dict],
    code_key: str,
    display_key: str,
    client: anthropic.Anthropic,
    cfg: Any,
) -> dict:
    """Ask Claude to select the best code from candidates. Falls back to top vector result if parsing fails."""
    if not candidates:
        return {
            "code": "UNMAPPED",
            "display": var_meta["label"],
            "confidence": "low",
            "rationale": "No candidates returned by vector search.",
        }

    candidates_text = "\n".join(
        f"  {i+1}. Code: {c.get(code_key, '?')} | "
        f"Term: {c.get(display_key, '?')} | "
        f"Units: {c.get('units', c.get('hierarchy', ''))} | "
        f"Vector distance: {c.get('distance', '?'):.4f}"
        for i, c in enumerate(candidates)
    )

    prompt = f"""You are a clinical terminologist helping to map EHR and wearable device
variables to standard LOINC or SNOMED codes for FHIR R4 interoperability.

VARIABLE TO MAP
  Name       : {var_meta['variable']}
  Label      : {var_meta['label']}
  Units      : {var_meta['units']}
  Type       : {var_meta['var_type']}
  Value range: {var_meta['min_val']} – {var_meta['max_val']}
  Notes      : {var_meta['notes'][:300]}
  Cluster    : {var_meta['cluster']}

CANDIDATE CODES (ranked by semantic similarity)
{candidates_text}

Select the single best code from the candidates above.
If none is a good fit, you may say "UNMAPPED" but this should be rare.

Respond ONLY with valid JSON in this exact schema:
{{
  "code": "<code string>",
  "display": "<preferred display name>",
  "confidence": "high|medium|low",
  "rationale": "<one sentence explaining why this code is the best match>"
}}"""

    response = client.messages.create(
        model=cfg.CLAUDE_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        result = json.loads(raw)
        # Validate expected keys
        assert "code" in result and "display" in result
        return result
    except (json.JSONDecodeError, AssertionError, KeyError):
        logger.warning("LLM returned non-JSON for %s: %s", var_meta["variable"], raw[:200])
        # Fall back to top vector result
        top = candidates[0]
        return {
            "code": top.get(code_key, "UNMAPPED"),
            "display": top.get(display_key, var_meta["label"]),
            "confidence": "low",
            "rationale": "Fallback to top vector result (LLM parse failed).",
        }
