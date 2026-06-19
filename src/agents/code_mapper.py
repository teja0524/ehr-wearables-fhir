"""
Node 2 — CodeMapper

Maps each VariableMeta to a standard code using a two-step RAG approach:
  1. ChromaDB vector search retrieves the top-K semantically similar terms
     from the terminologies relevant to the variable's cluster
     (LOINC, SNOMED CT and/or RxNorm).
  2. Claude selects the best code from the pooled candidates and explains why.

Which terminologies are searched is driven by the cluster's 'mapping' mode in
config.CLUSTER_REGISTRY (see config.MAPPING_VOCABS):
  rag_loinc   → LOINC
  rag_rxnorm  → RxNorm
  rag_clinical→ LOINC + SNOMED
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
SNOMED_SYSTEM = "http://snomed.info/sct"
RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"

# Per-vocabulary search adapters: store method + metadata keys + system URI.
_VOCAB = {
    "loinc": {
        "search": "search_loinc", "code_key": "loinc_code",
        "term_key": "long_common_name", "system": LOINC_SYSTEM, "label": "LOINC",
    },
    "snomed": {
        "search": "search_snomed", "code_key": "snomed_code",
        "term_key": "preferred_term", "system": SNOMED_SYSTEM, "label": "SNOMED CT",
    },
    "rxnorm": {
        "search": "search_rxnorm", "code_key": "rxnorm_code",
        "term_key": "ingredient_name", "system": RXNORM_SYSTEM, "label": "RxNorm",
    },
}


def map_codes(state: PipelineState, cfg: Any, store: TerminologyStore) -> PipelineState:
    """LangGraph node: map every VariableMeta to a code from its cluster's vocabularies."""
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
            logger.info("  [%s] %s → %s %s (%s)", var_meta["cluster"], var_meta["variable"],
                        mapping["code"], mapping["display"][:50], mapping["confidence"])
        except Exception as exc:
            state["warnings"].append(f"Code mapping failed for {var_meta['variable']}: {exc}")
            logger.warning("Mapping failed for %s: %s", var_meta["variable"], exc)

    state["code_mappings"] = mappings
    state["mapping_index"] = index
    mapped = sum(1 for m in mappings if m["code"] != "UNMAPPED")
    logger.info("CodeMapper: %d/%d variables mapped to a standard code", mapped, len(variables))
    return state


def _vocabs_for(cluster: str, cfg: Any) -> list[str]:
    mode = cfg.CLUSTER_REGISTRY.get(cluster, {}).get("mapping", "rag_loinc")
    return cfg.MAPPING_VOCABS.get(mode, ["loinc"])


def _map_single_variable(var_meta, client, store, cfg) -> CodeMapping:
    """Pool candidates from the cluster's vocabularies, then let Claude pick the best."""
    vocabs = _vocabs_for(var_meta["cluster"], cfg)
    query = _build_search_query(var_meta)

    candidates: list[dict] = []
    for vocab in vocabs:
        spec = _VOCAB[vocab]
        raw = getattr(store, spec["search"])(query, top_k=cfg.VECTOR_TOP_K)
        for c in raw:
            candidates.append({
                "code": c.get(spec["code_key"], "?"),
                "term": c.get(spec["term_key"], "?"),
                "system_uri": spec["system"],
                "system_label": spec["label"],
                "units": c.get("units", ""),
                "distance": c.get("distance", 9.99),
            })
    # Best candidates first (closest vector distance) across all vocabularies.
    candidates.sort(key=lambda c: c["distance"])

    selected = _llm_select_best_code(var_meta, candidates, client, cfg)

    # Resolve system + UCUM unit from the chosen candidate.
    system_uri, ucum_unit = (candidates[0]["system_uri"] if candidates else LOINC_SYSTEM), ""
    for c in candidates:
        if str(c["code"]) == str(selected["code"]):
            system_uri = c["system_uri"]
            ucum_unit = c.get("units", "")
            break

    return CodeMapping(
        variable=var_meta["variable"], cluster=var_meta["cluster"],
        code_system=system_uri, code=selected["code"], display=selected["display"],
        confidence=selected["confidence"], rationale=selected["rationale"],
        candidates=candidates, ucum_unit=ucum_unit,
    )


def _build_search_query(var_meta: VariableMeta) -> str:
    """Build a rich natural-language query from VariableMeta for vector search."""
    parts = [var_meta["label"]]
    if var_meta["units"] and var_meta["units"] not in ("", "nan"):
        parts.append(f"units: {var_meta['units']}")
    if var_meta["cluster"]:
        parts.append(f"cluster: {var_meta['cluster']}")
    if var_meta["notes"] and len(var_meta["notes"]) > 5:
        parts.append(var_meta["notes"][:150])
    return " | ".join(parts)


def _llm_select_best_code(var_meta, candidates, client, cfg) -> dict:
    """Ask Claude to select the best code. Falls back to the closest vector result on parse failure."""
    if not candidates:
        return {"code": "UNMAPPED", "display": var_meta["label"],
                "confidence": "low", "rationale": "No candidates returned by vector search."}

    candidates_text = "\n".join(
        f"  {i+1}. [{c['system_label']}] Code: {c['code']} | Term: {c['term']} | "
        f"Units: {c.get('units', '')} | Vector distance: {c['distance']:.4f}"
        for i, c in enumerate(candidates)
    )

    prompt = f"""You are a clinical terminologist mapping study variables to standard
codes (LOINC / SNOMED CT / RxNorm) for FHIR R4 interoperability.

VARIABLE TO MAP
  Name       : {var_meta['variable']}
  Label      : {var_meta['label']}
  Units      : {var_meta['units']}
  Type       : {var_meta['var_type']}
  Value range: {var_meta['min_val']} – {var_meta['max_val']}
  Notes      : {var_meta['notes'][:300]}
  Cluster    : {var_meta['cluster']}

CANDIDATE CODES (pooled across terminologies, ranked by semantic similarity)
{candidates_text}

Select the single best code from the candidates above. Only return a code that
appears in the list. If none is a genuine match, return "UNMAPPED" (the value
will fall back to a local code) — do not force a poor match.

Respond ONLY with valid JSON:
{{
  "code": "<code string or UNMAPPED>",
  "display": "<preferred display name>",
  "confidence": "high|medium|low",
  "rationale": "<one sentence>"
}}"""

    response = client.messages.create(
        model=cfg.CLAUDE_MODEL, max_tokens=400,
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
        assert "code" in result and "display" in result
        result.setdefault("confidence", "low")
        result.setdefault("rationale", "")
        return result
    except (json.JSONDecodeError, AssertionError, KeyError):
        logger.warning("LLM returned non-JSON for %s: %s", var_meta["variable"], raw[:200])
        top = candidates[0]
        return {"code": top["code"], "display": top["term"],
                "confidence": "low", "rationale": "Fallback to closest vector result (LLM parse failed)."}
