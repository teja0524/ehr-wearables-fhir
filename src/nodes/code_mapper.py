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

from src import llm, overrides
from src.fhir.resources import SYS_LOINC as LOINC_SYSTEM, SYS_RXNORM as RXNORM_SYSTEM, SYS_SNOMED as SNOMED_SYSTEM
from src.state import CodeMapping, PipelineState, VariableMeta
from src.vector_store.store import TerminologyStore

logger = logging.getLogger(__name__)

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


def _cache_path(state: PipelineState, cfg: Any):
    from pathlib import Path
    subdir = state.get("dataset", {}).get("output_subdir", "")
    base: Path = cfg.OUTPUT_DIR / subdir if subdir else cfg.OUTPUT_DIR
    return base / "mapping_cache.json"


def map_codes(state: PipelineState, cfg: Any, store: TerminologyStore) -> PipelineState:
    """LangGraph node: map every VariableMeta to a code from its cluster's vocabularies."""
    registry = state.get("dataset", {}).get("clusters", getattr(cfg, "CLUSTER_REGISTRY", {}))
    variables = state.get("variable_metadata", [])

    # Reuse a previous run's mappings if requested and available. Mapping is
    # per-variable (not per-subject), so the cache is valid across subject counts.
    cache_file = _cache_path(state, cfg)
    if state.get("skip_mapping") and cache_file.exists():
        cached = json.loads(cache_file.read_text())
        index = {k: v for k, v in cached.items()}
        state["code_mappings"] = list(index.values())
        state["mapping_index"] = index
        logger.info("CodeMapper: loaded %d cached mappings from %s", len(index), cache_file.name)
        if state.get("apply_overrides", True):
            return overrides.apply_overrides(state, cfg)
        return state

    model = state.get("model") or cfg.CLAUDE_MODEL
    mappings: list[CodeMapping] = []
    index: dict[str, CodeMapping] = {}

    logger.info("CodeMapper: mapping %d variables with model '%s'", len(variables), model)

    for var_meta in variables:
        try:
            mapping = _map_single_variable(var_meta, model, store, cfg, registry)
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

    # Persist for fast --skip-mapping reruns.
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    except Exception as exc:
        logger.warning("Could not write mapping cache: %s", exc)

    # Technician overrides win over the automatic mapping (applied last so they
    # survive re-mapping); the raw cache above stays override-free. A run can opt
    # out (apply_overrides=False) to get a clean mapping ignoring saved edits.
    if state.get("apply_overrides", True):
        return overrides.apply_overrides(state, cfg)
    return state


def _vocabs_for(cluster: str, cfg: Any, registry: dict) -> list[str]:
    mode = registry.get(cluster, {}).get("mapping", "rag_loinc")
    return cfg.MAPPING_VOCABS.get(mode, ["loinc"])


def _map_single_variable(var_meta, model, store, cfg, registry) -> CodeMapping:
    """Pool candidates from the cluster's vocabularies, then let the LLM pick the best."""
    vocabs = _vocabs_for(var_meta["cluster"], cfg, registry)

    # Retrieval ablation (config.NO_RAG). No shortlist is built and the store is
    # never queried; the model must recall a code unaided. Returns early so the
    # retrieval path below is untouched when the flag is off.
    if getattr(cfg, "NO_RAG", False):
        selected = _llm_recall_code(var_meta, vocabs, model, cfg)
        system_uri = _VOCAB.get(selected.get("vocab", ""), {}).get("system", LOINC_SYSTEM)
        return CodeMapping(
            variable=var_meta["variable"], cluster=var_meta["cluster"],
            code_system=system_uri, code=selected["code"], display=selected["display"],
            confidence=selected["confidence"], rationale=selected["rationale"],
            # Empty by construction. Downstream consumers already handle a
            # variable with no candidates (it is the UNMAPPED case), and the
            # empty list is itself the record that this run had no retrieval.
            candidates=[], ucum_unit="",
        )

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

    selected = _llm_select_best_code(var_meta, candidates, model, cfg)

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


def _llm_select_best_code(var_meta, candidates, model, cfg) -> dict:
    """Ask the LLM to select the best code. Falls back to the closest vector result on parse failure."""
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

    raw = llm.complete(prompt, model=model, max_tokens=400,
                       anthropic_api_key=cfg.ANTHROPIC_API_KEY).strip()
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
        # Guard against null/empty values the model may emit (e.g. for UNMAPPED).
        if not result.get("display"):
            result["display"] = var_meta["label"]
        if not result.get("code"):
            result["code"] = "UNMAPPED"
        return result
    except (json.JSONDecodeError, AssertionError, KeyError):
        logger.warning("LLM returned non-JSON for %s: %s", var_meta["variable"], raw[:200])
        top = candidates[0]
        return {"code": top["code"], "display": top["term"],
                "confidence": "low", "rationale": "Fallback to closest vector result (LLM parse failed)."}


def _llm_recall_code(var_meta, vocabs, model, cfg) -> dict:
    """Ask the LLM for a code from its own knowledge, with no candidates supplied.

    Used only by the NO_RAG ablation. The variable block, the option to decline,
    the confidence vocabulary and the response schema are deliberately identical
    to `_llm_select_best_code`, so that the presence or absence of the retrieved
    shortlist is the only difference between the two conditions.

    Returned codes are NOT checked here. Whether the model invents a plausible
    but non-existent identifier is the measurement, so the output is passed
    through unaltered and left to the terminology validation layer to judge.
    """
    allowed = ", ".join(_VOCAB[v]["label"] for v in vocabs if v in _VOCAB) or "LOINC"

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

No candidate list is provided. Recall the single best code from your own
knowledge of these terminologies. Permitted terminologies for this variable:
{allowed}.

Give the code exactly as it appears in the source terminology. If you cannot
recall a specific code for this variable with reasonable certainty, return
"UNMAPPED" (the value will fall back to a local code) — do not guess.

Respond ONLY with valid JSON:
{{
  "code": "<code string or UNMAPPED>",
  "vocab": "<loinc|snomed|rxnorm>",
  "display": "<preferred display name>",
  "confidence": "high|medium|low",
  "rationale": "<one sentence>"
}}"""

    raw = llm.complete(prompt, model=model, max_tokens=400,
                       anthropic_api_key=cfg.ANTHROPIC_API_KEY).strip()
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
        if not result.get("display"):
            result["display"] = var_meta["label"]
        if not result.get("code"):
            result["code"] = "UNMAPPED"
        # Constrain the system to one this cluster is allowed to use, so a
        # recalled code is never filed under a terminology the RAG arm could
        # not have chosen either.
        vocab = str(result.get("vocab", "")).strip().lower()
        result["vocab"] = vocab if vocab in vocabs else (vocabs[0] if vocabs else "loinc")
        return result
    except (json.JSONDecodeError, AssertionError, KeyError):
        # No shortlist exists to fall back to, so a parse failure is UNMAPPED.
        logger.warning("LLM returned non-JSON for %s: %s", var_meta["variable"], raw[:200])
        return {"code": "UNMAPPED", "vocab": vocabs[0] if vocabs else "loinc",
                "display": var_meta["label"], "confidence": "low",
                "rationale": "LLM parse failed and no retrieval fallback exists."}
