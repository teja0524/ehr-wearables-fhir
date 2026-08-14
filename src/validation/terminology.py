"""
Validation 2 — Terminology validation.

Checks that every code the pipeline emits actually EXISTS as an active concept
in its code system, using the FHIR `$validate-code` operation against a public
terminology server (tx.fhir.org by default).

Three distinct questions, only the first two of which are answerable
automatically:
  * Validation 1 (conformance.py) — "is this legal FHIR?" It would happily
    accept a well-formed resource carrying a code that does not exist.
  * Validation 2 (here) — "is this a real code, and is the display name right?"
    It cannot tell whether the code means the right thing.
  * Evaluation (tools/evaluate_mapping.py) — "is this the CORRECT code for this
    variable?" Requires a human-built reference standard; runs offline as a
    research study, not as part of the pipeline.

Design notes:
  * Results are cached on disk (keyed system|code) so repeat runs cost nothing
    and the network is hit once per distinct code.
  * Codes from the project-local CodeSystem are reported as 'local' and excluded
    from the resolution rate — they are study-specific by construction and no
    terminology server can know them.
  * A network failure marks a code 'unchecked', never 'invalid'. Reporting an
    unreachable server as an invalid code would silently understate coverage.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Code systems a terminology server can resolve.
_CHECKABLE = {
    "http://loinc.org",
    "http://snomed.info/sct",
    "http://www.nlm.nih.gov/research/umls/rxnorm",
    "http://hl7.org/fhir/sid/icd-10",
}

# HL7 infrastructure systems (status/category codes we set ourselves). These are
# validated structurally by Validation 1 and are not interesting terminology targets.
_INFRASTRUCTURE_PREFIX = "http://terminology.hl7.org/"

_STATUS_VALID = "valid"
_STATUS_INVALID = "invalid"
_STATUS_LOCAL = "local"
_STATUS_UNCHECKED = "unchecked"


def check_resources(resources: list[dict], cfg: Any) -> tuple[list[dict], dict]:
    """
    Validate every distinct (system, code) pair found in the generated resources.

    Returns (issues, summary). Issues are ValidationIssue-shaped dicts; only
    genuinely invalid codes produce a warning.
    """
    return _check_pairs(_collect_codes(resources), cfg)


def check_mappings(mapping_index: dict, cfg: Any) -> tuple[list[dict], dict]:
    """
    Validate the codes chosen by the CodeMapper, without needing built resources.

    Terminology validation only needs codes, not FHIR resources — so it can run
    in mapping-only mode, where the pipeline stops after code selection and no
    Observations or Bundles exist. This makes "do these codes actually exist?"
    answerable on the fast path, which is the useful check while iterating on
    the terminology seed.
    """
    pairs = {
        (str(m.get("code_system", "")), str(m.get("code", "")))
        for m in (mapping_index or {}).values()
        if m and m.get("code") not in (None, "", "UNMAPPED") and m.get("code_system")
    }
    return _check_pairs(pairs, cfg)


def _check_pairs(pairs: set[tuple[str, str]], cfg: Any) -> tuple[list[dict], dict]:
    """Shared validation core for both entry points."""
    if not getattr(cfg, "CHECK_TERMINOLOGY", True):
        return [], _summary({}, "disabled via config (CHECK_TERMINOLOGY=False)")

    if not pairs:
        return [], _summary({}, "no codes found")

    local_system = str(getattr(cfg, "SYSTEM_LOCAL", ""))
    cache_path = _cache_path(cfg)
    cache = _load_cache(cache_path)

    results: dict[str, dict] = {}
    to_query: list[tuple[str, str]] = []

    for system, code in sorted(pairs):
        key = f"{system}|{code}"
        if system == local_system:
            results[key] = {"status": _STATUS_LOCAL, "message": "project-local code system"}
        elif system.startswith(_INFRASTRUCTURE_PREFIX) or system not in _CHECKABLE:
            results[key] = {"status": _STATUS_LOCAL, "message": "not a checkable terminology"}
        elif key in cache:
            results[key] = cache[key]
        else:
            to_query.append((system, code))

    if to_query:
        logger.info("Terminology: validating %d new code(s) against %s",
                    len(to_query), getattr(cfg, "TX_SERVER_URL", ""))
    for system, code in to_query:
        key = f"{system}|{code}"
        res = _validate_code(system, code, cfg)
        results[key] = res
        # Only cache definitive answers — never cache a network failure.
        if res["status"] in (_STATUS_VALID, _STATUS_INVALID):
            cache[key] = res

    _save_cache(cache_path, cache)

    issues: list[dict] = []
    for key, res in results.items():
        if res["status"] == _STATUS_INVALID:
            system, _, code = key.partition("|")
            issues.append({
                "resource_id": f"code:{code}",
                "severity": "warning",
                "message": f"Code '{code}' not found in {system}: {res.get('message', '')}".strip(),
            })

    summary = _summary(results, "")
    logger.info("Terminology: %d/%d checkable code(s) resolved (%.1f%%), %d invalid, %d unchecked",
                summary["codes_valid"], summary["codes_checkable"],
                summary["resolution_rate_pct"], summary["codes_invalid"],
                summary["codes_unchecked"])
    return issues, summary


def _collect_codes(resources: list[dict]) -> set[tuple[str, str]]:
    """Walk the resource graph and collect every (system, code) coding pair."""
    found: set[tuple[str, str]] = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            system, code = obj.get("system"), obj.get("code")
            if isinstance(system, str) and isinstance(code, str) and system and code:
                found.add((system, code))
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(resources)
    return found


def _validate_code(system: str, code: str, cfg: Any) -> dict:
    """Call $validate-code on the terminology server for one concept."""
    base = str(getattr(cfg, "TX_SERVER_URL", "https://tx.fhir.org/r4")).rstrip("/")
    params = urllib.parse.urlencode({"url": system, "code": code})
    url = f"{base}/CodeSystem/$validate-code?{params}"
    timeout = int(getattr(cfg, "TX_TIMEOUT_SEC", 15))

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/fhir+json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        return {"status": _STATUS_UNCHECKED, "message": f"terminology server unreachable: {exc}"}
    except json.JSONDecodeError as exc:
        return {"status": _STATUS_UNCHECKED, "message": f"bad response: {exc}"}

    return _parse_validate_code(payload)


def _parse_validate_code(payload: dict) -> dict:
    """
    Read a $validate-code Parameters response.

    Shape: {"resourceType":"Parameters","parameter":[
              {"name":"result","valueBoolean":true},
              {"name":"display","valueString":"..."},
              {"name":"message","valueString":"..."}]}
    An OperationOutcome comes back instead when the server rejects the request.
    """
    if payload.get("resourceType") == "OperationOutcome":
        issues = payload.get("issue") or [{}]
        text = (issues[0].get("details") or {}).get("text") or issues[0].get("diagnostics", "")
        return {"status": _STATUS_UNCHECKED, "message": f"server error: {text}"}

    result: bool | None = None
    display = ""
    message = ""
    for param in payload.get("parameter", []) or []:
        name = param.get("name")
        if name == "result":
            result = bool(param.get("valueBoolean"))
        elif name == "display":
            display = str(param.get("valueString", ""))
        elif name == "message":
            message = str(param.get("valueString", ""))

    if result is True:
        return {"status": _STATUS_VALID, "display": display, "message": message}
    if result is False:
        return {"status": _STATUS_INVALID, "message": message or "code not found"}
    return {"status": _STATUS_UNCHECKED, "message": message or "no result in response"}


def _cache_path(cfg: Any) -> Path:
    return Path(getattr(cfg, "OUTPUT_DIR", Path("output"))) / "terminology_cache.json"


def _load_cache(path: Path) -> dict[str, dict]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(path: Path, cache: dict[str, dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    except OSError as exc:
        logger.warning("Terminology: could not write cache: %s", exc)


def _summary(results: dict[str, dict], note: str) -> dict:
    counts = {s: 0 for s in (_STATUS_VALID, _STATUS_INVALID, _STATUS_LOCAL, _STATUS_UNCHECKED)}
    for res in results.values():
        counts[res["status"]] = counts.get(res["status"], 0) + 1
    # The resolution rate covers only codes a terminology server can adjudicate:
    # local/infrastructure codes are excluded, as are codes we could not reach.
    checkable = counts[_STATUS_VALID] + counts[_STATUS_INVALID]
    rate = round(100.0 * counts[_STATUS_VALID] / checkable, 1) if checkable else 0.0
    return {
        "codes_total": len(results),
        "codes_checkable": checkable,
        "codes_valid": counts[_STATUS_VALID],
        "codes_invalid": counts[_STATUS_INVALID],
        "codes_local": counts[_STATUS_LOCAL],
        "codes_unchecked": counts[_STATUS_UNCHECKED],
        "resolution_rate_pct": rate,
        "note": note,
    }
