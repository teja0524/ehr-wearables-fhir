"""
Validation 1 — Conformance validation against the FHIR specification.

Wraps the official HL7 FHIR validator (`validator_cli.jar`, the reference
implementation maintained by HL7) instead of hand-written field checks, so the
reported conformance rate is authoritative rather than a self-defined subset of
rules.

What it checks: structural conformance to FHIR R4 (4.0.1) — cardinality,
datatypes, required elements, and the spec's invariants — plus any profile a
resource declares in `meta.profile`. Because the FHIRBuilder already stamps
`meta.profile` (e.g. the vitalsigns profile on vital-sign Observations), those
profile claims are verified automatically without extra configuration.

What it does NOT check: whether a code is the *right* code for a variable
(the evaluation study), or whether a code exists in its code system — terminology checking is
handled separately in `terminology.py` (Validation 2), so this runs with `-tx n/a`
(terminology server disabled) for speed and to keep the two layers'
results cleanly separated.

Requires: Java 11+ on PATH and validator_cli.jar. When either is missing the
caller falls back to the built-in structural checks; nothing hard-fails.
h
"""

from __future__ import annotations

import functools
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Severity levels the FHIR validator emits in an OperationOutcome.
_FATAL = {"fatal", "error"}


@functools.lru_cache(maxsize=1)
def java_available() -> bool:
    """
    True only if Java can actually EXECUTE.

    `shutil.which("java")` is not sufficient: macOS ships a /usr/bin/java stub
    that exists on PATH but exits non-zero with "Unable to locate a Java
    Runtime" when no JDK/JRE is installed. Trusting `which` there makes the
    pipeline believe the validator is available, skip the fallback, and report
    zero issues — a falsely clean result. So actually run `java -version`.
    """
    exe = shutil.which("java")
    if not exe:
        return False
    try:
        proc = subprocess.run([exe, "-version"], capture_output=True, timeout=30)
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def jar_path(cfg: Any) -> Path | None:
    """Resolve the validator jar from config/env, if it exists."""
    raw = getattr(cfg, "FHIR_VALIDATOR_JAR", None)
    if not raw:
        return None
    p = Path(raw)
    return p if p.exists() else None


def validator_available(cfg: Any) -> tuple[bool, str]:
    """(is_available, reason_if_not) — used to decide primary vs fallback path."""
    if not getattr(cfg, "USE_OFFICIAL_VALIDATOR", True):
        return False, "disabled via config (USE_OFFICIAL_VALIDATOR=False)"
    if not java_available():
        return False, ("no working Java runtime (install a JRE/JDK, e.g. "
                       "`brew install openjdk@17` on macOS)")
    if jar_path(cfg) is None:
        return False, f"validator jar not found at {getattr(cfg, 'FHIR_VALIDATOR_JAR', '(unset)')}"
    return True, ""


def validate_bundles(bundles: dict[str, dict], cfg: Any) -> tuple[list[dict], dict]:
    """
    Validate per-subject Bundles with the official validator.

    Node 4 runs before bundles are exported, so the JSON is written to a
    temporary directory and validated there. Every file is passed to a single
    JVM invocation (JVM startup dominates runtime, per-file cost is small).

    Returns (issues, summary). `issues` are dicts shaped like ValidationIssue.
    """
    if not bundles:
        return [], _summary(0, 0, 0, 0, "no bundles to validate", ok=False)

    limit = int(getattr(cfg, "VALIDATOR_MAX_BUNDLES", 25) or 0)
    items = list(bundles.items())
    sampled = items[:limit] if limit > 0 else items

    with tempfile.TemporaryDirectory(prefix="fhir_validate_") as tmp:
        tmpdir = Path(tmp)
        paths = []
        for subject_id, bundle in sampled:
            p = tmpdir / f"{subject_id}_bundle.json"
            p.write_text(json.dumps(bundle, ensure_ascii=False))
            paths.append(p)

        out_file = tmpdir / "_outcome.json"
        cmd = _build_command(paths, out_file, cfg)
        logger.info("Conformance: running official FHIR validator on %d bundle(s)", len(paths))
        logger.debug("Validator command: %s", " ".join(str(c) for c in cmd))

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=int(getattr(cfg, "VALIDATOR_TIMEOUT_SEC", 900)),
            )
        except subprocess.TimeoutExpired:
            logger.error("Conformance: validator timed out")
            return [], _summary(len(paths), 0, 0, 0, "validator timed out", ok=False)
        except Exception as exc:                       # noqa: BLE001 - report, don't crash the run
            logger.error("Conformance: validator failed to run: %s", exc)
            return [], _summary(len(paths), 0, 0, 0, f"validator failed to run: {exc}", ok=False)

        if not out_file.exists():
            # The validator exits non-zero when a resource is invalid, which is
            # expected; a missing output file means it genuinely failed to start.
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
            msg = "; ".join(tail) or f"exit code {proc.returncode}"
            logger.error("Conformance: no validator output (%s)", msg)
            return [], _summary(len(paths), 0, 0, 0, f"no validator output: {msg}", ok=False)

        try:
            data = json.loads(out_file.read_text())
        except json.JSONDecodeError as exc:
            return [], _summary(len(paths), 0, 0, 0, f"could not parse validator output: {exc}", ok=False)

    subject_ids = [sid for sid, _ in sampled]
    issues = _parse_output(data, subject_ids)
    errs = [i for i in issues if i["severity"] == "error"]
    # Count bundles with NO errors. Deriving this by subtracting the number of
    # distinct error ids is wrong whenever attribution falls back to a shared
    # placeholder — it silently inflates the conformance rate.
    errored = {i["resource_id"] for i in errs}
    clean = sum(1 for sid in subject_ids if sid not in errored)
    if errs and "?" in errored:
        # Attribution failed for at least one outcome; report conservatively
        # rather than crediting bundles we cannot prove are clean.
        clean = 0
    summary = _summary(len(paths), clean, len(errs),
                       sum(1 for i in issues if i["severity"] == "warning"), "")
    summary["profiles"] = _igs(cfg)
    logger.info("Conformance: %d/%d bundle(s) conform (%d error(s), %d warning(s))",
                clean, len(paths), len(errs), summary["warnings"])
    return issues, summary


def _build_command(paths: list[Path], out_file: Path, cfg: Any) -> list[str]:
    cmd: list[str] = [
        "java", "-Xmx2g", "-jar", str(jar_path(cfg)),
        *[str(p) for p in paths],
        "-version", str(getattr(cfg, "FHIR_VERSION", "4.0.1")),
        "-output", str(out_file),
    ]
    # Terminology is validated separately in Validation 2; disabling the tx server
    # here keeps this pass fast and offline-capable.
    if getattr(cfg, "VALIDATOR_DISABLE_TX", True):
        cmd += ["-tx", "n/a"]
    for ig in _igs(cfg):
        cmd += ["-ig", ig]
    return cmd


def _igs(cfg: Any) -> list[str]:
    return list(getattr(cfg, "VALIDATOR_IG_PACKAGES", []) or [])


def _parse_output(data: dict, sources: list[str] | None = None) -> list[dict]:
    """
    Normalise validator output into ValidationIssue-shaped dicts.

    One input file → a single OperationOutcome; several files → a Bundle whose
    entries are OperationOutcomes. Both shapes are handled.

    `sources` are the subject ids of the validated bundles, in the order they
    were passed to the validator. Outcomes come back in that same order, so
    positional mapping is used to attribute each issue to its bundle. This is
    more reliable than reading a source extension, which the validator does not
    always emit — and mis-attribution directly corrupts the conformance rate.
    """
    outcomes: list[dict] = []
    if data.get("resourceType") == "Bundle":
        for entry in data.get("entry", []):
            res = entry.get("resource") or {}
            if res.get("resourceType") == "OperationOutcome":
                outcomes.append(res)
    elif data.get("resourceType") == "OperationOutcome":
        outcomes.append(data)

    # Positional mapping only when the counts line up; otherwise fall back to
    # the extension, and finally to "?" meaning "could not attribute".
    aligned = bool(sources) and len(sources) == len(outcomes)

    issues: list[dict] = []
    for idx, outcome in enumerate(outcomes):
        source = sources[idx] if aligned else _outcome_source(outcome)
        for issue in outcome.get("issue", []):
            severity = str(issue.get("severity", "information")).lower()
            # The validator's "information" notes are noise for a conformance
            # rate; keep them as info so they can be filtered out downstream.
            if severity == "fatal":
                severity = "error"
            elif severity not in ("error", "warning"):
                severity = "info"
            issues.append({
                "resource_id": source,
                "severity": severity,
                "message": _issue_text(issue),
            })
    return issues


def _outcome_source(outcome: dict) -> str:
    """
    Fallback attribution from the OperationOutcome's source extension.

    Returns "?" when the source cannot be determined, which the caller treats as
    "attribution failed" and reports conservatively — never as a clean bundle.
    """
    for ext in outcome.get("extension", []):
        url = str(ext.get("url", ""))
        if url.endswith("source") or url.endswith("file"):
            val = ext.get("valueString") or ext.get("valueUri") or ""
            if val:
                return Path(str(val)).name.replace("_bundle.json", "")
    return "?"


def _issue_text(issue: dict) -> str:
    text = (issue.get("details") or {}).get("text") or issue.get("diagnostics") or "Unspecified issue"
    loc = issue.get("expression") or issue.get("location") or []
    where = loc[0] if isinstance(loc, list) and loc else ""
    return f"{text} [{where}]" if where else str(text)


def _summary(checked: int, conformant: int, errors: int, warnings: int, note: str,
             ok: bool = True) -> dict:
    """
    Build the conformance summary.

    `ok` reports whether the validator actually produced a verdict. It must be
    False on every failure path: a run that could not execute has NOT shown the
    resources to be conformant, and the caller uses this flag to fall back to
    the structural checks rather than reporting a falsely clean result.
    """
    rate = round(100.0 * conformant / checked, 1) if checked else 0.0
    return {
        "ok": ok,
        "validator": "official HL7 validator_cli",
        "bundles_checked": checked if ok else 0,
        "bundles_conformant": conformant,
        "conformance_rate_pct": rate if ok else None,
        "errors": errors,
        "warnings": warnings,
        "note": note,
    }
