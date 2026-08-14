"""
Local web server that runs the FHIR transformation pipeline from the dashboard.

Run it (in the project venv, with the deps installed):

    python server.py            # then open http://127.0.0.1:8000

The browser dashboard (viewer/fhir_viewer.html) lets you upload data files,
pick a model and output type (mapping report vs full bundles) and a patient
count, then runs the pipeline and shows the results — all locally. Nothing is
uploaded anywhere except your own machine; the server binds to localhost only.

Endpoints:
    GET  /            → the dashboard
    GET  /models      → the model dropdown choices (+ whether each key is set)
    POST /run         → run the pipeline on uploaded files, return JSON results
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_file

import config as cfg

app = Flask(__name__)
log = logging.getLogger("server")

VIEWER = cfg.BASE_DIR / "viewer" / "fhir_viewer.html"

# The terminology vector store is expensive to build, so build it once and reuse
# it across runs.
_STORE = None

# Remembers the most recent run so /rebuild can reuse its uploaded data.
LAST_RUN: dict = {}


def _get_store():
    global _STORE
    if _STORE is None:
        from src.vector_store.store import TerminologyStore
        _STORE = TerminologyStore.from_config(cfg)
        _STORE.build_or_load()
    return _STORE


@app.get("/")
def index():
    return send_file(VIEWER)


@app.get("/models")
def models():
    """Model choices for the dropdown, flagged with whether their API key is set."""
    out = []
    for m in cfg.MODEL_CHOICES:
        out.append({**m, "key_present": bool(os.getenv(m["key"], ""))})
    return jsonify({"models": out, "default": cfg.CLAUDE_MODEL})


def _detect_dataset(upload_dir: Path) -> str:
    """Pick the dataset template from the uploaded files.

    An ACE upload includes the Excel dictionary (and semicolon-delimited CSVs);
    anything else is treated as the MFU/ComfortAge layout.
    """
    for p in upload_dir.rglob("*"):
        if p.suffix.lower() in (".xlsx", ".xls"):
            return "ace"
    return "mfu"


def _save_uploads(files) -> Path:
    """Save uploaded files into a temp dir, preserving any folder structure."""
    tmp = Path(tempfile.mkdtemp(prefix="fhir_upload_"))
    for f in files:
        # webkitRelativePath (folder upload) arrives in f.filename as a/b/c.csv
        rel = Path(f.filename).as_posix().lstrip("/")
        dest = tmp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.save(dest)
    return tmp


def _resolve_data_root(upload_dir: Path) -> Path:
    """
    Find the directory the cluster paths are relative to.

    A browser folder upload puts the selected folder's own name at the front of
    every file's relative path (`sample_mfu/blood/blood_labs.csv`), so the files
    land one level deeper than the dataset config expects
    (`<data_dir>/blood/blood_labs.csv`). Descend through any single-directory
    wrappers so both a folder upload and a flat multi-file selection resolve to
    the same root.
    """
    entries = [p for p in upload_dir.iterdir() if not p.name.startswith(".")]
    # Exactly one wrapper level is added by the browser, so descend at most one.
    # Descending greedily would walk into a cluster directory whenever a dataset
    # happens to have a single cluster, which is worse than not descending.
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return upload_dir


def _dataset_config(name: str, data_dir: Path) -> dict:
    """Clone a dataset template from config and point it at the uploaded dir."""
    base = dict(cfg.DATASETS[name])
    base["data_dir"] = data_dir
    if base.get("dict_format") == "ace_xlsx":
        xlsx = next((p for p in data_dir.rglob("*.xls*")), None)
        if xlsx is not None:
            base["dict_file"] = xlsx.name
    return base


@app.post("/run")
def run():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    model = request.form.get("model") or cfg.CLAUDE_MODEL
    mode = request.form.get("mode", "mapping")            # "mapping" | "full"
    ds_choice = request.form.get("dataset", "auto")
    try:
        max_subjects = int(request.form.get("max_subjects", "20"))
    except ValueError:
        max_subjects = 20
    skip_mapping = request.form.get("skip_mapping", "false") == "true"
    apply_overrides = request.form.get("apply_overrides", "true") == "true"

    # Validate that the chosen model's API key is configured.
    choice = next((m for m in cfg.MODEL_CHOICES if m["id"] == model), None)
    if choice and not os.getenv(choice["key"], "") and not (choice["key"] == "ANTHROPIC_API_KEY" and cfg.ANTHROPIC_API_KEY):
        return jsonify({"error": f"{choice['provider']} model selected but {choice['key']} is not set in your environment/.env."}), 400

    upload_dir = _resolve_data_root(_save_uploads(files))
    log.info("Upload root resolved to %s", upload_dir)
    dataset_name = _detect_dataset(upload_dir) if ds_choice == "auto" else ds_choice
    resp, code = _execute(upload_dir, dataset_name, model, mode, max_subjects, skip_mapping, apply_overrides)
    if code == 200:
        LAST_RUN.update({"upload_dir": upload_dir, "dataset_name": dataset_name,
                         "model": model, "max_subjects": max_subjects})
    return jsonify(resp), code


def _execute(upload_dir: Path, dataset_name: str, model: str, mode: str,
             max_subjects: int, skip_mapping: bool, apply_overrides: bool = True):
    """Run the pipeline and return (response_dict, http_status)."""
    dataset = _dataset_config(dataset_name, upload_dir)

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    prev_level = root.level
    root.setLevel(logging.INFO)
    try:
        from src.graph import initial_state, export_output
        from src.agents.schema_parser import parse_schema
        from src.agents.code_mapper import map_codes
        from src.agents.fhir_builder import build_fhir
        from src.agents.validator import validate_fhir

        store = _get_store()
        state = initial_state(
            clusters=list(dataset["all_clusters"]),
            max_subjects=max_subjects if max_subjects > 0 else None,
            dataset=dataset, skip_mapping=skip_mapping, model=model,
            apply_overrides=apply_overrides,
        )
        # Write to the stable output/<dataset>/ folder (same as the CLI), so the
        # mapping_report.json and bundles on disk are always the current ones —
        # including any applied overrides — and are easy to find/reopen.
        state = parse_schema(state, cfg)
        state = map_codes(state, cfg, store)
        if mode == "full":
            state = build_fhir(state, cfg)
            state = validate_fhir(state, cfg)
        else:
            # Mapping-only: no resources exist, so FHIR conformance (Validation 1)
            # cannot run — but terminology validation (Validation 2) only needs the
            # codes, so it still applies.
            from src.validation import terminology
            _, term_summary = terminology.check_mappings(state.get("mapping_index", {}), cfg)
            state["terminology_summary"] = term_summary
        state = export_output(state, cfg)

        subdir = dataset.get("output_subdir", "")
        out_base = (cfg.OUTPUT_DIR / subdir) if subdir else cfg.OUTPUT_DIR
        report_path = out_base / "mapping_report.json"
        mapping_report = json.loads(report_path.read_text()) if report_path.exists() else None
        bundles = [{"id": sid, "bundle": b} for sid, b in state.get("fhir_bundles", {}).items()]
        errors = [i for i in state.get("validation_issues", []) if i["severity"] == "error"]
        return {
            "mode": mode, "dataset": dataset_name, "model": model,
            "results_dir": str(out_base),
            "subjects": len(state.get("fhir_bundles", {})),
            "validation_errors": len(errors),
            "mapping_report": mapping_report, "bundles": bundles,
            "log": buf.getvalue(),
            "warnings": state.get("warnings", [])[:20],
            "errors": state.get("errors", [])[:20],
        }, 200
    except Exception as exc:
        log.exception("Run failed")
        return {"error": str(exc), "log": buf.getvalue()}, 500
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)


@app.post("/overrides")
def save_overrides_endpoint():
    """Persist technician code overrides for a dataset (sticky, always applied)."""
    from src import overrides as ov
    data = request.get_json(force=True, silent=True) or {}
    dataset_name = data.get("dataset") or (LAST_RUN.get("dataset_name") if LAST_RUN else "")
    edits = data.get("overrides") or {}
    if not dataset_name:
        return jsonify({"error": "No dataset specified for overrides."}), 400
    full = ov.save_overrides(dataset_name, cfg, edits, merge=True)
    return jsonify({"ok": True, "dataset": dataset_name, "count": len(full)})


@app.post("/overrides/clear")
def clear_overrides_endpoint():
    """Delete all saved overrides for a dataset (so future runs map cleanly)."""
    from src import overrides as ov
    data = request.get_json(force=True, silent=True) or {}
    dataset_name = data.get("dataset") or (LAST_RUN.get("dataset_name") if LAST_RUN else "")
    if not dataset_name:
        return jsonify({"error": "No dataset specified."}), 400
    removed = ov.clear_overrides(dataset_name, cfg)
    return jsonify({"ok": True, "dataset": dataset_name, "removed": removed})


@app.post("/rebuild")
def rebuild_endpoint():
    """Rebuild FHIR bundles from the last run's data, reusing cached mappings +
    the now-updated overrides. Lets a technician correct codes and see the result
    without re-running the LLM mapping."""
    if not LAST_RUN.get("upload_dir"):
        return jsonify({"error": "Nothing to rebuild yet — run the pipeline once first."}), 400
    resp, code = _execute(
        LAST_RUN["upload_dir"], LAST_RUN["dataset_name"], LAST_RUN["model"],
        mode="full", max_subjects=LAST_RUN.get("max_subjects", 0), skip_mapping=True,
    )
    return jsonify(resp), code


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    # Bind to 127.0.0.1 for local dev; containers set HOST=0.0.0.0 so the port
    # is reachable from the host via the published mapping.
    host = os.getenv("HOST", "127.0.0.1")
    print(f"\n  FHIR pipeline dashboard → http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}\n")
    app.run(host=host, port=port, debug=False)
