#!/usr/bin/env python3
"""
Evaluation, step 1 — build the reference-standard annotation set.

Produces a BLINDED worksheet covering every variable the code mapper handled,
so semantic mapping accuracy can be measured against a human reference.

Why blinded: if the annotator can see the code the pipeline chose, they anchor
to it and the evaluation measures agreement-with-the-system rather than
correctness. The system's answers are therefore written to a SEPARATE key file
that the annotator does not open; `evaluate_mapping.py` joins the two afterwards.

Outputs (default under evaluation/):
    annotation_worksheet.csv   ← you fill this in (no system answers inside)
    annotation_key.csv         ← the pipeline's answers; do not open while annotating

Each worksheet row carries the variable's dictionary metadata plus candidate
codes retrieved from UMLS, so the annotator chooses from authoritative options
and records the evidence rather than recalling codes from memory.

Usage:
    python tools/build_annotation_set.py                     # all datasets
    python tools/build_annotation_set.py --datasets ace
    python tools/build_annotation_set.py --no-candidates     # skip UMLS lookup
    python tools/build_annotation_set.py --sample 150 --seed 42   # stratified sample
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg                                        # noqa: E402
from src.dictionary import load_cluster_variables, is_rag_mappable   # noqa: E402
from tools.umls_seed_builder import UMLSClient, _umls_term, VOCAB_SAB  # noqa: E402

logger = logging.getLogger("build_annotation_set")

DEFAULT_DIR = PROJECT_ROOT / "evaluation"

# What the annotator fills in. Kept deliberately small: every extra column is
# friction across 229 rows.
WORKSHEET_COLUMNS = [
    "annotation_id", "dataset", "cluster", "variable", "label", "units", "type",
    "candidates",            # UMLS-suggested options (read-only aid)
    # --- to be completed by the annotator ---
    "reference_code",        # the correct code, or NONE if none exists
    "reference_system",      # loinc | snomed | rxnorm | icd-10 | NONE
    "acceptable_alternates", # other defensible codes, semicolon-separated
    "evidence",              # URL or UMLS CUI justifying the choice
    "certainty",             # certain | probable | unsure
    "notes",
]

KEY_COLUMNS = ["annotation_id", "dataset", "cluster", "variable",
               "system_code", "system_code_system", "system_display",
               "system_confidence"]

# Vocabularies to query per cluster mapping mode, reusing the runtime routing.
_MODE_VOCABS = {
    "rag_loinc": ["loinc"],
    "rag_rxnorm": ["rxnorm"],
    "rag_clinical": ["loinc", "snomed"],
}


def _report_path(dataset_name: str) -> Path:
    sub = cfg.DATASETS[dataset_name].get("output_subdir", "")
    base = cfg.OUTPUT_DIR / sub if sub else cfg.OUTPUT_DIR
    return base / "mapping_report.json"


def _load_system_mappings(dataset_name: str) -> dict[str, dict]:
    """{cluster::variable: mapping} from a dataset's mapping_report.json."""
    path = _report_path(dataset_name)
    if not path.exists():
        logger.warning("No mapping report for '%s' at %s — run the pipeline first",
                       dataset_name, path)
        return {}
    data = json.loads(path.read_text())
    return {f"{m['cluster']}::{m['variable']}": m for m in data.get("mappings", [])}


def _dictionary_meta(dataset_name: str) -> dict[str, dict]:
    """{cluster::variable: meta} for every variable documented in a dictionary."""
    dataset = cfg.DATASETS[dataset_name]
    out: dict[str, dict] = {}
    for cluster_name in dataset["all_clusters"]:
        cluster_cfg = dataset["clusters"].get(cluster_name, {})
        if cluster_cfg.get("mapping", "") not in _MODE_VOCABS:
            continue
        try:
            variables = load_cluster_variables(dataset, cluster_cfg, cluster_name)
        except Exception as exc:          # noqa: BLE001 - a missing dict shouldn't abort
            logger.warning("Could not load dictionary for %s/%s: %s",
                           dataset_name, cluster_name, exc)
            continue
        for var, meta in variables.items():
            out[f"{cluster_name}::{var}"] = meta
    return out


def _variables_for_dataset(dataset_name: str,
                           system_map: dict[str, dict]) -> list[dict]:
    """
    Every variable the CodeMapper actually processed, with dictionary metadata.

    The authoritative source is the pipeline's own mapping_report.json, NOT a
    walk of the dictionaries: some clusters derive their variables from the data
    rather than a dictionary (wearables quantity-kinds are expanded from
    quantity_kind values; medication variables are the distinct drug names).
    Walking dictionaries alone silently omits those, so the worksheet would not
    cover everything the evaluation claims to cover.

    Only the variable NAMES are taken from the report — never the chosen code or
    display, which would break blinding.
    """
    dataset = cfg.DATASETS[dataset_name]
    dict_meta = _dictionary_meta(dataset_name)
    out: list[dict] = []

    if system_map:
        for key in system_map:
            cluster_name, _, var = key.partition("::")
            cluster_cfg = dataset["clusters"].get(cluster_name, {})
            mode = cluster_cfg.get("mapping", "rag_loinc")
            meta = dict_meta.get(key) or {
                # Data-derived variable (wearable quantity-kind, drug name):
                # no dictionary row exists, so fall back to a neutral label
                # built from the variable name itself.
                "label": var.replace("_", " ").strip(),
                "units": "", "type": "", "notes": "",
            }
            out.append({
                "dataset": dataset_name, "cluster": cluster_name,
                "variable": var, "meta": meta,
                "vocabs": _MODE_VOCABS.get(mode, ["loinc"]),
            })
        return out

    # No mapping report yet — fall back to the dictionary walk so the worksheet
    # can still be prepared, but warn that coverage may be incomplete.
    logger.warning("No mapping report for '%s': falling back to dictionary walk. "
                   "Data-derived variables (wearables kinds, drug names) will be "
                   "MISSING — run the pipeline first for full coverage.",
                   dataset_name)
    for key, meta in dict_meta.items():
        cluster_name, _, var = key.partition("::")
        if var == dataset.get("id_col") or not is_rag_mappable(meta):
            continue
        mode = dataset["clusters"].get(cluster_name, {}).get("mapping", "rag_loinc")
        out.append({
            "dataset": dataset_name, "cluster": cluster_name,
            "variable": var, "meta": meta, "vocabs": _MODE_VOCABS.get(mode, ["loinc"]),
        })
    return out


def _candidates_text(entry: dict, client: UMLSClient | None, top_n: int) -> str:
    """Authoritative candidate codes for one variable, as a readable cell."""
    if client is None:
        return ""
    term = _umls_term(entry["meta"])
    if not term:
        return ""
    parts: list[str] = []
    for vocab in entry["vocabs"]:
        sab = VOCAB_SAB.get(vocab, (None, None))[0]
        if not sab:
            continue
        for hit in client.search(term, sab, top_n):
            parts.append(f"{vocab}:{hit['code']} = {hit['term']}")
    return " | ".join(parts)


def _stratified_sample(entries: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional-by-cluster sample, so no cluster is accidentally excluded."""
    rng = random.Random(seed)
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_cluster[f"{e['dataset']}::{e['cluster']}"].append(e)

    total = len(entries)
    picked: list[dict] = []
    for key, group in sorted(by_cluster.items()):
        rng.shuffle(group)
        # At least one per cluster; otherwise proportional to cluster size.
        take = max(1, round(n * len(group) / total))
        picked.extend(group[:take])
    rng.shuffle(picked)
    return picked[:n] if len(picked) > n else picked


def build(dataset_names: list[str], out_dir: Path, use_umls: bool,
          top_n: int, sample: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    system_maps: dict[str, dict[str, dict]] = {}
    for name in dataset_names:
        system_maps[name] = _load_system_mappings(name)
        entries.extend(_variables_for_dataset(name, system_maps[name]))

    if not entries:
        logger.error("No mappable variables found — nothing to annotate.")
        return

    if sample and sample < len(entries):
        entries = _stratified_sample(entries, sample, seed)
        logger.info("Stratified sample: %d variables (seed=%d)", len(entries), seed)
    else:
        logger.info("Full census: %d variables", len(entries))

    client = None
    if use_umls:
        api_key = cfg_env_umls_key()
        if api_key:
            client = UMLSClient(api_key)
            logger.info("Fetching UMLS candidates (top %d per vocabulary)…", top_n)
        else:
            logger.warning("UMLS_API_KEY not set — worksheet will have no candidates. "
                           "Set it in .env to pre-fill authoritative options.")

    worksheet_rows: list[dict] = []
    key_rows: list[dict] = []

    for idx, entry in enumerate(sorted(entries, key=lambda e: (e["dataset"], e["cluster"], e["variable"])), 1):
        ann_id = f"A{idx:04d}"
        meta = entry["meta"]
        worksheet_rows.append({
            "annotation_id": ann_id,
            "dataset": entry["dataset"],
            "cluster": entry["cluster"],
            "variable": entry["variable"],
            "label": meta.get("label", ""),
            "units": meta.get("units", ""),
            "type": meta.get("type", ""),
            "candidates": _candidates_text(entry, client, top_n),
            "reference_code": "", "reference_system": "",
            "acceptable_alternates": "", "evidence": "",
            "certainty": "", "notes": "",
        })
        sysm = system_maps.get(entry["dataset"], {}).get(
            f"{entry['cluster']}::{entry['variable']}", {})
        key_rows.append({
            "annotation_id": ann_id,
            "dataset": entry["dataset"],
            "cluster": entry["cluster"],
            "variable": entry["variable"],
            "system_code": sysm.get("code", ""),
            "system_code_system": sysm.get("code_system", ""),
            "system_display": sysm.get("display", ""),
            "system_confidence": sysm.get("confidence", ""),
        })
        if idx % 25 == 0:
            logger.info("  prepared %d/%d", idx, len(entries))

    ws_path = out_dir / "annotation_worksheet.csv"
    key_path = out_dir / "annotation_key.csv"
    _write_csv(ws_path, WORKSHEET_COLUMNS, worksheet_rows)
    _write_csv(key_path, KEY_COLUMNS, key_rows)

    covered = sum(1 for r in key_rows if r["system_code"])
    logger.info("")
    logger.info("Worksheet : %s  (%d rows to annotate)", ws_path, len(worksheet_rows))
    logger.info("Key       : %s  (%d rows, %d with a system mapping)",
                key_path, len(key_rows), covered)
    logger.info("")
    logger.info("Do NOT open the key file while annotating — blinding is what makes")
    logger.info("the resulting accuracy figure meaningful.")


def cfg_env_umls_key() -> str:
    import os
    return os.getenv("UMLS_API_KEY", "")


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the evaluation annotation worksheet")
    ap.add_argument("--datasets", default="", help="Comma list (default: all)")
    ap.add_argument("--out", default=str(DEFAULT_DIR), help="Output directory")
    ap.add_argument("--no-candidates", action="store_true", help="Skip UMLS candidate lookup")
    ap.add_argument("--top-n", type=int, default=5, help="Candidates per vocabulary")
    ap.add_argument("--sample", type=int, default=0, help="Stratified sample size (0 = all)")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed (reproducibility)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    names = [d.strip() for d in args.datasets.split(",") if d.strip()] or list(cfg.DATASETS)
    build(names, Path(args.out), not args.no_candidates, args.top_n, args.sample, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
