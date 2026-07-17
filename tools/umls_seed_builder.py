#!/usr/bin/env python3
"""
UMLS seed enrichment

This is a *build-time* step and NOT a pipeline stage and NOT tied to any one dataset.

This step can be used to run occasionally, offline, to restock the SHARED seed 
(data/terminology/*_seed.csv) that every dataset's runtime retrieval reads.

Because the seed is shared, enrichment defaults to ALL datasets in the registry, 
there is no "default dataset". You can only narrow it if you want to.

Primary entry point is main.py:
    python main.py --enrich-seed                 # query UMLS for all datasets' unmapped-able vars
    python main.py --apply-seed                  # append reviewed rows to the seed CSVs
    python main.py --seed-test "clock test"      # diagnostic: raw UMLS hits for one term

This module also runs standalone (same functions) for power use:
    python tools/umls_seed_builder.py build|apply|test ...

FLOW:

  variable label  --_umls_term()-->  short clinical term
        -> UMLS /search (per target vocab: LOINC / SNOMED / RxNorm)
        -> candidate codes -> review CSV (you approve: accept=Y)
        -> appended to data/terminology/*_seed.csv (deduped by code)
        -> rebuild the vector store (embeds only the new rows: CPU, seconds)

UMLS /search is a LEXICAL lookup: by default all query words must appear in one
concept name. So we send a short clean term (not the verbose vector query) and
fall back to partialSearch when the strict pass finds nothing.

AUTH: free UMLS UTS licence + API key (https://uts.nlm.nih.gov/uts/profile),
read from the UMLS_API_KEY env var (add it to .env).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

# --- make the project importable whether run as script or imported ---------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg                                   # noqa: E402
from src.dictionary import (                           # noqa: E402
    load_cluster_variables,
    is_rag_mappable,
)

logger = logging.getLogger("umls_seed_builder")

UTS_SEARCH = "https://uts-ws.nlm.nih.gov/rest/search/current"

# candidate review CSV lives with the terminology it feeds — NOT in output/
SEED_CANDIDATES_DIR = cfg.TERMINOLOGY_DIR / "seed_candidates"
DEFAULT_REVIEW = SEED_CANDIDATES_DIR / "umls_seed_candidates.csv"

# our internal vocab key -> (UMLS source abbreviation, seed CSV filename)
VOCAB_SAB = {
    "loinc":  ("LNC", "loinc_seed.csv"),
    "snomed": ("SNOMEDCT_US", "snomed_seed.csv"),
    "rxnorm": ("RXNORM", "rxnorm_seed.csv"),
}

# exact column order of each seed CSV (must match data/terminology/*_seed.csv)
SEED_COLUMNS = {
    "loinc": ["loinc_code", "long_common_name", "component", "property",
              "time_aspect", "system", "scale", "method", "units",
              "category", "search_text"],
    "snomed": ["snomed_code", "preferred_term", "synonyms", "hierarchy",
               "search_text"],
    "rxnorm": ["rxnorm_code", "ingredient_name", "tty", "brand_examples",
               "search_text"],
}

REVIEW_COLUMNS = ["accept", "dataset", "cluster", "variable", "label", "vocab",
                  "rank", "code", "term", "umls_score", "query",
                  "proposed_search_text"]


# ---------------------------------------------------------------------------
# UMLS REST client (thin; swap this one class for umls-python-client if desired)
# ---------------------------------------------------------------------------
class UMLSClient:
    def __init__(self, api_key: str, sleep: float = 0.1):
        self.api_key = api_key
        self.sleep = sleep
        self.session = requests.Session()

    def _search_once(self, term: str, sab: str, top_n: int,
                     partial: bool) -> list[dict[str, Any]]:
        """One /search call. returnIdType=code so `ui` is the actual
        LOINC/SNOMED/RxNorm code and `name` is the term."""
        params = {
            "string": term,
            "sabs": sab,
            "returnIdType": "code",
            "searchType": "words",
            "pageSize": max(top_n * 2, 10),
            "apiKey": self.api_key,
        }
        if partial:
            params["partialSearch"] = "true"
        try:
            r = self.session.get(UTS_SEARCH, params=params, timeout=20)
            r.raise_for_status()
            results = r.json().get("result", {}).get("results", []) or []
        except Exception as exc:  # noqa: BLE001 - stay resilient across many calls
            logger.warning("  UMLS query failed for %r [%s]: %s", term, sab, exc)
            return []
        finally:
            if self.sleep:
                time.sleep(self.sleep)

        out = []
        for hit in results:
            code = str(hit.get("ui", "")).strip()
            name = str(hit.get("name", "")).strip()
            if not code or code.upper() == "NONE":  # UMLS sentinel for no match
                continue
            out.append({"code": code, "term": name})
            if len(out) >= top_n:
                break
        return out

    def search(self, term: str, sab: str, top_n: int) -> list[dict[str, Any]]:
        """Two-pass: strict 'all words' first (precision); if empty, retry with
        partialSearch (recall). Returns [] when both come up empty."""
        if not term:
            return []
        hits = self._search_once(term, sab, top_n, partial=False)
        if not hits:
            hits = self._search_once(term, sab, top_n, partial=True)
        return hits


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _vocabs_for_cluster(cluster_cfg: dict) -> list[str]:
    """Same routing the runtime mapper uses: cluster mapping mode -> vocab list."""
    mode = cluster_cfg.get("mapping", "rag_loinc")
    return cfg.MAPPING_VOCABS.get(mode, [])


# noise prefixes seen in ACE labels that hurt lexical matching
_NOISE_PREFIX = re.compile(
    r"^(cognitive domain\s+\w+\s*[-–]\s*|global cognition\s*[-–]\s*)",
    re.IGNORECASE,
)


def _umls_term(meta: dict) -> str:
    """Build a SHORT, clean clinical term for UMLS's lexical search.

    The runtime vector query is verbose on purpose (good for embeddings), but
    UMLS '/search' with searchType=words needs *all* words to appear in one
    concept name — so strip metadata, trailing description sentences,
    parentheticals, and unit encodings down to the core concept phrase.
    """
    label = (meta.get("label") or "").strip()
    label = re.split(r"[.\n]", label)[0]                 # first sentence/line only
    label = _NOISE_PREFIX.sub("", label)                 # drop "Cognitive domain X - "
    label = re.sub(r"\([^)]*\)", " ", label)             # drop (parentheticals)
    label = re.sub(r"\[[^\]]*\]", " ", label)            # drop [brackets]
    label = re.sub(r"[-–/]", " ", label)                 # separators -> space
    label = re.sub(r"\s+", " ", label).strip(" -:;,")
    return label


def _proposed_search_text(term: str, meta: dict) -> str:
    """Recall text for the new seed row: UMLS term + the variable's own
    label/notes. (Synonyms could be added via a CUI/atoms follow-up call.)"""
    parts = [term, meta.get("label", "")]
    notes = (meta.get("notes") or "").strip()
    if len(notes) > 5:
        parts.append(notes[:150])
    seen, uniq = set(), []
    for p in parts:
        p = (p or "").strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            uniq.append(p)
    return " | ".join(uniq)


def _load_unmapped(path: str) -> set[str] | None:
    """Read a mapping cache OR report -> {"cluster::variable", ...} currently UNMAPPED.

    Handles both shapes the pipeline writes:
      - mapping_cache.json  : flat dict {"cluster::variable": {mapping...}}
      - mapping_report.json : {"summary":..., "mappings":[{mapping...}], ...}
    """
    p = Path(path)
    if not p.exists():
        logger.warning("--only-unmapped file not found: %s (ignoring filter)", path)
        return None
    data = json.loads(p.read_text())
    if isinstance(data, dict) and "mappings" in data:       # report shape
        items = data["mappings"]
    elif isinstance(data, dict):                            # cache shape
        items = data.values()
    else:                                                   # bare list
        items = data
    unmapped: set[str] = set()
    for m in items:
        if isinstance(m, dict) and str(m.get("code", "")).upper() == "UNMAPPED":
            unmapped.add(f"{m.get('cluster','')}::{m.get('variable','')}")
    logger.info("Restricting to %d UNMAPPED variables from %s", len(unmapped), path)
    return unmapped


# ---------------------------------------------------------------------------
# build: query UMLS across dataset(s) -> review CSV
# ---------------------------------------------------------------------------
def run_build(dataset_names: list[str] | None = None,
              review_path: Path = DEFAULT_REVIEW,
              only_unmapped: str = "",
              clusters: set[str] | None = None,
              top_n: int = 3,
              sleep: float = 0.1) -> Path:
    """Enrich the shared seed. Defaults to ALL datasets (the seed is shared)."""
    api_key = os.environ.get("UMLS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ERROR: set UMLS_API_KEY "
                         "(get one free at https://uts.nlm.nih.gov/uts/profile)")

    dataset_names = dataset_names or list(cfg.DATASETS.keys())
    unmapped = _load_unmapped(only_unmapped) if only_unmapped else None
    client = UMLSClient(api_key, sleep=sleep)

    rows: list[dict[str, Any]] = []
    n_vars = 0
    vars_with_hits: set[str] = set()

    for ds_name in dataset_names:
        dataset_cfg = cfg.DATASETS[ds_name]
        registry: dict = dataset_cfg["clusters"]
        logger.info("=== dataset: %s ===", ds_name)

        for cluster_name, cluster_cfg in registry.items():
            vocabs = _vocabs_for_cluster(cluster_cfg)
            if not vocabs:                       # 'local' clusters aren't rag-mapped
                continue
            if clusters and cluster_name not in clusters:
                continue

            try:
                variables = load_cluster_variables(dataset_cfg, cluster_cfg, cluster_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load cluster %s/%s: %s", ds_name, cluster_name, exc)
                continue

            for var, meta in variables.items():
                if not is_rag_mappable(meta):
                    continue
                key = f"{cluster_name}::{var}"
                if unmapped is not None and key not in unmapped:
                    continue

                n_vars += 1
                term = _umls_term(meta)
                before = len(rows)
                for vocab in vocabs:
                    sab, _ = VOCAB_SAB[vocab]
                    for rank, hit in enumerate(client.search(term, sab, top_n), start=1):
                        rows.append({
                            "accept": "Y" if rank == 1 else "",  # pre-suggest top hit
                            "dataset": ds_name,
                            "cluster": cluster_name,
                            "variable": var,
                            "label": meta.get("label", ""),
                            "vocab": vocab,
                            "rank": rank,
                            "code": hit["code"],
                            "term": hit["term"],
                            "umls_score": "",  # UTS search returns rank, not a score
                            "query": term,
                            "proposed_search_text": _proposed_search_text(hit["term"], meta),
                        })
                got = len(rows) - before
                if got:
                    vars_with_hits.add(f"{ds_name}::{key}")
                logger.info("[%s/%s] %-36s term=%r hits=%d",
                            ds_name, cluster_name, var, term, got)

    review_path.parent.mkdir(parents=True, exist_ok=True)
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    logger.info("")
    logger.info("Scanned %d rag-mappable variables across %s -> %d candidate rows "
                "(%d variables got >=1 candidate)",
                n_vars, ", ".join(dataset_names), len(rows), len(vars_with_hits))
    logger.info("Review file: %s", review_path)
    logger.info("Edit 'accept' (Y keep / blank drop), then: python main.py --apply-seed")
    return review_path


# ---------------------------------------------------------------------------
# apply: reviewed CSV -> append accepted rows to seed CSVs
# ---------------------------------------------------------------------------
def _to_seed_row(rec: dict, vocab: str) -> dict[str, str]:
    row = {c: "" for c in SEED_COLUMNS[vocab]}
    st = rec.get("proposed_search_text", "").strip() or rec.get("term", "")
    if vocab == "loinc":
        row.update(loinc_code=rec["code"], long_common_name=rec["term"],
                   category=rec.get("cluster", ""), search_text=st)
    elif vocab == "snomed":
        row.update(snomed_code=rec["code"], preferred_term=rec["term"], search_text=st)
    elif vocab == "rxnorm":
        row.update(rxnorm_code=rec["code"], ingredient_name=rec["term"],
                   tty="IN", search_text=st)
    return row


def run_apply(review_path: Path = DEFAULT_REVIEW) -> None:
    if not review_path.exists():
        raise SystemExit(f"ERROR: review file not found: {review_path} "
                         f"(run 'python main.py --enrich-seed' first)")

    with open(review_path, newline="", encoding="utf-8") as f:
        accepted = [r for r in csv.DictReader(f)
                    if r.get("accept", "").strip().lower() in ("y", "yes", "1", "true")]
    if not accepted:
        raise SystemExit("No rows marked accept=Y in the review file — nothing to apply.")

    by_vocab: dict[str, list[dict]] = {}
    for rec in accepted:
        by_vocab.setdefault(rec["vocab"], []).append(rec)

    for vocab, recs in by_vocab.items():
        seed_path = cfg.TERMINOLOGY_DIR / VOCAB_SAB[vocab][1]
        code_col = SEED_COLUMNS[vocab][0]

        existing: set[str] = set()
        file_exists = seed_path.exists()
        if file_exists:
            with open(seed_path, newline="", encoding="utf-8") as f:
                existing = {r[code_col] for r in csv.DictReader(f)}

        added = 0
        with open(seed_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SEED_COLUMNS[vocab])
            if not file_exists:
                w.writeheader()
            for rec in recs:
                if rec["code"] in existing:
                    continue
                w.writerow(_to_seed_row(rec, vocab))
                existing.add(rec["code"])
                added += 1
        logger.info("%-7s : +%d new rows -> %s", vocab, added, seed_path)

    logger.info("")
    logger.info("Done. Rebuild the vector store to embed the new rows:")
    logger.info("  python main.py --dataset <any> --rebuild-store --mapping-only --max-subjects 0")


# ---------------------------------------------------------------------------
# test: diagnostic — hit UMLS with one term and print what comes back
# ---------------------------------------------------------------------------
def run_test(term: str, vocabs: list[str] | None = None) -> None:
    api_key = os.environ.get("UMLS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ERROR: set UMLS_API_KEY (https://uts.nlm.nih.gov/uts/profile)")
    client = UMLSClient(api_key)
    for vocab in (vocabs or ["loinc", "snomed"]):
        sab = VOCAB_SAB[vocab][0]
        hits = client.search(term, sab, top_n=5)
        logger.info("[%s / %s] %d hit(s):", vocab, sab, len(hits))
        for h in hits:
            logger.info("    %s  %s", h["code"], h["term"])


# ---------------------------------------------------------------------------
# standalone CLI (secondary; main.py is the primary entry point)
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="query UMLS -> review CSV (all datasets by default)")
    b.add_argument("--datasets", default="", help="comma list; default: all")
    b.add_argument("--clusters", default="")
    b.add_argument("--only-unmapped", default="")
    b.add_argument("--top-n", type=int, default=3)
    b.add_argument("--sleep", type=float, default=0.1)

    a = sub.add_parser("apply", help="append accepted rows to seed CSVs")
    a.add_argument("--review-file", default="")

    t = sub.add_parser("test", help="print raw UMLS hits for one term")
    t.add_argument("term")
    t.add_argument("--vocabs", default="loinc,snomed")

    args = p.parse_args()
    if args.command == "build":
        run_build(
            dataset_names=[d.strip() for d in args.datasets.split(",") if d.strip()] or None,
            only_unmapped=args.only_unmapped,
            clusters=set(c for c in args.clusters.split(",") if c) or None,
            top_n=args.top_n, sleep=args.sleep,
        )
    elif args.command == "apply":
        run_apply(Path(args.review_file) if args.review_file else DEFAULT_REVIEW)
    elif args.command == "test":
        run_test(args.term, [v.strip() for v in args.vocabs.split(",") if v.strip()])


if __name__ == "__main__":
    main()
