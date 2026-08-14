#!/usr/bin/env python3
"""
Evaluation, step 2 — measure semantic mapping accuracy against the reference standard.

Joins the completed annotation worksheet with the (blinded) key of pipeline
outputs and reports the metrics an evaluation chapter needs.

OUTCOME CATEGORIES
Every variable falls into exactly one, which is what makes precision and recall
meaningful here — "no code exists" is a legitimate correct answer, so accuracy
alone would be misleading:

    correct          system code == reference (or an accepted alternate)
    wrong_code       system produced a code, but not the right one
    missed           a code exists, system returned UNMAPPED   (recall failure)
    spurious         no code exists, but system produced one   (precision failure)
    correct_unmapped both agree no suitable code exists         (a correct decision)

METRICS
    accuracy   = (correct + correct_unmapped) / N        overall decision accuracy
    precision  = correct / (correct + wrong_code + spurious)
                 of the codes it emitted, how many were right
    recall     = correct / (correct + wrong_code + missed)
                 of the codes that existed, how many it found
    F1         = harmonic mean of the two

Also reported: per-cluster and per-vocabulary breakdowns, a calibration table
(is "high confidence" actually more accurate?), and an error taxonomy.

Optionally computes intra-rater agreement (Cohen's kappa) from a re-annotation
round — the accepted substitute for inter-annotator agreement when a single
annotator builds the reference standard.

Usage:
    python tools/evaluate_mapping.py
    python tools/evaluate_mapping.py --reannotation evaluation/annotation_round2.csv
    python tools/evaluate_mapping.py --out evaluation/results.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("evaluate_mapping")

DEFAULT_DIR = PROJECT_ROOT / "evaluation"

# Values in reference_code meaning "no suitable standard code exists".
_NONE_VALUES = {"", "none", "na", "n/a", "unmapped", "-"}

# Map the annotator's short vocabulary names to the system URIs the pipeline emits.
_SYSTEM_URIS = {
    "loinc": "http://loinc.org",
    "snomed": "http://snomed.info/sct",
    "rxnorm": "http://www.nlm.nih.gov/research/umls/rxnorm",
    "icd-10": "http://hl7.org/fhir/sid/icd-10",
    "icd10": "http://hl7.org/fhir/sid/icd-10",
}

CORRECT, WRONG, MISSED, SPURIOUS, CORRECT_UNMAPPED = (
    "correct", "wrong_code", "missed", "spurious", "correct_unmapped")


def _norm(code: str) -> str:
    return str(code or "").strip().lower()


def _is_none(code: str) -> bool:
    return _norm(code) in _NONE_VALUES


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _alternates(row: dict) -> set[str]:
    raw = row.get("acceptable_alternates", "") or ""
    return {_norm(p) for p in raw.replace(",", ";").split(";") if _norm(p)}


def classify(ref_row: dict, sys_row: dict) -> str:
    """Assign one outcome category to a single variable."""
    ref_code = ref_row.get("reference_code", "")
    sys_code = sys_row.get("system_code", "")

    ref_none = _is_none(ref_code)
    sys_none = _is_none(sys_code) or _norm(sys_code) == "unmapped"

    if ref_none and sys_none:
        return CORRECT_UNMAPPED
    if ref_none and not sys_none:
        return SPURIOUS
    if not ref_none and sys_none:
        return MISSED

    accepted = {_norm(ref_code)} | _alternates(ref_row)
    return CORRECT if _norm(sys_code) in accepted else WRONG


def _prf(counts: Counter) -> dict:
    c = counts[CORRECT]
    emitted = c + counts[WRONG] + counts[SPURIOUS]
    existing = c + counts[WRONG] + counts[MISSED]
    n = sum(counts.values())
    precision = c / emitted if emitted else 0.0
    recall = c / existing if existing else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (c + counts[CORRECT_UNMAPPED]) / n if n else 0.0
    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "counts": dict(counts),
    }


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> dict:
    """
    Cohen's kappa between two labelings of the same items.

    Used for intra-rater (test–retest) agreement: the same annotator re-labels a
    subset after a gap, quantifying how reproducible the reference standard is.
    """
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return {"n": 0, "observed_agreement": None, "kappa": None}

    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    count_a, count_b = Counter(labels_a), Counter(labels_b)
    expected = sum((count_a[k] / n) * (count_b[k] / n)
                   for k in set(count_a) | set(count_b))
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {
        "n": n,
        "observed_agreement": round(observed, 4),
        "expected_agreement": round(expected, 4),
        "kappa": round(kappa, 4),
        "interpretation": _kappa_label(kappa),
    }


def _kappa_label(k: float) -> str:
    """Landis & Koch (1977) benchmarks — the conventional reporting scale."""
    if k < 0.00: return "poor"
    if k < 0.21: return "slight"
    if k < 0.41: return "fair"
    if k < 0.61: return "moderate"
    if k < 0.81: return "substantial"
    return "almost perfect"


def evaluate(worksheet: Path, key: Path, reannotation: Path | None) -> dict:
    ws_rows = _read_csv(worksheet)
    key_rows = {r["annotation_id"]: r for r in _read_csv(key)}

    # Guard against silent misalignment. The two files are joined on
    # annotation_id, which is assigned by sort order when the worksheet is
    # built. If the key is regenerated from a different pipeline run whose
    # variable set has changed, ids shift and each annotation would be compared
    # against a DIFFERENT variable's system answer — producing plausible but
    # meaningless metrics. Verify identity, and refuse to continue if it breaks.
    mismatches = []
    for r in ws_rows:
        k = key_rows.get(r["annotation_id"])
        if k is None:
            mismatches.append(f"{r['annotation_id']}: missing from key")
        elif (k.get("variable"), k.get("cluster")) != (r.get("variable"), r.get("cluster")):
            mismatches.append(
                f"{r['annotation_id']}: worksheet has {r.get('cluster')}/{r.get('variable')}, "
                f"key has {k.get('cluster')}/{k.get('variable')}")
    if mismatches:
        raise SystemExit(
            "Worksheet and key are misaligned — refusing to produce metrics.\n"
            f"{len(mismatches)} mismatch(es); first 5:\n  " + "\n  ".join(mismatches[:5]) +
            "\n\nThe key was probably regenerated from a run with a different "
            "variable set. Rebuild the worksheet and key together, or re-run the "
            "pipeline so the variable set matches the annotated worksheet."
        )

    annotated = [r for r in ws_rows if str(r.get("reference_code", "")).strip()]
    skipped = len(ws_rows) - len(annotated)
    if not annotated:
        raise SystemExit(
            "No completed annotations found. Fill in 'reference_code' in the "
            "worksheet (use NONE where no suitable code exists) and rerun."
        )

    overall = Counter()
    by_cluster: dict[str, Counter] = defaultdict(Counter)
    by_vocab: dict[str, Counter] = defaultdict(Counter)
    by_confidence: dict[str, Counter] = defaultdict(Counter)
    by_certainty: dict[str, Counter] = defaultdict(Counter)
    per_item: list[dict] = []

    for row in annotated:
        aid = row["annotation_id"]
        sys_row = key_rows.get(aid, {})
        outcome = classify(row, sys_row)

        overall[outcome] += 1
        by_cluster[f"{row.get('dataset','')}/{row.get('cluster','')}"][outcome] += 1
        by_vocab[_norm(row.get("reference_system", "")) or "none"][outcome] += 1
        by_confidence[sys_row.get("system_confidence", "") or "none"][outcome] += 1
        by_certainty[_norm(row.get("certainty", "")) or "unstated"][outcome] += 1

        per_item.append({
            "annotation_id": aid,
            "dataset": row.get("dataset", ""), "cluster": row.get("cluster", ""),
            "variable": row.get("variable", ""),
            "reference_code": row.get("reference_code", ""),
            "system_code": sys_row.get("system_code", ""),
            "system_confidence": sys_row.get("system_confidence", ""),
            "outcome": outcome,
            "evidence": row.get("evidence", ""),
        })

    results = {
        "summary": {
            "variables_in_worksheet": len(ws_rows),
            "variables_annotated": len(annotated),
            "variables_skipped": skipped,
            **_prf(overall),
        },
        "by_cluster": {k: _prf(v) for k, v in sorted(by_cluster.items())},
        "by_reference_vocabulary": {k: _prf(v) for k, v in sorted(by_vocab.items())},
        "calibration_by_system_confidence": {k: _prf(v) for k, v in sorted(by_confidence.items())},
        "by_annotator_certainty": {k: _prf(v) for k, v in sorted(by_certainty.items())},
        "error_taxonomy": _taxonomy(per_item),
        "per_item": per_item,
    }

    if reannotation and reannotation.exists():
        results["intra_rater_agreement"] = _intra_rater(annotated, _read_csv(reannotation))

    return results


def _taxonomy(per_item: list[dict]) -> dict:
    """Frequency of each failure mode, with examples for the write-up."""
    tax: dict[str, dict] = {}
    for outcome in (WRONG, MISSED, SPURIOUS):
        items = [p for p in per_item if p["outcome"] == outcome]
        tax[outcome] = {
            "count": len(items),
            "examples": [
                {"variable": i["variable"], "reference": i["reference_code"],
                 "system": i["system_code"]}
                for i in items[:10]
            ],
        }
    return tax


def _intra_rater(round1: list[dict], round2: list[dict]) -> dict:
    """Compare the two annotation rounds on the items present in both."""
    r2 = {r["annotation_id"]: r for r in round2 if str(r.get("reference_code", "")).strip()}
    shared = [r for r in round1 if r["annotation_id"] in r2]
    if not shared:
        return {"n": 0, "note": "no overlapping annotated items between rounds"}

    a = [_norm(r["reference_code"]) or "none" for r in shared]
    b = [_norm(r2[r["annotation_id"]]["reference_code"]) or "none" for r in shared]
    out = cohens_kappa(a, b)
    out["note"] = ("Intra-rater (test-retest) agreement of the single annotator; "
                   "reported in place of inter-annotator agreement.")
    return out


def _print(results: dict) -> None:
    s = results["summary"]
    print("\n" + "=" * 68)
    print("  LAYER 3 — SEMANTIC MAPPING ACCURACY")
    print("=" * 68)
    print(f"  Annotated      : {s['variables_annotated']} / {s['variables_in_worksheet']}"
          f"   (skipped {s['variables_skipped']})")
    print(f"  Accuracy       : {s['accuracy']:.1%}")
    print(f"  Precision      : {s['precision']:.1%}")
    print(f"  Recall         : {s['recall']:.1%}")
    print(f"  F1             : {s['f1']:.3f}")
    print("\n  Outcomes:")
    for k in (CORRECT, CORRECT_UNMAPPED, WRONG, MISSED, SPURIOUS):
        print(f"    {k:<18} {s['counts'].get(k, 0)}")

    cal = results["calibration_by_system_confidence"]
    if cal:
        print("\n  Calibration (does stated confidence track correctness?):")
        print(f"    {'confidence':<12} {'n':>4}  {'accuracy':>9}")
        for conf, m in cal.items():
            print(f"    {conf:<12} {m['n']:>4}  {m['accuracy']:>8.1%}")

    ka = results.get("intra_rater_agreement")
    if ka and ka.get("kappa") is not None:
        print(f"\n  Intra-rater kappa: {ka['kappa']:.3f} ({ka['interpretation']}, n={ka['n']})")

    print("\n  Error taxonomy:")
    for k, v in results["error_taxonomy"].items():
        print(f"    {k:<18} {v['count']}")
    print("=" * 68 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate semantic mapping accuracy (the evaluation study)")
    ap.add_argument("--worksheet", default=str(DEFAULT_DIR / "annotation_worksheet.csv"))
    ap.add_argument("--key", default=str(DEFAULT_DIR / "annotation_key.csv"))
    ap.add_argument("--reannotation", default="", help="Round-2 worksheet for intra-rater kappa")
    ap.add_argument("--out", default=str(DEFAULT_DIR / "evaluation_results.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = evaluate(Path(args.worksheet), Path(args.key),
                       Path(args.reannotation) if args.reannotation else None)
    _print(results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  Full results → {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
