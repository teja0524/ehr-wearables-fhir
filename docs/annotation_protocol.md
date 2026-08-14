# Annotation Protocol — Reference Standard for Semantic Mapping Accuracy

This protocol defines how the reference standard ("gold standard") for the
evaluation study is constructed. It exists so that the labelling is
**reproducible**:
another person following these rules on the same variables should arrive at
substantially the same codes.

> **Stated limitation.** The reference standard is constructed by a single
> non-clinical annotator. Every label is anchored to an authoritative
> terminology resource and its evidence is recorded, and reproducibility is
> quantified through intra-rater (test–retest) agreement. It has **not** been
> adjudicated by a licensed clinician. This is disclosed in the thesis rather
> than implied to be expert consensus.

---

## 1. Scope

All variables that the pipeline's CodeMapper processes — i.e. every variable in
a cluster whose `mapping` mode is `rag_loinc`, `rag_rxnorm` or `rag_clinical`,
and whose dictionary type is mappable. Clusters coded deterministically
(codes already in the data, local codes) are out of scope: no mapping decision
is made for them.

Population: **229 variables** (143 MFU + 86 ACE) — annotated as a full census,
so no sampling error applies.

---

## 2. Blinding (mandatory)

The annotator must **not** see the pipeline's chosen code while annotating.
`build_annotation_set.py` enforces this by writing the system's answers to a
separate `annotation_key.csv`.

Do not open the key file until annotation is complete. Seeing the system's
answer causes anchoring, which converts the measurement from *correctness* into
*agreement with the system* — a fundamentally weaker claim.

---

## 3. Authoritative sources

A code may only be assigned if it was found in one of these:

| Vocabulary | Source | Use for |
|---|---|---|
| LOINC | <https://search.loinc.org> | Lab results, measurements, survey instruments, vital signs |
| SNOMED CT | <https://browser.ihtsdotools.org> | Findings, conditions, procedures, qualifiers |
| RxNorm | <https://mor.nlm.nih.gov/RxNav/> | Medication ingredients |
| ICD-10 | <https://icd.who.int/browse10> | Diagnoses where ICD is the natural fit |
| UMLS | <https://uts.nlm.nih.gov> | Cross-vocabulary search; resolving CUIs |

The `candidates` column of the worksheet is pre-filled with UMLS suggestions as
a convenience. **Candidates are a starting point, not an authority** — verify
the chosen code in the browser above before accepting it, and feel free to
assign a code that does not appear among the candidates.

---

## 4. Decision rules

Apply in order.

**R1 — Concept match over string match.** The code must denote the same clinical
concept as the variable. Similar wording is not sufficient; a "positivity"
flag is not the same concept as the underlying quantitative measurement.

**R2 — Granularity.** Prefer the most specific code that is fully correct. A
more general code is acceptable *only* if it does not assert anything false, and
must then be recorded in `acceptable_alternates` rather than as the primary.

**R3 — Property and specimen must agree (LOINC).** For laboratory variables the
LOINC COMPONENT, PROPERTY and SYSTEM (specimen) must all be consistent with the
variable. A plasma measure must not receive a CSF code, and a ratio must not
receive a mass-concentration code.

**R4 — Units are evidence, not proof.** Matching units support a candidate but
do not by themselves establish concept equivalence.

**R5 — Vocabulary choice.** Measurements and observations → LOINC. Findings,
conditions, and qualitative states → SNOMED CT. Medication ingredients →
RxNorm. If a defensible code exists in more than one vocabulary, choose the one
matching the cluster's intent and list the other as an alternate.

**R6 — "NONE" is a valid and important answer.** If no suitable standard code
exists, enter `NONE`. Study-specific composites, derived indices, and
instrument-internal subscores frequently have no standard representation.
Forcing a poor match corrupts the reference standard; recording `NONE`
correctly is what allows the evaluation to distinguish a *justified* UNMAPPED
from a *missed* mapping.

**R7 — Record alternates deliberately.** Where several codes are genuinely
defensible, put the best in `reference_code` and the others in
`acceptable_alternates` (semicolon-separated). The evaluation counts a system
answer as correct if it matches the primary **or** any alternate — this avoids
penalising the pipeline for a defensible choice.

**R8 — Evidence is required.** Every non-`NONE` label must carry a URL or UMLS
CUI in `evidence`. An unverifiable label is not part of a reference standard.

**R9 — Declare uncertainty.** Set `certainty` to `certain`, `probable` or
`unsure`. This is reported separately, so results can be re-examined excluding
low-certainty labels.

---

## 5. Columns to complete

| Column | Required | Content |
|---|---|---|
| `reference_code` | yes | The correct code, or `NONE` |
| `reference_system` | yes | `loinc` \| `snomed` \| `rxnorm` \| `icd-10` \| `NONE` |
| `acceptable_alternates` | no | Other defensible codes, `;`-separated |
| `evidence` | yes (unless NONE) | URL or CUI justifying the choice |
| `certainty` | yes | `certain` \| `probable` \| `unsure` |
| `notes` | no | Reasoning, ambiguities, points to revisit |

Leave a row entirely blank to skip it; skipped rows are excluded from the
metrics and reported as such.

---

## 6. Procedure

1. Generate the worksheet:
   ```bash
   python tools/build_annotation_set.py
   ```
2. Annotate `evaluation/annotation_worksheet.csv` in Excel/Numbers. Work in
   sessions of ~50 rows; fatigue degrades label quality.
3. **Test–retest round.** After a gap of at least one week, re-annotate a random
   ~30% of rows *without consulting round 1*. Save as
   `evaluation/annotation_round2.csv`.
4. Compute results:
   ```bash
   python tools/evaluate_mapping.py --reannotation evaluation/annotation_round2.csv
   ```

---

## 7. Reported metrics

Each variable is assigned exactly one outcome:

| Outcome | Meaning |
|---|---|
| `correct` | System code matches the reference (or an accepted alternate) |
| `wrong_code` | System emitted a code, but not the right one |
| `missed` | A code exists; the system returned UNMAPPED (recall failure) |
| `spurious` | No code exists; the system emitted one (precision failure) |
| `correct_unmapped` | Both agree no suitable code exists (a correct decision) |

From these: **accuracy** `(correct + correct_unmapped) / N`, **precision**
`correct / (correct + wrong_code + spurious)`, **recall**
`correct / (correct + wrong_code + missed)`, and **F1**.

Treating `correct_unmapped` as a success is deliberate: recognising that no
standard code exists is a correct decision, and scoring it as failure would
reward forced mappings.

Also reported: per-cluster and per-vocabulary breakdowns, a **calibration**
table (whether the pipeline's stated confidence tracks correctness), an **error
taxonomy** with examples, and **intra-rater κ** interpreted on the Landis & Koch
scale.

---

## 8. Reproducibility

Recorded alongside the results: pipeline commit hash, mapping model id,
`VECTOR_TOP_K`, terminology seed versions, annotation dates, and the sampling
seed if a sample was used instead of the census.
