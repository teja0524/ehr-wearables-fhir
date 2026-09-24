# UMLS seed enrichment (build-time)

Offline terminology management for the CodeMapper vector store. It restocks the
**shared** seed (`data/terminology/*_seed.csv`) so the existing runtime cosine +
LLM-select step finds codes it currently misses (the reason ACE stalled at
~18/86). It does **not** change the runtime pipeline and never runs inside a
per-subject run.

## Where it sits

```
BUILD TIME (this tool, occasional, offline)          RUN TIME (unchanged)
──────────────────────────────────────────          ────────────────────────
variable label → _umls_term() → UMLS /search         _build_search_query()
      → candidate codes → review (accept=Y)                 ↓ cosine over seed
      → append to data/terminology/*_seed.csv          top-k candidates
      → rebuild vector store (CPU, seconds)             _llm_select_best_code
                                                        → code or UNMAPPED
```

UMLS runs only at build time. The seed is shared across every dataset, so
enrichment defaults to **all** datasets. There is no default dataset to pick.

## Prerequisites

Free UMLS UTS licence + API key: https://uts.nlm.nih.gov/uts/profile
Add it to `fhir_mvp/.env`:

```
UMLS_API_KEY=your-key
```

## Use (via main.py — the primary entry point)

Run from `fhir_mvp/`, venv active.

1. Sanity-check the API path with one term (UMLS `/search` is lexical i.e. it needs
   short clinical terms, so we send a cleaned label + a partialSearch fallback):

   ```bash
   python main.py --seed-test "history of alcoholism"
   ```

2. Enrich. Defaults to all datasets. Recommended: only backfill currently
   UNMAPPED variables so the working ones are left alone:

   ```bash
   python main.py --enrich-seed --seed-only-unmapped output/ace/mapping_cache.json
   ```

   Writes `data/terminology/seed_candidates/umls_seed_candidates.csv`. The log
   prints `term=... hits=N` per variable and a final "X variables got ≥1
   candidate".

3. Review that CSV. The `accept` column is pre-set `Y` on the top hit per
   variable/vocab, change bad ones to blank/`N`. This is the safety gate.

4. Apply the approved rows to the shared seed CSVs (deduped by code):

   ```bash
   python main.py --apply-seed
   ```

5. Rebuild the vector store so the new rows get embedded:

   ```bash
   python main.py --dataset ace --rebuild-store --mapping-only --max-subjects 0
   ```

Then run the pipeline as usual. The mapper is unchanged, only the seed gets richer.

Narrowing options: `--seed-datasets ace` (source labels from one dataset only),
`--seed-only-unmapped PATH`.

## Standalone (secondary)

The same functions are runnable directly for power use / scripting:

```bash
python tools/umls_seed_builder.py build [--datasets ace] [--only-unmapped PATH]
python tools/umls_seed_builder.py apply
python tools/umls_seed_builder.py test "clock test"
```

## Honest gaps

- Expect to recover the *codeable* variables (named instruments like Boston Naming,
  Clock test, HAD, digit span, WMS/WAIS subtests; clinical findings like alcoholism
  history, visual/hearing problems, handedness; anatomical volumes), **not** all the unmapped attributes. Composite SEM scores, freesurfer-adjusted volumes, and `PRS_AD` have no
  LOINC/SNOMED equivalent and correctly stay local.
- **Units** and LOINC part fields / SNOMED hierarchy are left blank on new rows;
  only the term + `search_text` drive recall. Units come from the variable dict
  at runtime anyway.
- `search_text` = UMLS term + variable label/notes. To add true synonyms, extend
  `_proposed_search_text` with a CUI/atoms follow-up call.
- If you prefer the `umls-python-client` package over raw REST, Swap the `UMLSClient`
  class (see the header comment in the script).
- **Review before trusting.** UMLS surfaces plausible-but-wrong hits too (recall
  GDS→CDR / PRS_AD→APOE). The `accept` gate + your override UI are the net.
```
