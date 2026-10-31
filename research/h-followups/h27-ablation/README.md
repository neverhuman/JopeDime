# HWV-05: H27 fingerprint routing ablation

CPU preparation; the requested new DL-Val experiment has **not run**. No new Elo,
confidence interval, ADOPT/REJECT verdict or ranker fit is claimed by this directory.
The current [execution closeout](EXECUTION-STATUS.json) is
**BLOCKED_INPUT_QUALIFICATION**, not a scientific rejection or completed study.

## Design note

Preserve the preceding worker's preregistration byte for byte in
[analysis-preregistration.json](analysis-preregistration.json), SHA-256
`7800becbb4c3cdd45420a9c8210ac871c3c135aeb569f2b8bd4e3c88abd2a124`.
Two equal-capacity width-64 H27 M0 interaction rankers use the same initialization,
schedules and frozen DOPE state, three paired seeds, and the final 400 updates.
The control zeros eight GP fingerprint coordinates after training-only scaling.
Only DL-Train fits rankers/scalers; DL-Val supplies the final paired endpoint.
Reuse the 34,000 existing probe fits. No new probe fits, DOPE updates, holdout reads,
seed selection, early-checkpoint selection or serving promotion are authorized here.
Host execution uses nice 10 and at most four build/math threads under the owner's
current limit, superseding the older eight-thread resource setting without changing
the scientific recipe or its six-fit/2,400-update budget.

The endpoint averages paired wins/ties over all contexts and seeds within each
authoritative family, then weights families equally. Relative Elo is
`400 log10(p / (1-p))`; 100,000 whole-family resamples with seed 20261030 produce
the central 97.5% interval. Preserve infinite endpoints. ADOPT requires its lower
bound above zero **and** completed scientific input/execution qualification.
This relative Elo is distinct from Golden45 method-relative ratings. Binary is
NOT_RUN and never blended with regression.

## Required input recovery

The historical [36-model study](../../../docs/research/studies/dope-neural-ranking-20261021/h27/nn-fixed400-study.md)
already found no established lift on 850 historically exposed TRAIN contexts.
Its component crossfit is not the prospective DL-Val contrast. The old eight-probe
CV5, archived candidate S1 two-fold targets and neural component folds are distinct.

Before fitting, recover and hash the qualified fingerprints (aggregate identity
`c544d47f766b1dea941f7eb97d6d6bb035229b0a67b996ec51d280fe81d208db`),
candidate descriptors/fitness, frozen-state encodings, source capsule and exact
context/dataset/cut/evaluator/vocabulary identities. Join metadata to approved real
S3 catalog and authoritative DL-Train/DL-Val family rosters before opening payloads.
Require full validation fingerprint coverage and no connected-family overlap;
historical TRAIN rows cannot be relabeled as validation. Missing coverage blocks
execution. Do not create replacement probes without a documented, prospective
change to the zero-new-probe preregistration.

The retained metadata receipt [INPUT-COVERAGE.json](INPUT-COVERAGE.json) records
728/1,219 matching DL-Train
contexts and 67/91 matching DL-Val contexts, with 24 validation contexts absent.
Historical source roles are 690 FIT and 160 SELECT; none of the matched historical
components crosses the current train/dev boundary. No feature/pool/table payload
was opened. This is metadata coverage, not numerical or source qualification.
The pinned manifest itself was not recovered in the current bounded metadata
search; these inherited coverage counts have not been recomputed.
`qualify-inputs` can reproduce the receipt from that materialization manifest
and the current packet's `selected.json`, without opening any referenced payload.
When those pinned files are available, its receipt lists each unmatched DL-Val
context digest and the next identity-admission step. Execution remains
**BLOCKED_INPUT_QUALIFICATION**, with null Elo/CI/verdict.

## CPU-preparation additions (2026-10-31)

`routing-config.json` records the preregistered paired recipe and explicitly
keeps analysis inadmissible. `routing.py` is a pure helper for an already-scaled
feature row: the control arm zeros exactly the trailing eight fingerprint
coordinates after upstream DL-Train-only scaling. It performs no fitting, file
reads, model loading or scientific scoring. The doctest suite exercises this
path with synthetic values only.

`MISSING-DL-VAL-CONTEXTS.json` records 24 unresolved slots from the existing
67/91 aggregate coverage. It does **not** claim to list their context digests:
the pinned H27 materialization manifest is unavailable, and the retained
aggregate receipt has no per-context identities. Each unresolved slot points to
the same next steps: recover and hash-verify that manifest, run
`qualify-inputs` against the pinned selected roster, then complete full input
identity admission. Analysis remains pending until that succeeds. No null slot
may be interpreted as a scientific context or substituted with a synthetic
outcome.

`analyze` validates paired selection coverage and computes the frozen endpoint;
it does not train models or independently verify the producer's input provenance,
gradient isolation, matched schedules, model hashes or optimizer updates. Review
those receipts before interpreting its verdict. No analysis output is present yet.

Its CLI first hash-verifies the H27 materialization manifest and selected roster,
recomputes metadata coverage, rejects missing validation fingerprints or connected
components crossing train/validation, and requires the complete approved DL-Val
context/family roster. A caller cannot obtain a result for a chosen subset of
validation contexts. These checks precede paired-outcome reads and output creation.
They do not certify the producer's optimizer, numerical features or historical
exposure; the scientific qualification requirements above still apply. The pure
arithmetic function used by tests is not an execution-admission interface.

The input manifest has `dope_loaded_sha256`, the fixed `seeds`, `train_families`,
`paired_selections_sha256`, and `validation_contexts`. Each context supplies
`context_sha256`, `family_id`, `dl_split`, `slate_sha256`, `evaluator_sha256`,
`cut_id` and `vocabulary_sha256`. The paired JSON array repeats these identities
for every context/seed plus `with_fingerprint` and `without_fingerprint` objects.
Each object has `status: ok` with finite raw constrained `fitness`, or explicit
`status: failed` with null fitness. Missing/unknown outcomes stop the analysis.

```sh
research/h-followups/h27-ablation/analyze \
  --h27-manifest H27_MATERIALIZATION_MANIFEST.json --selected SELECTED.json \
  --manifest QUALIFIED_MANIFEST.json --pairs PAIRED_SELECTIONS.json --output NEW_RESULT.json
pytest research/h-followups/h27-ablation -q
just hygiene
just lint-paths
```

Tests use labeled arithmetic/refusal fixtures only; they are not experimental
results. Runtime data and model artifacts belong outside the working tree.
