# H124 independent CPU, family-harm and Elo audit

2026-10-31, `sol_v1_obj02`, owner `unpark_t0855_sol_v1_obj02.md`.
**Disposition: partial run corroborated; corrections required before another
scoring attempt. No H124 gain, harm rate, Elo or admission result exists.**

This audit covers merged main `c989ac5cdcb6769380027a700d2822a54d32b3da`
(#1058), executed source `e3c3b317c9170f8e1c8bf5c3f438b71230837dbf`,
and preregistration commit `01d8c8c06d9cf9b0ccc15c521d5acc26bbdfd332`.
The executed and merged [runner](runner.py) bytes match SHA-256
`ec1c0bc2f0ec737fb75c9a4fc65ffddb75347d6a944298933a441d78ccdaa4d0`.
The [preregistration](PREREG.md) matches SHA-256
`38925d26c0571d1eaa9971835a14ab2a65cfc5b3c890aea8189597a68aeba2ab`
and is an ancestor of the executed source. This is an audit of that version,
not authorization to resume or change its protocol.

## Findings requiring owner correction

1. **H124-A1, high: the executed input closure includes DL-Val features.**
   `load_data` reads all four arrays from the mixed H58 `context_features.npz`
   and constructs features for every cached context. Its closure check explicitly
   requires `train + dev`, not train alone. The pinned cache manifest describes
   1,310 contexts: 1,219 train and 91 validation. A synthetic archive passed
   through the actual loader returned all 91 validation feature entries; no real
   feature payload was opened by this audit. `baseline` also loads the mixed
   pool-prior arrays before selecting train indices. Thus “no DL-Val data or
   payloads” in the preregistration is stronger than this execution provides.
   Train table/pool selection remains separate; this finding does **not** establish
   use of validation labels in fitting, and no GBM fit completed. Resolve the
   boundary with approved, pinned train-only feature/prior inputs before scoring;
   merely filtering a dictionary after decoding the mixed arrays does not undo
   those reads. Any changed input pins need an explicit pre-run protocol amendment.

2. **H124-A2, medium: the preregistered median seed Elo is missing.**
   `elo_h124` returns per-seed ratings, but its headline `ratings` and
   `candidate_ci95` describe Elo after averaging each context's utilities across
   seeds. It never returns the preregistered median of the three seed Elo ratings
   or an interval for that estimator. These quantities are not interchangeable.
   In a synthetic 205-family control, seed deltas `+0.03, +0.03, -0.09` produce
   seed ratings `3224.71, 3224.71, -224.71`: median `3224.71`, versus the actual
   pooled-score headline `-224.71`. These are software counterexamples, not H124
   results. Report the registered estimator and label any additional pooled-score
   estimate explicitly; retain family pairing when bootstrapping the median.

3. **H124-A3, medium: RSS checks do not enforce a continuous peak ceiling.**
   `headroom` checks peak RSS at phase/fit boundaries, with no check in the
   baseline context loop or inside an estimator fit. The retained stop receipt
   reports 7,052,092 KiB against the 6,291,456 KiB ceiling. This is a resource-limit
   failure, not proof that a strict cap was maintained. The receipt attributes
   termination to SIGTERM; no raw worker exit status or signal trace is retained,
   so the exact termination mechanism is not independently established. Require a
   memory-bounded implementation and resource preflight before any further run;
   do not repeat this attempt to recover a metric.

## What the retained CPU evidence establishes

All six files listed in `partial-run-blocker.json` match both recorded lengths
and SHA-256 hashes. The retained split matches the pinned selected train roster:
1,219 unique contexts, 205 families, and five fresh 41-family folds reconstructed
with RNG 12400. The baseline NPZ contains exactly those 1,219 context keys and
1,404,287 finite candidate costs, with each vector matching its metadata count.
It contains predictions, not a completed paired blend panel.

The copied r882 checkpoint hashes to
`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
Its load receipt has no missing/unexpected keys. The retained DOPE guard reports
the frozen `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
before/after, CPU execution, no CUDA initialization and zero optimizer updates.
This audit verified those receipts and checkpoint bytes; it did not deserialize
the checkpoint or independently repeat the model pass.

Launch metadata records niceness 19, four OMP/MKL/OpenBLAS threads, load 31.80,
93,973,460 KiB available memory and an empty CUDA visibility setting. The stop
receipt reports zero fit receipts; none is present in the retained inventory.
The committed [metrics](metrics.json) correctly leave primary gain, harm and Elo
null with `STOPPED_RESOURCE_LIMIT` / `NOT_ASSESSED`. The incomplete attempt does
not support “shrinkage reduces harm,” a winner, or a serving/default change.

Metadata cross-check: the 1,219 train contexts cover 257 catalog dataset IDs
grouped into 205 cohort families. All 257 have cached real/regression/non-meta-test
flags. Thirty-one cohort families merge multiple catalog keys; no catalog key
is split across cohort families. This verifies the crosswalk, not a new
deduplication assessment. Three deterministic sampled dataset-manifest
GETs from the real S3 dataset bucket matched their exact inventory VersionIds and
SHA-256 hashes. Only metadata was fetched, not dataset blobs; the other 254 were
checked against cached metadata rather than re-fetched.

## Family-harm and Elo averaging

The following are **software controls over synthetic panels**, not empirical
H124 estimates. The arithmetic checked here passed; A1–A3 still apply.

| Quantity | Actual aggregation and independent check |
| --- | --- |
| Primary utility | Average context U4 within each family for each seed; subtract matching r882; average the 205 family deltas equally within a seed; take the median of the three seed means. Context-rich families get no extra weight. |
| Family harm | A family is harmed when its **three-seed mean family delta** is strictly below -0.01. Divide the harmed-family count by 205; do not count harmed contexts or any-seed events. |
| Utility bootstrap | 10,000 family-cluster draws, RNG 124000, retain all three seed values for every sampled family, then recompute the median seed mean. The one-sided LCB is its 5th percentile. |
| Elo headline | First average utility across seeds **per context**; classify that context as win/tie/loss with margin 0.0001. Average context outcomes within each family, then give every family total weight one in two-contender Bradley–Terry, prior 0.01 and r882 anchored at 1500. This is not Elo after thresholding mean family utility. |
| Elo interval | Resample those family fractional-win weights 1,000 times with RNG 124001. The existing interval belongs to the pooled-score Elo above, not the missing median seed Elo. |

Independent two-contender Elo solves
`1500 + 400*log10((W+0.01)/(F-W+0.01))`, where `F=205` and `W` is the sum
of family mean context-win weights. This matched the pinned canonical helper
and the runner's interval. In an unbalanced synthetic panel with 100 winning
contexts in one family and 204 single-context losing families, actual Elo is
`577.87`, matching equal-family weighting; incorrectly weighting contexts gives
`1376.16`. Its 95% interval independently replays to `[-224.71, 769.28]`.
A separate mixed-context family gives `1499.44` under the actual context-WTL
procedure versus `1501.69` after thresholding family mean utility.

Harm controls also distinguish the units: 100 harmed contexts belonging to one
family produce `1/205 = 0.4878%` family harm, not `100/304 = 32.8947%` context
harm. Twenty harmed families pass the 10% cap (`9.7561%`), while 21 fail
(`10.2439%`), despite positive gain and LCB in both synthetic controls. Independent
bootstrap reduction to sampled harm counts matched both LCBs. Mean-across-seed
harm and the strict -0.01 threshold passed separate controls.

## Verification and custody

The existing H124 module tests executed **6 passed, 0 failed, 0 skipped**.
The independent controls separately verified retained identities, folds,
baseline coverage, unequal-family weighting, harm boundaries, the utility
bootstrap, closed-form Elo/interval replay, the missing-median counterexample
and the mixed-cache scope counterexample. They used no estimator fits, real
feature/table/pool payloads, GPU or qualification lock. Auditing peak RSS was
124,064 KiB. No existing checkpoint, dataset, receipt, protocol or peer source was
changed.

Durable receipt: `~/jope_plan/master/status/H124-audit-t0855.md`.
Commands, replay script, JSON controls, artifact hashes, catalog crosswalk and
exact-version S3 readbacks: `~/jope_plan/evidence/H124/audit-t0855/`.
Replay the retained `audit_controls.py` with `nice -n 10`, one OMP/MKL/OpenBLAS
thread, hidden CUDA and Python `-B`; it binds runner bytes to the audited source.
The original worker's missing exit status and absence of a complete blend panel
remain explicit limits. The H124 owner retains runner/protocol correction scope;
the central merger receives this audit report only.
