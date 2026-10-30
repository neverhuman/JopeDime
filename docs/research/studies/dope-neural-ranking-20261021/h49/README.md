# H49: SAM / SWA / ES2 under strict-margin selection

Status: **complete — no strict win** (2026-10-30, `cuda:0`). H49 tested whether flatter-minimum optimizers (SAM, corrected SWA) or a larger-budget antithetic ES (ES2, fresh seeds 3–5, ~141 s/seed = 2× H48-PF median wall) could break the r785 adapter plateau. Only the 29,211-parameter adapter trained; DOPE stayed frozen. Baseline r785 was reused from H48 (not rescored on DL-Val).

## Hypothesis

The flywheel plateau (val utility ~1.134–1.138, 70+ consecutive rejections) comes from sharp, fold-specific minima. Sharpness-aware updates or weight averaging should transfer better to unseen families, and a longer ES search should find utility moves that gradient methods miss.

## Protocol and decision rule

Same cohort as H48: 1,219 DL-Train contexts / 205 families, 91 DL-Val contexts / 19 families, zero context or family overlap. Five fixed whole-family DL-Train folds; 0/SAM/SWA select the best fold checkpoint on inner held-out families (with the SWA selector fixed to keep the incumbent on ties/regressions — see Integrity). ES2 optimizes the inner family-CV objective over all folds. Each arm/seed got exactly one terminal DL-Val read (12 total). Strict win: median family-mean final Δ > 0 **and** 10k family-bootstrap 95% lower bound > 0 **and** Elo > 1500. Ties fail. The SEL audit (`H49-SEL-audit.json`) checked the pre-registered strict-margin rule (family-bootstrap lower bound of inner-CV gain > 0.001) against three known historical noise events (r782 and r882 exact ties, r859 single-context +0.000199): 0/3 survive.

## Terminal results

| Arm | Median family-mean Δ [family 95% CI] | Median context Δ | Elo vs r785 (1500) | W/T/L vs r785 | Min LOFO Δ | Effective seeds | Strict win |
|---|---:|---:|---:|---:|---:|---:|---|
| H49-0 | -0.0021 [-0.0066, +0.0016] | -0.0022 | 1492.68 | 4/81/6 | -0.0030 | 1 | No |
| H49-SAM | -0.0320 [-0.0995, +0.0002] | -0.0334 | 1496.34 | 10/66/15 | -0.0790 | 3 | No |
| H49-SWA | -0.0095 [-0.0227, +0.0005] | -0.0099 | 1485.36 | 6/75/10 | -0.0108 | 2 | No |
| H49-ES2 | -0.0308 [-0.0986, +0.0000] | -0.0321 | 1492.68 | 0/89/2 | -0.0354 | 3 | No |

Paired vs H49-0 (per-family median of the three control seeds): SAM −0.0299 [−0.0976, +0.0023]; SWA −0.0074 [−0.0193, +0.0005]; ES2 −0.0287 [−0.0985, +0.0055]. Every arm's median is negative and every Elo is below 1500. ES2 seed 5 is an exact tie (Δ = 0.0), which the strict rule rejects.

### Per-seed terminal values

| Arm | Seed | Selected fold | Inner-CV selected Δ (exploratory) | DL-Val utility | Context Δ | Family-mean Δ | Selected inner-CV train−Val gap (optimistic) | Adapter SHA |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| H49-0 | 0 | 1 | +0.0173 | 1.1354 | -0.0022 | -0.0021 | 0.2386 | 8956d33965 |
| H49-0 | 1 | 1 | +0.0309 | 1.1354 | -0.0022 | -0.0021 | 0.2416 | 37437d1aff |
| H49-0 | 2 | 1 | +0.0288 | 1.1354 | -0.0022 | -0.0021 | 0.2409 | d7b134b633 |
| H49-SAM | 0 | 1 | +0.0470 | 1.1193 | -0.0183 | -0.0174 | 0.2598 | f7c12e9d11 |
| H49-SAM | 1 | 1 | +0.0411 | 1.1042 | -0.0334 | -0.0320 | 0.2731 | 563dd55830 |
| H49-SAM | 2 | 1 | +0.0324 | 1.0618 | -0.0758 | -0.0725 | 0.3119 | 071c39ccbc |
| H49-SWA | 0 | 1 | +0.0092 | 1.1277 | -0.0099 | -0.0095 | 0.2443 | 3e7172bea9 |
| H49-SWA | 1 | 1 | +0.0093 | 1.1294 | -0.0082 | -0.0079 | 0.2427 | 8ad564a0ce |
| H49-SWA | 2 | 1 | +0.0098 | 1.1277 | -0.0099 | -0.0095 | 0.2444 | a07c9cbf4c |
| H49-ES2 | 3 | all | +0.0084 | 1.1026 | -0.0350 | -0.0336 | 0.2749 | 799ee23555 |
| H49-ES2 | 4 | all | +0.0109 | 1.1055 | -0.0321 | -0.0308 | 0.2746 | 74681ca99e |
| H49-ES2 | 5 | all | +0.0093 | 1.1376 | +0.0000 | +0.0000 | 0.2423 | 364106dd95 |

## Red-team findings

- **Winner's curse, again.** Every 0/SAM/SWA seed selected fold 1 (H48 selected fold 3 every time). All twelve inner-CV selected deltas are positive (+0.008 to +0.047); no terminal delta is. SAM has the largest inner-CV gains and the worst DL-Val deltas, and the train−Val gap widens with them (0.26 → 0.31). Inner-CV gain is anti-informative at this scale.
- **Pseudoreplication.** The three H49-0 seeds produced three different adapters (distinct SHAs) with *identical* DL-Val utility (1.135439), so effective replicates = 1. DL-Val utility is locally flat: 81/91 contexts tie r785 for the control. This is direct evidence for H50-S's question of whether the adapter can move utility at all.
- **Tie acceptance.** ES2 seed 5 = exact 0.0, and the ES2 CI upper bound is +1e−16. Both are rejected as ties by the strict rule, as intended.
- **SWA selector bug.** Caught before any DL-Val read: the first SWA code took the averaged candidate even when its CV utility was below r785. That run was stopped (only the H49-owned wave) and preserved under `~/jope-data/h49/failures/SWA-baseline-selection-bug/`. SWA seeds 0–2 were rerun with the fixed selector and a regression test. No terminal file existed for the buggy run.
- **Label discrepancy (minor).** Early progress notes quoted per-seed "family-CV selected" deltas (e.g. +0.0035 for H49-0 seed 0) that differ from the analysis `selected_inner_cv_delta` (+0.0173). These are different aggregations of exploratory, train-only numbers and don't affect any terminal result. The table above uses the analysis file.
- **Metric direction.** Utility is higher-is-better; Δ = candidate − r785. Signs agree with Elo W/L counts.
- **Leakage / sealed data.** 93 DL-Holdout contexts excluded, 0 holdout payloads opened, 0 outer rows delivered, train/val overlap 0. The live flywheel loop was never read or modified (checkpoint copies only).
- **DOPE freeze.** DOPE SHA256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` was unchanged in every record; trainable params = 29,211.
- **Resources.** Ran serialized on cuda:0 under `qualification.lock`. Host MemAvailable stayed ≥ 84 GiB, CPU < 30%, and there was no OOM. The terminal scorer waited for an unrelated gpu_bench lock holder and never preempted it.

## Verdict

No strict win; this is a clean negative. Flat-minimum methods (SAM, SWA) and a 2× ES budget all *lower* DL-Val utility relative to r785. Taken with H46–H48, the evidence points to adapter selection overfitting the inner folds, and DL-Val utility being nearly insensitive to adapter changes (identical utility from distinct adapters). Optimizer tricks on the adapter are not where the plateau breaks.

## Next step

H50 per preregistration: first **H50-S**, a utility sensitivity map (L2 radii 0.01/0.03/0.1/0.3/1.0 × 64 seeded directions, inner held-out DL-Train folds only). If > 90% of perturbations give exactly zero utility change at every radius, stop the adapter search and conclude the adapter is not the bottleneck; the lever is then the GP program pool / DOPE-side features (DOPE stays frozen without a separate ruling). Otherwise run H50-ES3 / sep-CMA / ES-EMA on fresh seeds 6–8 under the same strict rule.

## Integrity and artifacts

Checkpoint SHA256 `ef6696f6369122a162d58b9b522fd83cae110a5576ef3760e3b43ba85d129e0a`; source-manifest SHA256 `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`. The compact [`metrics.json`](metrics.json) has per-arm/per-seed summaries. Full per-family/per-context analysis is kept outside Git at `~/jope-data/h49/runs/h49-final-analysis.json` on xbabe2 (SHA256 `827b999b0c271f3bb64b083759803a7c4153c04b800b0d8357209330f88fdffe`). Harness: `research/flywheel_selfplay/h49/`.
