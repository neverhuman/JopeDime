# H244 — zero-update weight-perturbation prediction smoothing

**COMPLETE / NO_ADVANCE.** WSMOOTH_SELECT retained r882 in 14/15 FIT cells. The one non-baseline cell did not yield a positive DL-Val seed result; zero optimizer updates and no serving promotion.

## Hypothesis and setup

Averaging costs under small adapter-weight perturbations may smooth argmin cliffs. [Preregistration](https://github.com/neverhuman/JopeDime/pull/2090) merged before inference; execution `e61adfe566e784ca3182c248df428eb717fe2a3a`. [Fixed protocol](PREREG.md), [tensor RMS and seeds](perturbations.json), [family folds](fold_manifest.json).

r882 plus eight seeded Gaussian perturbations per sigma; per-tensor RMS scaling, sigma grid 0/.01/.03/.1. Three seeds 244101–244103 and fresh family folds 244141–244143; 15 FIT cells. WSMOOTH_SELECT applies the strict FIT-only selector; WSINGLE uses member 1 at that selected sigma and is report-only. All selections were fsync-locked before post-Native91 DL-Val scoring. Zero optimizer updates; DOPE/anchor/non-adapter state unchanged and r882 restored between batches.

## Outcome

| Arm vs r882 | Per-seed family gains | Median gain [CI95] | Positive seeds | Holm p | Elo [CI95] | Harm families per seed |
|---|---|---|---|---|---|---|
| CONTROL | +0.0000000 / +0.0000000 / +0.0000000 | +0.0000000 [0.0, 0.0] | 0/3 | report-only | 1500.000 [1500.0, 1500.0] | 0 / 0 / 0 of 19 |
| WSMOOTH_SELECT | -0.0004432 / +0.0000000 / +0.0000000 | +0.0000000 [0.0, 0.0] | 0/3 | 1.0 | 1500.000 [1500.0, 1500.0] | 2 / 0 / 0 of 19 |
| WSINGLE | -0.0006410 / +0.0000000 / +0.0000000 | +0.0000000 [0.0, 0.0] | 0/3 | report-only | 1500.000 [1500.0, 1500.0] | 3 / 0 / 0 of 19 |

Selected sigma-policy counts across 15 cells: `{"CONTROL": {"r882": 15}, "WSINGLE": {"ONE10": 1, "r882": 14}, "WSMOOTH_SELECT": {"SM10": 1, "r882": 14}}`. A fallback is the same r882 policy, not an independent replication. Per-seed DL-Val utilities average five selected cell policies; cells share the same 91-context exposed development panel. The paired family bootstrap conditions on this checkpoint, corpus and selector. It does not establish generalization to an untouched test panel. Harm means family utility loss >1e-4, as preregistered.

Only seed 244101 / outer fold 0 accepted σ=0.1: FIT family gain +0.00456316, CI95 [0.00065331, 0.00973397]. Its seed-level DL-Val gain after averaging the five selected cell policies was −0.00044321. The other two seeds used only r882. Consequently, the median gain and every median-bootstrap draw are zero; CI95 [0, 0] does not establish precise equivalence. Median Elo is likewise pinned at 1500 by those fallbacks; WSMOOTH_SELECT Elo by seed is 1494.51949 / 1500 / 1500.

TRAIN OOF (secondary, 205 families):

| Arm | Per-seed gains | Median gain [CI95] | Elo [CI95] |
|---|---|---|---|
| CONTROL | [0.0, 0.0, 0.0] | +0.0000000 [0.0, 0.0] | 1500.000 [1500.0, 1500.0] |
| WSMOOTH_SELECT | [-0.0006521715340419104, 0.0, 0.0] | +0.0000000 [0.0, 0.0] | 1500.000 [1500.0, 1500.0] |
| WSINGLE | [-0.0063176516548695185, 0.0, 0.0] | +0.0000000 [0.0, 0.0] | 1500.000 [1500.0, 1500.0] |

DL-Val context-mean utilities (descriptive comparison to historical plateau 1.1389572):

| Arm | Seed 244101 | Seed 244102 | Seed 244103 |
|---|---:|---:|---:|
| r882 | 1.137794190 | 1.137794190 | 1.137794190 |
| CONTROL | 1.137794190 | 1.137794190 | 1.137794190 |
| WSMOOTH_SELECT | 1.137331498 | 1.137794190 | 1.137794190 |
| WSINGLE | 1.137124999 | 1.137794190 | 1.137794190 |

WSMOOTH_SELECT context-mean differences from plateau 1.1389572 are −0.00162570 / −0.00116301 / −0.00116301.

The decision uses paired family-weighted gain against r882; the plateau context mean has a different weighting and is not an admission test. All numerical outputs, CONTROL and placebo contrasts, and harm-family identities are in [metrics.json](metrics.json).

## Context #27

Zero-based dev context `43022ae6a8bc83e9e25ea4a2882389152856226f5b2bdd4668db07f2f510ae91`, family `fam-9f67f1449b77152cbf20a6c82926f90496dacd7f05a222376edf0fdfb45f667f` was retained in every primary panel. Baseline: `{"values": 1.1226949552792034, "choices": 803, "invalid": false}`. Fixed policy readout (diagnostic, never used for selection):

| Fixed seed/policy | Utility | Choice | Invalid |
|---|---:|---:|---|
| r882 | 1.122694955 | 803 | False |
| 244101-SM01 | 1.122694955 | 803 | False |
| 244101-ONE01 | 1.122694955 | 803 | False |
| 244101-SM03 | 1.122694955 | 803 | False |
| 244101-ONE03 | 1.122694955 | 803 | False |
| 244101-SM10 | 1.122694955 | 803 | False |
| 244101-ONE10 | 1.128496258 | 996 | False |
| 244102-SM01 | 1.122694955 | 803 | False |
| 244102-ONE01 | 1.122694955 | 803 | False |
| 244102-SM03 | 1.122694955 | 803 | False |
| 244102-ONE03 | -2.000000000 | 190 | True |
| 244102-SM10 | -2.000000000 | 1010 | True |
| 244102-ONE10 | 1.128496258 | 996 | False |
| 244103-SM01 | 1.122694955 | 803 | False |
| 244103-ONE01 | -2.000000000 | 1010 | True |
| 244103-SM03 | 1.122694955 | 803 | False |
| 244103-ONE03 | -2.000000000 | 1010 | True |
| 244103-SM10 | -2.000000000 | 190 | True |
| 244103-ONE10 | -2.000000000 | 1010 | True |

The selected WSMOOTH_SELECT policy preserves the baseline context #27 choice and utility in all 15 cells. WSINGLE changes it only in seed 244101 / outer fold 0, to valid choice 996 and utility 1.128496258. Fixed σ=0.1 smoothing still fails the gate in seeds 244102 and 244103; these are descriptive diagnostics for policies not selected in those seeds. Thus smoothing did not remove the cliff uniformly. Selected per-cell outcomes are retained in metrics.json; these diagnostics do not alter the primary verdict.

## Verdict and next step

**NO_ADVANCE** under the registered median >0.002, family LCB95 >0, Holm m=1 p<0.05 and ≥2/3 positive-seed rule. WSINGLE is report-only. No serving promotion. Any follow-up requires a separate hypothesis and preregistration; no post-outcome dose/seed changes or repeat of this fixed run.

## Verification, resources and custody

Native91 reproduced all 91 r882 utilities exactly (mean 1.1377941898164525). Terminal cost/choice/gate/utility replay passed 23,580 checks, all 15 selectors and both panels; family CI/Elo decisions re-executed with the same frozen implementation. This is a replay audit, not an independent estimator or independent agent review. Its first attempt stopped on a helper-only file-timestamp variable error; that helper was corrected and the entire audit passed on the same saved results. There was no additional inference or study restart. The archived full access receipt lists 1,328 opened paths, zero refusals and zero protected reads; compact metrics retain its count/hash.

xbabe2 CPU 48–51, nice 19, one numeric thread; A1 guard admitted at RAM ≥30 GiB and load5 ≤100 and cooperatively paused below 20 GiB or above load5 108. Recovery requires ≥25 GiB sustained for 30 s and load5 ≤100; hard stop at RAM <12 GiB sustained for 30 s or <6 GiB instantly. Single process run; 0 pause/resume pairs; elapsed 3644.6 s, pause-request time 0.0 s, peak sampled RSS 3,273,158,656 bytes and 2 aggregate threads. RSS cap 4 GiB, free-disk floor 300 GB and 07:30Z deadline retained. No speed verdict.

Runtime archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H244/astra-t0440/closeout/runtime.tar.gz`, VersionId `whrTeDWe_ODgu2iff1qOgyb_kZY0vw4L`, SHA256 `df6009425a69d6957f889b40230237499179867db72364a91d00dc0593bb153d`.

Archive manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H244/astra-t0440/closeout/manifest.json`, VersionId `DRo.DdUIdzVLpU0RETyvG5.urO1f5QAT`, SHA256 `f5dd1ca65fb933e25a3b27a0907a6e51aea1584b854732b67df04af2ec69d3f0`. Exact-version recovery and all member hashes verified. Full costs/source/input copies and receipts stay in S3; only compact metrics/manifests are committed.

Catalog metadata checks matched all 276 admitted real dataset IDs, with no overlap with the 14 named sealed exclusions or BeyondArena/sealed key metadata. This checks the supplied catalog, not an independent authoritative sealed-family roster.

Real S3 provenance and source/input pins verified; no sealed/holdout/meta-test or live-loop access. Exclusions apply to H244, not historical r882 training/selection. Local 11 tests/Ruff/five records/schema/source/index hashes/diff checks and same-agent exact-head review accompany closeout. Full remote CI NOT_RUN under explicit hw_t0440 owner exception. PR #2071 untouched.
