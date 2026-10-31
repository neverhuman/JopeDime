# H134 — prospective single-primary confirmation of H129 BLEND_025 vs r882

**Owner:** `rd_opus`. Claim: `~/jope_plan/ns/receipts/H134_claim.txt`. Brief: hwave overseer
`rd_opus_t1435_h134`. CPU only.

This file is committed and published **before** H134 opens any pool, checkpoint or feature payload for scoring.

## Motivation

[H129](../h129_gbm_blend_harm_gate/README.md) admitted BLEND_025, merged as #1174 `d00d8d038` and cleared as
CANDIDATE by the overseer red-team. On fold 129029 with seeds 91–93 it showed:
- median seed gain +0.0155 against r882, LCB95 +0.0078;
- Holm p 0.0003;
- harm 5.9%;
- Elo 1523.5.

H134 asks one question on a **new** family fold with **new** seeds: does that result hold with a single
preregistered primary? Nothing is selected from H129 outcomes. λ, the arm and the recipe are fixed below.

## Fixed protocol

**Code.**
- [runner.py](runner.py) imports the merged H129 runner
  `research/flywheel_selfplay/h129_gbm_blend_harm_gate/runner.py`, pinned at SHA-256
  `e34d4a3f7ecb82971fa303f05cca665bb1108509abb90021d1a54717fe02cb6c`.
- That runner in turn imports the H124 module, pinned at `9ebce9652a7caadc7e50896a9c1b667de5aea53f1e3cf4f8797f815d8a16285a`.
- H134 rebinds **only** `SEEDS`, `FOLD_SEED` and `SAMPLE_SEED`.
- Unchanged H129 and H124 code does everything else: data loading with train-row-only cache reads, the 45 HistGBM
  fits, positive-slope fit-only calibration, the r882 reference pass and arm scoring (`tree_packets`).
- H134 adds only its single-primary readout and run guards.
- The runner is not edited after this commit. Any change needs an amendment committed before any outcome is read.

**Data.**
- The pinned real-S3 H50 DL-Train packet: 1,219 regression contexts in 205 families.
- The pinned H58 features.
- A byte-verified copy of r882.
- DOPE stays frozen at `c51a56d0…` and is checked before and after.

**Split and seeds.**
- Family-grouped fivefold cross-fit: sort the family ids, permute them with `default_rng(134141)`, and cut five
  41-family folds.
- Sampling RNG **134142**: 192 unrevealed training programs per context.
- Model seeds **141, 142, 143**, none used before.
- 15 fits per seed: 5 outer folds plus 10 inner pairs, **45 fits** in total.
- The H119/H85 HistGBM recipe is unchanged: `max_iter=300`, `learning_rate=0.08`, `max_leaf_nodes=63`,
  `l2_regularization=1.0`.

**Arms.** All are scored on held-out folds against matching fresh r882 costs, with the H119 Track-2 U4 endpoint.
Contexts are averaged within family, and the 205 families are weighted equally.

| Arm | Score | Role |
|---|---|---|
| `CONTROL` | r882 | reference (Elo 1500) |
| **`BLEND_025`** | **r882 + 0.25·(GBM − r882)**, λ fixed, no search | **sole primary** |
| `PURE_GBM` | GBM | report-only, no admission |
| `BLEND_050` | r882 + 0.5·(GBM − r882) | report-only, no admission |

The shared `tree_packets` also computes H129's HARM_GATED arm. H134 neither analyzes nor reports it.

**Admission (BLEND_025 only; all required, with no rounding or re-thresholding).**

1. The median of the three seed-mean family gains is **> +0.001**, strictly.
2. The family-cluster bootstrap LCB95 is **> 0**. This is the one-sided 5th percentile of the bootstrap median of
   seed means over 10,000 draws with RNG 134000; each draw resamples families and keeps all three seeds.
3. **All three** seed-mean gains are > 0.
4. Harm is **≤ 10%**: the share of families whose three-seed mean gain is < −0.01.
5. The **trimmed mean is > 0**: the mean family gain after dropping the 10 families with the largest |gain|.

A tie or any failed gate is **NO_ADVANCE**. A pass is **CONFIRMED: BLEND_025**.

**Reported, pre-declared tail checks.** These are computed on the three-seed mean family gains:
- the family median gain;
- the share of total gain that comes from the top 10 families;
- the trimmed mean defined in gate 5.

**Elo.**
- The headline is the median-of-seed Bradley–Terry Elo against r882 = 1500, with a 95% family-paired bootstrap
  interval (1,000 draws, RNG 134001). It is computed by the merged H124 helper.
- The pooled-score Elo is labelled additional.
- Elo is reported, never AUC.

**Completeness.**
- No outcome is written unless all 45 fits complete.
- The size of `metrics.json` is checked before it is written (< 100 KB).
- Per-family arrays go to custody, with their SHA-256 recorded in the metrics.

## Caveat (plain)

**H134 re-splits the same 205 DL-Train families that H124 and H129 used.** A pass shows the result is robust to the
fold assignment and model seed. It is **not** evidence on new data or new families. r882 was exposed to these
families earlier, so this remains conditional development evidence.

The pinned loader also reads DL-Val *identity and layout metadata*:
- the `dev` rows of `selected.json`, used for the family-alias check;
- the context-id column of the H58 caches;
- the proj16 offsets.

It decodes **no** DL-Val feature, table or pool payload (H124 Amendments 1–2). This is disclosed rather than
changed, because the brief requires the runner to be reused unchanged.

## Guards

- **Placement.** CPU only: no GPU, CUDA or `qualification.lock`. Run at `nice 19`, with OMP, MKL and OpenBLAS at 4
  threads.
- **Memory.** Worker RSS ≤ 6 GiB through the H124 watchdog, and MemAvailable ≥ 30 GiB at launch and before every fit.
- **Disk.** Abort if `/` reaches **93.8%**; this is checked at launch and before every fit.
- **Scratch.**
  - Scratch stays under 2 GB, at `~/jope-data/h134/run-*`.
  - Evidence goes to `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H134/`, written
    create-only with a version-pinned readback.
  - Scratch is deleted after custody is verified.
- **Isolation.** Zero DL-Val, sealed, holdout or meta-test payload reads. The live flywheel (xbabe1
  `flywheel_gp_fused`) is never touched.
- **Abort rule.** A guard abort makes the attempt INCOMPLETE. Any resume needs an owner decision and an amendment
  committed before it.
