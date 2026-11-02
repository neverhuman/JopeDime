# H216: random-init lottery with B-locked strict-margin selection — NO_ADVANCE

**Outcome: NO_ADVANCE.** The sole primary arm `LOTTERY16` achieved a positive median family-equal utility gain of `+0.004441` versus `r882` across all three replicate seeds (`+0.004330`, `+0.010295`, `+0.004441`) and kept family harm within the `<= 10%` cap (`12 / 205 = 5.85%`), but its `10,000`-draw paired-family 95% confidence interval was `[-0.010316, +0.029053]` (`LCB95 <= 0`) and its Holm `m = 2` adjusted one-sided `p`-value was `0.662734` (unadjusted `p = 0.347265`). The primary gate fails. All five outer folds (`1,219` contexts / `205` families) and all three replicate seeds completed cleanly in a single guarded CPU attempt (`487.21 s`). No model was adopted.

See [PREREG.md](PREREG.md), [metrics.json](metrics.json), [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json), and [provenance.json](provenance.json).

## Hypothesis

`H180`, `H183`, and `H184` established that the `adapter_10tok` architecture (`47` tensors / `29,211` parameters) acts as a structural kernel carrier, trained weights add nothing detectable over initial states (`ENS10 - RENS10 = -0.001153` in `H183`), and the untrained 10-member ensemble `RENS10` scored `+0.016972` (`95% CI [-0.014237, +0.050233]`). `H216` tested whether **selecting a single untrained adapter initialization on the reserved `B` split**—requiring its `10,000`-draw paired family-bootstrap median utility gain over `r882` on `B` to have 10th-percentile `> +0.005` (`LCB80 > +0.005`), else keeping `r882`—yields statistically reliable held-out lift on `A`. This is a single-model selection test, not an ensemble and not an optimizer sweep.

## Setup

- **Cohort & Outer-Fold Partition:** Real S3 `TRAIN` cohort (`1,219` `DL-Train` contexts across `205` families; `91` `DL-Val` contexts across `19` disjoint families for the `r882` native baseline parity gate `1.1377941898164525`, verified bit-exact before opening any `TRAIN` outcome pool). Reuses the `H183` family-grouped 5-fold outer partition (`fold_seed = 183140`, `A_outer = fold[outer]` `41` families, `B_outer = fold[(outer + 1) % 5]` `41` families, `FIT_outer = remaining 123 families` unused). Zero optimizer updates (`actual_optimizer_updates = 0`). `DOPE` (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) and `r882` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) verified unchanged before and after execution.
- **Replicate & Candidate Seeds:** `3` replicate seeds (`216101`, `216102`, `216103`), `K = 16` fresh constructor-initialized `adapter_10tok` states per replicate seed (`216201–216216`, `216301–216316`, `216401–216416`; `48` distinct tensor SHA-256 digests verified).
- **Arms:**
  1. `R882` (control): frozen `round-882.pt`.
  2. `LOTTERY16` (sole primary arm): on `B_outer` (`41` families), draws `10,000` paired family bootstrap samples (`stream_seed("H216", "B_BOOT", seed, outer)`) of each candidate `k in 1..16`'s equal-family utility gain over `R882`; selects the candidate maximizing `B_outer` bootstrap-median gain among those with `b_boot_q10_gain > +0.005` (and `b_boot_median_gain > +0.005`), else keeps `R882`.
  3. `PLACEBO` (null-selection control): applies the exact same strict-margin `B_outer` rule after family-shuffling `B_outer` gains across families and candidates (`stream_seed("H216", "PLACEBO_SHUFFLE", seed, outer)`).
  4. `ORACLE_A` (diagnostic only, cannot win): hindsight single-model ceiling selecting the best single model in `{"R882", 1..16}` on `A_outer` equal-family utility.
- **Stage-1 Lock Before `A`:** All `15` `(seed, outer)` `B`-split selections were computed with `A_outer` pools explicitly absent from `engine.pools`, persisted and `fsync`ed to `b_selection_lock.json` (`SHA-256: 70323592e1c6764a83091224cd039a0fae784d357f54881d5d997f95bef11c61`), and create-only uploaded/verified on S3 (`VersionId: nxllTYYPBxoLZFRzx2lmwRZW5PzoVy9Q`) before Stage 2 scored `A`.

## Outcome and metrics

| Arm | Median utility | Gain vs `r882` [95% CI] | Unadj. `p` / Holm `m=2` `p` | Harmed families (`<-0.01`) | Elo (`r882=1500`) [95% CI] | Exact `-2.0` counts by seed |
|---|---:|---|---:|---:|---|---:|
| `R882` | `1.385355` | `+0.000000 [+0.000000, +0.000000]` | — | `0 / 205 (0.00%)` | `1500.00 [1500.00, 1500.00]` | `17 / 17 / 17` |
| `LOTTERY16` (primary) | `1.389796` | `+0.004441 [-0.010316, +0.029053]` | `0.347265 / 0.662734` | `12 / 205 (5.85%)` | `1501.00 [1493.95, 1506.71]` | `18 / 16 / 19` |
| `PLACEBO` | `1.386346` | `+0.000991 [-0.015707, +0.029754]` | `0.533147 / —` | `47 / 205 (22.93%)` | `1504.24 [1492.57, 1512.70]` | `18 / 16 / 20` |
| `ORACLE_A` (diagnostic) | `1.413701` | `+0.028345 [-0.000624, +0.061806]` | `0.047495 / —` | `74 / 205 (36.10%)` | `1510.14 [1494.78, 1522.30]` | `12 / 13 / 13` |

Per-seed equal-family utility gains on `A` (`205` families):

| Contrast | Seed `216101` | Seed `216102` | Seed `216103` | Median [95% CI] | Holm `m=2` `p` |
|---|---:|---:|---:|---|---:|
| `LOTTERY16 - R882` | `+0.004330` | `+0.010295` | `+0.004441` | `+0.004441 [-0.010316, +0.029053]` | `0.662734` |
| `LOTTERY16 - PLACEBO` | `+0.004430` | `-0.000122` | `+0.003450` | `+0.003450 [-0.008566, +0.012671]` | `0.662734` |
| `PLACEBO - R882` | `-0.000099` | `+0.010417` | `+0.000991` | `+0.000991 [-0.015707, +0.029754]` | — |
| `ORACLE_A - R882` | `+0.028637` | `+0.028345` | `+0.027570` | `+0.028345 [-0.000624, +0.061806]` | — |
| `ORACLE_A - LOTTERY16` | `+0.024307` | `+0.018050` | `+0.023129` | `+0.023129 [-0.002363, +0.047667]` | — |

### B-Split Selection Behavior (`15` cells = `5` folds $\times$ `3` seeds)

- **`LOTTERY16` adopted a non-`r882` initialization in `3 / 15` cells** (all 3 seeds on outer fold `1`, selecting `CAND1` on `216101` with `B` bootstrap median `+0.051734, q10 = +0.016592`; `CAND13` on `216102` with `B` median `+0.047112, q10 = +0.014115`; and `CAND12` on `216103` with `B` median `+0.049195, q10 = +0.015973`) and fell back to `R882` in the remaining `12 / 15` cells (`folds 0, 2, 3, 4`) where cross-family variance on the 41-family `B` split kept `max q10 <= +0.005` (`-0.050558` to `+0.002769`).
- **`PLACEBO` adopted a non-`r882` initialization in `7 / 15` cells** (because family-shuffling broke correlated family tail risk across candidates on `B`), resulting in nearly `4x` higher family harm on `A` (`47 / 205 = 22.93%` vs `12 / 205 = 5.85%` for `LOTTERY16`) and lower median utility gain (`+0.000991` vs `+0.004441`).
- **Diagnostic `ORACLE_A` ceiling:** Even choosing the best single untrained adapter per `(seed, outer)` in hindsight on `A_outer` yields `+0.028345` (`95% CI [-0.000624, +0.061806]`) while harming `74 / 205` (`36.10%`) families. A single random `adapter_10tok` initialization induces family-heterogeneous cost shifts that help some families while hurting others; unlike multi-adapter cost averaging (`RENS10`), picking one untrained ticket cannot capture the variance-reduction benefit of an ensemble.

## Red-team notes (H-wave routine, 2026-11-02 ~11:05Z)

- **Deviation D1 (process, disclosed):** the full 3-seed run was launched from the committed prereg `b3bdbd847` while PR #1809 was still open, instead of after its merge (hw_t0325 step 2). `PREREG.md`, `runner.py`, `analysis.py`, `parents.py` and `test_h216.py` are byte-identical between `b3bdbd847` and the closeout commit `5ff5ced54`; only README, manifest and metrics changed. The B-selection lock was written before any A read. No effect on the verdict.
- **Shared partition caveat:** all three replicate seeds reuse the H183/H213 outer-fold manifest by design, so seed agreement is not independent replication of the fold split. All 3 non-r882 adoptions fell on outer fold 1 (one per seed), which is a fold effect, not seed-robust evidence.
- **Checks:** DOPE c51a56d0 before/after, native91 bit-exact 1.1377941898164525, 0 optimizer updates, no sealed/holdout/meta-test rows, no GPU. Ties reject; the verdict stands as NO_ADVANCE.

## Verdict and next step

- **Verdict:** **NO_ADVANCE** (`advance_candidate = false`, `serving_promotion = false`).
- **Next step:** Keep `r882` as reference. Do not pursue single-ticket random-init lottery selection without weight adaptation or outcome-free context-level gating (as in `H210`). Any promotion candidate continues to require full `H177`/`H178` qualification.

## Execution resources and S3 custody

- **Host & Guards:** `xbabe2` CPU only (`nice -n 19`, `CUDA_VISIBLE_DEVICES=""`, `5` worker processes $\times$ `1` thread, no `GPU0`, no `qualification.lock`). Start/end `load1`: `35.18 / 36.29` (`< 85.0`), `MemAvailable`: `84.74 / 81.23 GiB` (`>= 16.0 GiB`, swap `19.44 / 19.50 GiB` recorded), disk used: `93.55% / 93.52%` (`<= 96.0%`), process-tree RSS: `0.42 / 3.17 GiB`, elapsed `487.21 s`.
- **Versioned S3 Custody (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H216/argon-t0300/`):**
  - `b_selection_lock.json`: `VersionId` `nxllTYYPBxoLZFRzx2lmwRZW5PzoVy9Q`, `SHA-256` `70323592e1c6764a83091224cd039a0fae784d357f54881d5d997f95bef11c61` (`476,325` B).
  - `OOF-panels.npz`: `VersionId` `GlCjyLXa8lH.1zc.lGou6JqgFpoTeTTc`, `SHA-256` `f5f45c7a85ecf7cf03646e06ae03a582f4d5c8e86f56fe6d2120f3de9065bee1` (`159,637` B).
  - `audit.json`: `VersionId` `eP7RVKX494wnXbdQYTF8pe2TffYfK_kC`, `SHA-256` `718fdbcb1c32b829e06d3458aaf51eed47bb046ddfc540c584b5545f88094701` (`153,521` B).
  - `initial_hashes.json`: `VersionId` `18LbKMiNZAuZxEd_AkTkc6IFUKNaUdrI`, `SHA-256` `bbb4f3e2d2a81c585462d17862f61fbe68a7171ba58267708d541dd58729a831` (`3,843` B).
  - `metrics.json`: `VersionId` `Meayw7e0yLIqdb.zWUZ5o9y7oXQLyCv2`, `SHA-256` `0ab037b10b899e640451377e334fb46b86619d2b700f10fa7a32322e43b78f02` (`18,742` B).
  - `evidence.tar.gz` (`541` members including all `B`/`A` candidate score `.npz` shards): `VersionId` `8ZcuWWmkIhh8qZ4SPNej.JZfqCIQTQYP`, `SHA-256` `613e9b29fc041738405407318e344e9c66f08d920e45b8c1118fbb3e967a9eac` (`565,539,646` B).
