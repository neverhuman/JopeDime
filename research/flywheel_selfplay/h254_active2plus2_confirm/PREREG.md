# H254 Preregistration — Independent Frozen-Constant Confirmation of `ACTIVE_2PLUS2_HURDLE` vs `GATED_025` and `R882` (`h254-active2plus2-confirm-20261103`)

## 1. Authority, Ownership, and Snapshot Custody

- **Wave ID**: `h254-active2plus2-confirm-20261103` (`H254`)
- **Study path**: `research/flywheel_selfplay/h254_active2plus2_confirm`
- **Owner**: `rd_argon` (`[ARGON]`, screen `4187849.rd_argon`)
- **Authority**: `/home/ubuntu/nudges/hw_t1305_argon_h254.md`
- **Claim receipt**: `/home/ubuntu/jope_plan/ns/receipts/H254_claim.txt`
- **Evidence capsule root**: `/home/ubuntu/jope_plan/evidence/H254/argon-t1305`
- **Byte-exact snapshot of H252 Stage-1 source files** (copied read-only to `/home/ubuntu/jope_plan/evidence/H254/argon-t1305/snapshot/` and pinned in [manifest.json](manifest.json); `/home/ubuntu/audit/JopeDime_main/research/flywheel_selfplay/h252_closed_loop_flywheel/` and `/home/ubuntu/jope-data/h252/` are never edited or written):
  - `/home/ubuntu/jope_plan/evidence/H254/argon-t1305/snapshot/common_h252.py`: SHA-256 `abe974ed10a0cd0d4b08440015a72a7b0adcebb22118c234732b812e9f0b0f92`
  - `/home/ubuntu/jope_plan/evidence/H254/argon-t1305/snapshot/stage1_hurdle_sos_critic.py`: SHA-256 `fc0dd0e2fa543f0540d677609989c1785ef33b07f6cf7e54f4f8bedbd0e5e797`
- **Read-only input snapshot** (`/home/ubuntu/jope_plan/evidence/H254/argon-t1305/inputs/`):
  - `r882/round-882.pt`: `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64` (`DOPE` composite `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`)
  - `r882/best_dev_checkpoint.json`: `30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a`
  - `r882-train-costs.npz`: `57004a91a315b372cde62b126be20edd4eb0f337942e08b19bd85438b520f69f`
  - `r882-secondary-costs.npz`: `0e812aa7ea4ce46a85cbe17cb7eba3a58678cf13ec3b215d48a334dd5626f11f`
  - `promotions.jsonl`: `c9afe08ed4d76f59aa807102423b502b57f8c6bddcfedb6a3ae89f8b85821550`

## 2. Scientific Hypothesis and Frozen Arms

Wave H252 (`agy_research_wave`) reported Stage-1 `ACTIVE_2PLUS2_HURDLE` gains over both `R882` and passive 4-reveal `GATED_025`. Wave H254 conducts an independent, preregistered, frozen-constant replication of H252 Stage-1 `ACTIVE_2PLUS2_HURDLE` vs `GATED_025` (equal 4-reveal budget) and `R882` on `DL-Val` (`Native91`, `91` contexts / `19` families) and `DL-Train` OOF (`1,219` contexts / `205` families) under a fresh family-grouped fold seed (`254140`) and fresh evaluation seeds (`254101`, `254102`, `254103`).

### Frozen Arms (byte-exact from snapshot `stage1_hurdle_sos_critic.py`, zero constant changes)

1. **`R882`**: Frozen `round-882.pt` baseline (`argmin(rcost)`). Must reproduce `Native91 = 1.1377941898164525` (`91/91` bit-exact against `best_dev_checkpoint.json`) before any treatment evaluation.
2. **`GATED_025`**: H217 passive 4-reveal `sem24` RBF kernel correction on `idx4 = rng.choice(n, size=4, replace=False)`, gated by outcome-free `top_zgap < 0.25` (`sigma = 2.0`, `ridge = 0.2`, `w_y = 0.05`).
3. **`ACTIVE_2PLUS2_HURDLE`**: Sequential `2+2` active reveal under the exact same 4-reveal budget (`|rev4| == 4`) and `top_zgap < 0.25` gate:
   - Stage A reveals `idx_a = idx4[:2]` (`2` probes), updates interim cost with `sem24` RBF kernel (`sigma = 2.2`, `ridge = 0.2`, `w_y = 0.035`) + cross-fitted Decomposed Continuous Hurdle + Self-Play prior (`w_h = 0.010`).
   - Stage B selects `2` active contenders `chosen_b = [b1, b2]` (`b1 = argmin_{c not in idx_a} interim_cost[c]`, `b2 = argmin_{c in top6 \ (idx_a U {b1})} (interim_cost[c] - 0.015 * sqrt(d_min))`), forming `rev4 = idx_a + chosen_b` (`4` unique reveals).
   - Evaluates multi-scale `sem24` kernel (`0.6 * exp(-d^2/(2*1.8^2)) + 0.4 * exp(-d^2/(2*3.2^2))`, `ridge = 0.15`, `w_y = 0.05`) on `y4_comb = 0.70 * (y4_f - mean(y4_f)) + 0.30 * (y4_lr - mean(y4_lr))` + Hurdle prior (`w_h = 0.010`) + direct revealed-slot shrinkage (`+0.08 * (y4_f[j] - mean(y4_f))` on `rev4`).

## 3. Splits, Seeds, and Cross-Fitting Hygiene

- **Cohort**: `1,219` `DL-Train` contexts (`205` families) and `91` `DL-Val` (`Native91`) contexts (`19` families); regression only; `has_meta_test == False` enforced on every context.
- **Fresh Fold Seed**: `fold_seed = 254140` ([fold_manifest.json](fold_manifest.json)), partitioning the `205` `DL-Train` families into 5 disjoint folds of `41` families each.
- **Refit Hurdle & SP Priors**:
  - `DL-Train` OOF priors for fold `f` are fit strictly on the remaining 4 `DL-Train` folds (`{j != f}`).
  - `DL-Val` priors are fit strictly on `DL-Train` only (`0` `DL-Val` contexts or families ever enter any prior or promotion fit).
- **Evaluation Seeds**: `254101`, `254102`, `254103` (`rng_train = default_rng(seed)`, `rng_val = default_rng(seed + 1000)`).
- **Bootstrap Seed**: `boot_seed = 254150` (`10,000` paired family bootstrap draws; `400`-iteration MM Bradley-Terry Elo anchored at `R882 = 1500.0`).

## 4. Primary/Secondary Endpoints, Required Audits, and Decision Rule

- **PRIMARY Endpoint**: `ACTIVE_2PLUS2_HURDLE` minus `GATED_025` median family gain on `DL-Val` (equal 4-reveal budget).
- **SECONDARY Endpoint**: `ACTIVE_2PLUS2_HURDLE` minus `R882` median family gain on `DL-Val`.
- **Multiplicity Correction**: Holm-Bonferroni `m = 2` across the Primary (`ACTIVE_2PLUS2_HURDLE - GATED_025`) and Secondary (`ACTIVE_2PLUS2_HURDLE - R882`) endpoints on `DL-Val` (reporting both Holm-adjusted `p`-values and Holm-adjusted `97.5%` family bootstrap CIs `[1.25%, 98.75%]` alongside `95%` CIs `[2.5%, 97.5%]`). Also report `DL-Train` OOF and `400`-iteration MM Bradley-Terry Elo vs `R882 = 1500.0`.
- **Four Required Audits in [metrics.json](metrics.json)**:
  1. **Audit (a) — Zero `DL-Val` / Holdout Leakage**: Assert `0` `DL-Val` families or contexts in any Hurdle prior or SP promotion fit, `0` family overlap across the 5 `DL-Train` folds, and `0` `has_meta_test` contexts.
  2. **Audit (b) — Equal 4-Reveal Budget**: Assert per-context reveal count `== 4` (`4` unique candidate indices per context) for both `ACTIVE_2PLUS2_HURDLE` (`|rev4| == 4`, `|idx_a| == 2`, `|chosen_b| == 2`, disjoint) and `GATED_025` (`|idx4| == 4`) across all contexts and seeds, and verify identical `top_zgap < 0.25` gate activation counts.
  3. **Audit (c) — Negative Control (Permuted Revealed Fitness)**: Evaluate `ACTIVE_2PLUS2_HURDLE` with revealed `fitness` permuted within each context (seeded deterministically per `(seed, cid)` so revealed `fitness` is uncorrelated with candidate identity while final choices are scored on true `utility`). The `95% CI` on `DL-Val` must cover `0`.
  4. **Audit (d) — Descriptive `±50%` Constant Sensitivity**: One-at-a-time `-50%` (`0.5x`) and `+50%` (`1.5x`) perturbation of each constant in `ACTIVE_2PLUS2_HURDLE` (`18` constants, `36` perturbation runs), reported descriptively in `metrics.json` (never used for arm or hyperparameter selection).
- **Preregistered Decision Rule**:
  - **`ADVANCE`** if and only if:
    1. Primary (`ACTIVE_2PLUS2_HURDLE - GATED_025` on `DL-Val`) median family gain `> 0`, AND
    2. Holm-adjusted CI lower bound `> 0` (and `Holm p < 0.05` for `m = 2`), AND
    3. `>= 2/3` positive seeds on `DL-Val`, AND
    4. Negative control (`Audit (c)`) is null (`95% CI` covers `0`).
  - Ties = **`NO_ADVANCE`**. Lift is reported as Elo/utility; no speed verdicts.

## 5. Compute Envelope and Main-Hold Custody Protocol

- **Host & Affinity**: `xbabe2` CPU only, `taskset -c 57-60`, `nice -n 19`, `threads <= 4` (`1` math thread), `RSS <= 4 GiB`, free disk `>= 300 GB`, H239 A1 memory guard (`admit >= 30 GiB`, cooperative pause whenever `MemAvailable < 30 GiB` because swap is full, resume at `>= 30 GiB` sustained `30 s`), compute stop `2026-11-03T15:30:00Z`.
- **Main Hold (`CI_FREEZE`) Custody**: Preregistration PR is opened **UNMERGED** on `neverhuman/JopeDime` and backed by S3 exact-`VersionId` custody (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H254/prereg-<sha>/`) before any outcome is evaluated.
