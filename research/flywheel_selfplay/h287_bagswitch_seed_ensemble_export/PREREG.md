# H287 Preregistration — `h287_bagswitch_seed_ensemble_export`

## 1. Authority & Lineage
- **Wave**: `H287` (`h287_bagswitch_seed_ensemble_export`)
- **Lane / Agent**: `rd_argon` (`[ARGON]`)
- **Authority**: `/home/ubuntu/nudges/hw_t0005_argon_h287.md`
- **Parents**:
  - `H284` (`h284_bagswitch_frozen_fresh`, commit `98d5b9f58487471e987ed482b4fef305b19178a6`, S3 export `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H284/export/export_manifest.json` `VersionId = AHFLVzBwd_fOfBAFGwHd.ABmkbgIbUu3`)
  - `H286` (`h286_fresh_cohort_v2_even200`, commit `9a79996f9d29225e1e4f0e0621e280f68b967621`)
  - `H279` (`h279_bagged_switch_nested_q`, commit `f3ba4095c3f0190bf438c5cc20c7f2eb648dd334`)

## 2. Hypothesis
A seed-ensemble of the three `H284` frozen `BAG_SWITCH` policies (`seeds 279101–279103`) is at least as good as the median single seed under 4-fold family-grouped inner CV on `DL-Train` (`205` families, `1,219` contexts), with lower or equal harm, and can be frozen as ONE deterministic policy for out-of-sample evaluation in `H288`.

## 3. Data & Isolation Contract (`fresh_pools_read = 0`)
- All fitting, inner-CV evaluation, and model/ensemble freezing use `DL-Train` (`205` families, `1,219` contexts) ONLY.
- `fresh_pools_read` must remain `0` for the entire wave (`sys.addaudithook` blocks any open of `/sealed`, `/holdout`, `meta-test`, `qualification.lock`, `fresh_pools`, `fresh_cohort`, `fresh_top8`, `fresh-top8`, `fresh-build`).
- Zero neural weight updates (`total_updates = 0`); `DOPE` composite `sha256` (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) verified before and after.
- `DL-Val91` (`91` contexts, `44` families) is evaluated after `fit_lock.json` is fsync-locked as a descriptive-only check and explicitly labeled `"reused": true`.

## 4. Regression Checks (Step 1)
1. Reproduce `Native91 = 1.1377941898164525` (`91/91` exact context matches) and `TOP4_PICK@r882` (`1.2403601682250796` context mean, `1.2519997168684853` family mean).
2. Reproduce `H279` cell `KM276140-S279101-C0` bit-exact (`receipts`, `assess_picks`, `outer_votes`, `shuf_votes`, `invalid`, `changed`).

## 5. Arms & 4-Fold Family-Grouped Inner-CV Design (Steps 2–3)
- Evaluated across the 3 `H284` 4-fold family-grouped partitions of `DL-Train` (`205` families, `1,219` contexts; fold seeds `279101–279103`, refit seeds `287101–287103`):
  1. `BAG_SWITCH` (single-seed reference, $q=6$, $B=8$).
  2. `ENS_MEAN`: switch from `slot 0` to $j^* = \arg\max_k \text{masked}(\bar{s})[k]$ (where $\bar{p}_{\text{inv}}[k] = \frac{1}{3}\sum_{m=1}^3 p_{\text{inv},m}[k] < \tau = 0.10$ for $k \ge 1$ and $\bar{s}[k] = \frac{1}{3}\sum_{m=1}^3 s_m[k]$) iff mean ranker margin $\frac{1}{3}\sum_{m=1}^3 (s_m[j^*] - s_m[0]) > \bar{\mu}$ AND total bag votes across the 3 seeds' $24$ bagged rankers $\ge 18$ (mean bag vote $\ge 6.0$); else `slot 0`. Also reported without the 24-bag vote gate as `ENS_MEAN_PROPOSER`.
  3. `ENS_VOTE2`: switch away from `slot 0` to candidate $j \ge 1$ iff $\ge 2$ of $3$ member `BAG_SWITCH` policies switch AND agree on the exact same candidate $j$; else `slot 0`.
  4. `ENS_VOTE3` (unanimous): switch away from `slot 0` to candidate $j \ge 1$ iff all $3$ of $3$ member `BAG_SWITCH` policies switch AND agree on the exact same candidate $j$; else `slot 0`.
  5. `CLIP` (pre-declared exploratory): `BAG_SWITCH` ($q=6$) refit on `DL-Train` with per-context training utility clipped at `-1.0` (`np.maximum(utility_k, -1.0)`) so `-2.0` invalid picks do not dominate the inner hyperparameter selection loss (`3` seeds), plus its majority-vote ensemble `CLIP_ENS_VOTE2`.
- Bootstrap: `10,000` family-grouped draws with seed `287150`; ties `|d| <= 1e-4` are not wins; harmed family threshold `loss > 0.01`; Elo anchored at `slot 0 = 1500`.

## 6. Verdict Rule for `H287` (Step 5)
- `EXPORTED_ADVANCE` iff at least one `ENS` arm (`ENS_MEAN`, `ENS_VOTE2`, `ENS_VOTE3`) achieves on `DL-Train` 4-fold family-grouped inner CV:
  1. Inner-CV median family-mean gain vs `slot 0` $\ge$ single-seed `BAG_SWITCH` median gain, AND
  2. `per-seed LCB95 > 0` on `3/3` inner-CV replicates, AND
  3. Harmed families (`loss > 0.01`) median $\le$ single-seed `BAG_SWITCH` harmed median.
- Otherwise `EXPORTED_NO_ADVANCE`. Fresh out-of-sample verdict belongs to `H288`.
