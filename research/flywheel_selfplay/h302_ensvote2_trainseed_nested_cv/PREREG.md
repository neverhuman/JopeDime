# H302 (`h302_ensvote2_trainseed_nested_cv`) — Preregistration

- **Wave ID**: `H302`
- **Slug**: `h302_ensvote2_trainseed_nested_cv`
- **Owner**: `rd_argon` (`[ARGON]`)
- **Mode**: `xbabe2` CPU only (`nice -n 19 taskset -c 0-7`, `<= 8` idle cores, `RSS <= 8 GiB`, scratch `<= 5 GB` deleted at close, `0` GPU use, `0` neural updates, `0` fresh/sealed/holdout reads, DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` untouched)
- **Preregistered UTC**: `2026-11-04T05:02:00Z` (pushed to `origin/v1/next` **before** any outer-test score is computed; stop compute `07:15Z`, closeout `07:30Z Nov 4`)

## 1. Hypothesis & Context

`H301` (`ed8956e2c1`) established `SELECTION_NOISE_PLATEAU` across all paired-control trajectories from `r3175` (and `H296` SWA/Polyak closed `NOT_ADVANCED`, `b915eaf1`), showing that adapter-optimizer continuation waves on `r3175` are dominated by selection noise and multi-step drift, whereas the selection layer (`ENS_VOTE2`) is the productive path. `H294` confirmed `ENS_VOTE2` (`CONFIRMED_FRESH`), but its 3 "seeds" (`294150, 294151, 294152`) were bootstrap seeds on a single deterministic policy (`0.0` point-estimate spread across seeds). `H300` (true training-seed confirmation on fresh `v6A`) is `BLOCKED_NO_COHORT` (`H299`, `3b8253f68f`).

`H302` tests the exact `H300` training-seed confirmation question **without fresh data** via 4-fold nested family-grouped cross-validation inside `DL-Train` (`205` coarse families, `1,219` contexts): do three independently refitted `ENS_VOTE2` policies (root training seeds `30001, 30002, 30003`, `9` refitted `BAG_SWITCH` members per outer fold, `36` refitted `BAG_SWITCH` members / `360` LightGBM models across the 4 outer folds) consistently beat slot `0` out-of-fold across training seeds?

## 2. Step 1 Regression Verification (Completed Before Preregistration Push)

Recorded in [`regression_receipt.json`](regression_receipt.json):

- Verified `selected.json` (`sha256 = 981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`, `1,219` `DL-Train` contexts across `205` coarse families) and pinned `H287` export commit `213b30ad52e1ce48927bba960ed609bd98ab4dfe` (`ensemble_spec.json` `sha256 = 32ca19f30f427ae135f5f6c2aacc0fe28097d956f4e2ca0e7fd5f41ffec49995`, `fit_lock.json` `sha256 = 8a20c58bdb4bfcbadfa5ba927aa0577b630e3174e6dc4a7bfd579175ca08f35f`).
- Re-ran the published `H287` serialization/scorer regression on the `64` check `DL-Train` contexts (`context_cids_sha256 = 4005568e59fc5fddf0893e3f30ddbc5e1b7e8b599a4310d60e21ec26e52de7ba`): all `60` exported LightGBM models and all deterministic and per-seed arms (`ENS_VOTE2` `19` changed picks, `picks_sha256 = 8ef2837ae43e22721b7f7ffdf30371c29ba18e1ad3d9f0f788462d7c350d6d01`) matched **bit-for-bit (`True`)**.

## 3. Outer Split & Verbatim `H300` Member Fitting Protocol

Following [`research/flywheel_selfplay/h299_fresh_cohort_v6_scout/H300_PREREG.md`](../h299_fresh_cohort_v6_scout/H300_PREREG.md) ("TRAIN fitting" section) verbatim for member procedure, seeds, and pins:

1. **Pre-Declared 4-Fold Outer Family Split**:
   - Each of the `205` coarse `DL-Train` families $f$ is assigned to outer fold
     $$k(f) = \text{SHA256}(\texttt{"H302-outer-v1\textbackslash x00"} \mathbin{\Vert} \text{UTF8}(f))[-1] \bmod 4.$$
   - Exact partition sizes (recorded in [`regression_receipt.json`](regression_receipt.json) before any fit):
     - **Outer Fold `0`**: `57` outer-test families (`335` contexts) | `148` outer-train families (`884` contexts)
     - **Outer Fold `1`**: `45` outer-test families (`239` contexts) | `160` outer-train families (`980` contexts)
     - **Outer Fold `2`**: `50` outer-test families (`269` contexts) | `155` outer-train families (`950` contexts)
     - **Outer Fold `3`**: `53` outer-test families (`376` contexts) | `152` outer-train families (`843` contexts)
   - Outer-test rows never enter fitting, inner fold construction, hyperparameter/margin selection, or bag resampling.
2. **Root Training Seeds & Member Seeds (`9` Members per Outer Fold, `36` Members Total)**:
   - Member seed is `core.stream_seed(root_seed, "H300_ENS_VOTE2_MEMBER", member_index)` (`h265_core.py`):
     - **Root Seed `30001`**: Member `0` = `685781424`, Member `1` = `1430007429`, Member `2` = `1330439319`
     - **Root Seed `30002`**: Member `0` = `1869595753`, Member `1` = `786733481`, Member `2` = `1272701215`
     - **Root Seed `30003`**: Member `0` = `1510579523`, Member `1` = `89540748`, Member `2` = `1415857569`
3. **Per-Member Fitting Inside `outer_train` Only**:
   - Use the `H287` `DL-Train` top-8 packet/features (`568` feature columns, `r882-train-costs.npz` proposal features, unclipped utility/relevance targets).
   - For each outer fold $k \in \{0, 1, 2, 3\}$ and each member seed $m_{\text{seed}}$:
     - Derive 4 inner family-grouped folds over `outer_train` families ONLY via `ensemble.family_folds(outer_train_rows, outer_train_fam_by_cid, m_seed, n=4)`, with `cids_by_fold[0] = [cids_by_fold[1][0]]` as the fold-0 plumbing check (not an assessment fold; contributes no selection evidence).
     - Run `veto.run_cell_veto(m_seed, 0, cids_by_fold, outer_train_packets, outer_train_fam_by_cid, k=8, salt="")` to select `lr in (0.03, 0.05, 0.10)`, `n_estimators in (50, 100, 200)`, and `margin` from the unchanged `H287` grid with strict fit admission (`best_gain > STRICT_FIT_MIN_GAIN` and `LCB95 > 0`, else infinite-margin fallback `margin = inf`), and refit the full-`outer_train` proposer ranker and invalid gate (`tau = 0.10`).
     - Fit the `B = 8` family-bootstrap rankers (`bagged.fit_ranker_n`) on `outer_train` families using `core.stream_seed(m_seed, 0, "BAG_SWITCH", "outer_boot", b)` and `core.stream_seed(m_seed, 0, "BAG_SWITCH", "outer_bag", b)` (`q_fixed = 6` of `8`).
     - Serialize all `10` LightGBM models per member (`90` models per outer fold, `360` models total) to scratch disk, verify bit-exact reload on the first `64` `outer_train` contexts (`verify_reload`), and write/fsync `fold_{k}_fit_lock.json` **before** scoring `outer_test` families once from disk.
   - Build `ENS_VOTE2` for each root seed $s \in \{30001, 30002, 30003\}$: switch away from slot `0` to candidate $j \ge 1$ iff $\ge 2$ of its `3` member `BAG_SWITCH` policies switch to the same nonzero slot $j$; else keep slot `0`.

## 4. Estimand, Uncertainty, and Decision Rule (Ties Never Count)

- **Family-Equal Utility Estimand**:
  - For each root training seed $s \in \{30001, 30002, 30003\}$ and each of the `205` coarse `DL-Train` families $f$, let $d[s, f]$ be seed $s$'s out-of-fold mean utility difference vs slot `0` within family $f$ (`utility = -fitness` if `gate` passes, `-2.0` if invalid, matching the pinned `H294` evaluator).
  - Each family has equal weight: $\text{gain}[s] = \frac{1}{205}\sum_{f=1}^{205} d[s, f]$. Primary statistic is $\text{median}_{s} \text{gain}[s]$.
- **Shared Whole-Family Bootstrap (`boot_seed = 302150`)**:
  - `10,000` whole-family bootstrap draws over the `205` families using fixed RNG seed **`302150`** shared across all 3 root training seeds (`boot_seeds = (302150, 302150, 302150)`).
  - For each draw, sample the same `205` family indices with replacement for every seed, recompute each seed's mean gain, and take their median to form the pooled `95%` percentile CI (`2.5% / 97.5%`). Per-seed one-sided `95%` LCB is the `5%` quantile (`0.05`), reported alongside two-sided `95%` CIs (`2.5% / 97.5%`).
- **Primary Decision Rule (`INNERCV_CONFIRMED_TRAINING_SEEDS` vs `NOT_CONFIRMED_TRAINING_SEEDS`)**:
  - **`INNERCV_CONFIRMED_TRAINING_SEEDS`** iff all 4 `H300` conditions hold:
    1. Median gain across training seeds is strictly **`> +0.001`**,
    2. Per-seed one-sided `95%` family-bootstrap LCB (`5%` quantile) is strictly **`> 0.0` on `3/3` training seeds**,
    3. On every training seed, harmed-family share ($d[s, f] < -0.01$) is **`<= 35%`**,
    4. On every training seed, invalid-pick excess over slot `0` is **`<= max(2, ceil(0.01 * 205)) = 3`**, and the exact one-sided McNemar test on discordant invalid picks gives **`p >= 0.05`**.
  - Otherwise **`NOT_CONFIRMED_TRAINING_SEEDS`**. Family ties ($|d| \le 10^{-4}$) never count as wins.
- **Required Secondary & Red-Team Diagnostics**:
  1. Secondary family-median gain per seed and its median across seeds.
  2. Training-seed spread (`range`, `SD`, `IQR` of point gain, LCB95, changed picks, and Elo across `30001, 30002, 30003`) vs `H294`'s bootstrap-seed spread (`0.0` point-estimate spread, `0.000377` LCB95 range across `294150, 294151, 294152`).
  3. Whether the 3 fitted `ENS_VOTE2` policies make identical picks (exact pairwise and 3-way pick agreement across `1,219` contexts and Jaccard overlap of switched contexts).
  4. MM Bradley-Terry Elo anchored to `slot0 = 1500.0` (`bt_two`, `prior = 0.01`) with 205-family `10k` bootstrap `95%` CI (`boot_seed = 302150`).
  5. Descriptive `H294`-threshold check (`median > +0.010`, one-sided sign test $p < 0.05$ on `3/3` seeds, and both `NULL_RANDOM_VALID_SLOT` `302101..302103` and `NULL_RANDOM_SLOT_NO_LABELS` `302201..302203` failing).
