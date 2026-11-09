# H279 Preregistration: Bagged Strict-Margin Re-Ranker Switching (`h279_rerank8_bagged_strict_switch`), `xbabe2` CPU

- **Owner:** `rd_argon` (`[ARGON]`)
- **Authority:** `/home/ubuntu/nudges/hw_t2140_argon_h279.md` (Grok Bot H-wave pass for Jepson, `2026-11-03T21:40Z`, decided)
- **Claim:** `/home/ubuntu/jope_plan/ns/receipts/H279_claim.txt`
- **Total neural optimizer updates:** `0` (`round-882.pt` `sha256 = 0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64` and `DOPE` composite `sha256 = c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` verified before and after)
- **Single-pick contract:** `0` test-time fitness/gate reveals per context; sealed meta-test, holdout, and `DL-Test` never opened (enforced by `sys.addaudithook`)

## 1. Hypothesis

In `H276` (`c70eeb2af`), `RERANK8` (`+0.0577`) and `RERANK8_VETO` (`+0.0686`) passed all leave-family-cluster-out generalization gates, but `61–63 / 205` families still lost relative to `r882` per seed (`40–43 / 205` with loss `> 0.01`), `WARD` cluster 1 was flat, and `fam-769243ee…` on `DL-Val91` lost across every wave.

**`H279` tests:** Does switching away from `r882`'s pick (`slot 0`) to `RERANK8_VETO`'s pick (`j_ref`) **only when a bootstrap-bagged ensemble of $B = 8$ family-resampled re-rankers agrees by the strict `H265` margin** cut harmed families (`loss > 0.01` vs `r882`) by at least `25%` (`<= 0.75 x REF`) while retaining at least `80%` (`>= 0.80 x REF`) of `RERANK8_VETO`'s out-of-cluster lift?

## 2. Frozen Parent Science & Regression Gate

- [topk.py](topk.py) (`sha256 = fcd2d100c30d97dfe70fab7b981332f0c8dbffbf8bf7b7badaaef09550983d43`), [h265_core.py](h265_core.py) (`sha256 = 5c17ddad3993b5cebb702d2c58882fa2e54dc15572649c663169416f6965de76`), [veto.py](veto.py) (`sha256 = bf9c06ddd97f793b43f8ddbd31488768be3567db3f485f2d9377248e9b878ec0`), and [fold_manifest.json](fold_manifest.json) (`sha256 = e89f9aa130cf1529a684c83c2dc8bd1658dbae69a5c1e0af87760ed8d92208f2`) are **byte-identical** to `H276`.
- Before primary compute, `--smoke` verifies:
  1. `Native91` (`91/91` bit-exact, mean `1.1377941898164525`) and `TOP4_PICK@r882`;
  2. Exact reproduction of `H272` cell `F272140-S272101-K0` (`RERANK8` receipt, `2` invalid, `131` changed, `84` deep picks);
  3. Exact identity of the 4 outcome-blind family clusterings against `fold_manifest.json`.

## 3. Arms (`Holm m = 2` over `BAG_SWITCH` and `BAG_SWITCH_NESTED`)

- **`r882` (`BASE`)**: Frozen `r882` top-1 pick (`slot 0`).
- **`RERANK8_VETO` (`REF`)**: Byte-identical `veto.run_cell_veto` (`tau = 0.10`, `K = 8`). Reference only.
- **`BAG_SWITCH` (`A`, gated)**:
  - Fits $B = 8$ bootstrap-bagged `LGBMRanker` models (`bagged.fit_ranker_n`, exact `topk.fit_ranker` hyperparameters at the cell's inner-selected `(lr*, n_estimators*)`) on family-level bootstrap resamples of the cell's `TRAIN` families (`fit_fams` sampled with replacement, size `len(fit_fams)`).
  - Given `RERANK8_VETO`'s candidate pick `j_ref` and strict `H265` margin `strict_margin`, bagged ranker $b \in \{0, \dots, 7\}$ casts a switch vote iff `j_ref != 0` and `bag_score[b, j_ref] - bag_score[b, 0] > strict_margin`.
  - Switches from `slot 0` to `j_ref` iff `votes >= q` with **$q = 6$ fixed**.
- **`BAG_SWITCH_NESTED` (`B`, gated)**:
  - Same outer $B = 8$ bagged voting rule, with $q \in \{5, 6, 7\}$ selected per outer cell by 4-fold inner family-grouped CV over the `TRAIN` clusters only (`bagged.select_q_inner`: Tier 1 requires both inner OOF `harmed_001 <= 0.75 * ref` and inner OOF `gain >= 0.80 * ref`, maximizing `gain - 0.001 * harmed_001`; Tier 2 requires `harmed_001 <= 0.75 * ref` and `gain > +0.010`; Tier 3 fallback maximizes `gain - 0.0015 * harmed_001`; ties break toward larger $q$). Inner CV never touches outer `ASSESS` labels.
- **`PLACEBO` (`BAG_SWITCH_SHUF_PLACEBO`)**:
  - `BAG_SWITCH` ($q = 6$, $B = 8$) where both the candidate proposer and all $B = 8$ bagged agreement voters are trained on within-family shuffled relevance labels (`topk.shuffle_within_family` on `TRAIN` families only, masked by the cell's outer invalid gate). Must NOT pass (`median <= +0.005`).

## 4. Evaluation Panels, Seeds & Decision Gates

- **PRIMARY (`DL-Train` leave-family-cluster-out):**
  - `4` outcome-blind partitions (`KM276140`, `KM276141`, `KM276142`, `WARD`) $\times$ `5` held-out clusters $\times$ `3` model seeds (`279101, 279102, 279103`) = **`60` cells** across `1,219` contexts / `205` families.
  - Per seed, the primary statistic is the median over the 4 partitions of equal-family OOF gain.
  - Uncertainty via **10,000-draw two-level cluster bootstrap** at `boot_seed = 279150`.
  - All 60 cells' selections (`selection.json`) and `primary_panel.npz` are fsync-locked before any `DL-Val91` read.
- **SECONDARY (`DL-Val91`, supporting only):**
  - `91` contexts / `19` families scored by the `KM276140` cells' 5-model mixture per seed.
- **Strict `ADVANCE` Gates (per gated arm `BAG_SWITCH` or `BAG_SWITCH_NESTED`, `Holm m = 2`; ties count as NOT better):**
  1. `median_gt_0_010`: Median per-family gain vs `r882 > +0.010`;
  2. `three_seeds`: Per-seed gain `> 0` on `3/3` seeds;
  3. `holm_lcb_gt_0`: Holm-level (`m = 2`) two-level cluster bootstrap LCB `> 0` (and Holm $p < 0.025$);
  4. `harmed_le_075_ref`: Harmed families (`loss > 0.01` vs `r882`, median over seeds) `<= 0.75 * REF`'s harmed count;
  5. `gain_ge_080_ref`: Median per-family gain vs `r882 >= 0.80 * REF`'s median gain;
  6. `invalid_le_ref`: Total `-2.0` invalid picks on `DL-Train <= REF`;
  7. `clusters_4_of_5_every_partition`: `>= 4/5` clusters non-negative in `>= 2/3` seeds in **every** partition (`KM276140`, `KM276141`, `KM276142`, `WARD`);
  8. `placebo_le_0_005`: `PLACEBO` median gain `<= +0.005` and `PLACEBO` does not pass;
  9. `dlval_gain_gt_0`: `DL-Val91` median per-family gain vs `r882 > 0`.

## 5. Resource Envelope & Deadlines

- **Host:** `xbabe2` CPU only, `taskset -c 57-60`, `nice -n 19`, `threads <= 4`, `RSS <= 4 GiB`.
- **Supervisor (`monitor.py`):** Pause on `load5 > 108`, `MemAvailable < 20 GiB`, or swap-in rate `> 2,000` pages/s (resume `< 200` pages/s for `30 s`); hard stop on `MemAvailable < 12 GiB` sustained `30 s` or `< 6 GiB` on any sample.
- **Deadlines:** Preregistration on `origin/v1/next` by **`22:25Z` (`2026-11-03`)**; compute stop **`00:30Z` (`2026-11-04`)**; closeout by **`01:15Z` (`2026-11-04`)**.
