# H301 (`h301_paired_control_drift_meta`) — Preregistration

- **Wave ID**: `H301`
- **Slug**: `h301_paired_control_drift_meta`
- **Owner**: `rd_argon` (`[ARGON]`)
- **Mode**: **ANALYSIS ONLY** (`xbabe2`, `nice 19`, `taskset -c 0-1`, `RSS <= 2 GiB`, no new training, no new `dev91` scoring, no GPU, DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` untouched)
- **Preregistered UTC**: `2026-11-04T04:19:00Z` (pushed to `origin/v1/next` **before** executing the H301 meta-analysis; stop compute `05:35Z`, closeout `05:45Z Nov 4`)

## 1. Hypothesis

Every recent adapter-training wave from `r3175` lost to `r3175` (`1.1389572403748425`), **including its paired baseline-loss control** (`H297` `CONTROL` median `-0.008461`, `H298` `PAIRED_CONTROL` median `-0.036830`, `t1845` `control_v2` `-0.003518` after 2 updates). We test whether `r3175` is a selected maximum of a noisy `dev91` search (winner's curse over `~4,800` flywheel rounds) so that any further `TRAIN` update regresses toward `~1.13` (`SELECTION_NOISE_PLATEAU`), or whether multi-step unconstrained `TRAIN` updates actively drift away from a genuinely superior basin (`OPTIMIZER_PLATEAU`).

## 2. Step 1 Regression Verification (Completed Before Preregistration Push)

Recorded in [`regression_receipt.json`](regression_receipt.json):

1. **`H297` (`h297_robust_ctxnorm_clipped_loss`)**:
   - Fetched `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H297/astra-t0240/closeout-manifest.json` (`VersionId = 9rFcHFjvo72_WQ6kmiAAurahBKPPu.Yw`) and `results.tar.gz` (`VersionId = K7zNgkztjDJ32iTNlNc9clWVrzZDw2tF`, `sha256 = d3829dd9591d39428062f8c50ea99b276872c4126452b7c853fe83d088cf2216`).
   - Recomputed all arm summaries via committed `H297/analysis.py`: primary `CTXNORM_CLIP` `median_lift = -0.008461025318455606`, `median_lift_ci95 = [-0.019228082690923506, 0.0003517660130095988]`, `median_elo = 1488.5444469574622`, `median_elo_ci95 = [1467.9687997051508, 1504.185164900202]`; `CONTROL` `median_lift = -0.008461025318455606`, `median_lift_ci95 = [-0.019228082690923506, 0.0003517660130095988]`. **Bit-for-bit match (`True`)** against committed `H297/metrics.json`.
2. **`H298` (`h298_family_balanced_gradaccum`)**:
   - Fetched `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H298/s3_manifest.json` (`VersionId = 6qFHIFeiLNUQNWqxCbcVNkKYJiAeJyvx`) and `h298_checkpoints_and_results.tar.gz` (`VersionId = lG1UYi3tUz3vNfkFllycDAWzTvOLIbOE`, `sha256 = 95494ce2efc99eead40e5f40d503aa750f8783e1b6f5cb472bfcdb8d5f72865a`).
   - Recomputed all arm summaries via committed `H298/analysis_h298.py`: primary `FAMBAL_ACC4` `median_lift_vs_r3175 = -0.033528458894059865`, `median_seed_family_bootstrap_ci95_vs_r3175 = [-0.09813891237891575, 0.0035028817498420945]`; `PAIRED_CONTROL` `median_lift_vs_r3175 = -0.03683030173874502`, `median_seed_family_bootstrap_ci95_vs_r3175 = [-0.10260735527823622, 0.0029918138749592755]`. **Bit-for-bit match (`True`)** against committed `H298/metrics.json`.

## 3. Pre-Declared Trajectory Inventory (`dev91` Identity `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`)

Every already-scored paired-control / baseline-loss (`SmoothL1 + 0.1 KL`) trajectory initialized from `r3175` (`round-3175.pt`, `1.1389572403748425`):

1. **`t1820` (`fw-plateau-t1820-20261102`)**: `639` consecutive 2-update (`n_updates = 2`) baseline-loss proposals from `r3175` (`rounds 3176..3814`), cross-linked to the 91-context utility matrix in `H289` `prepared-closure.tar.gz` (`VersionId = dj7fcZrNhjkNwY2s5DkfOah1nfAUObnL`, `rounds 3175..4675`).
2. **`t1845` (`fw-plateau-t1845-20261102`)**: `3` paired-control seeds (`184501`, `184502`, `184503`) on `control_v2` (`n_updates = 2`), plus the exact `184501` 2-update control replays in `H297` and `H298`.
3. **`H297` (`h297_robust_ctxnorm_clipped_loss`)**: `3` paired-control seeds (`29701`, `29702`, `29703`) on `CONTROL` (`n_updates = 8`, fixed `lr = 3.6880929895544625e-05`, `sigma = 2.7751362402065425e-05`).
4. **`H298` (`h298_family_balanced_gradaccum`)**: `3` paired-control seeds (`29801`, `29802`, `29803`) on `PAIRED_CONTROL` (`n_updates = 8`, v2 warm-restart schedule).
5. **`t2115` (`fw-plateau-t2115-20261102`)**: `9` baseline-loss (`SmoothL1 + 0.1 KL`) trajectories from `r3175` across `n_updates in (16, 64, 256)` and `lr in (2.5e-5, 1e-4, 3e-4)`.
6. **`H296` (`h296_swa_polyak_adapter_averaging`)**: Included automatically if closed on `origin/v1/next` before `05:20Z` (otherwise recorded as unclosed at cutoff).

For every `(wave, seed/label, n_updates)` trajectory we report `dev91` mean, lift vs `r3175`, 19-family `10k` bootstrap `95%` CI, `W/T/L` contexts, changed contexts, and MM Bradley-Terry Elo (`r3175 = 1500`) with 19-family `10k` bootstrap `95%` CI.

## 4. Pre-Declared Estimators & Decision Rule (Ties Never Count)

- **Bootstrap Specification**: `10,000` cluster bootstrap draws (`bootstrap_seed = 30100`) resampling the `19` `dev` families with replacement (`counts @ family_sums / (counts @ family_sizes)`).
- **(a) Drift Slope & Median-Over-Waves Control Lift**:
  - Compute the median-over-waves control lift across the wave-level paired controls (`t1820`, `t1845`, `H297`, `H298`, plus `t2115` `16/64/256`-step horizons; and sensitivity restricted to the 2–8 update waves `t1820, t1845, H297, H298`) and its 19-family `10k` bootstrap `95%` CI `[LCB95, UCB95]`.
  - Compute the ordinary least-squares slope $\hat{\beta}_1$ of lift vs `n_updates` (and vs $\log_2(\text{n\_updates})$) across horizons `n_updates in {2, 8, 16, 64, 256}` (and across `n_updates in {2, 8}`), with 19-family `10k` bootstrap `95%` CI.
- **(b) Winner's-Curse Analysis on Flywheel `dev91` History**:
  - From the read-only copy of `adapter_admissions.jsonl` (all rounds evaluated under fixed comparison identity `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`, and specifically all post-`r3175` 2-update candidates from `r3175`) and `H289` `history.npz`:
    - Compute the empirical round-to-round `dev91` candidate utility distribution: mean $\mu_{\text{round}}$, SD $\sigma_{\text{round}}$, median, IQR, `95th` percentile $q_{0.95}$, `99th` percentile $q_{0.99}$, and maximum rejected candidate (`1.1385566140621184`).
    - Compute the expected maximum under Gaussian order statistics $\mu_{\text{round}} + \sigma_{\text{round}}\Phi^{-1}(\frac{N}{N+1})$ and empirical bootstrap order statistics vs observed `r3175` (`1.1389572403748425`).
    - Compute the 19-family `10k` cluster bootstrap standard error $\text{SE}_{\text{fam}}(r3175)$ of `r3175`'s `dev91` mean utility (and $\text{SE}_{\text{fam}}(\Delta_{q95})$ of `r3175` minus the `95th`-percentile round candidate).
- **(c) Verdict Rule**:
  - Verdict is **`SELECTION_NOISE_PLATEAU`** iff:
    1. Median-over-waves control lift `< 0.0` with its 19-family `10k` bootstrap `95%` CI strictly excluding `0` (`UCB95 < 0.0`), **AND**
    2. `r3175` (`1.1389572403748425`) exceeds the `95th` percentile ($q_{0.95}$) of the round-to-round `dev91` val distribution by **less than `2 x` its 19-family bootstrap SE**:
       $$(u_{r3175} - q_{0.95}) < 2 \cdot \text{SE}_{\text{fam}}.$$
  - Otherwise verdict is **`OPTIMIZER_PLATEAU`** (if control lift `< 0` with `CI95` excluding `0` but $(u_{r3175} - q_{0.95}) \ge 2 \cdot \text{SE}_{\text{fam}}$) or **`INCONCLUSIVE`**. Ties never count.
