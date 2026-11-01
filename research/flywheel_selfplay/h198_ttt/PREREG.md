# H198 — per-context test-time training (TTT) of the r882 adapter on revealed candidates only

Owner: `rd_argon` / `[ARGON]`. Authority: `/home/ubuntu/nudges/hwave_t1205_h198_rdargon.md`.

## Intervention and input closure

Evaluate per-context test-time training (`TTT`) starting from a fresh **copy** of the `r882`
adapter (`/home/ubuntu/jope-data/h198/inputs/round-882.pt`, SHA-256
`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`; `47` trainable adapter
tensors, `29,211` parameters) on **revealed candidates only** (`revealed_indices = [0, 1, 2, 3]`),
resetting the adapter to `r882` bit-exact (`rt.tensor_hash` verified) after every context.
The `DOPE` backbone (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) and
all `291` non-adapter tensors remain frozen in eval mode; the revealed-history structural kernel,
prior correction, and cross-attention stay active as live (`enable_prior_correction=True`,
`enable_struct_kernel=True`, `enable_cross_attn=True`, `enable_ucb=False`).

Use the pinned real-S3 corpus lineage (`1,219` `DL-Train` contexts across `205` families and
`91` `DL-Val` contexts across `19` families). No outer-test, holdout, or `has_meta_test` payloads
are opened; runtime `sys.addaudithook` enforces closed input paths. `r882` was historically
selected on `DL-Val` and exposed to `DL-Train`, so all evaluations are conditional continuation /
test-time adaptation readouts (`diagnostic_only=True`, `serving_promotion=False`).

## Arms and leakage red lines

1. **`R882` / `TTT0` (`0` steps)**:
   - Unmodified `r882` adapter (`0` SGD steps). Must reproduce native `DL-Val` `91/91` bit-exact
     (`mean = 1.1377941898164525`, per-context error `0.0`) before any `TRAIN` scoring, and
     single-context `adapter_forward` at `0` steps must match batched native `r882` candidate
     costs bit-for-bit.
2. **`TTT_K` (PRIMARY, Holm $m=1$)**:
   - For each scored context, run `k` plain SGD steps (`torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=0.0)`)
     on the live ranking loss (`rt.loss` from `h146_lookahead_exec/runtime.py`) restricted
     strictly to the `4` revealed candidate rows (`0..3`) of that single context via
     `revealed_live_loss`.
   - Grid: `k in {1, 3, 5} x lr in {1e-5, 3e-5, 1e-4}` (`9` grid points). Because plain SGD with
     `momentum=0, weight_decay=0` is a prefix trajectory in `k`, a `5`-step trajectory at a fixed
     `lr` produces exact snapshots at `k = 1, 3, 5`.
   - In each `(seed, outer)` cell, `(k*, lr*)` is selected **ONLY on that cell's `B` families**
     by maximizing equal-family mean utility on `B` (ties broken deterministically toward smaller
     `k`, then smaller `lr`).
3. **`TTT_SHUF` (placebo, descriptive)**:
   - Identical locked `(k*, lr*)` plain-SGD steps per cell as `TTT_K`, except the `4` revealed
     targets (`revealed_raw_f`, `revealed_gate`) entering the TTT loss are permuted within the
     context via a deterministic fixed-point-free permutation (derangement of `{0, 1, 2, 3}`,
     `DERANGEMENTS_4`) seeded by `stream_seed("TTT_SHUF", seed, cid)`. Forward scoring after TTT
     still conditions on the true revealed history (`raw`, `gate`).
4. **`TTT_L2` (anchored, descriptive)**:
   - Identical locked `(k*, lr*)` plain-SGD steps per cell as `TTT_K`, adding an L2-to-`r882`
     quadratic anchor `0.5 * beta * sum((p - p_r882)^2)` with `beta = 10.0` fixed a priori.

### Leakage red lines (enforced in code, unit tests, and runtime self-check)

- **Revealed-only loss**: Only revealed rows (`0..3`) of the scored context enter the TTT loss.
  `Engine.context_inputs` slices `fitness[:4]` and `gate[:4]` before constructing tensors and
  never includes full-pool `fitness` or `gate`; `ttt_context_trajectory` raises `ValueError` if
  `"fitness"` or `"gate"` keys are present in `context_inputs`.
- **Post-argmin label access**: Unrevealed candidate labels (`4..M-1`) are never read before
  `pick = int(np.argmin(costs))` is finalized; only then does `Engine.realized_utility(cid, pick)`
  look up `-pool["fitness"][pick] if pool["gate"][pick] else -2.0`.
- **Synthetic poisoned-unrevealed-label runtime proof**: Both `test_h198.py` and the live runner
  (`verify_leakage_mask_invariance` on the first `8` `TRAIN` contexts before Stage 1) verify that
  overwriting unrevealed labels `fitness[4:] = np.nan` and `gate[4:] = False` yields bit-identical
  post-TTT candidate cost vectors, argmin picks, and adapter L2 drifts across `TTT_K`, `TTT_SHUF`,
  and `TTT_L2`.
- **Zero cross-context state flow**: `ttt_context_trajectory` constructs a fresh `SGD` instance
  per context, restores all `47` adapter tensors from `initial` and clears gradients inside a
  `finally:` block, and verifies `rt.tensor_hash == initial_hash` before and after every context.

## Cross-fit stage order and independent fold seeds

Seeds **`198101, 198102, 198103`** use independent family-fold seeds **`198141, 198142, 198143`**
(pinned in `fold_manifest.json`; pairwise fold-assignment diagonal Jaccard means `0.0965`,
`0.1149`, `0.0993`). Each seed partitions the `205` `TRAIN` families into `5` cyclic folds of
`41` families (`A = fold j`, `B = fold (j + 1) % 5`, `FIT = remaining 3 folds`), giving exact-once
`B` and `A` coverage across `j = 0..4`.

1. **Stage 0 (Baseline & leakage proof)**: Verify bit-exact `91/91` `DL-Val` `r882` baseline
   (`1.1377941898164525`), `1,219` `TRAIN` native `r882` baseline, and `8`-context runtime
   poisoned-unrevealed-label invariance.
2. **Stage 1 (`B`-only grid selection & lock)**: Evaluate the `9` `(k, lr)` grid points on `B`
   contexts, select `(k*, lr*)` on each of the `15` `(seed, outer)` `B` cells using `B`
   equal-family mean utility only, and `fsync` `ttt_selection_lock.json` (binding `fold_manifest`
   SHA-256, `initial_adapter_sha256`, per-cell `B` grid utilities, and per-seed modal `B` choice
   for descriptive `DL-Val`) before any Stage-2 `A` or treatment `DL-Val` readout.
3. **Stage 2 (`A` readout & descriptive `DL-Val`)**: Verify `ttt_selection_lock.json` on disk
   before every cell, assemble pooled `A` (`205` families, `1,219` contexts per seed) for `R882`,
   `TTT_K`, `TTT_SHUF`, and `TTT_L2` (plus descriptive fixed-grid panels `K1_LR1e-05 .. K5_LR1e-04`),
   and evaluate descriptive `DL-Val` (`91` contexts / `19` families) at each seed's modal `B`-locked
   `(k*, lr*)`.

## Statistical inference and verdict rule

- **Bootstrap**: `boot_seed = 198150`, `10,000` paired family-bootstrap draws over the `205`
  `TRAIN` families (average contexts within family first, resample families with replacement using
  the same index matrix across seeds, and reduce the 3 seeds via `np.median(..., axis=0)` for
  median-seed 95% CIs and `np.mean(..., axis=0)` for mean-seed CIs).
- **Primary decision (`TTT_K` vs `R882`, Holm $m=1$)**:
  `ADVANCE` requires all of the following:
  1. `TTT_K` vs `R882` `LCB95 > 0` (`median_ci95[0] > 0.0` and `mean_ci95[0] > 0.0`) and strict
     margin `median_gain > 0.001 + 1e-12` (`mean_gain > 0.001 + 1e-12`);
  2. `Holm p < 0.05` ($m=1$, both $p_{\text{one-sided}} < 0.05$ and $p_{\text{vs } +0.001} < 0.05$);
  3. Family harm tail `<= 10%` (`max_seed_harm_fraction <= 0.10` and `mean_seed_harm_fraction <= 0.10`,
     where a family is harmed if $\Delta_f < -0.01$);
  4. Placebo separation `TTT_K > TTT_SHUF` (`median_gain(TTT_K vs R882) > median_gain(TTT_SHUF vs R882)`,
     `median_gain(TTT_K - TTT_SHUF) > 0.0`, and `mean_gain(TTT_K - TTT_SHUF) > 0.0`);
  5. Imported `H177` pooled selector admission (`m=1`) and `H178` calibration receipt gate
     (`CALIBRATED_PASS` & `POWER_OK`).
  Otherwise the verdict is `NO_ADVANCE`. Strict improvement only; ties reject.
- **Required reporting**: Equal-family utility and 95% CI, paired Bradley–Terry Elo vs `R882 = 1500`
  (`400` MM iterations, prior `0.01`, tie tolerance `1e-4`, `1,000` family-bootstrap draws at seed
  `198151`), family harm fraction ($\Delta_f < -0.01$), `-2.0` invalid pick counts per seed,
  `(k*, lr*)` selection frequencies across the `15` cells, mean per-context adapter L2 drift,
  and pairwise fold Jaccard across the 3 fold seeds. No AUC or tok/s metrics.

## Execution guardrails and S3 custody

One guarded CPU attempt on `xbabe2` at `nice -n 19`, math threads `1`, aggregate threads `<= 4`,
`RSS < 6 GB`, wall cap `10,800 s` (`3h`), host disk usage `<= 93.8%` (`free >= 40 GB`),
`MemAvailable >= 16 GiB`, owned local output root `< 300,000,000` bytes (`300 MB`), and
`metrics.json < 100 KB`. Per-candidate cost arrays and panel archives are streamed to versioned S3
(`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H198/rd_argon-t1205/`) and
verified by exact-version SHA-256 readback in `custody.json`.
