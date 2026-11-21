# Wave H458 Track G — Test-Informed `Hyper-CV` Multi-Gen GP Flywheel (`h458_track_g_hyper_cv_flywheel`)

> [!IMPORTANT]
> **In-Corpus `dev91` Disclosure & Quarantine:** All `dev91` flywheel metrics and `allow_d91_random_splits=1` runs in this directory are **in-corpus** (`canonical_promotion_forbidden = true`). Startup and runtime quarantine guards enforce `0` `CONFIRM`, `0` `FMGAP` (Parts B/C or any FMGAP cohort), `0` `HOLDOUT`, `0` `SEALED`, and `0` `DL-VAL` opens, plus `S8` exclusion of all `281` `mmap_v1` tables whose raw S3 `family_key` maps to the canonical `CONFIRM` hash bucket (`S8_BUCKET_AUDIT.json`). Inference and GP evolution strictly consume `(Xs, ys, Xa)` with `0` `ya` bytes read.

## Summary of Track G Components

1. **Held-Out Generalizability & Overfit-Gap Surrogate (`train_qhat_v4_generalizability_h458.py`):**
   - Fits frozen `DOPE` (`c51a56d0...`, `0` trainable `DOPE` parameters, SHA-256 verified before and after) projected via `DL-Train`-only (`1,219` contexts, `0` `dev91`) 32-dim SVD + 12-dim `LM12` into non-zero `44d` structural context vectors (`ctx44`).
   - Evaluates `S8`-clean real episodes in `gp_teacher_shards_v1` strictly on held-out query rows `(Xq_real, yq_real)` (`70%` interior IID + `30%` high-leverage extrapolation) against closed-form GCV ridge: while `100.0%` of `v1` shards beat GCV ridge on `CV-7`, `6.46%` (`792 / 12,251` shards) lose on held-out `Episode-Test`.
   - Filters via optimal convex GP+ridge blend $w^* \ge 0.25$, `blend_regret >= +0.020`, `overfit_gap <= 0.120`, and extrapolation guard to bootstrap **`9,097` zero-loser shards** in `gp_teacher_shards_v2/` (`0.0%` `Episode-Test` losers) and index `gp_family_seed_library_v4.json`, while emitting Track S Interface `(iv)` telemetry (`v1_backfill_cv_to_test_ledger.npz`).
   - Trains `ContextualSurrogateQHatV4` (`q_pred`, `stab_pred`, `overfit_gap_pred >= 0`, `val_spearman_r = 0.736817` vs `QHatV2` `0.719844` across `379` unseen held-out families) with `PretrainedQHatV4Prior(..., require_ok=True)`.

2. **Test-Verified Multi-Gen GP Teacher Harvester v2 (`hyper_cv_test_gp_harvester_h458.py`):**
   - Evolves exact `bootstrap.s1main.superset` GP pipelines causally on `(Xc, yc)` (`y_joint = [yc, 0]`, `0.0` `yq_real` leakage), screening with real `44d` `ctx44` + `QHatV4` overfit-gap penalty and admitting into `gp_teacher_shards_v2/` only candidates passing `2×CV-7` + held-out `Episode-Test` (`0.0%` ridge losers) while writing dense multi-candidate trajectories to `teaching_shards_v4/`.

3. **3-Tier `Train Hyper-CV` `Gen 1 -> 10..1000` Evolution & `M = 1` Debiased Stacker (`ultra_gp_cycle_h458.py`, `stack_hyper_cv_h458.py`):**
   - **Tier A (`Search Hyper-CV`):** 5-island multi-generation search anchored on Stage 1–3 `L+M` valley-soup `2×CV-7` OOF residuals (`1865.36` stacked / `1838.04` pure neural) on `rep=0,1` minus fold-spread SE and `QHatV4` gap penalty.
   - **Tier B (`Vaulted Horizon Ratchet`):** Unseen `rep=77` `7`-fold split on `(Xs, ys)` selecting at most `1` champion AST per horizon (`g10, g50, g100, g250, g500, g1000`) monotonically.
   - **Tier C (`Untouched Stacker Cross-Fit`):** Unseen `rep=101,102` `2×CV-7` cross-fitting **only** the `M = 1` locked AST per horizon (`0` `ya` bytes read), emitted as Track S Interface `(i)` `colblocks_g{G}/` + `sidecar.json` (`closure="fair-search-cv"`, `cross_fitted=true`).
   - **Shared `K = 5` `skill-xfit5` Feature-Byte Hash (`make_k5_feature_byte_strata`):** Canonical `blake2b(..., key=b"h458-k5-strata-v1")` over canonicalized IEEE-754 `<f8` feature rows (`-0.0 -> +0.0`, canonical quiet `NaN`), verified identically across 3 independent implementations (`f2893c38 = [1170, 1274, 1230, 1238, 1195]`, `0` `1e-9` near-twin escapes across all `19/19` `dev91` families).

4. **`X3-B` Evaluation-Aligned Max-Variety `104M` Trainer (`train_104m_h458_x3b_eval_variety.py`):**
   - Trains `ICL_L` (`104M`, warm-started from `r4_L_A` EMA) on `bank_v2` `r3` (`9df95437...`) with exact `S8 + B7 + S4` eligibility masking (`258` `B7` `DL-TRAIN` canonical-`CONFIRM` CIDs masked on `r3`, `530` on `v1`), context-only `|corr|` wide-table column selection (`0.0` query leakage), `100%` random feature-ID slots (`cl_h458` `K=5` unseen-stratum proof), `55%` `eval_clean` + `23%` `subspace_clean` + `22%` zero-loser `v2` Exact-GP distillation, GPU OOM batch-halving guard, and bounded `--keep-latest-only 1` 2-slot valley checkpoint ring (`ckpt_valley_slot{0,1}.pt`).

5. **Tracked Historical Stage Snapshots & Flock Coordination (`coord_shared_chat_h458.py`, `auto_launch_h458_track_g_after_stage3.py`):**
   - Re-exports the shared `h462_apex_prism.stack_guard_cv7_h452` implementation via `stack_guard_cv7_h452.py` and tracks clean Jankurai-verified snapshots of `train_104m_gp_flywheel_h452.py`, `train_104m_gp_flywheel_h454.py`, and `score_h452_dev91.py` alongside the 8 `H458` Track G modules so zero live flywheel modules remain untracked.

6. **Track S9 Arm A5 — Orthogonal Cross-Fitted Functional Gradient Boosted GP (`boosted_residual_gp_h459.py`, `run_h459_p201b_xbabe3_queue.py`, `build_h459_s9_budget_json.py`, `score_s9_boards_h459.py`):**
   - Multi-stage sequential symbolic residual boosting (`M in {5, 7, 10}` stages strictly on `medium / high / ultra` efforts; `fast` hard-refused) with rotating stratified OOF residual handoff (`rep = 200 + m`), 5-island multi-geometry evolution (L2 + fold-causal MAD Pseudo-Huber `arcsinh`/`sinh` + kernel/subspace specialists), stage-adaptive `DOPE` + `UnifiedQHatPrior` (`--prior dev91-free` `qhat_v2.pt` `bf05e040...` default, `F1` closure guard, `D1` `SETTINGS.json` lock, `H1` hybrid anchor guard, `H2` `--enforce-wall-cap 0` candidate parity, `H4` explicit `--closure`), non-negative Vault Coordinate Backfitting (`rep = 77`), and untouched Tier-C OOF export (`rep = 101, 102`).
   - Paired controls (`--budget-unit candidates`: `128 / 512 / 4,096` `effort.rs` unique candidates + full `calls_by_kind` ledger + `build_h459_s9_budget_json.py` `AMENDMENT_3` budget table): dual-seed `a0_s0`/`a0_s1` sharing fold partitions (`D5`), budget-dominating `a0_2x` (`2.0x`), `a6_comm` (`M`-program committee without residuals), and `a6b_runbag` (`0.5 * (a0_s0 + a0_s1)`), selected on `P201B` (`--cid-sha-top-n 80`, scored solely by `cl_mtc` via `s9_p201b_stack.py`) and confirmed frozen on `FITSEL` (`all-106` + `clean-88`) before any `S7` probe or `dev91` look.

## Verification
```bash
CUDA_VISIBLE_DEVICES="" python3 research/flywheel_selfplay/h458_track_g_hyper_cv_flywheel/selftest_h458.py
CUDA_VISIBLE_DEVICES="" python3 research/flywheel_selfplay/h458_track_g_hyper_cv_flywheel/boosted_residual_gp_h459.py --selftest
```
