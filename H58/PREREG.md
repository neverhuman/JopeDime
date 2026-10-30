# H58 Preregistration — Flywheel Self-Play Defect-Isolated Multi-Arm Wave (`A` / `B` / `C` / `D`)

- **Wave ID**: `H58` (`t550`)
- **Date**: 2026-10-30
- **Owner**: `agy_fw_audit` (`agy_h58`)
- **Worktree / Branch**: `/home/ubuntu/JopeDime/.agent/worktrees/agy_h58` (`agy/h58`)
- **Base Audit**: `H56_audit/SUMMARY.md` (`H_AUDIT`, PR `#608`), Ranks 1–3, P0/P1
- **Execution Mode**: Offline isolated copy of the flywheel in `research/flywheel_selfplay/h58/` (never touching the live `xbabe1` loop, `xbabe0`, or `/home/ubuntu/admiralty*`)

---

## 1. Scientific Question & Scope

Does fixing the four structural defects identified in `H56_audit/SUMMARY.md` (Ranks 1–3, P0/P1) enable the `h40b` 10-token multimodal neural GP ranker (`adapter_10tok`) to learn positive family-grouped ranking utility and Elo gain from a **fold-clean initialization** (not `r882`) under 5-fold family-grouped inner cross-validation on `DL-Train` and a single terminal evaluation on `DL-Val`?

We test four cumulative, preregistered arms — each running an isolated copy of the flywheel loop from the **identical fold-clean initialization** per `(seed, fold)`:

| Arm | Code | Specification | Trainable Params |
| --- | --- | --- | ---: |
| **`A`** | `CONTROL` | Current flywheel harness (`h46/run.py` / `h48/run.py` / `automatic_promotion.py` / `run_fused_closed_loop.py` mechanics): `predict()` with `revealed_indices = [0, 1, 2, 3]` unmasked in loss and `argmin`; `order[:48]` (`np.argsort(pool["fitness"])[:48]`) placed deterministically in slots `0..47` during training so `order[0..3]` become the revealed anchors and leak via Nadaraya-Watson self-distance $d(i,i)=0$; zeroed `ctx_feat` (`zeros(B, 584)`), `base_prior_cost` (`zeros(B, M)`), `sos_prior` (`zeros(B, M, 2)`), and constant `scale = 0.4208`; dead `parent_delta_net` un-frozen (`29,211` params); HWM-only `promote()` (`score < incumbent - 1e-6`); and `smooth_l1_loss(pred, target, beta=1.0) + 0.1 * kl_div(τ=0.2)` across all 192 slots with failure target `2.0`. | `29,211` (`23,546` active + `5,665` dead `parent_delta_net`) |
| **`B`** | `FIX1` | **Fixes Bug 1 + Bug 3 + Bug 4** (`H56_audit` Rank 1 & Rank 4):<br>1. **Bug 1 (No-Peek Anchor / Slot Fix)**: Indices `0..3` of every slate are fixed to the context's 4 initial unsorted seed anchors `[0, 1, 2, 3]` in both training and evaluation. In training slates (`M = 192`), the 188 unrevealed candidates (`slots 4..191`, sampled from pool indices `>= 4`) are **randomly permuted** per step so the best candidate is never at a deterministic slot index. Revealed anchor positions `0..3` are **masked out of the training loss** (`pred[:, 4:]` vs `target[:, 4:]`) and **masked out of evaluation `argmin`** (`choice = 4 + argmin(pred[4:])`).<br>2. **Bug 3 (Real Context / Prior Features)**: Supplies real `ctx_feat` (`[B, 584]` with `dope629[:, :112]` at `88:200` and `lm48_z` at `536:584`), real `base_prior_cost` (`[B, M]` from frozen `_predict_calibrated_prior_batched_gpu(..., track_prefix="prior10")`), real `sos_prior` (`[B, M, 2]` from frozen `_compute_sos_prior_batched_gpu`), and per-context `scale = max(std(fitness[0:4]), 0.10)`.<br>3. **Bug 4 (Freeze Dead Subnetwork)**: Freezes `adapter_10tok.parent_delta_net.*` (`requires_grad = False`), leaving only the `23,546` active parameters trainable. | `23,546` (active `adapter_10tok` only) |
| **`C`** | `FIX1+2` | **`B` + Fixes Bug 2 (`H56_audit` Rank 2 — All Completed Evaluated GP Candidates as Hard Negatives)**:<br>In every round `r in 1..R`, the flywheel mutates training-fold `bank` parents via `propose_flywheel_candidates`, scores novel proposed ASTs with the current neural adapter, selects `1` top greedy + `1` exploration proposal (`2` per round) via `gp_selection_indices`, evaluates them with the native Rust two-fold CV evaluator `evaluate_gp_batch_rust`, and replaces HWM-only `promote()` with full completed-evaluation admission — admitting **all completed** evaluated GP candidates (`status == "completed"`) on training contexts (both round-0 historical `promotions.jsonl` records and per-round live Rust GP evaluations, including non-HWM evaluations `score >= incumbent`, model-proposed false positives, and genuine CV gate failures `status == "completed" and cv_gate_passed == False`, assigned training failure target `2.0`) into `pools[c]` and `pool["hard_negatives"]` (whereas Arms `A` and `B` admit only HWM winners `status == "completed" and cv_gate_passed == True and score < incumbent - 1e-6` and discard all non-HWM and gate-failing evaluations; execution errors `status != "completed"` are never admitted in any arm). During training slate construction (`188` unrevealed slots), up to `36` slots are reserved for evaluated hard negatives (prioritizing model false positives and gate failures), up to `48` slots for top passing unrevealed candidates, and the remaining slots for uniform random unrevealed candidates, all randomly permuted across slots `4..191`. | `23,546` (active `adapter_10tok` only) |
| **`D`** | `FIX1+2+3` | **`C` + Fixes Rank 3 (`H56_audit` Rank 3 — Shift-Invariant Listwise Top-Focused Loss: PL/ListNet + Gate Margin)**:<br>Replaces `smooth_l1 + 0.1 * kl(τ=0.2)` with a shift-invariant listwise top-focused objective computed strictly on the `188` unrevealed candidates (`slots 4..191`):<br>1. **Dual-Temperature Plackett-Luce / ListNet Cross-Entropy ($L_{\text{PL}}$)**: Within-context standardized passing regret targets $\tilde{r}_i = \text{clip}((c_i - \min_{\mathcal{P}} c) / \max(\text{std}_{\mathcal{P}}(c), 0.05), 0, 6)$ (with gate failures set to $\tilde{r}_j = +8.0$), optimizing $L_{\text{PL}} = 0.7 \cdot \text{KL}(\text{softmax}(-\tilde{r}/0.10) \parallel \text{softmax}(-\hat{c}/0.15)) + 0.3 \cdot \text{KL}(\text{softmax}(-\tilde{r}/0.35) \parallel \text{softmax}(-\hat{c}/0.40))$.<br>2. **Gate-Failure & Top-Choice Pairwise Hinge Margins ($L_{\text{gate}}, L_{\text{pair}}$)**: For passing set $\mathcal{P}$, failing set $\mathcal{F}$, best passing index $i^\star = \arg\min_{i \in \mathcal{P}} c_i$, and top-8 passing set $\mathcal{P}_8$: $L_{\text{gate}} = \text{mean}_{j \in \mathcal{F}} \max(0, 0.50 + \hat{c}_{i^\star} - \hat{c}_j) + 0.5 \cdot \text{mean}_{j \in \mathcal{F}} \max(0, 0.25 + \bar{\hat{c}}_{\mathcal{P}_8} - \hat{c}_j)$ and $L_{\text{pair}} = \text{mean}_{i \in \mathcal{P}_4, j \in \mathcal{P}_{\text{bot}}, c_i < c_j - 10^{-6}} \max(0, 0.15 + \hat{c}_i - \hat{c}_j)$.<br>3. **Total Objective**: $L_D = L_{\text{PL}} + 0.50 \, L_{\text{gate}} + 0.25 \, L_{\text{pair}}$ (exact shift invariance $\mathcal{L}(\hat{c} + \delta) = \mathcal{L}(\hat{c})$). | `23,546` (active `adapter_10tok` only) |

---

## 2. Pinned Data, Frozen Backbone, and Fold-Clean Initialization

1. **Offline Data Packet (`/home/ubuntu/jope-data/h50/input`)**:
   - `source_manifest.json` file SHA256: `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4` (`source_manifest_sha256` payload field: `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`).
   - **Inner CV Split (`DL-Train`)**: `1,219` contexts across `205` families (`context_split.json`).
   - **Terminal Validation Split (`DL-Val`)**: `91` contexts across `19` disjoint families (`context_split.json`).
   - **Holdout / Sealed Data**: `0` payloads opened (`excluded_payloads_opened == 0`, `outer_rows_delivered == 0`).
2. **Frozen DOPE Backbone & Pinned Prior / Evaluator Assets**:
   - Composite frozen DOPE SHA256 (`dope_freeze_guard.py`): `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.
   - `frozen_stage1_stage2_inference_params_v3.pt` SHA256: `136b2381c312fd4bf267249fe0543aa44f5b780a05d57f974d43dd4277588b6d`.
   - `dope_mlc_v3_multitask_best.pt` SHA256: `a36510b9e63050c01d182ccce7e4b489d9fb6ed7e39f8f2948cff420114e8952`.
   - `program-feature-names.json` SHA256: `c4e4373d1bc7e99bc38e44eb4acda22d0f9ecc9e332fa8cb9c6a0b463fe37962`.
   - `rust_evaluator.py` SHA256: `53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900`.
   - `model.py` SHA256: `0aa76360a7e5e691e420bd5a88a05b343b5b83edcc645e5f50fb33a3d5756f5a`.
   - `phase12_multimodal_model.py` SHA256: `7f5d30280de5cd47e8af5aa6787e5cb2bff321d984f9cae7d261eabd0a9c0d7a`.
   - `bootstrap.s1main` worker binary SHA256: `e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8`.
   - Verified before and after every cell; `requires_grad = False` on all non-`adapter_10tok` parameters.
3. **Fold-Clean Initialization (Zero `r882` / `DL-Train` Neural Contamination)**:
   - As proved in `H56_audit/SUMMARY.md` (§5), `round-882.pt` (`69ca5cb657c9`) was trained on 100% of `DL-Train` (all 5/5 inner folds), contaminating any inner-CV baseline or warm-start that loads `r882`.
   - Every `(arm, seed, fold)` run in `H58` starts from a **deterministic fold-clean random initialization** of both the frozen `gamenet_anchor` submodule (`anchor_seed = 20000 + seed`, scale `0.1`, `requires_grad = False`) and `adapter_10tok` (`init_seed = 10000 + seed`, scale `0.1`), with `prog_mean` and `prog_std` computed strictly from the 4 seed anchors of `fit_cids`.
   - The `INIT_G0` out-of-fold reference on `DL-Train` is likewise evaluated fold-by-fold using each inner fold's `fit_cids` normalization before scoring `held_cids`.
   - No `round-785.pt` or `round-882.pt` weights are loaded into any arm (`A`, `B`, `C`, or `D`) at initialization or via rollback.

---

## 3. Experimental Protocol: 3 Seeds × 5-Fold Inner CV + Single Terminal `DL-Val`

1. **Seeds**: `3` preregistered seeds: `SEEDS = (12, 13, 14)`.
2. **Inner CV on `DL-Train` Only**:
   - 5-fold family-grouped split (`K = 5`, `fold_seed = 4900`, matching `H52`/`H53`/`H53b`): every one of the `205` `DL-Train` families is assigned to exactly one validation fold `k in {0, 1, 2, 3, 4}`.
   - For each `(arm, seed, fold)`:
     - Train strictly on `fit_cids` (the 4 training folds of `DL-Train`, ~975 contexts / ~164 families).
     - **Cap runs by rounds, not wall time**: Fixed `R = 20` flywheel rounds per fold (`S = 2` optimizer steps per round = `40` gradient updates per fold, `batch_size = 256` family-balanced contexts per step, `slate_size = 192`, `AdamW(lr=3e-4, weight_decay=1e-4)` with cosine annealing to `3e-5`, gradient clip `1.0`, EMA shadow decay `0.90`).
     - **Live GP Proposal & Native Rust Two-Fold CV Budget**: In each round `r in 1..R`, the flywheel selects `gp_contexts_per_round = 1` family-balanced context from the training fold's fast two-fold JTT1 contexts (`table_bytes <= 65,536`, comprising `28` contexts across `7` families in `DL-Train`, each verified against its pinned `.jtt1.gz` SHA256 with no cross-context fallback), proposes `gp_proposals_per_context = 8` novel GP offspring ASTs via `propose_flywheel_candidates`, ranks them with the current neural adapter, selects `gp_eval_per_context = 2` proposals (`1` top greedy pick `order[0]` + `1` random exploration pick from `order[2:]` via `gp_selection_indices`), and evaluates them with `evaluate_gp_batch_rust(run_outer_test=False, max_workers=2)` (`40` native two-fold CV evaluations per 20-round cell). Only completed evaluations (`status == "completed"`) are eligible for pool admission (`HWM-only` in `A`/`B`; all completed evaluations including non-HWM and genuine CV gate failures `status == "completed" and not cv_gate_passed` in `C`/`D`). Execution failures (`status != "completed"`) are never admitted, and any cell with `evaluated_completed < R` fails closed.
     - Evaluate on the held-out inner fold `val_cids` (`DL-Train` fold `k`) **only at the terminal round `R`** (never selecting a best checkpoint on validation).
   - Pooling the terminal round-`R` predictions across the 5 disjoint inner folds yields an out-of-fold prediction for all `1,219` `DL-Train` contexts (`205` families) for each `(arm, seed)`.
3. **Single Terminal `DL-Val` Evaluation**:
   - For each `(arm, seed)`, train one full-`DL-Train` model from the same fold-clean initialization (`init_seed = 10000 + seed`) across all `1,219` `DL-Train` contexts for the exact same fixed round cap `R = 20` (`40` gradient steps).
   - Evaluate the resulting terminal round-`R` model **exactly once** on the `91` `DL-Val` contexts (`19` disjoint families).
   - **Never** evaluate intermediate rounds on `DL-Val` and **never** pick a peak round on `DL-Val`.

---

## 4. Primary Metrics, `t537` Gate, and Elo Reporting

We report **family-grouped Bradley-Terry Elo** (via `python/src/jope/north_star/elo_tracker.py`: `compute_family_grouped_bradley_terry_elo`) and **family-mean utility** (`u = clip(1.0 - reg_norm / 2.0, -1.0, 1.0)` on passing selections, `-1.0` on gate failures), **never AUC**:

1. **Primary Gate (`t537`) on 5-Fold Out-of-Fold `DL-Train` (`205` families, 3-seed mean per family)**:
   - Using 10,000 paired family-cluster bootstrap draws (`seed = 20261030`):
     - `mean_LCB95(arm - A) > +0.001`
     - `family_median(arm - A) >= 0.0` (strictly non-negative; negative median fails).
   - Pairwise step-up comparisons are also reported for `B vs A`, `C vs B`, `D vs C`, and `D vs A`.
2. **Bradley-Terry Elo (`elo_tracker.py`)**:
   - Computed across the 4 arms (`A`, `B`, `C`, `D`) + fold-clean init (`INIT_G0`) on the `205` out-of-fold `DL-Train` families and on the `19` terminal `DL-Val` families (with `A = CONTROL` anchored at `1500.0` Elo and 1,000 family-bootstrap draws for 95% Elo CIs).
3. **Secondary Diagnostic Metrics**:
   - Top-1 gate violation rate (`viol%`), normalized regret (`reg_norm`), Spearman rank correlation ($\rho$) on unrevealed candidates `4..P-1`, and Oracle top-5% hit rate (`hit5%`).

---

## 5. Execution & Resource Governance Rules

1. **CPU Prep & Unit Tests First**:
   - All CPU feature precomputation (`prepare_cache.py`) and unit tests (`test_h58.py`) run under `nice -n 10` with `CARGO_BUILD_JOBS=4` and host CPU utilization kept `<= 85%`.
2. **Red-Team Pass Before GPU**:
   - Before acquiring `/home/ubuntu/jope_plan/qualification.lock` for the GPU wave, obtain a red-team pass from `wp_agy_redteam` or a codex screen on `H58/PREREG.md`, `research/flywheel_selfplay/h58/`, and the unit test suite.
3. **GPU Execution Under Lock**:
   - Run the GPU training and evaluation wave strictly under `flock /home/ubuntu/jope_plan/qualification.lock` (`nice -n 10`).
4. **Deliverables & PR Cadence**:
   - PR 1: `H58/PREREG.md` (squash `--admin` merge within 30 minutes).
   - PR 2 / Final: `research/flywheel_selfplay/h58/` code, tests, audit receipts, and `H58/SUMMARY.md` (squash `--admin` merge within 30 minutes).
