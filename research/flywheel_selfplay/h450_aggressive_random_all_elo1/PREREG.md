# H450 Pre-Registration — Quarantined Random-Split In-Corpus Fit Diagnostic: 100% Online Random-Split Fine-Tuning on Combined (`train ∪ test`) Data + Relational Lead-Lag & 2-Stage Residual Boosting

> **QUARANTINE, INTERPRETATION & CUSTODY CONTRACT (`IN-CORPUS FIT DIAGNOSTIC ONLY — canonical_promotion_forbidden = true`)**
> Wave `H450` executes the owner-approved (`2026-11-12T17:20:51Z`, session `e066f19a`: *"I want to fine tune the MOST aggressive random train/test split on ALL of our data across EVERYTHING, but make the fine tuninig completely random, random seeds, that way they don't match the exact dev91 split, make sure you combine train/test before you do the random"*) quarantined study across all `3,525` pre-combined (`train ∪ test`), CSPRNG row-shuffled non-sealed tables:
> - **Explicit Scientific Interpretation:** Because `H450` includes the `91` `dev91` tables (plus `42` `dev91`-associated S3 cousins) inside its combined `train ∪ test` random-split training corpus, any `H450` score evaluated on `dev91` **measures in-corpus fit to `dev91` under random splits, NOT out-of-sample generalisation**, and is **kept strictly outside the canonical `dev91-audit-v1` ranking and separate from fair out-of-sample foundation-model comparisons**.
> - **Sole Canonical Fair Entry Preserved:** `h448.APEX_FAIR_CHAMPION` (`1821.5` Elo, Rank `#7/33`, `0%` `dev91` training exposure, S3-archived and published at `f11a2b57800ae637cdce4573b56a026fae29a034`) remains the sole official fair `dev91-audit-v1` entry.
> - **Strict Quarantine:** Zero writes to `research/elo/reference-board/boards/dev91-audit-v1/scores/`, `/home/ubuntu/jope-data/h335-stagev/refboard/`, `/home/ubuntu/jope-data/h335-cl/neural`, or S3 canonical prefixes. All `H450` artifacts reside strictly inside `/home/ubuntu/jope-data/h450-quarantine` (`chmod 0700`) and `/home/ubuntu/jope_plan/evidence/H450/`.
> - **Zero Sealed Holdout Opens:** `APEX-CONFIRM` (`142` families / `241` tables), `DL-Holdout` (`20` families), `1,780` outer folds, and `sealed` remain at **`0` opens** under `H317` guard v2.

## 1. Combined-First Corpus & 100% Online Random-Seed Splitting
1. **Merge & Pre-Shuffle Before Any Split:** `build_combined_f64_bank_h450.py` and `d91_h450.py` merge `train ∪ test` (`[X_search; X_audit], [y_search; y_audit]`) in exact **`float64`** across all `3,525` non-sealed tables (`661` `APEX` non-`CONFIRM` + `1,219` `DL-Train` + `1,512` S3 `T1` + `42` `dev91` S3 cousins + `91` complete `dev91` tables) and apply an irreversible CSPRNG full-table row permutation (`os.urandom(32)`) before serialization (`0/91` split identity matches).
2. **Step-Unique 3-Regime Random Splits (`0` Static Folds, `pub-r3` Owner + Review Alignment `2026-11-12T20:12Z`):** `train_aggressive_random_h450.py` uses a live multi-process ring-buffer drawing a brand-new `blake2b(run_nonce, worker_id, ep_seq)` seed on every single episode across all `3,525` combined tables at their **natural family-balanced corpus share (`--d91-mix 0.0`, no artificial `dev91` oversampling)**, shuffling combined rows anew and sampling random context ratios $p_{\text{ctx}} \in [0.20, 0.95]$ at untruncated row counts up to `N = 2,048` and `WIDE_GPU = 64` features across 4 multi-seed checkpoints (`2× 104M L` + `2× 24M M` on `xbabe2` + `xbabe3` `RTX 3090` GPUs; `xbabe1` untouched), randomizing every episode across **three regimes**:
   - **Regime 1 (`~32%` of real episodes):** Pure original features (`0` added columns; `85%` real target + `15%` continuous self-supervised feature target).
   - **Regime 2 (`~68%` of real episodes):** Split evenly between **Reviewed Signal Column Addition** (context-screened Lead-Lag/date-diff/log/sentinel/key/formula/OOF-base columns) and **Synthetic Column & Teacher Injection** (`2..18` distractor/collinear/bilinear/warp/step/Fourier columns + `28%` synthetic `GP+Tree+Warp` teacher targets on real `X` geometry).
   - **Regime 3 (`--synth-prob 0.45` of GPU micro-batches):** Full synthetic datasets (`icl_prior_v3s.Prior` + `PROBS_V3S` 12-family SCMs + synthetic `float64` Panel/Temporal Lead-Lag tables + synthetic Sentinel/Hurdle point-mass tables).
   *(Note: Any `0%`-neural-`dev91` upgraded control evaluated here is strictly labeled **post-hoc on `dev91`**, never `fair`, because the 7 upgrades were designed after `H448`/`H449` diagnostics were visible; `APEX-DEV` is also marked spent for `H449`/`H450` checkpoints while `H445`'s pre-frozen `DEV` read stays valid.)*

## 2. Seven Algorithmic Upgrades & Legal Test-Time Evaluation (`0` `y_audit` Read at Inference)
At evaluation time on `dev91` (`91` contexts / `19` families), all features, neighbors, warps, models, and stack weights are fitted strictly on `(X_search, y_search)` (`2×5 = 10` multi-repeat randomized 5-fold GPU cross-validation OOF/bag + full-`X_search` refit onto `X_audit`):
1. `float64` context-only median/IQR normalization inside `icl_model_h450.py` prior to `float32`/`bfloat16` cell projection.
2. Context-only within-entity 1D Lead-Lag neighbor interpolation (`y_prev`, `y_next`, distance-weighted `y_interp`), sentinel flags (`-99.99`), colinear offset differences, and signed-log transforms (`relational_struct_h450.py`).
3. Clipped SVD-LOO GCV Ridge + Two-Part Hurdle Sentinel detector (`robust_resid_h450.py`).
4. Exact 1,024-bin Yeo-Johnson probability integration (`mean_warped`) + 2-Stage Self-Boosted & Nonlinear-Base Residual `ICL` (`cols_h450_gpu.py`).
5. Specialist Lead-Lag + Laplacian/Matérn-ARD Kernel + multi-depth Duan-CatBoost/ExtraTrees/LightGBM block (`cols_tab_h450.py`) — also evaluated standalone (`0` neural `d91` training) to isolate pure out-of-training-weights tabular/stacker improvements from `d91` random-split neural fine-tuning!
6. Pool-agnostic global-best OOF anchor + adaptive OOF-validated refit blend + direct 5-fold OOF log-RMSE (`u`-metric) combiner (`stack_h450.py`).
7. Quarantined diagnostic readout & Token/Compute Cost Ledger (`analyze_h450.py`, `token_compute_ledger_h450.py`), explicitly separating fair out-of-sample entrants (`0%` `dev91`) from quarantined in-corpus `dev91` fit diagnostics.

## 3. Amendment pub-r4 (2026-11-12, cl_fwriter takes over H450)
**Ownership.** Jepson, in the cl_fwriter session at about 20:26Z: "I need you to take over. train_aggressive_random_h450.py". cl_fwriter now owns the trainer, the stages, the GPU windows and the closeout. agy_apex_w10's pub-r3 run (launched before the transfer, `run_h450.sh gpu-train-and-cols 0.48`) is kept as a separate pub-r3 data point; its files are not changed.

**Owner direction for the pub-r4 run** (quoted with times in `ns/receipts/H450_R4_OWNER_cl_fwriter.txt`): train on all regression data with train and test merged; a different random seed for every sample; 0-10 % injected synthetic columns; start from our best model; maximum data entropy; dev91 included with different random splits; track the fixed-split dev91 Elo at every major epoch; the goal is to beat Kumo. APEX-CONFIRM and DL-Holdout stay sealed, as Jepson decided at about 19:00Z in the same session. The trainer refuses any sealed or unexpected cohort.

**Trainer changes** (`train_aggressive_random_h450.py`; the other pub-r3 files are unchanged):
1. **Real episodes:**
   - fresh blake2b seed each, table by smart stratified weights over all 3,525 tables;
   - dev91 at its natural share, with no `--d91-mix`;
   - target: own 85 %, another real column 10 %, synthetic teacher on the real features 5 %, plus 30 % Huberised jitter;
   - rows never duplicated across context and query; the pub-r3 batched path padded short tables by repeating rows;
   - random p_ctx and a 75 % target-stratified split;
   - random column crop and order, with injected synthetic columns `k <= floor(0.10 F)`;
   - target space: raw 40 %, Ridge residual 40 %, Yeo-Johnson 20 %.

   The pub-r3 regimes that added 2-18 synthetic or 16 reviewed columns are removed.
2. **Synthetic share:** fully synthetic batches are `--synth-prob 0.30` of steps (v3s prior, panel and sentinel tables), unchanged from pub-r3 apart from the share.
3. **Outputs:**
   - an EMA epoch checkpoint `ckpt_epoch_NN.pt` and an exposure ledger `epoch_NN.json` every training hour;
   - `ckpt_last.pt` every 10 min, with SIGTERM saving and `--resume`, so GPU windows can pause and resume;
   - no SWA soup in the trainer.

   Sha256 checks of the init checkpoint and the bank are required.
4. **Smoke (CPU, 300 items, 812 episodes):**
   - 281 episodes/s on 7 workers;
   - injected share 4.3 %, maximum 10 %, 0 violations;
   - dev91 2.7 % of real episodes;
   - targets 689 own / 78 other column / 45 teacher.

   `selftest_h450.py` passes all 6 tests.

**Runs:**
- **Init:** the H449 best checkpoints, `all_non_d91_L` (b854b518...) and `all_non_d91_M` (5aadefdf...).
- **Pairs:** seed A (L + M) on the xbabe2 3090 and seed B (L + M) on the xbabe3 3090.
- **Schedule:** 10 h cosine, lr 4e-5 for L and 6e-5 for M, grad-accum 3, tokens 140k.
- **cl_apex:** its runs are paused under rule 7 and resumed from ckpt_last.pt with a loaded=true sync check.

**Per-epoch dev91 tracking:**
- for each epoch, `cols_h450_gpu.py` fits on X_search only with the epoch's L + M checkpoints;
- `analyze_h450.py` then reads y_audit under the H450 lock and receipt;
- results are reported as **in-corpus fit**: outside the ranking, never a generalisation claim, and never compared with Kumo as generalisation;
- the init (0 %-dev91) columns computed by pub-r3 serve as epoch 0.

The comparison with Kumo L (2070.2) is reported as an in-corpus gap only.

**pub-r5 addition (unique-dataset tracking).** Jepson: "I want to track the # of unique datasets this model has seen... it should NEVER see the exact same datasets due to our seed and synthetic injection".
- **Fingerprint:** every training dataset gets a 64-bit blake2b fingerprint. It is computed after every transform, so it covers the exact X, y and n_ctx the model sees. The context rows and the query rows are each sorted before hashing, so a reordered copy of the same split still counts as a duplicate.
- **What is logged** (every 30 s, in every `epoch_NN.json` and in `manifest.json`):
  - `unique_datasets_seen`;
  - `datasets_seen`;
  - `duplicate_real_datasets`;
  - `duplicate_synthetic_datasets`;
  - `distinct_source_tables_seen` out of 3,525.
- **Resume:** the fingerprint set persists in `ckpt_last.pt`, so the counts carry across pauses.
- **Expectation:** 0 duplicates. Any duplicate is reported as a finding.
- **CPU smoke:** 1,475 episodes, 1,475 unique, 0 duplicates, 1,056 distinct source tables.

## 4. Amendment pub-r6 (2026-11-12 about 21:08Z): all compute, 104M to exhaustion

**Owner order.** Jepson, in the cl_fwriter session, verbatim:

> I want you to take full ownership and get this running ASAP across ALL of our datasets and all of our compute xbabe0, xbabe1, xbabe2, xbabe3 and xbabe4 ... I want you to do this to the 104M model and I want you to track the elo on dev91 (using our original splits) every N? datasets, please start with our VERY best model and work to fine tune it to exhaustion please. Please make sure our GPUs are 80% memory utilization and our CPUs don't average above 85% utilization.

Coordination with Gemini goes through `/home/ubuntu/jope_plan/H450_CLAUDE_GEMINI_COORD.md`.

**Trainer.** The trainer bytes are unchanged from pub-r5. These runs use them:

| run | host | size | tokens | hours | lr | seed |
|---|---|---|---|---|---|---|
| r4_L_A | xbabe2 | L | 320k (bmax 16) | 48 | 4e-5 | 450401 |
| r4_M_A | xbabe2 | M | 140k | 48 | 6e-5 | 450402 |
| r4_L_B | xbabe3 | L | 140k | 48 | 4e-5 | 450411 |
| r4_M_B | xbabe3 | M | 140k | 48 | 6e-5 | 450412 |
| r4_L_C | xbabe3 | L | 180k | 48 | 4e-5 | 450421 |

- The token budgets put GPU memory at about 80%.
- r4_L_B and r4_M_B were paused at 21:13Z, through SIGTERM and ckpt_last.pt, after 0.157 h. They resumed on the 48 h schedule.
- The start model is unchanged: the H449 `all_non_d91_L` 104M checkpoint (b854b518...), the best measured 104M.

**Cadence (N).** One EMA checkpoint per training hour. That is about 25,000 datasets per 104M run on a 3090. Every reported Elo carries the run's `unique_datasets_seen` at that checkpoint.

**Evaluation (`eval_x0_h450.sh`).**
- dev91 columns: `cols_h450_gpu.py --device cpu` on xbabe0. Its GPU is in error and its 128 CPUs are idle. The run is 26 shards × 4 threads, which keeps it under the 85% cap.
  - Every checkpoint, epoch 0 included, is scored on the same host and backend.
  - Epoch 0 is the init L + M pair, which is also the zero-dev91 control arm (`--zero-d91-gpu-cols`) for every later epoch.
- Then `analyze_h450.py` on xbabe2.
- Labels are as in §3: in-corpus fit, `canonical_promotion_forbidden`.

**Exhaustion rule.** A run stops when its pair's in-corpus dev91 Elo (the champion stack and the ICL-only arm) has not risen for 4 consecutive checkpoints, or when its 48 h schedule ends.

**Hosts not used for training** (owner informed):
- xbabe1: a 4090 fully held by a non-H450 diffusers job.
- xbabe0: GPU in error. Fixing it needs a driver reset or a reboot, and neither was done.
- xbabe4: a Mac Studio with about 24/32 cores held by llama-server.

**Sealed.** APEX-CONFIRM and DL-Holdout stay out of the bank. This is the owner's earlier answer, "Keep them sealed (Recommended)", and it stands until he types otherwise.

## 5. Amendment pub-r7 (2026-11-12 about 21:37Z): GPU evaluation, 104M only, xbabe3 RAM fix

- **xbabe3 RAM incident.**
  - What happened: the kernel OOM-killed r4_L_B (21:20:11Z) and r4_L_C (21:21:12Z), then two other services, which have since recovered. Three trainers on the 62 GB host were too many. The trainers use about 8.3 GB (L) and 3.6–4.6 GB (M) of host RAM each.
  - Fix: one trainer per host from now on. r4_L_C is dropped; it had no saved checkpoint.
  - Loss: r4_L_B resumed from its 21:13:51Z ckpt_last.pt, so about 6 minutes of its training were lost.
- **104M only.** The owner asked for the 104M model, so r4_M_A and r4_M_B are paused. Their ckpt_last.pt files are kept. Every dev91 pair now uses the H449 init 24M checkpoint, so any Elo change comes from the 104M fine-tune alone.
- **Runs:**

  | run | host | batch tokens | GPU memory |
  |---|---|---|---|
  | r4_L_A | xbabe2 | 420k | about 15 GB; the dev91 evaluator also runs on this GPU |
  | r4_L_B | xbabe3 | 560k | about 80% of the GPU |

- **Evaluation (`eval_gpu_h450.sh`).**
  - `cols_h450_gpu.py --device cuda` runs on xbabe2. On xbabe0's CPUs it finished 0 of 91 contexts in 17 minutes, since each context runs about 1,000 model forwards, so `eval_x0_h450.sh` is retired.
  - Pairs: A = (L_A at epoch N, M_init) and B = (L_B at epoch N, M_init). Epoch 0 is the init L in both pairs and also serves as the zero-dev91 control arm.
  - Everything else is as in §3 and §4.
- **Log field.** The trainer's `ds_per_s` field counts micro-batches per second, not datasets. Datasets per second is computed from the `datasets` difference between log lines; the 104M runs about 29/s on a 3090. The trainer bytes are unchanged, and the mislabel is recorded here as a finding.
