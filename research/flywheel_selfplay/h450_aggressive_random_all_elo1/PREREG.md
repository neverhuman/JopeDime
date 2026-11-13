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

## 6. Amendment pub-r8 (2026-11-12 about 21:46Z): two-GPU sharded dev91 evaluator

On one 3090 that is also training, `cols_h450_gpu.py` takes about 2.4 minutes per context, or about 3.6 hours for all 91. `eval_gpu_h450.sh` now splits the contexts by index parity:
- **Shard 0:** the xbabe2 GPU.
- **Shard 1:** the xbabe3 GPU, which also verifies the stage, eval bank and checkpoint sha256.

Every epoch uses the same split. Both hosts run torch 2.7.0 and numpy 1.26.4.

The pub-r7 single-GPU baseline was stopped after 3 contexts, before any analysis. Its directory is kept as `g00_init.SUPERSEDED_single_gpu`.

**Cadence.** The evaluator scores the newest pair of epoch checkpoints as soon as the previous evaluation finishes. N is therefore about every 2 epochs, or about 150–200k datasets per 104M run. Each result is reported with each run's unique-dataset count.

## 7. Amendment pub-r9 (2026-11-12 about 22:05Z): Gemini's review findings, third GPU

**Review.** agy_apex_w10 reviewed the trainer at 1030eeda92. The receipt is `~/jope_plan/evidence/H450/review_train_aggressive_random_1030eeda92.md`, and its findings were posted in the coord file at 21:34Z.

| Finding | Status in pub-r9 |
|---|---|
| (1) `ds_per_s` counts micro-batches | Now logged as `micro_batches_per_s`, plus a true `datasets_per_s` |
| (3i) duplicates were only counted | Every dataset whose fingerprint is already in the run's seen set, or that repeats within a batch, is now dropped before the forward and counted in `dedup_skipped`. The model can no longer train on the exact same dataset twice. |
| (3ii) and (3iii) fingerprint canonical form | Every non-finite value hashes as +inf, and -0.0 hashes as +0.0. Fingerprints of datasets with no NaN, inf or -0.0 are unchanged, so the seen sets of resumed runs carry over. |
| (4) out-of-memory skip | Restarts the gradient-accumulation cycle, so a step never sums a partial cycle |
| (2) no injection on narrow tables (F < 10, floor(0.1·F) = 0) | By design: 10% of fewer than 10 columns is less than one column. Narrow tables still get columns removed (crops) and the other-column and teacher targets. |

**Launcher.** `run_r4_h450.sh train` now accepts xbabe1 (root `~/jope-data/h450-r4-x1`). Every unit carries `MemoryMax` (default 20G), so a trainer that grows is killed inside its own cgroup and no other service is touched.

**Third GPU.** xbabe1's 4090 became free about 21:55Z, when its non-H450 diffusers job ended. It runs r4_L_X1: L, 420k tokens, 48 h, lr 4e-5, seed 450431, from the same H449 init. r4_L_A and r4_L_B move to the pub-r9 trainer by `--resume`, with no loss of progress.

## 8. Amendment pub-r10 (2026-11-12 about 22:25Z): evaluation queue and exhaustion rule

`eval_queue_h450.sh` runs as the user unit `h450-eval-queue`. Each time the previous evaluation finishes, it scores the newest epoch that both r4_L_A and r4_L_B have written, as pairs (L_A, M_init) and (L_B, M_init).

For each scored epoch it appends one line to `~/jope_plan/evidence/H450/diag_results/r4_TRAJECTORY.jsonl` and posts that line to AGENT_CHAT and the coord file. Each line holds:
- the Elo arms;
- the gaps to Kumo L;
- each run's unique-dataset, duplicate and source-table counts from `epoch_NN.json`.

The §4 exhaustion rule is now mechanical. When neither the champion stack nor the ICL-only arm has beaten its best for 4 scored epochs in a row, the queue posts EXHAUSTED and stops. Stopping the trainers stays a decision cl_fwriter states in a post.

r4_L_A epoch 1 (train 1.00 h) had seen 113,533 datasets, all unique, with 0 duplicates. They came from 3,473 of 3,525 source tables, with 4.24% injected columns and a 1.97% dev91 share.

## 9. Amendment pub-r11 (2026-11-13 about 00:20Z): dynamic two-3090 evaluator

**Cross-host check.** Two 3090s give byte-identical columns for the same context.
- Context 016648dc… (index 0) of the epoch-0 baseline was computed on xbabe2. It was recomputed on xbabe3 with the same published `cols_h450_gpu.py`, the same checkpoints (sha verified) and torch 2.7.0 / numpy 1.26.4.
- `cmp` showed the two `.npz` files are identical.
- The fixed parity split of §6 is therefore not needed for comparability.

**Epoch 2.** The xbabe2 shard ran about 3 times slower than the xbabe3 shard (about 3 min against about 1 min per context) because xbabe2's CPUs are busy with other jobs.
- `helper_g02_steal.sh`, published here, had xbabe3 compute xbabe2's shard-0 contexts one at a time from the top index down.
- The local process skipped contexts already present. Every context was still computed exactly once by the published script, on a 3090.

**pub-r11 evaluator.** `eval_gpu_h450.sh` splits the 91 contexts into 91 one-context tasks (`--num-shards 91 --shard-idx I`), taken from a locked queue by whichever of the two 3090 workers (xbabe2, xbabe3) is free.

**Queue.** `eval_queue_h450.sh` starts its no-gain streak from the epoch-0 baseline and replays `r4_TRAJECTORY.jsonl`, so a restart keeps the streak.

**Epoch-0 baseline** (init L + init M, 0% dev91 in its training; original split):

| Arm | Elo | 95% interval |
|---|---|---|
| Champion stack | 1783.7 | [1663.5, 1950.7] |
| Legacy guard stack | 1852.8 | [1719.0, 2035.2] |
| ICL boost-only | 1652.0 | [1500.3, 1815.7] |

Receipt: `diag_results/r4_g00_init/`.

## 10. Amendment pub-r12 (2026-11-13 about 03:40Z): per-episode feature removal and injection

**Trigger.** ENG:watcher, in agy_apex_w10's session at 03:04:47Z (verbatim as relayed at 03:26Z): "confirm, with config flags per live worker (r4_L_A xbabe2, r4_L_B xbabe3, fleet_L_x1 xbabe1), that every batch uses a fresh random train/test split, random synthetic-feature injection, and random feature removal, and fix any worker that does not at its next checkpoint boundary". This matches the owner's "add synthetic data to and from the features each time randomly" and "maximum data entropy".

**pub-r11 behaviour, measured against the request:**
- Fresh random split on every episode: yes.
- Injection: k drawn from 0..floor(0.10·F), so some F ≥ 10 episodes got none.
- Feature removal: only on p ≥ 6 tables, in 40% of unbatched real episodes (lines 446–447). In the batched path, removal happened only when k > F − p.

**pub-r12 flags.** These are the fleet-r5 settings, so the two fleets match. Both flags default to 0, which reproduces pub-r11 exactly; `parity_r12_h450.py` checks this byte for byte.
- `--feature-drop-every-batch 1`: every real episode with p ≥ 2 removes at least 1 real column.
  - Unbatched episodes and each batch's anchor table remove 1..max(1, floor(0.35·p)) columns.
  - Batch mates remove whatever their width F_t requires (at least 1).
  - Tables wider than 64 also lose columns to the width cap, as before.
- `--feature-drop-max-frac 0.35`.
- `--synth-inj-every-batch 1`: every real episode with F ≥ 10 injects 1..floor(0.10·F) synthetic columns. The cap k ≤ floor(0.10·F) is unchanged, so F < 10 episodes still get none.
- Fully synthetic prior batches are unchanged: the open scope question in review finding 4 is still with the owner.

**Telemetry.** The epoch JSON and `log.jsonl` record the counters since the pub-r12 resume: `feature_drop_share_p2_since_r12` and `synth_inj_share_f10_since_r12`. Both should be exactly 1.0.

**Switch-over.** r4_L_A and r4_L_B take SIGTERM. Each trainer saves `ckpt_last.pt` on exit and resumes with the pub-r12 stage and the same argv plus the three flags. The seen-fingerprint set, ledger, optimiser, EMA and LR schedule carry over.
- Epochs written before the switch are pub-r11 protocol.
- The epoch JSON of every later epoch states `augmentation`.
- The dev91 trajectory marks the first mixed epoch, which contains both protocols.

## 11. Amendment pub-r13 (2026-11-13 about 12:40Z): keep scoring past the plateau

**Trigger.** ENG:watcher posted on AGENT_CHAT at 2026-11-13 05:32 MST, addressed to cl_fwriter. It relays Jepson's standing directive of 3:17 PM MST on Nov 12: keep training the 100M+ model on all S3 regression data across the fleet GPUs. Do not stop r4_L_A or r4_L_B, and do not hand the xbabe2/xbabe3 3090s back to cl_apex at queue exhaustion, unless Jepson says so in his chat. If the plateau rule trips, keep the trainers running and propose the next recipe change.

**Change.** `eval_queue_h450.sh` posts PLATEAU once when the no-gain streak reaches 4 scored epochs. It then keeps scoring every newest common epoch instead of exiting. The rule itself (champion stack or ICL-only must beat its best) and the dev91 protocol are unchanged.

**State at the change** (after epoch 13): the no-gain streak is 3/4. The best champion stack is 1791.7 (epoch 05) and the best ICL-only is 1672.9 (epoch 09).

**Not changed here.** ENG:watcher also relayed a checkpoint policy (Jepson, 5:20 AM MST Nov 13): keep only the latest checkpoint per run plus metrics and eval data, and switch to keep-latest saving at the next natural restart.
- The trainers have not restarted. The save code changes at that restart, in a later amendment.
- No existing checkpoint is removed until Jepson confirms in cl_fwriter's chat. This follows his 2026-10-13 no-delete ruling.
