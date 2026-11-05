# H381: rapid tests of 300-row low-fidelity GP search and a grown screener (TRAIN), then a 5-dataset dev91 direction check

**Owner:** `cl_fwriter` (Claude Opus 5.5).

**Status:** part A (§1–6) was written before any H381 S60 GP run or E2 training run. The part-B amendment (§7–8) was appended after the part-A TRAIN results and before any H381 dev91 read or R60 run.

**Request.** Owner, 2026-11-05, after the H369 ladder:
- Use synthetic data variety, and shorten the randomly sampled rows to about 300 if that trains much faster.
- Make the network larger while keeping what already exists, through weight locking or fine-tuning.
- Spot-check relative Elo on 5 random datasets from the dev91 holdout to see whether RMSE moves in the right direction.

**Constraints:**
- About 4 hours in total.
- dev91 is read only for the final 1–2 arms. Every decision is made on TRAIN.
- "Don't rerun models you already have": every baseline is a **stored** result.

## 1. Baselines (stored; none is rerun)
| Use | Baseline | Location |
| --- | --- | --- |
| E1, E2-in-loop | **X0** = H364 Phase C `AGG.g50.CV3@12` (the H335 AGG recipe), seed 364101 | `h335-cl/runs/H364_C_xbabe0` (xbabe0; pulled copy on xbabe2) |
| E2 | **S0** = H317 `H0.1-A` (scale S, opt1, A4 views, A525 subset, 3,000 steps, 2 seeds × 5 family folds) | `h317-cl/neural/runs/H0.1-A` |
| E2, descriptive only | H317 `H0.2-L` | `h317-cl/neural/runs/H0.2-L` |
| dev91 | The board's stored `h335.H335_AGG` `per_seed` rows | `dev91-audit-v1` |

## 2. E1: GP search on a 300-row SEARCH subsample, finished on full rows
### Panel and checks
- **Panel S60:** the first 60 P201d contexts (one per family) ordered by `stream_seed(381101, "S60", cid)`.
  - `panels.json` lists them; S60 cids sha256 `4f1ceefc5920dc2c…`.
  - P201d is the H364 panel; X0 already covers all 190 of its contexts.
- **Pre-checks, done before this PREREG:**
  - **Envelope identity** (`gp_h381.py idcheck`, 3 cids): `lf_rows=0` reproduces the X0 memo envelope shas.
    `lf_rows=300` has 150 + 150 SEARCH rows, and the same AUDIT envelope sha, `ba`, `ya` and `y_search_mean`.
  - **Replay** (R6, the first 6 S60 cids, with the H364 memo copied in): 0 physical worker runs, and `AGG.g50.single`
    and `AGG.g50.CV3@12` equal X0 exactly (Δ = 0, identical stack specs).
  - **Smoke** (DEV2, P201d positions 60–61, never scored): on the same candidates, 300-row evals cost 2.1× and 5.8×
    less wall time than full-row evals.

### Method
- **Code:** `gp_h381.py`. Runtime patches only; the 18 H317 pins and the H335 files are unchanged and verified.
- **SEARCH subsample:**
  - In each native SEARCH fold, sort the rows by y (stable), cut them into 150 equal strata (`np.array_split`), and keep
    one seeded-random row per stratum (`stream_seed(381111, "lf", cid, fold, 150)`).
  - Rows keep their original order. The subsample is fixed per context for the whole search.
- **Unchanged from X0:** the AUDIT envelope (members fit on all 1,120 SEARCH rows, 480 AUDIT rows), `ba`, the
  screener, the views, Seed F and Evo A (K = 6 live / P = 48 screened per generation), seed 364101 and the worker.

### Outputs
| Output | Built from |
| --- | --- |
| `LF.g<gc>.{single, CV3@12}` | Low-fidelity SEARCH results (CV3 weights fit on 300 rows) |
| `FR.g<gc>.{single, CV3@12}` | The `fr_k = 24` best low-fidelity candidates (`rank_key`) plus the LF CV3 members, re-evaluated on the full 1,120-row SEARCH envelope; outputs rebuilt from those results |

### Runs on xbabe0 (docker sandbox, 2 × 48 cores, at the same time)
- **A:** `--lf-rows 300 --gens 100 --checkpoints 50,100 --fr-k 24`.
- **B:** `--lf-rows 600 --gens 50 --checkpoints 50 --fr-k 24`.

### CPU accounting
- **Search CPU of an output** is the summed single-thread worker wall-seconds of the unique candidates it used.
  - It covers the low-fidelity SEARCH evals up to `gc`, plus the FR evals for FR outputs.
  - The runner enforces a CPU-time limit (RLIMIT_CPU), so wall time is close to CPU time.
- **X0's search CPU** is the same sum over its stored memo SEARCH records.
- **Reporting:** the ratio is Σ over S60. AUDIT CPU and Python (screener) wall time are reported separately.

### Decisions (TRAIN S60, vs stored X0 `AGG.g50.CV3@12`)
Δu is AUDIT u. The family = context here (60 families).
- **E1-win:** `FR.g100.CV3@12` passes `h335_stats.primary_test` and the search-CPU ratio ≤ 1.0.
  - `primary_test`: clip ±0.5, exact sign-flip p < 0.025, LOFO min > 0, median ≥ 0.
- **E1-speed:** `FR.g50.CV3@12` is non-inferior and the search-CPU ratio ≤ 0.40.
  - Non-inferior: the family-bootstrap 95% lower bound (10,000 resamples, seed 381150) of the clipped mean Δu is
    > −0.005.
- **Descriptive only:** all other outputs (LF vs FR, LF600, single) and the `headline` (unclipped gain, % RMSE).

## 3. E2: a deeper screener grown from S0, with frozen or fine-tuned weights
### Model and growth
- **S3:** S with 3 layers instead of 1 (d = 32, 4 heads, cand_hidden 228, ffn ×4.0; LayerNorm/GELU like S).
  - It adds `ctx_blocks.{1,2}` and `cross_blocks.{1,2}`. It is registered at runtime as `SCALES["S3"]`.
- **Growth:**
  - Each S0 fold model (`H0.1-A/seed{317101,317102}/fold{j}/model.pt`) loads into S3 with `strict=False`; the
    missing keys may only be new-block keys.
  - In the new blocks, the weight **and** bias of `att.o` and `ffn.w2` are zeroed, so the grown model starts out
    computing exactly S0. Checked before training: max |Δ| = 0 on 4 contexts.
  - The stored normalizer is kept.
- **Fold-cleanliness:** the S0 model's `train_cids` must equal this run's training contexts (same A525 subset, same
  family fold).

### Arms
All new. Shared settings: opt1 (AdamW, cosine), 3,000 steps, A525 subset, 2 seeds × 5 folds, the same (seed, fold)
run seeds as S0, and `keep_models`.

| Arm | Change from S0 |
| --- | --- |
| `S0cont` | S0, 3,000 further steps, every parameter at 8e-5 (isolates "more training") |
| `S3sc` | S3 from scratch at the standard opt1 settings |
| `S3fz` | Grown S3; the S0 parameters **locked**; new layers at 8e-4 |
| `S3ft` | Grown S3; all parameters; new at 8e-4, S0 parameters at 8e-5 |
| `S3ftB16` | `S3ft` trained on the B16 view subset: the existing synthetic row-crop, column and distractor variety |

### Decision and hosts
- **Metric:** `rapid_hypothesis_lab` top-1 native utility per held-out context (seed mean), validated by
  `validate_arm_dirs`.
- **Rule:** `decide` with `PLANNED["H381-E2"]`: baseline S0, m = 5, Holm, contract gate (Holm lower bound > +0.001,
  median ≥ 0, ≤ 10% of families harmed, LOFO > 0).
- **Hosts:** folds 0–2 on xbabe1 (RTX 4090), folds 3–4 on xbabe3 (RTX 3090), per `FOLD_HOST`.
- **GP in the loop:**
  - **Trigger:** only if the best arm's family-mean gain over S0 is ≥ +0.01 (gate result reported).
  - **Run:** its 10 models go to xbabe0. `gp_h381.py --lf-rows 0 --gens 50` on S60, seed 364101; the screener is the
    only change from X0.
  - **Judged by** `primary_test` vs X0.
  - **If E1 passed,** the best screener + LF300 combination may also run.

## 4. E3 (synthetic tasks)
Not run unless at least 75 min remain after E2. If it runs, it is preregistered separately first.

## 5. dev91 direction check (part B)
- **Arms:** at most 2 final arms.
  - The E1 output with the largest S60 clipped mean Δu vs X0.
  - The E2-in-loop arm, if run and its S60 mean Δu > 0.
- **Contexts:** 5 dev91 contexts from 5 distinct families.
  - First 5 families by `stream_seed(381501, "d91fam", family)`; in each, the cid with the smallest
    `stream_seed(381501, "d91ctx", cid)`.
- **Compared only with stored board rows:**
  - per (context, seed) RMSE `ba·e^(−u)` vs `h335.H335_AGG`;
  - a 5-context Elo with all board entrants.
  - Direction only; nothing is added to the board ladder, and the 5 contexts are recorded as exposed.
- **Mechanics:** before any dev91 read, a part-B amendment fixes the arms, the LOCK (code and models in `code_sha256`,
  no dev91 path there), the receipt and the one-shot unlock (H369 pattern), and is published.

## 6. Other rules
- **Hosts:** xbabe2 runs no compute (other lanes hold its CPU pools); xbabe4 is unused (macOS, no torch).
- **DOPE:** c51a56d0 is unchanged and checked by the lab harness.
- **Closed data:** sealed, holdout and DL-Test are never touched.
- **Deadline:** compute ends by **2026-11-05T20:15Z**. If time runs short, drop in this order: E3, `S3ftB16`, run B.

## 7. Part-B amendment (dev91), fixed before any H381 dev91 read
### 7.0 Part-A results that fix the choices below (S60, vs stored X0; `results/e1_all.json` sha `6a104911…`)
| Output | Clipped mean Δu | Median | LOFO | Sign-flip p | Raw search-CPU ratio |
| --- | --- | --- | --- | --- | --- |
| `E2L:AGG.g50.CV3@12` | **+0.0028** | 0 | −0.0056 | 0.424 | 0.56 |
| `B:FR.g50.CV3@12` (best E1) | −0.0074 | −0.0024 | −0.0160 | 0.633 | 0.66 |
| `A:FR.g100.CV3@12` (E1-win output) | −0.0520 | −0.0107 | −0.0613 | 0.974 | 0.56 |
| `A:FR.g50.CV3@12` (E1-speed output) | −0.0583 | −0.0115 | −0.0678 | 0.987 | 0.28 |

- **E1-win** and **E1-speed** both fail; every E1 output is below X0.
- **E2 offline:** no arm passes the contract gate. The best gain is `S3fz` (+0.0103), so the GP-in-loop trigger fired and `E2L` ran.
- **Not run:**
  - the combination run (it needs a passing E1);
  - E3, because E2 shows no data-hunger signal: `S3ftB16` −0.0032 vs `S3ft`, and `S0cont` −0.0069.
- **CPU-ratio caveat (descriptive, added after part A):**
  - On full-row SEARCH programs evaluated both in a part-A run and in X0, the part-A run's wall time is 0.64× X0's
    for `E2L` (6,513 programs), 0.74× for B (77) and 0.94× for A (116).
  - The raw CPU ratios therefore flatter the new runs, because the host was less loaded than during X0.
  - `analyze_e1_h381.py` now reports this as `runs.*.host_speed`. The outputs and rules are unchanged (identical
    `outputs` sha).

### 7.1 Final arms, by the §5 rule
- **E1:** `E1B` (`--lf-rows 600 --gens 50 --fr-k 24`, H0.1-A screener), output `FR.g50.CV3@12`.
  - It has the largest S60 clipped mean Δu of the E1 outputs (−0.0074). It is below X0 on TRAIN, and dev91 does not
    change that verdict.
- **E2-in-loop:** `E2L` (the S3fz screener, `--lf-rows 0 --gens 50`), output `AGG.g50.CV3@12`.
  - Its S60 mean Δu is +0.0288 (clipped +0.0028) > 0. It did not pass `primary_test` on TRAIN.
- **REPLAY (harness check, not an arm):** `--lf-rows 0 --gens 50` with the H0.1-A screener, which is exactly the H335
  AGG recipe.

| Name | `lf_rows` | `gens` | `checkpoints` | `fr_k` | `neural_run_dir` (fold-0 pair) | Output |
| --- | --- | --- | --- | --- | --- | --- |
| REPLAY | 0 | 50 | [50] | 0 | `h335-cl/neural/runs/H0.1-A` | `AGG.g50.CV3@12` |
| E1B | 600 | 50 | [50] | 24 | `h335-cl/neural/runs/H0.1-A` | `FR.g50.CV3@12` |
| E2L | 0 | 50 | [50] | 0 | `h335-cl/neural/runs/H381-E2-S3fz` | `AGG.g50.CV3@12` |

### 7.2 Mechanics
- **Sandbox:** `sandbox_h381_d91.sh`. `stage_d91_h381.sh` stages the published bytes of this commit.
- **LOCK:** `d91_h381.py lock` writes `h381-d91/stage_v/LOCK.json`.
  - `code_sha256` covers the code, the H317 pins, the grammar port, the superset worker, and each arm's fold-0
    `model.pt` for both seeds. It contains **no dev91 path**.
  - `inputs_manifests_sha256` covers the two H335 Stage V prep manifests. `select` checks every selected table and
    SEARCH view against them.
- **Receipt and unlock:** a create-once 0444 receipt `ns/receipts/H381_LOCK_sha256.txt` (xbabe0, mirrored on xbabe2).
  There is one `unlock`; every later command uses `resume_stage_v`.
- **Contexts:** `select` writes a create-once `cids5.json` by the §5 rule (seed 381501).
- **Runs:** phase V, seeds 335101 and 335102, 5 contexts × 2 seeds per arm, cores 80-127.
  - Each run dir starts from a read-only-sourced copy of the H335 Stage V memo for the 5 contexts. A program the H335
    run already evaluated is read from it, not recomputed. The worker is deterministic, and REPLAY tests this.
  - REPLAY runs first.
- **Readout:** `compare_d91_h381.py` against the stored `dev91-audit-v1` board (sha `40cb4828…`). No board entrant is
  rerun or rescored, and nothing is added to the board.

### 7.3 Checks and reading
- **REPLAY must reproduce the board:** 0 physical worker runs, and per-seed `AGG.g50.CV3@12` u equal to the board's
  `h335.H335_AGG` `per_seed` within 1e-9.
  - If it fails, that is reported. The arms are then compared first with the REPLAY output (same harness), and with
    the board rows second.
- **Per arm, the readout reports:**
  - per (context, seed): Δu and the % RMSE change (`ba·e^(−u)`) vs `h335.H335_AGG`;
  - contexts better or worse (seed mean, ±1e-4);
  - the 5-context AUDIT-Elo rank among all complete board entrants (anchor `h335.H257_REPLICA`).
- **"RMSE moving in the right direction"** means mean Δu > 0 with at least 3 of 5 contexts better. This is descriptive
  only: the best possible sign-flip p on 5 families is 1/32, so nothing is adopted from it.
- **Exposure:** the 5 contexts are recorded as exposed in the H381 receipts and in the board README.

### 7.4 Disclosure: board bytes read before this amendment
- **What was read:** during part A, a command that inspected the board's record format printed the first line of
  `dev91-audit-v1/scores/h335.H335_AGG.jsonl`. That line holds one context (`016648dc…`) with stored `per_seed` u
  3.409 and 2.222.
- **Nothing else:** no other dev91 table, view, memo or score was read.
- **Why it changes nothing:**
  - the arm choice above follows §5 from TRAIN outputs only;
  - the contexts come from the seeded rule;
  - the stored rows are fixed comparison values.
- `compare_d91_h381.py` flags `016648dc…` if it is drawn.

## 8. E2L replication on TRAIN (R60), fixed before it runs
- **Why:** `E2L` is the only H381 output above X0 on S60, and S60 alone cannot separate it from noise (p 0.424).
- **Panel R60:** P201d positions 62–121 in the S60 order (positions 60–61 are DEV2).
  - 60 families, disjoint from S60. cids sha256 `0c711a7606e3c9b4…` (`panels.json`).
  - S60, R6 and DEV2 are unchanged.
- **Run:**
  - `gp_h381.py launch --panel R60 --lf-rows 0 --gens 50 --checkpoints 50 --fr-k 0 --neural-run-dir .../H381-E2-S3fz`
    with seed 364101;
  - xbabe0 cores 16-63, code staged as `gpc-run2` from this commit.
  - The screener is the only change from X0. The stored X0 already covers R60 and is not rerun.
- **Reading** (`analyze_e1_h381.py --panel R60` and `--panel S60+R60`, output `AGG.g50.CV3@12` vs stored X0):
  - **Replicated:** the R60 clipped mean Δu > 0 **and** the pooled S60+R60 (120 families) `primary_test` passes.
  - **Otherwise:** `E2L` stays an unconfirmed lead.
  - **Deadline:** if R60 is not complete by 2026-11-05T20:15Z, it is reported as incomplete and not read as a
    replication.
- **Scope:** TRAIN only. It cannot change the §7 arms or their dev91 reading.

## 9. Erratum (closeout; the §7–8 text above is left exactly as published at `b16c12ff2b`)
- **§7.0, E3 bullet.** It reads "`S3ftB16` −0.0032 vs `S3ft`". **−0.0032 is `S3ftB16` vs S0.** The isolating
  contrast `S3ftB16 − S3ft` is **+0.0071** (95% CI −0.022 … +0.042; sign-flip p 0.34; median −0.003; 42% of families
  harmed by more than 0.01; `results/decide_h381_e2.json`).
- **Effect on decisions: none.**
  - E3 was conditioned on a data-hunger signal, and this contrast is not a reliable one.
  - The §5/§7 arm rule does not involve E2 offline numbers.
