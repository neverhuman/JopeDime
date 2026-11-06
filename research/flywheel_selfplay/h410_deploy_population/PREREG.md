# H410 PREREG — deploy the population, not a program (micro-tests on the 5 exposed dev91 contexts)

**Status: part A, published before any H410 fit, LOCK or dev91 read.**
- Exact presets and builder settings are fixed in code before the dev91 LOCK. The LOCK hashes the code.
- They are listed in a part-B amendment (§8) published before the unlock.

- **Owner:** `cl_fwriter` (Claude Opus 5.5, xbabe2). Claim: `ns/receipts/H410_claim.txt`.
- **Authority:** Jepson, 2026-11-05/06.
  - Request: "rapid micro testing of your 5-10 best ideas on the 5 datasets, I don't want you to worry too much about
    overfitting, we need your best possible micro experiments you can run to validate your MUCH bigger idea."
  - AskUserQuestion answers:
    - **Retrain the scale-L screener: yes.** This is an explicit, single exception to the earlier no-rerun rule.
    - **Time box: about 4 h, including the TRAIN confirmation.**
- **Overfitting waiver (owner).** H410 iterates adaptively on the 5 dev91 contexts that H381 already exposed:
  `ac07466e…`, `8eeca816…`, `016648dc…`, `6ecbb7d5…` and `b1a5c1a1…` (H381 `cids5.json`).
  - H410's dev91 numbers are direction-only, tuned-on evidence. They are never board results.
  - The 5 contexts are logged in the board README as adaptively tuned on.
  - No other dev91 context is read.
  - Every baseline is a stored result (board rows, H335/H381 memos, H364 X0). Nothing is rerun except C1's screener
    training.

## 1. Thesis
**The GP already finds good programs but deploys them badly.**
- The deployed model is one 12-member simplex blend. Its members are chosen by a noisy 2-fold, 560-row fitness, and it
  is refit once (`stacking_h317.cv3_members` / `weights_s1_direct`).
- On the 5 contexts the stored H335 seeds differ by up to 1.19 u (`016648dc`: 3.41 vs 2.22).
- Foundation models win by averaging over many strong hypotheses.
- **Claim:** an ensemble selected out-of-fold and bagged over the *whole* evaluated population (every run and seed,
  plus a strong-learner and target-transform library) closes much of the gap to TabPFN. No longer search is needed.

## 2. Inputs (stored, read-only; copied into H410 roots, never written in place)
- **Pool P per dev91 context:** the SEARCH records with stored out-of-fold predictions `p0`/`p1` and the AUDIT
  records `pa`. They come from:
  - the H335 Stage V memo (seeds 335101 and 335102 share it), as copied by H381 into its REPLAY run;
  - the H381 E2L memo (S3fz screener, same seeds).
- **Stored references:**
  - the deployed per-seed outputs from the H335 Stage V unit records and the H381 E2L unit records (members, `w_audit`);
  - board rows `dev91-audit-v1` (sha `40cb4828…`).
- **TRAIN (§6):**
  - H364 X0 `runs/H364_C_xbabe0` (P201d, seed 364101): memo plus stage records;
  - H381 E2L (S60) and E2R (R60) runs.

## 3. Arms on the 5 dev91 contexts
**Two variants of every builder:**
- **own-pool:** each seed's own H335 run pool only. This costs the same search as the stored H335 seed, so it isolates
  deployment.
- **pooled:** P.

**Tier A — no new SEARCH evals (only AUDIT refits of newly chosen members)**

| # | Arm | Definition |
| --- | --- | --- |
| A0 | Headroom (diagnostic) | Deployed u vs the best already-audited single program; deployed stack's SEARCH out-of-fold RMSE vs the best builder's |
| A1 | Run-bag | Uniform mean of the AUDIT predictions of the 4 stored deployed CV3@12 outputs (H335 s1/s2, E2L s1/s2); 0 worker runs |
| A2 | Wide stack | `cv3_members` with N ∈ {24, 48} and a wider fitness window, then the stock `weights_s1_direct` |
| A3 | Ensemble selection | Greedy forward selection with replacement on the pooled out-of-fold predictions (Caruana), uniform weights over picks |
| A4 | Relaxed stack | Non-negative least squares with an intercept on out-of-fold predictions of the top programs |
| A5 | Bagged members | A2/A3 members deployed as bagged twins (`histgbm2` more bags, `rforest`/`xtrees` 500 trees; other heads unchanged) |

**Tier B — a small new SEARCH stage**
- The memo is shared across seeds, so evals are counted per context.
- The worker timeout is raised to 300 s for this stage only.

| # | Arm | Definition |
| --- | --- | --- |
| B1 | Strong-learner library | About 30 fixed presets at the worker's upper ranges (`oblivious`, `histgbm2` with bags/huber, `rforest`/`xtrees`, `krr`/`gpr`); then the A builders over P ∪ library |
| B2 | Target transforms | `(target kind=log1p\|rank_gauss\|winsor)` prepended to the top programs of P and the B1 tree presets; then the A builders |
| B3 | Polish | Scaled-up neighbours of the top programs (more rounds at lower lr, more bags/trees, depth ±1); then the A builders |

**Tier C — the bigger screener**
- **C1:** retrain H317 `H0.2-L` exactly (scale L, opt1, views A4, subset A525, 3,000 steps, budget slate 128 / top_k 48
  / batch 16, seeds 317101 and 317102), **fold 0 only**, with `keep_models`. Runs on the xbabe1 RTX 4090 through the
  pinned `rapid_hypothesis_lab.py oof` on TRAIN.
  - **Check:** retrained fold-0 out-of-fold predictions vs the stored H0.2-L fold-0 predictions (reported; exact
    equality is not required).
  - Then AGG g50 with this fold-0 pair as the screener, on the 5 contexts × seeds 335101 and 335102. Its programs join
    P.

**BIG:** P ∪ C1 ∪ B1–B3 → the best Tier A builder → A5 bagging.

## 4. Readout (direction only)
- **Per context:** u_arm vs stored `h335.H335_AGG` (seed mean, and per seed for own-pool arms), TabPFN 3.5, TabFM-ens
  and Kumo.
- **Gap-to-TabPFN closed** = (u_arm − u_AGG) / (u_TabPFN − u_AGG), where the denominator is > 0.
- **5-context Elo:** `h335_stats.elo_panel` over all complete board entrants plus the arms, anchor
  `h335.H257_REPLICA`.

## 5. Validation rule on the 5 (fixed now)
**`THESIS_SUPPORTED_ON_5`** if both hold:
1. BIG beats stored H335_AGG on ≥ 4 of 5 contexts with mean Δu ≥ +0.10.
2. The best Tier A pooled arm reaches ≥ 50% of BIG's mean Δu.

Otherwise it is reported as `NOT_SUPPORTED`, with the component that carried any gain.

## 6. TRAIN confirmation (S60 + R60, 120 families; stored X0 never rerun)
- **Primary (deployment only, same programs as X0):**
  - Apply the best Tier A builder to X0's own pool (its memo).
  - Compare with X0's stored `AGG.g50.CV3@12` using `h335_stats.primary_test` (clip 0.5; sign-flip p < 0.025;
    LOFO > 0; median ≥ 0).
- **Secondary:**
  - the builder on the X0 ∪ E2L (S60) / X0 ∪ E2R (R60) pools;
  - plus B1, if B1 carried the dev91 gain;
  - family bootstrap LCB, seed 410150.
- **Cost:** AUDIT refits, plus B1 library evals if used.

## 7. Integrity
- **Code:** new modules only, in `research/flywheel_selfplay/h410_deploy_population/`, applied as runtime patches. The
  H317 pins, H335 files and H381 files are unchanged.
- **dev91:** a new LOCK with a create-once 0444 receipt `ns/receipts/H410_LOCK_sha256.txt` and a one-shot unlock.
  - `code_sha256` holds code, the worker and the C1 `model.pt` files; no dev91 path.
  - `cids5.json` is copied from H381 and checked by sha as an input.
  - H381's `ARMS.json` is not reused or edited.
- **Harness check before any new output:** the offline builder at N = 12 with stock settings on the H335 pool must
  reproduce the stored H335_AGG per-seed u (Δu ≤ 1e-9) with 0 worker runs.
- **Guards and closed data:**
  - 0 guard refusals; DOPE `c51a56d0` unchanged;
  - sealed, holdout and DL-Test closed;
  - H381's data root is read-only to H410.
- **Hosts:**
  - xbabe0 GP sandbox, cpuset `16-63,80-127`, host CPU ≤ 85%, MemAvailable ≥ 20 GiB;
  - xbabe1 GPU for C1;
  - xbabe2 control only.
- **Nothing is added to the board.**

## 8. Part-B amendment (published before the dev91 LOCK; no dev91 byte read by H410 yet)
**8.1 Exact arms** (LOCK `arms`, code `deploy_h410.py` / `d91_h410.py` at this commit)

| Arm | Sources replayed (memo hits) | Pools | Library | Timeout |
| --- | --- | --- | --- | --- |
| `AB` | H335 (`H0.1-A` fold-0 pair), E2L (`H381-E2-S3fz` fold-0 pair); seeds 335101, 335102 | `PH` = H335 both seeds (the stored entrant's own two runs); `P2` = H335 + E2L; own = each H335 seed alone | yes, over `P2` | 300 s |
| `C1` | — (new GP: `unit_lf`, lf_rows 0, g50, `H410-C1-L` fold-0 pair) | — | — | 60 s (recipe) |
| `BIG` | H335, E2L, C1; same seeds | `P3` = all three; `PC1` = C1 alone | yes, over `P3` | 300 s |

- **Builders** in every pool:
  - `W24`, `W48`: window 0.5.
  - `CAR`: top 200, 100 steps.
  - `NNLS`: top 48, with intercept.
  - `BAG_W24`, `BAG_W48`, `BAG_CAR`: bagged twins.
  - `RUNBAG` (pools with ≥ 2 runs).
  - With the library: `LIB.single` and `<pool>+LIB.*`.
- **Library:**
  - 20 presets (`LIB_PRESETS`);
  - log1p / rank_gauss variants of 6 of them;
  - log1p / rank_gauss / winsor variants of the top 16 of the pool;
  - ≤ 64 polish neighbours of the top 8.
- **Stock check:** every replayed run's stock `CV3@12` is re-audited under a new name. On dev91 it must equal the
  stored board `per_seed` row (H335) or the H381 E2L record (E2L).

**8.2 Seeding (host side, before the LOCK)**
- **Memo:** the `AB` and `C1` run-dir memos are byte copies of H381 E2L's memo. On all 5 contexts the H335 Stage V
  memo is a byte prefix of it, so one file serves every replay.
- **BIG:** the BIG memo is the concatenation of the AB and C1 memos after both finish. Duplicate keys hold identical
  results, and the loader overwrites them.
- **Screener / views:**
  - The C1 models are copied read-only to `h335-cl/neural/runs/H410-C1-L`.
  - The views npz is linked, and its sha is in the LOCK inputs.

**8.3 C1 retrain (TRAIN, done before this amendment)**
- **Run:** xbabe1 RTX 4090, `rapid_hypothesis_lab.py oof`, fold 0, seeds 317101 and 317102, about 5.5 min.
  - DOPE `c51a56d0` unchanged.
  - Model shas `c388c637…` and `2ca8c30e…`.
- **Not bit-identical to the stored H0.2-L**, as GPU training is nondeterministic. The fold-0 top-1 agreement is
  0.40 / 0.37.
- **Equal quality:** mean top-1 u on the 244 fold-0 contexts is 1.373 / 1.527, vs 1.343 / 1.523 stored.

**8.4 TRAIN confirmation: primary builder rule (fixed now, before any TRAIN H410 output is read)**
- **Launch:** the TRAIN S60 / R60 deploy runs were launched at 16:03Z, from staged code `train-s1` (the working copy
  that this commit publishes). Their outputs are not read until the dev91 choice below is recorded.
- **Primary TRAIN builder:**
  - The candidates are the Tier A builders in the `PH` pool (W24, W48, CAR, NNLS, BAG_W24, BAG_W48, BAG_CAR).
  - Pick the one with the highest mean Δu over the 5 dev91 contexts vs stored H335_AGG (seed mean).
  - Ties go to the simpler builder, in that order.
- **On TRAIN** it is applied to X0's own pool (`own.<builder>`) and tested vs stored X0 with `primary_test`.
- **Secondary:**
  - the same builder in `P2` (X0 + E2);
  - `P2+LIB.<builder>`, if a library output beats the `PH` builder's mean Δu on dev91.
- **Smoke (TRAIN DEV2, never scored):**
  - replays reproduce stored X0 exactly (0 replay worker runs);
  - 112 / 112 library evals completed; 0 timeouts;
  - AUDIT ledger consistent; 0 same-family bank hits.
