# H381: rapid tests of 300-row GP search and a grown screener (TRAIN), plus a 5-dataset dev91 direction check

**Verdict: `NO_SPEEDUP_LEAD_UNCONFIRMED`.**
- **300-row search (E1):** much worse. LF300 raises RMSE by about 18–29% vs X0.
- **Grown, locked screener inside the GP loop (E2L):** the only H381 output above X0. It is not confirmed; see R60.
- **dev91 (5 datasets, direction only):** E2L ends up better on 4 of 5 datasets. Most of the gain comes from one
  dataset where the stored H335 seed did badly.

**Bottom line for the owner's questions:**
- **"300 random rows to train much faster"** does not work for the GP search. It is faster per eval but loses badly.
  The 600-row version still loses about 10%, and after correcting for host load it saves almost no CPU.
- **"Larger network, reusing what we have":** growing the screener to 3 layers and **locking** the old weights is the
  best variant tried. It beats fine-tuning (p 0.04). Inside the GP it gives a consistent but unproven ~2.5% RMSE cut
  on 120 TRAIN families (p 0.23).
- **dev91 spot check:** on 5 datasets, RMSE moves the right way on 4 of 5, mostly by under 1%.

- **PREREG:** [`PREREG.md`](PREREG.md).
  - Part A at `5e1e881ba5`.
  - Part-B amendment (dev91 arms, R60 replication) at `b16c12ff2b`, published before the LOCK, the unlock and the
    R60 run.
  - A closeout erratum (§9).
- **Owner:** `cl_fwriter`, Claude Opus 5.5.
- **Request:** owner, 2026-11-05 — synthetic variety plus about 300 random rows to train much faster; a larger
  network that reuses the existing weights (locked or fine-tuned); a spot check on 5 random dev91 datasets of
  whether RMSE moves the right way.
- **Rule from the owner's follow-up:** "don't rerun models you already have". Every baseline is a stored result:
  - H364 X0 `AGG.g50.CV3@12` on TRAIN;
  - H317 `H0.1-A` / `H0.2-L` out-of-fold results;
  - the `dev91-audit-v1` board rows.

  No baseline, entrant or reference model was rerun.
- **Numbers:** [`metrics.json`](metrics.json).

## E1: GP search on a low-fidelity SEARCH subsample (TRAIN S60, 60 families, vs stored X0)
- **The idea:** evaluate GP candidates on 150 stratified rows per SEARCH fold (LF300) or 300 per fold (LF600), with
  the AUDIT envelope unchanged.
- **FR outputs** re-evaluate the 24 best low-fidelity candidates plus the CV3 members on the full 1,120 SEARCH rows,
  then rebuild the stack.

| Output (`CV3@12`) | Family-mean Δu | RMSE change | Families W/T/L | Clipped mean | Sign-flip p | Search-CPU ratio (raw) |
| --- | --- | --- | --- | --- | --- | --- |
| LF300 g50 | −0.214 | +24% | 18/0/42 | −0.069 | 0.995 | 0.22 |
| LF300 g50 + FR | −0.200 | +22% | 18/0/42 | −0.058 | 0.987 | 0.28 |
| LF300 g100 | −0.208 | +23% | 19/0/41 | −0.078 | 0.997 | 0.49 |
| LF300 g100 + FR | −0.165 | +18% | 21/0/39 | −0.052 | 0.974 | 0.56 |
| LF600 g50 | −0.104 | +11% | 28/0/32 | −0.018 | 0.787 | 0.60 |
| LF600 g50 + FR | −0.096 | +10% | 27/0/33 | −0.007 | 0.633 | 0.66 |

- **Both preregistered rules fail:**
  - **E1-win** (LF300 g100 FR beats X0): p 0.974.
  - **E1-speed** (LF300 g50 FR non-inferior): bootstrap LCB −0.109, where the rule needs > −0.005.
- **Why it fails:**
  - The 300-row fitness is too noisy to rank programs; the search drifts toward programs that look good on 150 rows.
  - The full-row finish recovers only a little.
  - On context 6940ec22, LF300 evals cost *more* wall time on average than X0's full-row evals (10.2 s vs 7.1 s): the
    search moved to more expensive programs. Fewer rows did not make each eval cheaper there.
- **The CPU ratios are flattered by host load.**
  - On full-row SEARCH programs that a run and X0 both evaluated, the run's wall time was 0.64× X0's for E2L (6,513
    programs), 0.74× for LF600 (77 programs) and 0.94× for LF300 (116 programs). X0 ran on a busier host.
  - Corrected for that, LF600 + FR is about 0.9× X0's CPU: no real saving, at a 10% RMSE loss.
- **Not run:** the planned combination run (best screener + LF300), because E1 did not pass.

## E2: a deeper screener grown from the existing weights (offline, 205 TRAIN families, vs stored S0 = H0.1-A)
- **S3** = scale S with 3 layers (2 new attention blocks per stack).
- **Growth:** the new blocks' output projections start at zero, so the grown model starts out computing exactly S0
  (checked bit-equal on 4 contexts).
- **Training:** 3,000 steps, 2 seeds × 5 family folds, both GPUs (folds 0–2 on xbabe1, 3–4 on xbabe3). DOPE unchanged.

| Arm | Family-mean gain vs S0 | Holm LCB | Contract gate |
| --- | --- | --- | --- |
| `S3fz` (grown, S0 weights **locked**) | **+0.0103** | −0.016 | fail (harm, LCB) |
| `S3sc` (S3 from scratch) | +0.0061 | −0.059 | fail |
| `S3ftB16` (fine-tune + B16 synthetic-variety views) | −0.0032 | −0.039 | fail |
| `S0cont` (S0, 3,000 more steps) | −0.0069 | −0.027 | fail |
| `S3ft` (grown, all weights, old at ×0.1 LR) | −0.0103 | −0.030 | fail |
| H317 `H0.2-L` (stored, descriptive) | +0.0718 | — | — |

- **Locking the old weights beat fine-tuning them:** `S3fz − S3ft` = +0.021, p 0.040, LOFO +0.014.
  - Fine-tuning the old weights loses what S0 learned; more training of S0 alone also loses (`S0cont`).
- **Synthetic variety on top of fine-tuning:** `S3ftB16 − S3ft` = +0.007 (p 0.34, 42% of families harmed). No
  reliable data-hunger signal, so the E3 synthetic-task pilot was not run (PREREG §4, erratum §9).

## E2L: the locked S3 screener inside the GP loop (TRAIN S60, vs stored X0)
| Output | Family-mean Δu | RMSE change | W/T/L | Clipped mean | Sign-flip p | LOFO |
| --- | --- | --- | --- | --- | --- | --- |
| `AGG.g50.CV3@12` | **+0.0288** | **−2.8%** | 28/7/25 | +0.0028 | 0.424 | −0.0056 |
| `AGG.g50.single` | −0.0169 | +1.7% | 17/1/42 | −0.0316 | 0.958 | −0.041 |

- The screener is the only change from X0 (same seed, panel, worker and recipe). It does not pass `primary_test`.

## R60: E2L replication on 60 fresh TRAIN families (PREREG §8)
- **Panel:** R60 = P201d positions 62–121, families disjoint from S60.
- **Run:** identical to E2L (S3fz screener, seed 364101), vs the stored X0, which is not rerun.

| Panel | Family-mean Δu | RMSE change | W/T/L | Clipped mean | Median | LOFO | Sign-flip p | Bootstrap LCB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S60 (E2L) | +0.0288 | −2.8% | 28/7/25 | +0.0028 | 0 | −0.0056 | 0.424 | −0.025 |
| R60 (fresh) | +0.0215 | −2.1% | 23/10/27 | **+0.0136** | 0 | +0.0053 | 0.220 | −0.018 |
| **S60 + R60** (120) | **+0.0251** | **−2.5%** | 51/17/52 | +0.0082 | 0 | +0.0040 | **0.230** | −0.012 |

- **§8 rule:** the R60 clipped mean is > 0, but the pooled `primary_test` does not pass. **E2L stays an unconfirmed
  lead.**
- **What the gain looks like:** the mean is positive on both panels, but the median family is unchanged.
  - 16 of 120 contexts gain more than 0.05; 13 lose more than 0.05.
  - The extremes are +2.88 (077d29c6) and −2.46 (d3f3bd42).
  - The gain is a shift in a minority of contexts, not a broad improvement.
- **CPU:** raw search-CPU ratio 0.71 vs X0. On 13,657 shared programs this host ran at 0.65× X0's wall time, so E2L
  costs about 1.1× X0's search CPU after the correction.

## dev91: 5-dataset direction check (stored board rows, nothing added to the board)
- **LOCK:** `b9375201…`, with a create-once receipt `ns/receipts/H381_LOCK_sha256.txt` and a one-shot unlock.
- **Contexts:** 5 dev91 contexts from 5 families, drawn by the seeded rule (seed 381501): `ac07466e…`, `8eeca816…`,
  `016648dc…`, `6ecbb7d5…`, `b1a5c1a1…`.
- **Seeds:** 335101 and 335102, as in H335.
- **Harness check (REPLAY):** the H335 AGG recipe run on a copy of the H335 Stage V memo reproduced the stored
  `h335.H335_AGG` `per_seed` u **exactly** (Δu = 0.0 on all 10 (context, seed) pairs), with 0 physical worker runs.

| Arm (output) | Contexts better / worse | Median context Δu | Mean Δu | Mean RMSE change | Without `016648dc`: mean Δu, better | 5-context AUDIT-Elo |
| --- | --- | --- | --- | --- | --- | --- |
| `h335.H335_AGG` (stored) | — | — | — | — | — | 1689.6 |
| E2L (`AGG.g50.CV3@12`) | **4 / 1** | +0.008 | +0.099 | −5.8% | −0.004, 3/4 | **1861.1** (= TabPFN 3.5) |
| E1B (`FR.g50.CV3@12`) | 3 / 2 | +0.003 | +0.036 | +2.2% | −0.088, 2/4 | 1750.2 |

Per-context Δu, seed mean:

| Arm | `ac07466e` | `8eeca816` | `016648dc` | `6ecbb7d5` | `b1a5c1a1` |
| --- | --- | --- | --- | --- | --- |
| E2L | +0.003 | +0.008 | +0.508 | +0.030 | −0.057 |
| E1B | +0.003 | +0.006 | +0.528 | −0.270 | −0.089 |

- **Read with care:**
  - `016648dc` carries most of both means. H335's stored seed 335102 scored u 2.222 there, against 3.409 for seed
    335101. Both H381 arms land near 3.3–3.4 on both seeds. That is variance in the stored baseline more than skill
    in the arms.
  - On the other four contexts, E2L moves RMSE by under 1% three times (all better) and is worse on `b1a5c1a1`
    (about 6% higher RMSE).
  - E1B loses badly on `6ecbb7d5` (about 31% higher RMSE), as it did on TRAIN.
  - The 5-context Elo inherits all of this. 5 families cannot reach significance (the best sign-flip p is 1/32).
- **Disclosure (PREREG §7.4):** `016648dc…`'s stored H335 row was printed during part A, before the amendment.
  - It was drawn by the seeded rule and is flagged in `results/d91_compare_arms.json`.
  - No choice depended on it.
- **Exposure:** these 5 contexts are recorded as exposed in the board README (H381 exposure entry).

## Integrity
- **Code:**
  - The 18 H317 pins and the H335 files are unchanged (checked at every stage).
  - New code exists only in this directory, and runs as runtime patches.
  - Code roots are staged read-only with `CODE.sha256` (`gpc-run1`, `gpc-run2`, `d91-run1`). The dev91 root holds the
    published bytes of `b16c12ff2b`.
- **Guards:**
  - 0 guard refusals in every GP and dev91 process;
  - `same_family_bank_hits` 0;
  - the DOPE composite `c51a56d0` is unmodified.
- **Closed data:** sealed, holdout and DL-Test were never touched. dev91 was read only after the LOCK, only for the 5
  selected contexts.
- **Fixed during the run:** the dev91 sandbox did not see the merged TRAIN view bank that the screener's context
  coder reads from `$H317_DATA/neural/`.
  - REPLAY failed at its first unit, before any eval.
  - The same file (sha `eb042276…`) was linked read-only into the H381 data root, and REPLAY was rerun.
- **Erratum:** PREREG §9 corrects the `S3ftB16` number in §7.0 (no decision changed).

## Files
| File | What it is |
| --- | --- |
| `gp_h381.py` | LF envelope, FR finish, `unit_lf`, GP runner (TRAIN) |
| `run_gp_h381_xbabe0.sh`, `stage_h381.sh` | xbabe0 launcher; read-only code staging |
| `grow_h381.py` | E2 growth / freeze / fine-tune harness; `decide` |
| `analyze_e1_h381.py` | E1 / E2L / R60 analysis vs stored X0 (`--panel`, host-speed check) |
| `d91_h381.py`, `sandbox_h381_d91.sh`, `stage_d91_h381.sh` | dev91 LOCK / unlock / select / run; sandbox; staging |
| `compare_d91_h381.py` | dev91 readout vs stored board rows (per-seed RMSE, 5-context Elo) |
| `panels.json` | S60, R6, DEV2, R60 |

## What this suggests next (not run; owner decision)
1. **Finish the E2L read on TRAIN.** P201d has 68 more unused families (positions 122–189). Running E2L there costs
   about 25 min on xbabe0 and would bring the pooled test to 188 families. Neither seed nor baseline is rerun.
2. **Test the scale-L screener in the loop.** The stored H317 `H0.2-L` (offline +0.072, the largest screener lead)
   kept no `model.pt`, so it cannot be dropped into the GP without retraining. This is the one larger-network test
   H381 could not do under the no-rerun rule.
3. **Drop low-fidelity row subsampling for the GP search.** The search budget is better spent on generations at full
   rows.
