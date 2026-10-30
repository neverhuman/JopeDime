# H46 — Gradient smoothing to break the DL-Val91 regression plateau (overnight wave 1)

Status: **PLANNED / RUNNING** (opened 2026-10-30 00:35 MDT). Results are appended by a follow-up PR.
Owner ask: Jepson, 2026-10-30 00:02 MDT (overnight H46+ waves, 3 seeds, offline on a checkpoint COPY).

## Symptom (evidence from live summary.jsonl, rounds 775–844, xbabe1 run-20261030)
- DL-Val91 regression utility (higher is better): mean 1.1317, sd 0.0105, max 1.13760 (r785 = current best).
- Since r785 **every** candidate has been rejected and the loop takes action
  `restore_best_model_and_optimizer` (best_checkpoint = round-785.pt). Each round then does only
  `optimizer_steps=2` (AdamW lr 3e-4, wd 1e-4, batch 8192) from the SAME restored state.
- Consequence: candidate scores are a **deterministic cycle**, not noise: 1.13558 / 1.13544 / 1.13665 / 1.13443 /
  1.13700 / 1.13745 repeat exactly (identical per-context vectors). The "~4-round period" is the batch-order cycle
  replayed from one frozen starting point. The model can never move more than 2 steps from r785 → structural plateau.
- Update l2 ≈ 0.033/round; grad norm ≈ 0.45 < clip 1.0 (clipping inactive). Most of the 91 per-context scores are
  bit-identical across candidates: utility is an argmin-over-programs (discrete), so tiny steps rarely flip a selection.
- r785 accept was mainly one context recovering from a −2.0 failure score; one exact-tie accept was observed earlier.
- train−val gap (reg) ≈ 0.288.

## Hypotheses (ranked by expected value)
- **H46-A (multi-step + EMA)**: escaping restore-to-best requires accumulating many steps. Train N=64 steps continuously
  from r785 and evaluate a Polyak/EMA copy (decay 0.98) every 8 steps → EMA smooths the batch-cycle oscillation while
  letting the model actually travel.
- **H46-B (A + warmup/cosine + tighter clip)**: 4-step linear warmup to 3e-4 then cosine to 3e-5; clip 0.5.
- **H46-C (A + robust loss)**: Huber/clipped per-context loss (failure scores floored, per-context rank/z normalisation)
  so −2.0 failures don't dominate gradients.
- **H46-S (selection fix, applied post-hoc to every arm)**: accept only if the family-grouped paired bootstrap
  (2,000 resamples over the 91 DL-Val contexts, grouped by family) gives lower 95% CI of Δ > 0 AND Δ > 1e-4;
  exact ties are rejected.
- Control **H46-0**: live protocol replay (restore-on-reject, 2 steps/round) for the same wall budget.

## Setup
- Host xbabe2 (RTX 3090), worktree `~/JopeDime/.agent/worktrees/h46-gradsmooth`, branch `research/h46-gradient-smoothing`.
- Checkpoint COPY: `xbabe2:~/jope-data/h46/ckpt/round-785.pt` (sha256 ef6696f6…), plus round-844.pt, best_dev_checkpoint.json
  (evaluation_identity e79a7715…), input copy `~/jope-data/h46/input` (442 MB). The live loop on xbabe1 is NOT touched.
- Data: the approved regression 1,219 DL-Train / 91 DL-Val contexts only. Sealed meta-test and holdout never read.
  Binary deferred.
- DOPE frozen: `dope_freeze_guard.py` checksum must equal c51a56d0… at start and after every eval; only the 29,211
  adapter params train. Abort the arm on mismatch.
- Seeds: 0, 1, 2 (batch shuffle + init of EMA). Report per arm: median val utility across seeds, 95% bootstrap CI,
  per-family deltas, number of contexts improved/worsened, regression Elo vs r785 (no AUC).
- GPU: hold `~/jope_plan/qualification.lock` (flock) per seed run; release between runs. CPU ≤85%, watch swap.

## Success criterion
An arm "beats the plateau" only if its median Δ vs r785 across 3 seeds is > 0 and the seed-pooled bootstrap lower
95% CI is > 0, and not driven by a single context (leave-one-context-out Δ still > 0).

## Outcome / metrics / verdict / next step
Pending — see `metrics.json` (filled by the results PR).
