# H47: utility-aligned ranking losses vs the H46 objective mismatch (overnight wave 2)

Status: **QUEUED/RUNNING** (opened 2026-10-30 01:15 MDT on xbabe2, screen `h47_runner`). Results land in a follow-up PR.
Owner ask: Jepson, 2026-10-30 00:02 MDT (overnight H46+ waves, 3 seeds, offline on a checkpoint COPY).

## Why (evidence from H46, interim, xbabe2 `~/jope-data/h46/runs`)
- H46-A (64 continuous steps + EMA 0.98): the training loss (Huber + 0.1·KL) falls from 3.690 to 3.359, yet the final EMA
  DL-Val91 utility goes from 1.137595 to 1.036767 / 1.037759 / 1.036767 (seeds 0/1/2, Δ ≈ −0.10), and the **train**
  utility also falls (r785 1.425042 → 1.3545 / 1.3522 / 1.3551). H46-B (warmup-cosine, clip 0.5) seed0: val 1.053521,
  train 1.358874.
- Train utility falling along with val means the problem is **objective mismatch, not overfitting**. Lowering absolute-cost
  Huber regression (mostly calibration/offset) does not improve, and actually hurts, the argmin-over-programs selection that
  utility measures. H46-0 (live restore-on-reject protocol) never beats r785 (best Δ −0.0021, or Δ 0 after retaining r785).
- Seeds are near-pseudo-replicates: full 1,219-context batch, and the seed changes only the random part of each 192-program
  slate (top-48 is deterministic). So treat the context/family bootstrap, not seed spread, as the uncertainty.

## Hypotheses / arms (all: r785 copy, 64 full-cohort AdamW steps, EMA 0.98 evaluated every 8 steps; raw weights also logged)
- **H47-R**: listwise top-1 cross-entropy, target softmax(utility/0.05) over the slate (gate fail = −2), model softmax(−cost/0.2).
  Shift-invariant in predicted cost.
- **H47-Z**: H46 loss with per-context centred cost (removes the offset/calibration term). A minimal test of the mismatch hypothesis.
- **H47-PW**: pairwise logistic, top-16 slate programs vs every slate program, weighted by the utility gap.
- **H47-RP**: H47-R + L2-SP proximal anchor to r785 (λ = 1e-2) and lr 3e-5 (trust region).
- Comparator: r785 (Δ = 0) and H46-0 control. Pre-registered verdict uses the **final EMA** weights (no step selection).
  The best step is reported separately and labelled selection-biased.

## Guards
DOPE frozen (guard checksum c51a56d0… checked before training and around every eval; abort on mismatch); only the 29,211
adapter params train. Data: the approved 1,219 DL-Train / 91 DL-Val regression packet copy only. No sealed meta-test or
holdout. Live xbabe1 loop untouched. GPU: one seed at a time under `~/jope_plan/qualification.lock`, starting only after
the H46 runs finish. Abort if MemAvailable < 20 GiB.

## Success criterion (same as H46)
Median Δ across seeds > 0, seed-pooled family-bootstrap lower 95% CI > 0, and min leave-one-context-out Δ > 0.
Secondary: train utility must rise (alignment check). Report regression Elo vs r785 (no AUC).

## Outcome / metrics / verdict / next step
Pending. See `metrics.json`.
