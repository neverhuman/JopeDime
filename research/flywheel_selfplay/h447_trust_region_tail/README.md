# H447: training-target arms under train-fold CV on the flywheel plateau (offline, adapter-only)

**Verdict: `NO_LIFT` (Elo-neutral).** `trust_big` passes the utility-strict rule on all 3 seeds, but its Elo vs the incumbent is about 0 on every seed (−0.6, −0.9, 0.0; wins ≈ losses ≈ 235). Its mean gain comes almost entirely from recovering 10–11 of the incumbent's 19 gate failures (−2.0 utility). On the 1,200 contexts where the incumbent passes, it wins and loses equally. Record it as **utility-strict pass, Elo-neutral (tail-driven)**. Under the Elo-only rule that is not a lift. `soft_list` is no lift, and `regret_cur` is short of strict. Every val91 regression check has a worse mean than the incumbent.

## Hypothesis
H443–H446 left about 90% or more of contexts tied with the incumbent. The guess was that the training target holds the plateau. Pointwise smooth-L1 on raw fitness plus a sharp KL (T=0.2) is dominated by easy contexts, so the argmin gets almost no gradient where the incumbent is wrong. H447 tests three fixes:
- a scale-invariant soft listwise target;
- a regret curriculum that moves weight onto the incumbent's hard contexts;
- a larger step held near the incumbent by an L2-SP trust region.

## Setup
- **Code:** `h447_trainfold_cv.py` (as run; CPU only, `nice 19`, 16 threads, batch 1,024, slate 192, abort if MemAvailable drops below 15 GB). It ran on xbabe1 against a scratch copy, `~/jope-data/hwave-h447`. The live run directory was never written to.
- **Start point:** incumbent `round-3175.pt` (sha `1d6ea900…`) plus its optimizer state, deep-copied per candidate, with the first moment zeroed. Parity reproduces the live val utility exactly (1.1389572403748425) on every seed.
- **DOPE frozen:** composite sha `c51a56d0…` before and after on every seed. Only the 29,211 adapter parameters train.
- **Data:** real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), 0 family overlap. The aggregator asserts every row is DL-Train or DL-Val, so no sealed or holdout data was touched.
- **Instrument:** the same as H444–H446.
  - 5 family-grouped TRAIN folds (244/244/244/244/243), with pooled out-of-fold paired gains vs the incumbent and a family-bootstrap 95% CI.
  - **Strict:** mean > 0.002 **and** CI lower bound > 0; ties never count as wins.
  - Utility is higher-is-better (negated fitness of the argmin program, −2.0 on a gate failure).
  - Elo is computed vs the incumbent, with ties scored 0.5.
- **Arms** (wd 1e-4, grad clip 1.0, init perturbation 5e-3 relative, seed-keyed data order):
  - `soft_list`: adds a per-row standardised soft listwise CE (target softmax(−z(fitness)/0.5), weight 1.0); lr 1e-4, 12 steps.
  - `regret_cur`: `soft_list` plus context sampling weight exp(β·z(regret_inc)), with β ramping 0 → 1.5 (TRAIN folds only); lr 1e-4, 12 steps.
  - `trust_big`: the `soft_list` loss with lr 3e-4 and 16 steps, plus an L2-SP anchor to the incumbent adapter (λ 5.0, relative).
- **Regression check:** the CV-best arm per seed (`trust_big` on all 3) is retrained on all TRAIN and scored once on val91.
- **Timing:** started 2026-11-12 10:46 UTC (03:46 MST), DONE 14:09 UTC (07:09 MST).

## Outcome (OOF over 1,219 TRAIN contexts)
| Arm | OOF gain per seed | Family-bootstrap 95% CI per seed | Elo vs incumbent per seed | W / T / L per seed | Strict |
| --- | --- | --- | --- | --- | --- |
| soft_list | −0.00005, +0.0017, −0.0005 | [−0.015, +0.013], [−0.012, +0.014], [−0.015, +0.013] | −5.4, −5.1, −6.0 | 82/1036/101, 82/1037/100, 82/1034/103 | 0/3 |
| regret_cur | +0.0030, +0.0077, +0.0039 | [−0.011, +0.017], [−0.007, +0.023], [−0.010, +0.017] | −1.7, +0.9, +0.3 | 91/1031/97, 97/1028/94, 95/1030/94 | 0/3 |
| trust_big | +0.0225, +0.0239, +0.0227 | [+0.006, +0.041], [+0.008, +0.042], [+0.006, +0.041] | −0.6, −0.9, 0.0 | 238/741/240, 234/748/237, 235/749/235 | 3/3 (utility only) |

Every `trust_big` fold is positive on every seed (fold gains +0.003 to +0.051).

**val91 regression check (one read per seed, `trust_big`):**

| Seed | Val utility | Gain vs incumbent | W / T / L | Elo | not_worse |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.13023 | −0.0087 | 16/57/18 | −7.6 | false |
| 2 | 1.13096 | −0.0080 | 19/55/17 | +7.6 | false |
| 3 | 1.12990 | −0.0091 | 17/59/15 | +7.6 | false |

- The gain CIs are about [−0.040, +0.013].
- One context (−1.025, the same on every seed) drives the negative val mean. Without it the val mean is +0.002 to +0.003.
- `trust_big` changes about 34 of 91 val decisions, against 2–4 in H444–H446, but wins ≈ losses there too.

Peak RSS was 6.25 GB per seed, and the minimum MemAvailable was 16.3 GB.

## Red-team
- **trust_big's gain is tail-driven, not win-driven** (`metrics.json` → `concentration`). Per seed:
  - The top-10 contexts carry 109–117% of the OOF gain, and the top-50 carry 196–216%. Without the top-10, the mean is −0.002 to −0.004.
  - 9 of the top-10 (and 10–11 of the top-50) are incumbent −2.0 gate failures. `trust_big` recovers 10–11 of the incumbent's 19 TRAIN failures and creates 0 new ones. Those flips alone are worth +0.021 to +0.023, about 100% of the mean.
  - On the 1,200 contexts where the incumbent passes, it scores W/T/L 227/733/240, 224/739/237 and 224/741/235, with a gain of −0.0008, +0.0025 and −0.0006. That is a coin flip.
  - The median per-context delta is 0 overall, and −0.0001 to −0.0004 over the changed contexts.
  - Per family: 88–90 families gain, 82–86 lose and 31–33 are unchanged, and the top-5 families hold 55–59% of the gain.
  - This repeats the r785 pattern (one −2.0 flip) at a larger scale. Fixing failures is real utility, but it is not more wins.
- **regret_cur's loss rise (≈10 → ≈42) is a sampling-composition effect, not divergence.**
  - loss0 is logged at β=0 (uniform sampling) and matches `soft_list` (9.8–10.4) on every fold.
  - lossN is logged at β=1.5, where batches are drawn mostly from high-regret contexts with large smooth-L1 residuals. It sits flat at 40.4–42.8 on all 15 fold×seed runs rather than growing.
  - Decisions changed vs the incumbent (32–50 per fold) are in `soft_list`'s range (30–41), and OOF utility is not worse.
  - The script logs only first and last step losses, so H448 also logs a uniform-batch loss at the last curriculum step.
- **Seed independence:** the per-arm fold-gain vectors differ for every seed pair, and every (arm, fold) and every regcheck has 3 distinct adapter shas. Seeds disagree on 51–64 of about 475 changed OOF contexts for `trust_big`, 14–18 for `regret_cur` and 17–18 for `soft_list`. That is better than H446 (7–8), but decisions are still strongly correlated, so seed spread is not a noise estimate, and the verdict rests on the family-bootstrap CIs.
- **Leakage:** folds are family-disjoint and exclude val families (asserted), with train/val family overlap 0. The regret weights use TRAIN-fold contexts only, so held-out fold contexts are never sampled.
- **Metric direction, ties and cherry-picking:**
  - The aggregator recomputes the strict rule. Utility is higher-is-better, and ties are 0.5 in Elo, never wins.
  - All 3 arms and 3 seeds are reported.
  - val91 was read once per seed, for the CV-selected arm only.
- **Other checks:** no sealed or holdout use. DOPE `c51a56d0` was intact on every DONE line. There was no OOM, and disk is fine (xbabe1 at 89%). The post-hoc concentration analysis recomputed the incumbent's per-context TRAIN utility with the run's own code path (it reproduces 1.420532014416241). It read nothing from val.

## Meaning
A larger trust-region step moves many argmin decisions: about 39% of TRAIN contexts change. The only earlier arm close to that was H446 `reveal_match` (32%), which landed in a worse basin. `trust_big` also fixes about half of the incumbent's gate failures without creating new ones. On passing contexts the moves are unbiased noise, though. The objective rewards mean utility, which a few −2.0 recoveries dominate, so nothing pushes the changed decisions toward wins.

## Next step
H448 keeps `trust_big`'s step (lr 3e-4, 16 steps, L2-SP 5.0) and changes what it optimises toward per-context wins:
- `trust_rank`: a rank-transformed listwise target, a tail-clipped pointwise target and per-context loss normalisation;
- `trust_regret`: `trust_big` plus the regret curriculum, with a uniform-batch loss probe;
- `trust_pair`: a Bradley-Terry pairwise loss vs the incumbent's own argmin.

The strict rule also requires Elo vs incumbent > 0 with family-bootstrap Elo CI lower bound > 0, and per-seed selection is by OOF Elo.

No adoption nudge was sent to the flywheel owner.

Artifacts: `metrics.json` (aggregate per arm and seed, fold gains, regression checks, decision diversity, concentration, red-team), `h447_trainfold_cv.py` (as run) and `h447_aggregate.py`. The raw logs, fold signatures and the concentration dump (`analysis_inc_train.json`) stay on xbabe1 in `~/jope-data/hwave-h447`.
