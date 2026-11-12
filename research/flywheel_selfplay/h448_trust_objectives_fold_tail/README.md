# H448: per-context objectives on the trust-region step (offline, adapter-only)

**Verdict: `NO_LIFT`.** No arm passes the strict rule (Elo > 0 with family-bootstrap Elo CI lower bound > 0) on any seed, and every Elo CI spans 0. `trust_regret` is the only arm with a positive Elo point estimate on all 3 seeds (+3.1, +4.6, +1.7), but wins ≈ losses (≈242 vs ≈232). `trust_rank` and `trust_pair` are Elo-negative on every seed. The val91 regression check (`trust_regret`, selected on every seed) is worse than the incumbent on mean on all 3 seeds (−0.055, −0.089, −0.056), and significantly worse on seed 2 (CI upper bound < 0).

## Hypothesis
H447 `trust_big` (soft listwise + lr 3e-4 × 16 steps + L2-SP anchor 5.0) was utility-strict but Elo-neutral: its gain came from recovered −2.0 gate failures, and its changed decisions on passing contexts were a coin flip. H448 kept that trust-region step and changed the objective so that every context counts equally:
- `trust_rank`: listwise target on per-row fitness **ranks**, smooth-L1 target clipped to [row best, row best + 0.5], per-context loss normalisation;
- `trust_regret`: `trust_big` + the regret curriculum (β 0 → 1.5, TRAIN folds only), with a uniform-batch loss probe at the last step;
- `trust_pair`: `trust_big` + a Bradley-Terry loss vs the incumbent's own argmin program (weight 1.0, T 0.1).

## Setup
- **Code:** `h448_trainfold_cv.py` (as run; CPU only, `nice 19`, 16 threads, batch 1,024, slate 192, abort if MemAvailable < 15 GB) on xbabe1 against a scratch copy, `~/jope-data/hwave-h448`. The live run directory was never written to.
- **Start point:** incumbent `round-3175.pt` (sha `1d6ea900…`) plus its optimizer state, deep-copied per candidate, first moment zeroed. Parity reproduces the live val utility exactly (1.1389572403748425) on every seed.
- **DOPE frozen:** composite sha `c51a56d0…` before and after on every seed; only the 29,211 adapter parameters train.
- **Data:** real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), 0 family overlap; the aggregator asserts every row is DL-Train or DL-Val. No sealed or holdout data was touched.
- **Instrument:** same as H444–H447 — 5 family-grouped TRAIN folds (244/244/244/244/243, `h444/` salt), pooled OOF paired gains vs the incumbent, family-bootstrap 95% CIs. **Strict (H448):** mean > 0.002 **and** gain CI low > 0 **and** Elo > 0 **and** Elo CI low > 0; ties never win (0.5 in Elo). Per-seed selection by OOF Elo.
- **Timing:** started 2026-11-12 14:39 UTC (07:39 MST), DONE 18:40 UTC (11:40 MST); ≈14–15 min per fold. Peak RSS 6.2 GB per seed, minimum MemAvailable 16.0 GB.

## Outcome (OOF over 1,219 TRAIN contexts)
| Arm | OOF gain per seed | Gain CI per seed | Elo per seed | Elo CI per seed | W / T / L per seed | Strict |
| --- | --- | --- | --- | --- | --- | --- |
| trust_rank | +0.0121, +0.0093, +0.0070 | [−0.013, +0.037], [−0.015, +0.034], [−0.017, +0.031] | −5.1, −4.8, −4.0 | [−21, +11], [−23, +12], [−20, +12] | 272/657/290, 271/660/288, 271/663/285 | 0/3 |
| trust_regret | +0.0133, +0.0151, +0.0150 | [−0.013, +0.037], [−0.009, +0.039], [−0.011, +0.037] | +3.1, +4.6, +1.7 | [−9.7, +16], [−8.0, +17], [−9.9, +14] | 243/744/232, 245/745/229, 240/745/234 | 0/3 |
| trust_pair | +0.0117, +0.0073, +0.0075 | [−0.013, +0.036], [−0.017, +0.033], [−0.018, +0.033] | −6.8, −10.0, −5.7 | [−23, +8.6], [−27, +6.1], [−21, +9.8] | 341/513/365, 334/516/369, 344/511/364 | 0/3 |

Fold gains (seed 1): `trust_rank` +0.035, +0.064, −0.029, −0.003, −0.006; `trust_regret` +0.022, +0.053, −0.011, −0.012, +0.015; `trust_pair` +0.037, +0.046, −0.001, −0.008, −0.016 (all seeds in `metrics.json`).

**val91 regression check (one read per seed, `trust_regret`):**

| Seed | Gain vs incumbent | Gain CI | W / T / L | Elo | not_worse |
| --- | --- | --- | --- | --- | --- |
| 1 | −0.0547 | [−0.129, +0.003] | 20/52/19 | +3.8 | false |
| 2 | −0.0889 | [−0.188, −0.006] | 17/53/21 | −15.3 | false |
| 3 | −0.0560 | [−0.131, +0.003] | 18/52/21 | −11.5 | false |

This is ~6–10× worse on mean than H447 `trust_big` (−0.009): the regret curriculum, which trains mostly on high-regret TRAIN contexts by the last steps, generalises worse to the 19 val families.

## Red-team
### (a) `trust_regret` loss rise (≈10 → ≈42): bookkeeping, not divergence
The script logs, at the final step and with the final-step parameters, the loss of a **uniform** context draw next to the curriculum-weighted one (`losses.jsonl` → `uniform_last`). On all 18 runs (15 fold runs + 3 full runs) the uniform-batch loss is 9.7–10.5, the same as the step-0 loss (9.7–10.4, β = 0), while the curriculum batch sits at 40.0–42.5. The loss curve (seed 1, cv0: 10.1, 11.2, 12.3, 13.8, 17.7, 24.2, 30.8, … 41.9) tracks β, saturating as β → 1.5. So the rise is the sampling composition shifting onto high-regret contexts with large smooth-L1 residuals. It is not divergence. It also shows the 16 steps barely move the uniform-batch loss — the trust-region step changes argmins, not fit.

### (b) The cross-wave "folds 0–1 positive, folds 2–4 negative" pattern
`fold_diag.py` (reads only TRAIN; the incumbent's per-context TRAIN utility comes from the H447 dump, which reproduces 1.420532 with the run's own code path):
- **Same partition every wave.** H444–H448 use the same deterministic `h444/`-salted family partition, so the pattern repeating across waves is one partition seen five times, not five independent confirmations.
- **Folds 0–1 hold 12 of the incumbent's 19 TRAIN gate failures** (5 + 7; folds 2–4 have 4, 2, 1). Fold 0 also has the lowest incumbent utility (1.239 vs 1.424/1.577/1.518/1.345) and contains the largest family (36 contexts, 15% of the fold).
- **The positive folds are gate-failure recovery.** Splitting each H448 fold's gain into passing contexts vs incumbent −2.0 contexts (mean over 3 seeds): recovered failures contribute +0.041 to +0.048 in folds 0–1 and +0.014 to +0.021 in folds 2–4. On passing contexts the gain is ≈0 in folds 0–1 (−0.012 to +0.007) and negative in folds 2–4 (−0.016 to −0.035).
- **On passing contexts the arms lose more than they win:** `trust_rank` W/L 258/290, 257/288, 258/285; `trust_pair` 326/365, 319/369, 329/364; `trust_regret` 230/232, 232/229, 227/234.
- **Interpretation.** The passing-context losses are worst where the incumbent fits best (fold 2, incumbent utility 1.577: `trust_rank` −0.062 on passing contexts). The incumbent was trained on all 1,219 TRAIN contexts, so on each held-out fold it is in-sample, while the candidate's extra steps never see that fold: moving away from the incumbent there is mostly forgetting, and it costs most where the in-sample fit was best. That makes this instrument conservative (it will not manufacture a lift), and it explains why mean gains come only from the −2.0 tail and Elo sits at or below 0.
- **Not fold leakage:** folds are family-disjoint and exclude val families (asserted in code); the curriculum weights use TRAIN-fold contexts only. **Not incumbent weakness on whole families:** the max family share per fold is 7–15%, and the gain is concentrated in individual gate-failure contexts, not families.
- Cross-wave fold gains for H445–H448 are in `metrics.json` → `fold_diagnosis.cross_wave_fold_gains`. H447 `trust_big` was positive in every fold (because it recovered 10–11 failures spread over all folds), and H445/H446 arms show the same partition effect at smaller scale (fold 2 positive, fold 3 negative for `look_ctxnorm` and `fail_hinge`).

### (c) Other checks
- **Seed independence:** per-arm fold-gain vectors differ for every seed pair; every (arm, fold) and every regcheck has 3 distinct adapter shas. Seeds disagree on 39–44 (`trust_regret`), 72–78 (`trust_rank`) and 102–139 (`trust_pair`) OOF contexts. Decisions are still strongly correlated, so the verdict rests on the family-bootstrap CIs.
- **Leakage:** train/val family overlap 0; folds family-disjoint (asserted); val91 read once per seed, only for the CV-selected arm.
- **Metric direction and ties:** utility is higher-is-better (negated fitness, −2.0 on gate fail); the aggregator recomputes the strict rule; ties are 0.5 in Elo and never wins.
- **Data:** DL-Train/DL-Val only; no sealed or holdout reads; DOPE `c51a56d0` intact on every DONE line; no OOM.

## Meaning
All three per-context objectives on the trust-region step leave the plateau where H447 left it: the only reliable effect is recovering incumbent gate failures, and on passing contexts the moves are noise that loses at least as often as it wins. Changing the objective did not convert the step into wins.

## Next step
H449 keeps the only Elo-positive objective (`trust_regret`) and attacks the noisy passing-context moves in weight space: `regret_ctrl` (H448 reproduction), `regret_blend` (WiSE-FT interpolation half-way toward the incumbent), `regret_soup` (uniform soup of 3 independently keyed sub-runs) and `regret_soup_blend` (both), with the same instrument and H448 strict rule.

No adoption nudge was sent to the flywheel owner.

Artifacts: `metrics.json` (aggregate per arm and seed, fold gains, regression checks, decision diversity, fold diagnosis, loss-rise check), `h448_trainfold_cv.py` (as run), `h448_aggregate.py` and `fold_diag.py`. Raw logs and fold signatures stay on xbabe1 in `~/jope-data/hwave-h448`.
