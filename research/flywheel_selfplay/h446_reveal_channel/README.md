# H446: revealed-history channel arms under train-fold CV on the flywheel plateau (offline, adapter-only)

**Verdict: `NO_LIFT`.** No arm passed the strict rule on any seed, and every val91 regression check was worse than the incumbent. `reveal_match` is clearly worse, `reveal_rand` is slightly negative with CIs spanning 0, and `fail_hinge` is flat.

## Hypothesis
The H445 red-team found a train/eval mismatch in the revealed-history channel. Training slates reveal the fitness and gate of the pool's true top 4, but eval reveals pool indices 0–3, which sit at a median fitness rank of about 554 of 1,152 (this run measured 558/538/566/554, with only 4.1% in the top 48). H446 tested whether removing that mismatch, or adding a hinge on gate-failing programs, lets the adapter change its argmin choices for a real lift over the round-3175 incumbent.

## Setup
- **Code:** `h446_trainfold_cv.py` (CPU only, `nice 19`, 8 threads, batch 1,024, slate 192, MemAvailable abort below 15 GB). It ran on xbabe1 against a scratch copy, `~/jope-data/hwave-h446`. The live run directory was never written to.
- **Start point:** incumbent `round-3175.pt` (sha `1d6ea900…`) plus its optimizer state, deep-copied for every candidate. Parity reproduces the live val utility exactly (1.1389572403748425) on every seed.
- **DOPE frozen:** composite sha `c51a56d0…` before and after on every seed. Only the 29,211 adapter parameters train.
- **Data:** real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), 0 family overlap. The aggregator asserts every row is DL-Train or DL-Val, so no sealed or holdout data was touched.
- **Selection:** the same instrument as H444/H445: 5 family-grouped TRAIN folds (244/244/244/244/243), pooled out-of-fold paired gains, and a family-bootstrap 95% CI. **Strict:** mean > 0.002 **and** CI lower bound > 0, with ties never counted as wins. Utility is higher-is-better (negated fitness of the argmin program, −2.0 on a gate failure). Elo is computed vs the incumbent with ties at 0.5.
- **Arms** (wd 1e-4, grad clip 1.0):
  - `reveal_match`: the first 4 slate entries are pool indices 0–3 (exactly what eval reveals), then the true best 48 and random fill; lr 1.5e-4, 20 steps.
  - `reveal_rand`: the first 4 slate entries are 4 random programs per row; lr 1e-4, 12 steps.
  - `fail_hinge`: unchanged reveal, the incumbent's predicted top 16 added as hard negatives, and a margin hinge (w 0.5) pushing gate-failing programs above the best passing one; lr 5e-5, 8 steps.
- **Regression check:** the CV-best arm per seed (`fail_hinge` on all 3) is retrained on all TRAIN and scored once on val91.

## Outcome (OOF over 1,219 TRAIN contexts)
| Arm | OOF gain per seed (median) | Family-bootstrap 95% CI per seed | Elo vs incumbent per seed (median) | W / T / L | Strict passes |
| --- | --- | --- | --- | --- | --- |
| reveal_match | −0.0425, −0.0414, −0.0459 (−0.0425) | [−0.063, −0.023], [−0.061, −0.021], [−0.067, −0.026] | −27.4, −26.8, −26.6 (−26.8) | 148–150 / 826–827 / 243–244 | 0/3 |
| reveal_rand | −0.0067, −0.0068, −0.0064 (−0.0067) | [−0.016, +0.003], [−0.017, +0.003], [−0.015, +0.0014] | −3.7, −3.1, −2.9 (−3.1) | 64–69 / 1,070–1,078 / 77–80 | 0/3 |
| fail_hinge | +0.0001, −0.0003, +0.0019 (+0.0001) | [−0.0054, +0.0055], [−0.0055, +0.0051], [−0.0013, +0.0066] | −1.7, −2.9, −0.6 (−1.7) | 10–14 / 1,187–1,189 / 16–20 | 0/3 |

**val91 regression check (one read per seed, `fail_hinge`):**
- seeds 1 and 2: −0.0088, Elo −15.3 (0 W / 87 T / 4 L), val 1.130129 on both.
- seed 3: −0.0035, Elo −7.6 (0 / 89 / 2).

None was `not_worse`. Peak RSS was 6.29 GB per seed, and the minimum MemAvailable was 18.0 GB.

## Red-team
- **Decisions are still seed-coupled.** The 3 regcheck adapters have distinct hashes, and every (arm, fold) has 3 distinct adapters, so this is not a code bug. Even so:
  - Seeds 1 and 2 score an **identical** val91 utility (1.13013, same 0/87/4) with different adapter shas. The two regcheck values (−0.008828 and −0.003518) are the same two values H444 and H445 produced, so near the incumbent val91 has only a couple of reachable outcomes.
  - Fold gains repeat across seeds: `fail_hinge` fold 2 is +0.0119 on seeds 2 and 3 (+0.0116 on seed 1), and fold 3 is −0.00884 on seeds 1 and 2.
  - Decision diversity (summed over 5 folds): `fail_hinge` changes 30–32 of 1,219 contexts vs the incumbent, and seeds disagree on only 7–8; `reveal_rand` changes 141–149, seeds disagree on 10–20; `reveal_match` changes 392–393, seeds disagree on 27–38.
  - Seed spread is therefore **not** a noise estimate, and the verdict rests on the family-bootstrap CIs.
- **reveal_match moves a lot of decisions, all in the wrong direction.** About a third of contexts change and losses outnumber wins 244 to 148. Matching eval's reveal makes training harder (loss ~14 vs ~4) without a better argmin, so the revealed-channel mismatch does not explain the plateau.
- **Output path slip (fixed, no data impact).** The runs were launched with a relative `--out`, and the script `chdir`s into the flywheel-gp-fused code worktree on xbabe1 before writing, so `summary.json`, `progress.log` and `fold_sig/` landed as untracked `results/` and `results_probe/` in that worktree's root. No tracked file or live run state was touched. They were moved back into `~/jope-data/hwave-h446` before aggregation. H447 resolves `--out` to an absolute path before the `chdir`.
- **Other checks:** the metric direction and the strict rule are recomputed in the aggregator, and ties never count. Folds are family-disjoint and val families are excluded from them. No sealed or holdout use. DOPE was intact on every DONE line, there was no OOM, and the pinned script sha256 checks out.

## Meaning
Four waves (H443–H446) of optimiser, loss-shape, candidate-choice and revealed-channel changes leave about 90% or more of contexts tied with the incumbent. Small adapter moves stay in the incumbent's argmin basin, and the larger move (`reveal_match`) lands in a worse one.

## Next step
H447 keeps the same instrument and changes the training target and its order: a soft listwise target with a temperature over true fitness, a hard-context curriculum, and a larger trust-region step anchored to the incumbent. Seeds also get independent data order and stronger per-seed perturbation, with a decision-diversity assertion.

No adoption nudge was sent to the flywheel owner.

Artifacts: `metrics.json` (per-arm, per-seed CV, fold gains, regression checks, decision diversity, red-team), `h446_trainfold_cv.py` (as run), `h446_aggregate.py`. The raw logs stay on xbabe1 in `~/jope-data/hwave-h446/results`.
