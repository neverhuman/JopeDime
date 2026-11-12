# H444: family-grouped TRAIN-fold CV selection for the flywheel plateau (offline, adapter-only)

**Verdict: `NO_LIFT`.** No arm passed the strict rule on any seed. `fbal_accum` is flat, while `swa_cos` and `hilr_wd` make the adapter clearly worse. The seeds of `fbal_accum` turned out not to be independent (see Red-team).

## Hypothesis
H443 showed the plateau at round 3175 is structural and that 91 val contexts cannot resolve gains below about 0.003. H444 moved selection onto a larger signal: family-grouped 5-fold CV over the 1,219 TRAIN contexts, with val91 used only as a one-shot regression check. Under that selection, it tested whether three training changes give a real lift: family-balanced sampling with gradient accumulation, SWA with cosine LR, and a higher LR with weight decay (basin escape).

## Setup
- **Code:** `h444_trainfold_cv.py` (CPU only, `nice 19`, 8 threads, micro 8, batch 1,024, slate 192, MemAvailable abort below 15 GB). It ran on xbabe1 against a scratch copy, `~/jope-data/hwave-h444`. The live run directory was never written to.
- **Start point:** incumbent `round-3175.pt` plus its optimizer state. Parity reproduces the live incumbent's val utility exactly (1.1389572403748425) on every seed.
- **DOPE frozen:** composite sha `c51a56d0…` before and after on every seed. Only the 29,211 adapter parameters train, and the optimizer excludes DOPE (guard asserted).
- **Data:** real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), 0 family overlap. Every row is asserted DL-Train or DL-Val, so no sealed or holdout data was touched (the aggregator re-checks this).
- **Selection:** 5 family-grouped folds over TRAIN (244, 244, 244, 244, 243 contexts), with no family in more than one fold and no val family in any fold. Out-of-fold paired gains vs the incumbent are pooled and CI'd by family bootstrap. **Strict:** mean OOF gain > 0.002 **and** CI lower bound > 0. Ties never count as wins. Utility is higher-is-better (negated fitness of the argmin program, −2.0 on a gate failure).
- **Arms:**
  - `fbal_accum`: family-balanced sampling, 4× accumulation, lr 1e-4, constant (2 optimizer steps at micro 8).
  - `swa_cos`: uniform sampling, warmup + cosine LR (peak 3e-4), SWA mean of the last half of the steps.
  - `hilr_wd`: family-balanced sampling, warmup + cosine LR (peak 1e-3), weight decay 1e-2.
- **Regression check:** the CV-best arm per seed is retrained on all TRAIN and scored once on val91.

## Outcome (OOF over 1,219 TRAIN contexts)
| Arm | Seeds | OOF gain per seed | Family-bootstrap 95% CI | Elo vs incumbent | W / T / L | Strict passes |
| --- | --- | --- | --- | --- | --- | --- |
| fbal_accum | 1, 2, 3 | +0.00021, +0.00030, +0.00021 | about [−0.0057, +0.0064] | −2.0, −1.7, −2.0 | 15–16 / 1,181–1,182 / 22 | 0/3 |
| swa_cos | 1, 3 (2 partial) | −0.0239, −0.0213 | [−0.044, −0.008], [−0.041, −0.006] | −11.4, −9.4 | 104–106 / 971–974 / 139–144 | 0/2 |
| hilr_wd | 1, 2, 3 | −0.078, −0.091, −0.087 | entirely below 0 (upper ≤ −0.042) | −47.3, −52.6, −50.8 | 233–241 / 566–576 / 406–418 | 0/3 |

**val91 regression check (one read each):**
- seed 1 `fbal_accum`: −0.0088, Elo −15.3 (0 W / 87 T / 4 L).
- seed 3 `fbal_accum`: −0.0035, Elo −7.6 (0 / 89 / 2).
- seed 2 `hilr_wd`: −0.0485, Elo −42.2 (19 / 42 / 30). This was not a CV choice, because the relaunch only had that arm.

None was `not_worse`. Peak RSS was 6.35 GB per seed, and the minimum MemAvailable on completed runs was 15.8 GB.

## Red-team
- **Seeds were not independent for `fbal_accum`.** Fold gains 1–4 are bit-identical on all three seeds, and seeds 1 and 3 match on fold 0 too, so only seed 2 fold 0 differs. The per-step losses *do* differ by seed, so the seed is wired into sampling (hash of tag/seed/step), slates (`np.random`) and modality dropout (torch RNG).
  - **Cause:** `fbal_accum` makes only 2 optimizer steps (micro 8 / accum 4) at lr 1e-4 from the **restored Adam state** (step 487, ‖m‖ 0.45 vs ‖√v‖ 0.25). With β1 0.9, the stored momentum supplies about 90% of each update direction, and it is the same on every seed. The seed-dependent part is a clipped 10% gradient share. The utility is a discrete argmin over a fixed pool, so the tiny differences rarely flip a program choice.
  - **Consequence:** the `fbal_accum` seed spread is not a noise estimate. Its verdict stands on the CI alone, and that CI straddles 0.
  - **Fix:** H445 adds a seeded init perturbation, seeds sampling, slates and dropout per fold, resets the Adam first moment, and asserts that per-seed fold gains are not all identical.
- **Seed 2 restart:** at 01:18Z (6:18 PM MST), seed 2 self-aborted on its MemAvailable < 15 GB guard during `swa_cos` fold 2, because other jobs on xbabe1 were using memory. Nothing was killed externally. It was relaunched at 01:40Z on `hilr_wd` only (`results/seed2_hilr`). Seed 2's `swa_cos` has folds 0–1 only (−0.0258, −0.0008), and these agree with seeds 1 and 3.
- **Metric direction and ties:** the aggregator recomputes `passes_strict` from the stored mean and CI. Ties score 0.5 in Elo and never count toward acceptance.

## Amendment (after merge): optimizer state leaked across folds
Building H445 turned up a second H444 bug. `opt.load_state_dict(ck["optimizer"])` does not copy tensors in torch 2.7: the new optimizer **aliases** the checkpoint's `exp_avg`, `exp_avg_sq` and `step` tensors, and AdamW updates them in place. So within each seed process, every fold, arm and the final regression-check retrain continued the Adam moments and step count left by the previous training, instead of restarting from the incumbent's state.
- This adds to the seed coupling above: every process starts from the same stored momentum and keeps reinforcing it.
- It mostly affects the arms that ran later in a process. `swa_cos` and `hilr_wd` on seeds 1 and 3 ran after `fbal_accum`. The seed-2 relaunch ran `hilr_wd` first, from fresh state, and still lost by the same amount (−0.091 vs −0.078 and −0.087). So the verdict does not change: still `NO_LIFT`.
- H445 deep-copies the optimizer state for every candidate and asserts that the stored step count (487) is unchanged.

## Meaning
Moving selection to TRAIN-fold CV works as an instrument: it resolves −0.02 effects with CIs clear of zero, which val91 cannot. But nothing in the optimiser or sampling space lifts the incumbent. Small updates stay in the same argmin basin, and larger ones (SWA, high LR) leave it for worse.

## Next step
H445 keeps the same CV and strict rule, with properly independent seeds. Its arms target the candidate choice rather than the optimiser schedule:
1. Ranking loss on larger slates.
2. Low-LR ASAM.
3. Lookahead with per-context loss normalisation.

No adoption nudge was sent to the flywheel owner.

Artifacts: `metrics.json` (per-arm, per-seed CV, fold gains, regression checks, seed-independence check), `h444_trainfold_cv.py`, `h444_aggregate.py`. The raw logs stay on xbabe1 in `~/jope-data/hwave-h444/results`.
