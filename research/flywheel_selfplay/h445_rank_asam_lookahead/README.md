# H445: candidate-choice arms under train-fold CV on the flywheel plateau (offline, adapter-only)

**Verdict: `NO_LIFT`.** No arm passed the strict rule on any seed, and every val91 regression check was worse than the incumbent. `rank_slate` is clearly worse, `asam_lowlr` is flat, and `look_ctxnorm` is a small positive whose CI spans 0 on every seed.

## Hypothesis
H444 showed that optimiser and sampling changes do not move the round-3175 incumbent. H445 kept the same instrument and targeted the candidate choice instead. It tested whether a ranking loss on larger slates, low-LR adaptive SAM, or Lookahead with per-context loss normalisation could change which program the argmin picks, for a real lift.

## Setup
- **Code:** `h445_trainfold_cv.py` (CPU only, `nice 19`, 8 threads, batch 1,024, slate 192, MemAvailable abort below 15 GB). It ran on xbabe1 against a scratch copy, `~/jope-data/hwave-h445`. The live run directory was never written to.
- **Start point:** incumbent `round-3175.pt` (sha `1d6ea900…`) plus its optimizer state, deep-copied for every candidate (the H444 aliasing fix). Parity reproduces the live val utility exactly (1.1389572403748425) on every seed.
- **DOPE frozen:** composite sha `c51a56d0…` before and after on every seed. Only the 29,211 adapter parameters train, and the optimizer excludes DOPE (guard asserted).
- **Data:** real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), 0 family overlap. All rows are DL-Train or DL-Val, so no sealed or holdout data was touched.
- **Selection:** the same as H444: 5 family-grouped TRAIN folds (244/244/244/244/243), pooled out-of-fold paired gains, and a family-bootstrap 95% CI. **Strict:** mean > 0.002 **and** CI lower bound > 0, with ties never counted as wins. Utility is higher-is-better (negated fitness of the argmin program, −2.0 on a gate failure). Elo is computed vs the incumbent with ties at 0.5.
- **Seed independence (H444 fix):** a seeded relative init perturbation (2e-3 × RMS) of the adapter, a reset of Adam's first moment, and sampling, slates and dropout keyed on (seed, arm, fold, step). Trained adapter hashes are asserted different across seeds.
- **Arms** (lr 5e-5, wd 1e-4, grad clip 1.0):
  - `rank_slate`: 384-program slates (best 96 plus 288 random), batch 512, plus a top-1 listwise CE (w 0.5), 8 steps.
  - `asam_lowlr`: adaptive SAM (rho 0.5), 4 steps × 2 passes.
  - `look_ctxnorm`: Lookahead (k 4, α 0.5) over AdamW with per-context loss normalisation, 8 steps.
- **Regression check:** the CV-best arm per seed is retrained on all TRAIN and scored once on val91.

## Outcome (OOF over 1,219 TRAIN contexts)
| Arm | OOF gain per seed (median) | Family-bootstrap 95% CI per seed | Elo vs incumbent per seed (median) | W / T / L | Strict passes |
| --- | --- | --- | --- | --- | --- |
| rank_slate | −0.0095, −0.0092, −0.0155 (−0.0095) | [−0.022, +0.0001], [−0.021, +0.0012], [−0.033, −0.0036] | −4.6, −6.0, −6.6 (−6.0) | 30–33 / 1,136–1,137 / 49–53 | 0/3 |
| asam_lowlr | +0.0021, +0.0003, −0.0006 (+0.0003) | [−0.00003, +0.0068], [−0.0002, +0.0009], [−0.0016, +0.00005] | +0.6, +0.3, −0.9 (+0.3) | 2–4 / 1,212–1,213 / 2–5 | 0/3 |
| look_ctxnorm | +0.0006, +0.0016, +0.0019 (+0.0016) | [−0.0052, +0.0068], [−0.0039, +0.0076], [−0.0038, +0.0081] | −1.4, −1.1, −0.6 (−1.1) | 18–19 / 1,177–1,179 / 21–23 | 0/3 |

**val91 regression check (one read per seed):**
- seed 1 `asam_lowlr`: −0.0035, Elo −7.6 (0 W / 89 T / 2 L).
- seeds 2 and 3 `look_ctxnorm`: −0.0088, Elo −15.3 (0 / 87 / 4), the same on both.

None was `not_worse`. Peak RSS was 6.24 GB per seed, and the minimum MemAvailable was 18.5 GB.

## Red-team
- **Seed independence: the weights are independent, but the decisions are coupled.** Every (arm, fold) has 3 distinct adapter hashes, and the aggregator's fold-gain-vector check passes. That rules out a code bug like H444's. Even so, the argmin decisions barely depend on the seed:
  - `look_ctxnorm` fold 4 flips the *same* 10 of 243 contexts on all three seeds (hence the identical −0.002265).
  - `asam_lowlr` changes only 0–3 contexts per fold.
  - Seeds disagree on at most 6 contexts in any fold.
  - Small moves of the adapter cross the same few fragile decision boundaries. So seed spread is **not** a noise estimate here, and the verdict rests on the family-bootstrap CIs. Per-fold counts are in `metrics.json` → `red_team.decision_diversity`.
- **Val91 regcheck repeats H444 exactly.** H445's −0.003518 (0/89/2) and −0.008828 (0/87/4) are the *same* values as H444's `fbal_accum` regchecks. Near the incumbent, val91 has only a couple of reachable outcomes, which are 2 or 4 fragile contexts flipping to worse programs. The H445 regchecks did not store adapter hashes, so seeds 2 and 3 being identical cannot be told apart from a same-weights case at the hash level. H446 stores and asserts them.
- **Revealed-history mismatch (new; inherited from the live loop).** The adapter is shown the fitness and gate of the first 4 slate programs as revealed history.
  - Training slates (`automatic_promotion.training_indices`, reused here) put the true best 48 first, so training always reveals the pool's top 4.
  - Eval (`utility`) reveals pool indices 0–3, which sit at a median fitness rank of about 554 of 1,152 after journal replay (only 4% are in the top 48).
  - The model is therefore trained on a revealed-history distribution it never sees at eval. This is a plausible structural cause of the plateau, and H446 tests it.
- **Mid-run crash:** at the first `asam_lowlr` step, all 3 seeds crashed on `p.grad None`, because some adapter params get no gradient on a batch. The patch skips no-grad params (backup `h445_trainfold_cv.py.bak_pre_samfix`, logs `run_crash1.log`). `rank_slate` folds resumed from `fold_sig`, and the keyed RNG streams make the resumed folds identical to a fresh run.
- **Other checks:** the metric direction and the strict rule are recomputed in the aggregator, and ties never count. There was no leakage (family-disjoint folds, val families excluded from folds) and no sealed or holdout use. DOPE was intact on every DONE line, there was no OOM, and disk was fine (xbabe1 at 87%).

## Meaning
Loss-shape and optimizer changes around the incumbent stay in its argmin basin, and the few contexts they flip are mostly the same fragile ones. The plateau is not an optimisation problem. The next lever is what the model is trained to see.

## Next step
H446 keeps the same instrument and tests the revealed-history channel:
1. `reveal_match`: training reveals pool indices 0–3, exactly as eval does.
2. `reveal_rand`: training reveals 4 random programs.
3. `fail_hinge`: a margin hinge on gate-failing programs, using the incumbent's predicted top 16 as hard negatives.

No adoption nudge was sent to the flywheel owner.

Artifacts: `metrics.json` (per-arm, per-seed CV, fold gains, regression checks, decision diversity), `h445_trainfold_cv.py` (as run, after the ASAM patch), `h445_aggregate.py`. The raw logs stay on xbabe1 in `~/jope-data/hwave-h445/results`.
