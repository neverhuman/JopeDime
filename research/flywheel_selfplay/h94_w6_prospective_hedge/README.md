# H94 — W6 prospective test of H92's HEDGE_CTX on new programs

Status: **NO_ADVANCE.** On programs never evaluated before, the hedged context learner does **not** beat the generic
program-only prior: −0.0105 U4 and −51.9 Elo. The unhedged context model is worse still, at −73.2 Elo. H92's
cached-screen gain does not transfer to new proposals.

## Hypothesis

[H92](../h92_w6_downside_controlled/README.md) found, on cached development pools, that hedging W1's context model
clears the common gate: keep the program-only prior's top-2 picks and add the context model's top-2 others. The
result was +0.036 U4 and +50.5 Elo over the prior. Because H92 was adaptive development on H83's families, its next
step was a prospective test: **on new programs, HEDGE_CTX should beat the program-only prior under the common gate.**

## Setup (fixed in the runner before any H94 outcome)

- **Roster.** The frozen [H67](../h67_w4_active_reveal_2plus2/README.md) roster: 50 DL-Train contexts in 44 families,
  each family inside one W1 fold. Seeds 701–703.
- **New programs.** For each context and seed, 16 new proposals from the pinned live proposer (the H81 generator with
  its own stream), excluding every program already in the admitted pool. Shortfalls: 0 of 150 context-seeds.
- **Scorers.** Two W1-recipe refits per W1 outer fold, identical except for their features:
  - `PRIOR`: program-only (prog412), `random_state` 700+k. Fitted here, sha256 `5938a354…`.
  - `CTX`: prog412 + lm48_z + dope629, `random_state` 800+k. This is the H81 refit `models.pkl`, sha256 `c8bfbe94…`.
  - A context is scored only by the model of the fold that excluded its whole family. Using two refits keeps the
    comparison internally consistent, so W1's lost fitted models are not needed.
- **Arms.** Each arm reveals four new programs by native original-cut training CV, after the four hash-ordered pool
  anchors:
  - `PRIOR_top4`: the prior's top 4.
  - `CTX_top4`: the context model's top 4.
  - `HEDGE_CTX`: the prior's top-2 plus the context model's top-2 among the rest.

  The endpoint is the best utility over the anchors and the arm's four new evaluations. An execution error makes an
  arm INCOMPLETE; there were none.
- **Primary.** HEDGE_CTX − PRIOR_top4 under the common gate: one-sided 5% LCB > +.001, family median ≥ 0, and at most
  10% of families losing more than .01.
- **Statistics.** Ten thousand family-cluster bootstrap draws (seed 20261031) give median and mean intervals. Elo is
  the canonical family-grouped Bradley–Terry, with PRIOR_top4 = 1500.
- **Consistency guard.** Before any proposal was scored, the refits had to reproduce H92's direction on the roster's
  cached pools. They did: HEDGE_CTX − PRIOR was **+0.0415** (1.9065 vs 1.8650), so the guard passed.
- **Budget.** At most 1,200 native fits. 556 were used (the arms share picks), with 0 failures and 0 timeouts.

## Outcome (44 families, 150 complete context-seeds)

| Contrast | Mean | 95% CI (mean) | One-sided LCB | Median [95% CI] | Losing > .01 | W/T/L | Gate |
|---|---:|---|---:|---|---:|---|---|
| **HEDGE_CTX − PRIOR_top4** (primary) | **−0.0105** | [−0.0401, +0.0068] | −0.0369 | 0.0000 [−0.0015, 0.0000] | 13.6% | 13/11/20 | **fail** |
| CTX_top4 − PRIOR_top4 | −0.0217 | — | −0.0715 | −0.0001 [−0.0031, 0.0000] | 22.7% | 15/7/22 | fail |
| HEDGE_CTX − CTX_top4 | +0.0112 | — | −0.0226 | 0.0000 [0.0000, 0.0000] | 6.8% | 12/24/8 | fail |

| Arm | Family-mean retained best | Mean best new program (descriptive) | Family Elo |
|---|---:|---:|---:|
| PRIOR_top4 | **2.0809** | 1.8024 | **1500.0** |
| HEDGE_CTX | 2.0704 | 1.7977 | 1448.1 |
| CTX_top4 | 2.0592 | 1.7642 | 1426.8 |

## Verdict and reading

- **The cached advantage reverses on new programs.** On the same roster, the refits give HEDGE_CTX +0.0415 over PRIOR
  among cached pool programs. Among new proposals, they give −0.0105. The context model ranks admitted pool programs
  well, but it does not rank the proposer's new programs better than the program-only prior. Those programs are
  mostly near-champion edits, a different distribution from the pool.
- **Hedging still limits the damage, as designed.** HEDGE_CTX loses less than CTX_top4 (−0.0105 vs −0.0217, harm 13.6%
  vs 22.7%), but that is protection from a worse scorer, not a gain.
- **W6 context conditioning has no prospective support so far.** Across H83, H92 and H94, context-conditioned scorers
  help only on cached pools, where H92's design was adaptive. The first new-program test does not support routing
  live candidate ranking through the context model.

## Red-team checks

- **Commensurable endpoint.** Anchor utilities come from the admitted pools, and new utilities come from native
  original-cut fits. To check that they are comparable, 10 pool programs (the roster champions present in
  `champions.json`) were refit natively: **all 10 reproduce the stored pool utility exactly.** The check script and
  its output are in S3 (`anchor_check/`).
- **Out of fold by family.** Every score comes from the fold model that excluded the context's whole family, and the
  roster keeps every family within one W1 fold.
- **No leakage from outcomes into selection.** Selection uses only model predictions on proposal features; no native
  result for the run is read before all picks are fixed.
- **Data scope.**
  - DL-Train only: admitted training pools, W1 train-only context features, and the H58 DL-Train champion bank for
    the proposer.
  - Zero DL-Val, outer-test, sealed or holdout payload reads.
  - DOPE c51a56d0 is not loaded, and no GPU is used.
- **Elo, not AUC; ties are not wins.** |Δ| ≤ 1e-4 counts as a tie, reported separately and scored 0.5.
- **Power.** 44 families is the W4/H67 roster size. The primary's 95% interval [−0.040, +0.007] excludes H92's
  cached effect (+0.036), so this is a reversal, not only a lack of power.
- **Apparent HWMs.** 45 revealed programs in 20 contexts beat their pool's best fitness on the single original cut.
  The pointer appended to `HWM_CANDIDATES.jsonl` is version `USIe0IJE…`, through a conditional put with the prefix
  preserved. These are apparent only, not confirmed on resplits. `found_by` names the first arm in `ARMS` order that
  revealed the program, because the arms share picks.

## Next step

1. **Do not adopt context-model ranking for live proposals.** If W6 continues, it should train and validate on
   proposer-distribution programs, for example a W4-style reveal log, rather than on admitted pools.
2. **Confirm the 45 apparent HWMs on resplits** (a W8-style protocol) before anyone treats them as improvements.
3. The program-only prior remains the comparator to beat for new-program selection.

## Reproduce

```sh
python3 -m pytest -q research/flywheel_selfplay/h94_w6_prospective_hedge/test_run_prospective_hedge.py   # 5 synthetic tests
cd research/flywheel_selfplay/h94_w6_prospective_hedge
OMP_NUM_THREADS=8 nice -n 15 python3 -I run_prospective_hedge.py --out ~/jope-data/h94/<run_id> --run-id <run_id> \
  --ctx-models <H81 refit models.pkl, sha256 c8bfbe94...> --workers 12 --threads 8
```

The runner imports the merged H81 helpers (pins, inputs, native fit, ledger), pinned by full sha256, and refuses
changed bytes.

## Evidence

Run `opus-h94-20261031a` used xbabe2 at nice 15 with 12 fit workers: 4:56 wall, 2.6 GB peak, 556 native fits and
zero GPU. The source sha256 recorded in the metrics equals the committed runner. [metrics.json](metrics.json) holds
every number above and the S3 object versions. The objects live under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W6/opus-h94-20261031a/`: manifest, metrics,
rows, fits, ledger, guard, apparent HWMs, the program-only models, logs, sources and the anchor check. Each was written
create-only with a version-pinned readback hash.
