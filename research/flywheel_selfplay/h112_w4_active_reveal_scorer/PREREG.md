# H112 preregistration — W4 active reveal (two-plus-two) with the proposal-distribution scorer as prior

Fixed on 2026-10-31, before any H112 roster outcome. `run_h112.py` implements this design. Its frozen sha256,
recorded in every receipt, is authoritative; any difference from this text is a defect.

## Hypothesis

Choosing the second pair of observations from the first pair's outcomes beats a fixed top four at four total native
fits, **when the prior has signal on new programs**.

## Why now

- W4's [registered gate](../rd_breakthrough/PROPOSALS.md#w4--active-reveal-within-four-total-evaluations): "Advance
  over nonadaptive prior-top4 at matched fits and elapsed time. Kill if the advantage disappears when history
  information and construction costs match, or comes from access to an unrevealed cached label."
- [H103](../h103_w4_two_plus_two_scored/README.md) tested two-plus-two with the cached-pool program prior and found
  NO_ADVANCE. That prior tied random4 on new programs, so the update had no signal to refine.
- [H110](../h110_w9_scorer_replay_scored/README.md) found a proposal-distribution scorer that beats random4 by
  +269 Elo.
- **Disclosure:** using it here follows the H103/H110 outcomes on this roster. No H112 outcome has been seen.

## Design

| Item | Fixed value |
|---|---|
| Roster and split | Frozen [H67](../h67_w4_active_reveal_2plus2/README.md) roster: 50 DL-Train contexts, 44 families. Family-grouped W1 outer folds (sha256 `dd211134…`); 0 families span folds ([fold_manifest.json](fold_manifest.json)). |
| Seeds | Fresh **901, 902, 903**; 16 new proposals per context-seed (stream `h112-stream`, admitted pool excluded) |
| Prior training memory | The H106 memory (H94 + H103 reveal logs) plus every H110 native fit (`fits.jsonl` sha256 `239b9d26…`). All are recovered by stream replay and de-duplicated per (context, program): 2,306 rows. |
| Prior | The H106 recipe: features `[prog412 − champion, prog412]` → within-context utility percentile; HistGBM 200 / .05 / 15 leaves / l2 1, `random_state` 1120+k. One model per W1 fold k, trained only on contexts outside fold k. |
| Utility scale | `mu = b_k · score`. `b_k` is the OLS slope of clipped (±1) within-context-centred utility on within-context-centred **out-of-fold** score from inner folds j ≠ k. No intercept. |
| `s_top4` | Comparator and Elo anchor: the prior's top 4 |
| `s_2plus2` | Reveal the prior's top 2. Residuals `clip(u − mu, ±1)` are centred across the pair (level-free) and drive the merged H81 RBF-GP posterior mean on proj16 (lengthscale = median pairwise distance over the context's pool; noise ratio 1). Then reveal the top 2 of the rest. |
| `s_2plus2_swap` | Placebo: the same, with the two centred residuals exchanged |
| `random4` | Seeded uniform four (`h112-random`) |
| Labels | All 16 proposals are evaluated natively, so every arm is scored on a fully labelled batch. The adaptive pair is chosen after only the first pair's outcomes are revealed, and all picks are fixed before the remaining fits run. |
| Endpoint | Best utility among the arm's four picks (zero history). Seeds are averaged within context, then contexts within family. |
| Primary | `s_2plus2 − s_top4` under the common gate: one-sided .05 LCB > +.001, family median ≥ 0, ≤ 10% of families losing > .01. A 10,000-draw paired family bootstrap, seed 20261031. |
| Advance rule | Primary gate passes **and** `s_2plus2 − s_2plus2_swap` has a positive family mean (the revealed outcomes must matter) |
| Kill rule | NO_ADVANCE if the gate fails or the placebo matches. Matched cost holds by construction: every arm reveals exactly four programs from the same batch with the same history. |
| Secondary | `s_top4 − random4` (H110 primary A replicated on fresh seeds); median 95% CIs; W/T/L at tie 1e-4 |
| Elo | Canonical family-grouped Bradley–Terry, `s_top4` = 1500, tie 1e-4, prior .01, 400 iterations. No AUC. |
| Completeness | A verdict only on the full roster × 4 arms × 3 seeds panel, with no shortfalls or off-roster rows (as in H96) |
| Budget | ≤ 2,400 native fits, deadline 2026-11-01T18:00Z, xbabe2 (x86-64 evaluator), nice 15, load-gated at 85%, 0 GPU |

## Data scope

DL-Train only: logged native reveals, the admitted H50 pools, W1 train-only features and the H58 DL-Train champion
bank. No DL-Val, outer-test, sealed or holdout row is read. DOPE stays frozen at `c51a56d0` and is not loaded.
