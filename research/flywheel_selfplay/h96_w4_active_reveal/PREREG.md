# H96 preregistration — W4 active reveal (two-plus-two) on new programs

Fixed on 2026-10-31, before any W4 arm outcome on the roster. The runner `run_w4.py` implements exactly this; where the
two differ, the runner's frozen source (sha256 recorded in every receipt) is authoritative and the difference is a defect.

## Question

W4 ([PROPOSALS.md](../rd_breakthrough/PROPOSALS.md#w4--active-reveal-within-four-total-evaluations)): does choosing the
second pair of observations from the first pair's outcomes beat a fixed prior-top4 at four total fits? The verbatim
gate: advance over nonadaptive prior-top4 at matched fits and elapsed time; kill if the advantage disappears when
history information and construction costs match, or comes from access to an unrevealed cached label.

## Why this comparator

[H81](../h81_w4_prospective_2plus2/README.md) built the prospective stratum with the W1 context-aware GBM and stopped
before scoring, because that comparator could not be reproduced. [H94](../h94_w6_prospective_hedge/README.md) then
found the program-only W1-recipe refit stronger than the context refit on new programs (the context model lost by
−51.9 Elo). H96 uses the program-only refit as the W4 comparator and the base of the adaptive arms. **Disclosure:**
that choice was informed by H94's outcome on the same roster. No W4 arm outcome on the roster has been observed.

## Design

| Item | Fixed value |
|---|---|
| Roster | Frozen [H67](../h67_w4_active_reveal_2plus2/README.md) roster: 50 DL-Train contexts, 44 families, each family inside one W1 fold |
| Seeds | 701, 702, 703 |
| Programs | 16 new proposals per context-seed from the pinned live proposer (`rust_evaluator.py` sha256 `53422343…`), stream `h96-stream`, excluding every program in the admitted pool. Context-seeds with < 8 proposals are shortfalls. |
| Prior | `mu = f_k(prog412) + mean admitted-pool utility of the context`. `f_k` is the H94 program-only refit for the W1 outer fold that excluded the context's family (`models_prog.pkl`, sha256 `5938a354…`). |
| History | Zero: the CPU arms see no outcome before their first reveal |
| `prog_top4` | Comparator and Elo anchor: the prior's top 4 |
| `prog_2plus2` | Reveal the prior's top 2. Their residuals `u − mu` update an RBF GP posterior mean on proj16 (lengthscale = median pairwise distance over the context's full pool, noise ratio 1). Reveal the top 2 of the rest. This is the merged H79/H81 rule, imported and not copied. |
| `prog_2plus2_swap` | Placebo: the same, with the two residuals swapped |
| `random4` | Four uniformly random proposals (`h96-random` seed) |
| `r882_top4` (GPU, queued) | The r882 checkpoint copy scores the same 16 proposals given the four hash-ordered pool anchors. It has more history than the CPU arms, so it is descriptive continuity only, never the comparator. See [GPU_QUEUE.md](GPU_QUEUE.md). |
| Endpoint | Best utility among the arm's four new reveals (native original-cut training CV, `bootstrap.s1main` sha256 `e3be9386…`). Utility is `−fitness` on gate pass, else −2. An execution error makes the arm INCOMPLETE. |
| Completeness gate | A verdict is issued only on the full panel: exactly one completed row per preregistered seed for every roster context × arm, no off-roster context, no proposal shortfall and no incomplete context-seed. Otherwise `analyze` reports INCOMPLETE with the gap counts and **no verdict**. A surviving subset is never scored. Execution errors are stored by the fit store, so a deterministic failure keeps the panel INCOMPLETE; any rerun policy is a separate, preregistered owner decision. |
| Aggregation | Seeds averaged within context, then contexts within family; families weighted equally |
| Primary | `prog_2plus2 − prog_top4`, under the common gate: one-sided 5% bootstrap LCB > +.001, family median ≥ 0, ≤ 10% of families losing > .01. Paired family-cluster bootstrap with 10,000 draws, seed 20261031 and one shared index matrix. |
| Secondary | `prog_2plus2 − prog_2plus2_swap` (does the outcome content matter?) and `prog_top4 − random4` (prior signal); median 95% CIs; W/T/L at tie 1e-4 |
| Elo | Canonical family-grouped Bradley–Terry (`elo_tracker.py` sha256 `9ece9832…`), `prog_top4` = 1500, tie 1e-4, prior .01, 400 iterations. No AUC. |
| Cost | Native wall and child CPU per arm, plus selection CPU; fit counts per arm |
| Prior-signal guard | Before any proposal is fitted, zero-history `prog_top4` must beat `random4` on the roster's cached pools (family-mean best of four). Otherwise the run stops. |
| Budget | ≤ 1,800 native fits (W4 ceiling 2,400 cumulative; H81 spent 0), deadline 2026-11-01T12:00Z, xbabe2 only (x86-64 evaluator), nice 15, load-gated at 85% |

## Decision

- **ADVANCE** to W4's next stage, the held-family and matched-elapsed-time check, if the primary passes the common gate
  *and* `prog_2plus2 − prog_2plus2_swap` has a positive family mean. Otherwise the gain does not come from the revealed
  outcomes.
- **NO_ADVANCE** otherwise. A pass is development evidence on fit-only contexts. It does not authorize a live change.

## Data scope

DL-Train only: the admitted H50 pools, W1 train-only context features (proposer landmarks) and the H58 DL-Train
champion bank. No DL-Val, outer-test, sealed or holdout rows are read. DOPE stays frozen at `c51a56d0`; the CPU arms
do not load it.
