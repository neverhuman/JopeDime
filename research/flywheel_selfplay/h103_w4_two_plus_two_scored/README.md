# H103 — W4 two-plus-two on new programs: scored CPU run of the H96 preregistration

Status: **NO_ADVANCE.** The full panel completed with no gaps. Adaptive two-plus-two does not beat prior-top4 on new
programs: +0.0145 [−0.007, +0.047], +7.9 Elo, median 0. It is indistinguishable from its residual-swap placebo
(+0.0011). **On new proposals the program-only prior does not beat random4** (−0.051; random4 is +4.6 Elo against it).

## Hypothesis and protocol

This run executes [H96's preregistration](../h96_w4_active_reveal/PREREG.md), merged in #964 before any roster
outcome, with the runner exactly as merged (`run_w4.py` sha256 `beeb76b1…`, the same value in the run and analyze
receipts).

- **Hypothesis.** Revealing the prior's top two and updating a GP posterior with their outcomes picks a better second
  pair than a fixed prior-top4, at four total native fits.
- **Roster.** The frozen [H67](../h67_w4_active_reveal_2plus2/README.md) roster: 50 DL-Train contexts in 44 families,
  seeds 701–703.
- **Programs.** 16 new proposals per context-seed from the pinned live proposer.
- **Prior.** The H94 program-only W1-recipe refit (`5938a354…`), plus the context's admitted-pool mean.
- **Arms.**
  - `prog_top4`: the comparator and Elo anchor.
  - `prog_2plus2`: the merged H79/H81 RBF-GP update on proj16.
  - `prog_2plus2_swap`: a placebo with the two residuals swapped.
  - `random4`.
- **Endpoint.** The best utility among each arm's four new reveals (zero history), with seeds averaged within context
  and contexts within family.
- **Gate and statistics.** The common gate and a 10,000-draw family bootstrap (seed 20261031). The completeness gate
  allows a verdict only on the full roster × arm × seed panel.

## Outcome (44 families, 150 / 150 context-seeds complete)

| Check | Result |
|---|---|
| Prior-signal guard (cached roster pools) | `prog_top4` 1.8187 against `random4` 1.5048: **pass** |
| Completeness gate | 0 panel gaps, 0 shortfalls, 0 off-roster contexts, 0 incomplete context-seeds |
| Native fits | 850 (cap 1,800), 0 failures, 0 timeouts; 1,124 s summed native wall |

| Contrast | Mean [95% CI] | One-sided LCB | Median [95% CI] | Losing > .01 | W/T/L | Gate |
|---|---|---:|---|---:|---|---|
| **prog_2plus2 − prog_top4** (primary) | **+0.0145** [−0.0071, +0.0470] | −0.0050 | 0.0000 [0, 0] | 9.1% | 9/27/8 | **fail** (LCB) |
| prog_2plus2 − prog_2plus2_swap | +0.0011 | −0.0043 | 0.0000 [0, 0] | 4.5% | 8/28/8 | — |
| prog_top4 − random4 | −0.0508 | −0.1339 | −0.0005 [−0.0055, +0.0021] | 31.8% | 18/2/24 | — |

| Arm | Family-mean best of four | Family Elo | Native wall (s) |
|---|---:|---:|---:|
| prog_top4 | 1.8474 | 1500.0 | 900 |
| prog_2plus2 | 1.8619 | 1507.9 | 911 |
| prog_2plus2_swap | 1.8608 | 1505.9 | 902 |
| random4 | **1.8982** | 1504.6 | 734 |

Elapsed cost is matched within 1.3% across the prior arms. random4 is cheaper because it reveals fewer gate-failing
programs (1.3% against 2.8%).

## Verdict and reading

- **W4 does not advance.** The preregistered decision needs the primary gate *and* a positive placebo contrast. The
  primary fails on its LCB, and the placebo contrast is +0.0011 with W/T/L 8/28/8. Two-plus-two changed the revealed
  set in 75 of 150 context-seeds, and its placebo differed from it in 46. When the picks did change, the content of
  the outcomes did not matter.
- **The program prior has no usable signal among new proposals.** On cached pools the same prior beats random by
  +0.314. Among the proposer's 16 new programs per context, a uniform pick is better on mean (−0.051 for the prior)
  and roughly even on families (18/2/24). This extends H94: both the program-only and the context W1-recipe models
  rank admitted pool programs well but not the proposer's near-champion edits. Within one context's proposal batch,
  the features they use barely vary in a way that predicts outcome.
- **Implication.** The flywheel cannot rely on cached-pool priors to choose which new proposals to evaluate. W4's
  premise of adaptively spending a small reveal budget needs a scorer trained on the proposal distribution, for
  example from logged proposal reveals such as H94, H95 and H103's own 850 fits.

## Red-team checks

- **Preregistration before outcomes.** H96 (#964) merged at 10:46:49Z. This run started afterwards from an archive of
  the merged main (`b2c04a84`), and its runner hash equals main's. No roster outcome was used to design the run.
  Dry-run contexts came from non-roster families.
- **Complete panel.** There are no gaps, no shortfalls and no execution errors, so the verdict is the gated one, not
  a surviving subset.
- **Leakage.**
  - Every context is scored by the fold model that excluded its whole family.
  - The adaptive arms see only their own first-pair outcomes, through the `reveal` callback.
  - Zero DL-Val, outer-test, sealed or holdout reads.
  - DOPE is not loaded, and no GPU is used.
- **Elo, not AUC.** Ties count as 0.5 and are reported separately.
- **The random4 result is not an artifact of failures.** random4 has fewer gate failures (1.3% against 2.8%), but its
  advantage is broad: it wins 24 of 44 families and loses 18 (2 ties), and at the context level it wins 27 of 50
  and loses 21. The prior is not merely being dragged down by −2 utilities.
- **Not run.** The `r882_top4` continuity arm stays queued after H57 → Fused D → W1 (see H96 `GPU_QUEUE.md`). It is
  descriptive only and does not affect this verdict.

## Next step

1. **Close W4's prior-driven variant.** Record W4 two-plus-two as NO_ADVANCE on new programs. The r882 GPU arm, when
   its turn comes, is continuity evidence only.
2. **A proposal-distribution scorer.** H94, H95 and H103 together hold about 1,860 native fits of proposer
   programs on the H67 roster. A scorer trained on such reveals, with whole-family held-out evaluation, is the
   precondition for any W4/W6 learner to beat random selection on new programs.

## Reproduce

```sh
cd research/flywheel_selfplay/h96_w4_active_reveal
nice -n 15 python3 -I run_w4.py run --out ~/jope-data/h103/<run_id> --run-id <run_id> \
  --prog-models <H94 models_prog.pkl, sha256 5938a354...> --workers 12
nice -n 15 python3 -I run_w4.py analyze --out ~/jope-data/h103/<run_id> --run-id <run_id> --prog-models <same>
```

## Evidence

Run `opus-h103-20261031a` used xbabe2 at nice 15 with 12 fit workers: 9:47 wall and 173 MB peak, with zero GPU.
[metrics.json](metrics.json) holds every number above and the S3 object versions. The objects live under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W4/opus-h103-20261031a/`: manifest, metrics,
run receipt, rows, fits, ledger, shortfalls, logs and the runner. Each was written create-only with a version-pinned
readback hash.
