# H118 — W6 cross-family transfer on new families: scored run of the H115 preregistration

Status: **NO_ADVANCE.** Conditioning the proposal scorer on the new family's dataset features adds nothing: X_SCORER −
G_SCORER = −0.0003, with 31 of 38 families tied. It also fails the context-shuffle placebo and the unseen-template
check, so W6's context-transfer hypothesis is rejected under its own kill rule.

The **generic proposal scorer does transfer to families outside the training roster**: +0.055 over random4, LCB
+0.017, harm 2.6%, about +136 Elo. Current retrieval ranks below random again.

## Protocol

This run executes [H115's preregistration](../h115_w6_xfamily/PREREG.md), merged in #1027, with the merged runner
(`run_h114.py` sha256 `ed7b30a0…`; the frozen `h114` identifiers are explained there).

- **Panel.** 38 DL-Train contexts, one per family, all outside the H67 roster. 0 cross-family aliases. Seed 1001.
- **Programs.** 16 new proposals per context, all evaluated natively: **608 fits, 0 failures**. W6's native-fit total
  is 1,180, within its 1,200 ceiling.
- **Training memory.** 2,306 logged outcomes from 50 roster contexts. The scorers and panel manifest are
  byte-identical to the dry run's (`fd567b9f…`, `0afd4bfe…`).

## Outcome (38 families; completeness gate passed)

| Contrast | Mean [95% CI] | Adj. LCB | Median [95% CI] | Losing > .01 | W/T/L | Gate |
|---|---|---:|---|---:|---|---|
| **X_SCORER − G_SCORER** (primary) | **−0.0003** [−0.0035, +0.0037] | −0.0035 | 0.0000 [0, 0] | 7.9% | 2/31/5 | **fail** |
| **X_SCORER − RETR** (primary) | +0.0946 [+0.0161, +0.2371] | +0.0161 | +0.0039 [+0.0001, +0.0225] | 2.6% | 25/9/4 | pass |
| X_SCORER − X_SHUF (placebo) | −0.0024 | −0.0083 | 0.0000 | 5.3% | 3/32/3 | — |
| **G_SCORER − random4** (secondary) | **+0.0551** [+0.0125, +0.1157] | **+0.0167** | +0.0005 [0, +0.0104] | 2.6% | 19/12/7 | thresholds met |
| PROG_PRIOR − random4 | +0.0201 [−0.031, +0.091] | −0.0249 | 0.0000 | 39.5% | 15/5/18 | — |
| RETR − random4 | −0.0398 [−0.141, +0.027] | −0.1223 | −0.0001 | 39.5% | 14/5/19 | — |

The primary LCBs are one-sided at .025 (Bonferroni m = 2); the secondary LCBs are one-sided at .05.

**Unseen-template panel.** 37 of 38 families qualify, with on average 12.6 of 16 proposals using templates no donor
row has. On it, X_SCORER − G_SCORER = −0.0004, so the check **fails**.

| Arm | Family-mean best of four | Family Elo (G_SCORER = 1500) |
|---|---:|---:|
| X_SHUF | 1.5823 | 1471.2 |
| G_SCORER | 1.5802 | **1500.0** |
| X_SCORER | 1.5799 | 1476.2 |
| PROG_PRIOR | 1.5452 | 1306.9 |
| random4 | 1.5251 | 1363.6 |
| RETR | 1.4853 | 1286.0 |

## Verdict and reading

- **W6's context-transfer hypothesis is rejected.** Adding the new family's `lm48_z` features to a scorer trained on
  other families' outcomes neither helps nor beats its shuffled-context placebo, and it fails on unseen templates.
  Together with H83/H92 (cached) and H94 (prospective), context conditioning has now failed every prospective test.
- **What does transfer is the generic proposal scorer.**
  - Trained only on roster families, it picks better new programs than random on 38 entirely new families. The mean
    is +0.055 with LCB +0.017, the harm tail is 2.6%, and it wins 19 families against 7.
  - This is the out-of-roster confirmation that H110 asked for. It is secondary evidence here: preregistered, but not
    a primary contrast.
  - The median gain is small (+0.0005) because 12 of 38 families tie. Most of the advantage is in a minority of
    families with large gains.
- **Current retrieval and the cached-pool prior do not help on new families.** Retrieval sits below random (−0.040),
  matching H83, and the program prior ties random (harm 39.5%), matching H103.

## Red-team checks

- **Preregistration came first.** #1027 merged before the run, and the runner hash equals main's. The 16-fit dry
  run's context is outside the panel.
- **Out of family.** No panel family is in the roster, and the W1 manifest records 0 cross-family table and dataset
  aliases. Exact-program overlap between panel proposals and donor rows is 34 of 608 programs (5.6%). Those outcomes
  are context-specific, so this is not leakage.
- **Complete panel.** 0 execution errors and 0 shortfalls.
- **Elo, not AUC.** Ties are scored 0.5, and 31 of 38 X/G families tie.
- **Data scope.**
  - DL-Train only.
  - Zero DL-Val, outer-test, sealed or holdout reads.
  - DOPE is not loaded, and no GPU is used.

## Next step

Make the **generic proposal scorer** the candidate for an owner-gated live-loop ablation: rank the live proposer's
candidates by it, rather than by raw retrieval or the cached prior. It has now beaten random on held roster families
(H110, H116) and on new families (H118). Context conditioning should be dropped from W6.

## Evidence

Run `opus-h115-20261031a` used xbabe2 at nice 15 with 12 workers: 2:07 wall, 608 native fits and zero GPU.
[metrics.json](metrics.json) holds every number above and the S3 object versions. The objects live under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W6/opus-h115-20261031a/`: manifest, metrics,
run receipt, panel manifest, scorers, rows, census, fits, ledger, logs and the runner. Each was written create-only
with a version-pinned readback hash.
