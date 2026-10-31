# H68 — W9 causal flywheel closeout preparation

Status: CPU-only preparation. No W9 experiment, outcomes, model fit, GPU work, or statistical claim is included. H68 is the lane claim for the preparation files in this directory; it does not reserve an experiment run.

## Hypothesis (verbatim from `rd_breakthrough/PROPOSALS.md`, W9)

> updated guidance acquires better data, and those data causally improve the next guidance model at equal resource cost.

## W9 advance gate (verbatim)

> **Metrics / success / kill:** raw family utility, Elo, U4 and actual optimizer/parameter changes. Require fresh+updated to beat both one-sided controls and the strongest non-neural matched baseline, and require its memory to improve the common learner over uniform-acquisition memory. Report the 2×2 interaction, both cycle increments and all costs. A positive interaction alone is insufficient; a higher pool oracle without learner improvement is not a neural flywheel. If the two directions of benefit are not both supported, retain the narrower search or supervised-learning result and reject the breakthrough claim.

Common advance threshold (verbatim):

> The proposed advance gate is a positive adjusted family-mean lower bound above **+.001**, median family delta ≥0, and at most 10% of families losing more than .01, against the strongest matched control. A **+5 Elo** point gain is a practical target, not a forecast or substitute for raw utility.

Any future scored result also follows the common contract: regression only; exact admitted training-row and family closure; five family folds with seed 4900; fixed experiment seeds 701–703; matched initialization, data, updates, budgets and inference except the intervention; one fixed terminal checkpoint per cell; paired family-cluster bootstrap of 10,000 draws with seed 20261031; all failures, unknowns and exclusions reported. There are zero new DL-Val, outer-test, or sealed reads. Unknown row closure refuses admission.

## Design to be tested after upstream gates

W9 crosses frozen/updated ranker with frozen/fresh memory (four cells). Frozen-memory cells retain outcomes only in a shadow log. Every cell has the same acquisition budget. The strongest W1 non-neural policy, including the context-aware tree if it wins, is a comparator. Each produced memory is replayed into the same training recipe once with the neural learner and once with the strongest W1 tree learner, matching row counts, status mix, fit budget and optimizer updates.

Upstream dependencies consumed: W1 supplies the selected development recipe, the strongest non-neural comparator and the context-aware-tree disposition. W2/W8 provide diagnostic constraints, not pooled outcomes. W3–W7 artifacts or outcomes may enter only if their source rows are admitted under the same common closure and each cell receives identical initial memory; absent or unverified artifacts remain absent. Do not substitute legacy S3 retained scores: they are not eligible raw inputs without a separate role-verified, field-allowlisted export. The W9 run additionally requires a frozen development recipe, fit-only timing pilot, admitted prospective training outcomes and full per-row provenance before any model work.

The proposal ceiling is seeds 701–703, five family folds, two acquisition/learning cycles, at most 9,600 native calls, four GPU-hours and four CPU-hours. Scale the predeclared context-round roster down prospectively if the timing pilot cannot fit. No live deployment, chosen peak checkpoint, or promotion follows from this gate.

## CPU smoke boundary

`runner.py dry-run` reads only the pinned public dataset-corpus manifest and its content-addressed **train** blob for catalog entry `b3b41e3413953c73` (25 rows, 840 columns; train SHA256 `fc765fe9d309884cebe08fb9e2be0d9a6df219f1407a05684badcfee609d1ccf`). It verifies the manifest identity, train blob hash, byte count and rectangular CSV shape, then discards the bytes. It never requests the test blob, does no fitting or scoring, and emits metadata only. This catalog smoke is input-plumbing evidence only: it is not part of the 1,219-context W9 roster, not an admitted W9 row, and not a result. The W9 learning-store prefix and its outcome data are not accessed.

## Handoff

Before any W9 experiment, the owner must review the exact upstream artifacts, row-closure manifest, alias/table hashes, source/evaluator/vocabulary identities, optimizer-update accounting, and resource plan. Obtain a separate run authorization and required GPU lease; this preparation does not grant either. Keep bulk inputs, predictions, checkpoints and receipts outside Git.
