# H67 — W4 active reveal with four total observations

Status: **CPU preparation and protocol smoke only; no W4 effect estimate or verdict.**

## Hypothesis and registered advance gate

**Hypothesis:** selecting observations for the decisions they enable beats a fixed history at the same total fit budget.

The verbatim W4 gate from [`rd_breakthrough/PROPOSALS.md`](../rd_breakthrough/PROPOSALS.md#common-experimental-contract) is: “Advance over nonadaptive prior-top4 at matched fits and elapsed time. Kill if the advantage disappears when history information and construction costs match, or comes from access to an unrevealed cached label.” The common gate additionally requires an adjusted family-mean lower bound above `+.001`, median family delta at least zero, and no more than 10% of families losing more than `.01`, against the strongest matched control. Four-fit retained-best utility and family Elo are primary; legacy `val_reg_inner` is separate and reported only where its contract applies.

## Planned W4 protocol

Use regression-only DL-Train contexts and original training rows. Seeds are `701`, `702`, and `703`; assign five whole-family folds using seed `4900`. Freeze 50 fit-only acquisition contexts (10 per fold), then compare r882, program prior, random, and two-plus-two policies over three seeds and four fits per context. The active policy selects two distinct candidates before outcomes, reveals only those two, then selects two distinct additional candidates. Compare all policies at four observations and matched elapsed time against nonadaptive prior-top4. Retrospective cached-pool screening and prospective-new-program results are separate strata. Record every selection probability and revealed program ID.

The initial manifest is a preparation roster from the already admitted H50 packet (`source_manifest_sha256=3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`, file SHA256 `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`). It uses the packet's `DL-Train` rows only, resolves family and canonical-table aliases before assigning folds, and stores no outcomes. The manifest records all 1,219 context-to-fold assignments plus the 50 frozen acquisition contexts. The packet records 1,219 DL-Train contexts / 205 families, 91 DL-Val contexts / 19 families, and 93 excluded holdout contexts with zero holdout payloads opened and zero outer rows delivered. The smoke uses one hash-verified training pool from this real-S3-derived packet; it is a protocol-integrity check, not a candidate comparison or W4 evidence.

For the eventual scored run, report family-grouped raw utility and Elo per the common contract: fixed complete contender panel, r882=1500 within each track, tie margin `1e-4`, prior `.01`, 400 iterations, and the scorer source SHA. Include win/tie/loss counts and family-bootstrap intervals. The primary contrast is retained-best utility against nonadaptive prior-top4, using 10,000 paired family-cluster bootstrap draws with seed `20261031`; preserve all three seed rows together, report family mean/median and the fraction losing more than `.01`, and adjust the interval for the preregistered primary contrasts. Matched elapsed time and fit counts are required. These calculations are not implemented by this preparation harness.

No sibling labels, DL-Val, outer-test, or sealed data are admitted. Do not read a new S3 prefix. Keep DOPE frozen at `c51a56d0`. This preparation spends zero GPU hours and makes no native-fit claim; the proposal's ≤2,400 fit / four CPU-hour W4 budget remains cumulative across future execution.

## Reproduce preparation

```sh
python3 research/flywheel_selfplay/h67_w4_active_reveal_2plus2/prepare.py \
  --input ~/jope-data/h50/input \
  --manifest research/flywheel_selfplay/h67_w4_active_reveal_2plus2/family_fold_manifest.json \
  --smoke
```

The runner refuses a packet with a changed source identity, non-training rows, incomplete pool inventory, or a pool hash mismatch. Its smoke loads one cached pool into the smoke process, but the acquisition callback receives candidate outcomes only at the explicit reveal boundary; the second pair is chosen from the first pair's revealed history. The receipt distinguishes loaded arrays from unrevealed outcomes used for selection. It reports no utility comparison and cannot establish the registered gate. A separate admitted fit/scoring implementation and resource-budget ledger are still required before any W4 result run.
