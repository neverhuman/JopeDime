# H118 reproduction index

This index points to the committed [report](README.md) and
[compact metrics](metrics.json). It adds no copied source, model, fit rows, or
dataset payload.

## Protocol and inputs

- **Protocol:** W6 cross-family transfer evaluation under H115's preregistration
  ([PREREG.md](../h115_w6_xfamily/PREREG.md), merged #1027). The evaluated
  source file `run_h114.py` has SHA-256
  `ed7b30a0b822401928fb610df45ae8d051da3e782b0c5b64f47fa4507131456b`.
  The protocol, comparator definitions, endpoints, and gates are in the
  report.
- **Dataset and panel:** 38 DL-Train contexts, one per family, all outside the
  50-context H67 donor roster; 2,306 donor outcomes; 16 proposals/context
  (608 fits), zero cross-family aliases. The panel manifest SHA-256 is
  `0afd4bfe4f8afd80bde434133b1fdfb893e12498d4d8656c30fb29984394aef4`;
  the exact-version manifest and panel IDs are in `metrics.json.s3_objects`.
- **Seed:** panel seed `1001`. No DL-Val, outer-test, holdout, or sealed data
  was read; DOPE was not loaded; GPU use was zero.

## Source, metrics, and loss record

- **Run:** `opus-h115-20261031a`. Run-time main commit recorded in metrics:
  `5e74f9eaa8a15ff54a268429cff0d43b3104642a`. Report PR
  [#1041](https://github.com/neverhuman/JopeDime/pull/1041), reviewed source
  head `0bd95e7003f5bef3914548595704ca79b24c81c9`, merge
  `398133750f1ff570c8098c56a162183d7aaced0b`.
- **Elo:** `G_SCORER` is the 1500 anchor; `X_SCORER` is 1476.18. The generic
  scorer's secondary comparison to `random4` is about +136 Elo. The
  preregistered context-transfer hypothesis is `NO_ADVANCE`; see the complete
  family-weighted ratings and intervals in `metrics.json`. Elo, not AUC.
- **Loss availability:** the run fitted/evaluated proposal scorers; neither
  `metrics.json` nor its exact-version `run.log` contains per-epoch train or
  validation loss values. The log's SHA-256 is
  `e5a07a31e1fb57c04e7cb7b2deb192aa8f7cc5b66130528649a760f79431572a`.
  No loss curve is available in the archived run record; none is synthesized.

## Reproduction

Use the exact-version source, panel, scorer, fit, and run-log objects listed
in `metrics.json.s3_objects`. The panel identity is hash-bound above; bulky
objects remain in their versioned S3 custody.
