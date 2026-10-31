# H120 reproduction index

This index supplements the committed [report](README.md),
[metrics](metrics.json), and [preparation receipt](prep_receipt.json). No source
or bulk panel/results are copied into this file.

## Protocol and inputs

- **Protocol:** W4 offline reveal-policy screen, opened as H119 and explicitly
  renumbered H120 to preserve the earlier H119 claim. Uses H112's prior and
  rules unchanged; exact decision thresholds, comparisons, and test command
  are in the report. This is an offline screen, not a prospective fit run.
- **Dataset and panel:** reused, fully labelled H110/H116 proposal batches;
  300 batches across 50 DL-Train contexts / 44 families, six batches/context.
  The integrity check reproduces all 150 H116 batches exactly. The exact
  versioned `rows_h120.jsonl` and screen manifest identities are in
  `metrics.json.s3_objects`; no dataset rows are embedded here.
- **Seeds:** source batches use H110 seeds `801–803` and H116 seeds `901–903`;
  dry-run-only context seed `1001`; family Elo bootstrap seed `20261031`.
  No new sample split or fit seed was drawn. Native fits: `0`; GPU: `0`.

## Source, metrics, and loss record

- **Source:** screen source commit `9e46eea4cbc0e4a89e2bc6dd218d9bdaad694c8d`;
  runner SHA-256 `cc5219f5013ff6bd08ddfde66f3d06336677c471bdd0f6838287e222074e4d0c`.
  Prereg/screen-prep merge `569324b7760a95be26e817ce843b2430e32095fb`.
  Closeout PR [#1067](https://github.com/neverhuman/JopeDime/pull/1067),
  reviewed source head `515e3f431415e30952e90e38d2178fcdaed3df8d`, merge
  `8353bad2988c3da06096622d86b5eb495d8def82`.
- **Elo:** `S_TOP4` is the 1500 anchor; `GP_2PLUS2` 1519.0
  [1496.3, 1547.1], `UCB_2PLUS2` 1504.7 [1475.6, 1538.1], `GP_1PLUS3`
  1467.9 [1402.4, 1525.4], and `random4` 1211.8 [1062.3, 1317.3]. All
  screen rules fail the preregistered common gate; W4 adaptive reveal at 4/16
  is closed. Elo, not AUC.
- **Loss curves:** not applicable. The screen performs no native fitting or
  model optimization; therefore no train/validation loss curve exists.

## Reproduction

Use the version-pinned screen and prep objects in `metrics.json.s3_objects`
and the exact command in README §“Reproduce”. The manifest and row artifact
remain in S3; this Git folder contains only compact reports and test code.
