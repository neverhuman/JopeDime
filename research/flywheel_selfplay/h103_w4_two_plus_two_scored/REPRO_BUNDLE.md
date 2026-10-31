# H103 reproducibility bundle

This index points to the preregistered protocol and compact outcome metadata. It contains no
copied source, dataset, fit output, checkpoint, or large run receipt.

## Protocol and configuration

- Protocol/configuration: [H96 PREREG.md](../h96_w4_active_reveal/PREREG.md) and
  [H67 roster README](../h67_w4_active_reveal_2plus2/README.md).
- Results and run details: [README.md](README.md) and [metrics.json](metrics.json).
- Arms: `prog_top4`, `prog_2plus2`, `prog_2plus2_swap`, `random4`; 16 new proposals per
  context-seed, four reveals per arm, zero history.

## Data, splits, seeds, source

- Dataset identity: H67 admitted DL-Train roster, 50 contexts / 44 families; zero DL-Val,
  outer-test, sealed, or holdout reads. The cited H67 packet lineage is H50 source manifest
  SHA-256 `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`.
- Split identity: [family_fold_manifest.json](../h67_w4_active_reveal_2plus2/family_fold_manifest.json),
  fold seed `4900`, file SHA-256
  `823b0240d98c6ef11704c3fdfe853afdaca7074e13e84a9a2ddf8b82004c8a7b`. Outcome seeds are
  `701, 702, 703`; family bootstrap seed is `20261031`.
- Execution source: H96 `run_w4.py`, SHA-256
  `beeb76b124db47b17794710ff66f9709d20d61f3f3b09a1da44c6e5d743f0a9b`; H81 source SHA-256
  `9ad50be577a30c616f7668d52f621d5411e70019ba9f1d1960eb8a45a448c999`; proposer model SHA-256
  `5938a35497ccde1a1489c4d6b13d122196aed16739a3272afea06d57d5fff692`.
- The execution README records source checkout `b2c04a84` (short SHA only); the exact full Git
  commit is not preserved in this run's committed metadata. Execution run ID:
  `opus-h103-20261031a`.

## Elo and loss curves

- Family-grouped Elo: `prog_top4` 1500.00, `prog_2plus2` 1507.89,
  `prog_2plus2_swap` 1505.92, `random4` 1504.60. Verdict: **NO_ADVANCE**; the primary gate
  failed, and the program prior did not beat random4 on new proposals.
- Train/validation loss curves: **not applicable**. H103 is a native scoring/evaluation study,
  not a model training run; it records native evaluator fits, not optimizer training losses.

## Custody

Version-pinned S3 result objects and their hashes are listed in `metrics.json` and described in
the README. They remain external; this bundle carries only their identifying metadata.
