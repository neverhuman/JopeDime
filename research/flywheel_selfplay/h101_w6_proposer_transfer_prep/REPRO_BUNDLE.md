# H101 reproducibility bundle

This is a metadata index for the W6 transfer and prospective audit. H101 is a deterministic
summary/transfer of an immutable execution; it is not a new training run. No source, dataset,
model, or bulky receipt is copied into this folder.

## Protocol and configuration

- Registered design and interpretation: [README.md](README.md), including the W6 registration
  at `PROPOSALS.md:209` and the H94/H95 prospective audit definition.
- Derived result and configuration pins: [metrics.json](metrics.json).
- Execution: five family-grouped W1 folds over 1,219 contexts / 205 families; prospective audit
  covers 50 contexts / 44 families / 150 context-seeds. H101 itself performed **zero native
  fits and zero GPU seconds**.

## Data, splits, seeds, source

- Dataset identity: DL-Train only for W6 (1,219 contexts / 205 families); prospective audit is
  the restricted H94/H95 evaluated subset. No DL-Val, outer-test, sealed, or holdout read.
- Split identities in `metrics.json.pins`: H72 train roster
  `9d3656d9817239d07ac27d1cb4f90d688bf3421b492e15b62ac078a9d183190d`, H72 fold manifest
  `129002bcd6cfc4b2f3a0b9e0562967cd4510dcbf88364c2d11d0bbe8801c2c73`, W1 fold manifest
  `dd211134bba127960b1612687478064969dca422e35dfa889790760f316961c1`. The five W6 folds use
  seed `0`; prospective H94/H95 context-seeds are `701, 702, 703`.
- Executed source file SHA-256:
  `3e6eda5859e9ec387ee57b8dc188fb0f82baedf9567eec583aa623889c63bdcc`; post-review formatted
  H101 source SHA-256: `d79358c7c668e72034df0595872749baaf0a45040487d023a04af1557dd6e802`.
  The historical execution Git commit SHA is not recorded in the committed metrics/README;
  these source hashes are the available exact execution identities.
- Derived H101 summary commit: `3a7031afd79683fde0731d1bfa4ce78dbd19710`.
- Immutable executed metrics SHA-256:
  `0ca473ed3de4ad0012148bb9f670e5a4562888617879a7285aea7b7899efe47e`; derived H101 metrics
  SHA-256: `78186916cfcc341f189a3f60474da7f9040d0837fea43389865ee76980206a80`.

## Elo and loss curves

- Registered W6 verdict: **KILL_NO_ADVANCE**. Track-2 cached U4 family Elo: PRIOR 1500.00,
  TREAT_FAMBAL 1524.97, RETR 1459.58, SHUF 1509.07. The PRIOR contrast is diagnostic only;
  retrieval, unseen-template, and context-shuffle gates fail.
- Prospective restricted evaluated-subset family Elo: H94_PRIOR_top4 1500.00,
  TRUE_W1_PRIOR_EVAL_SUBSET 1508.79, TRUE_W1_HEDGE_CTX_EVAL_SUBSET 1431.99. This restricted
  subset comparison is not a full-slate result; see README.
- Train/validation loss curves: **not applicable**. H101 ran no fits and has no optimizer loss
  trajectory. The prior H85/W1 model fit is referenced as an input, not repeated here.

## Custody

The README identifies version-pinned S3 execution metrics and external shard receipts. The
metadata here is sufficient to locate and verify those receipts without copying their payloads.
