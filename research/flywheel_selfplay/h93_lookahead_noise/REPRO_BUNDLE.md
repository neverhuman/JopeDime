# H93 reproducibility bundle

This index contains metadata only. The protocol, source, inputs and compact closeout remain in
the linked files; no source copy, dataset, model checkpoint, or run payload is included here.

## Protocol and configuration

- Protocol and fixed arm configuration: [PREREG.md](PREREG.md) (CONTROL, LA, GN, EB4; 20 rounds,
  two optimizer updates per round; 40 updates; batch 16 except EB4 accumulation; slate 192).
- Actual incomplete closeout and attempt accounting: [README.md](README.md) and
  [metrics.json](metrics.json).
- Frozen anchor: r882 SHA-256
  `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
- Frozen DOPE SHA-256:
  `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.

## Data, splits, seeds, source

- Dataset identity: admitted real-S3 H50 packet; 1,219 DL-Train contexts / 205 families and
  91 DL-Val contexts / 19 disjoint families. H50 source manifest SHA-256:
  `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`.
- Split lineage: H50 family-fold lineage, fold seed `4900`; dev readout is preregistered over
  the 19 dev families. These are identifiers, not copied split rows.
- Paired outcome seeds: `30, 31, 32`; bootstrap seed `93000`. Only CONTROL/30 completed as an
  outcome; EB4/30 was timing-only; LA/30 attempts voided before scoring. The other outcome
  cells are missing, not zero.
- Preregistration source commit: `d48d32d7b2ceff7d3e959feeb5d1f7b25e483856`.
  Incomplete-closeout source commit: `33beae0a0d4d25b2c5591ddbf2ace71e710ae5e3`.

## Elo and loss curves

- Elo: **not estimable**. The paired panel is 1/12 complete; `metrics.json` records `panel.elo`
  and family-grouped CI as null. No outcome comparison or admission estimate is reported.
- Train/validation loss curves: **incomplete / unavailable**. The committed closeout contains
  no per-update curve series, and the partial panel cannot provide paired train/dev curves.
  No values are reconstructed from summaries.

## Custody

Only compact metadata is in Git. Raw run receipts, logs and attempt state remain in the external
H93 run root and the linked custody notes; this bundle does not claim to reproduce those files.
