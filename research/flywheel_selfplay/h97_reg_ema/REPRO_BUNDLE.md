# H97 reproducibility bundle

This index contains metadata only. It points to the frozen protocol and compact result summary;
it does not include source copies, datasets, adapter snapshots, checkpoints, or run payloads.

## Protocol and configuration

- Protocol and arm configuration: [PREREG.md](PREREG.md) (CONTROL, WD, DROP, EMA; 20 rounds,
  two AdamW updates per round; 40 updates; batch 16; slate 192).
- Results and interpretation: [README.md](README.md) and
  [metrics_summary.json](metrics_summary.json).
- Frozen anchor: r882 SHA-256
  `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
- Frozen DOPE SHA-256:
  `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.

## Data, splits, seeds, source

- Dataset identity: admitted real-S3 H50 packet; 1,219 DL-Train contexts / 205 families and
  91 DL-Val contexts / 19 disjoint families. H50 source manifest SHA-256:
  `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`.
- Split lineage: H50 family-fold lineage, fold seed `4900`; registered scoring reads the 91
  dev contexts / 19 families. No separate terminal validation/admission read was performed.
- Paired outcome seeds: `33, 34, 35`; bootstrap seed `95000`. All 12 outcome cells completed.
- Outcome worker source commits: `13b91f036d18b5e6b6ed08d015329e50ea840208` and
  `6a5b8f74a6662fb78d8151dd229661a0a8c1b25c`. Closeout commit:
  `9203d61d343ef42fbe8e2528ad9230de1fa0729b`.

## Elo and loss curves

- Family-grouped Elo vs CONTROL: CONTROL 1500.0, WD 1500.0, DROP 1500.0, EMA 1536.7
  (median; EMA CI95 `[1481.7, 1634.3]`). Median Elo vs fresh r882: CONTROL 1321.1,
  WD 1321.1, DROP 1344.0, EMA 1425.7 (EMA CI95 `[1296.7, 1536.7]`). See the JSON summary
  for all arms and confidence intervals. Verdict remains **NO ADOPTION**.
- Train/validation loss curves: **not included in the committed run summary**. The compact
  outcome receipts and `metrics_summary.json` provide endpoint metrics, not per-update loss
  series. Raw run receipts are in external H97 custody; no curves are inferred here.

## Custody

The H97 closeout records the external receipt audit and its integrity checks. Large raw receipts,
family vectors, and adapter snapshots remain outside Git.
