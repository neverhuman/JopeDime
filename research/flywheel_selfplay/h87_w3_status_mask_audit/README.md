# H87 — independent W3 status-mask integrity audit

Status: **consumer-side integrity check only**. H87 does not run W3, fit or update a model, establish completion provenance, or decide W3 admission.

H86 exported a field-allowlisted status mask for the 1,219 admitted H50 DL-Train pools. H87 independently reads the exact H86 S3 object versions and checks the mask against the pinned pool bytes, ordered `program_sha256` identities, H65 fold manifest and H86 descriptive metrics. It reimplements the declared value-signature status rule instead of importing H86's classifier. Its only training-payload role is DL-Train; it does not open DL-Val, outer, sealed or holdout payloads.

The audit checks exact archive members and status dtype/domain, context ordering, offset coverage, per-pool hash and candidate count, program-order digest, mask-to-pool status equality, status totals by fold, and the H86 arm-c counts. A passing check establishes integrity of this exported inference mask. It does not turn signature-inferred labels into recorded completion status.

## Inputs and execution

The H50 packet and H65 fold manifest are pinned in `audit.py`. H86 `manifest.json`, `metrics.json` and `status_mask.npz` are fetched from the exact S3 version IDs and SHA-256 values recorded below. The script reads only paths for the H50 `train` roster and its DL-Train pool files.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 \
  python3 -B research/flywheel_selfplay/h87_w3_status_mask_audit/audit.py \
  --artifacts ~/jope-data/h87_w3_mask_audit/<run_id>/input-artifacts \
  --out ~/jope-data/h87_w3_mask_audit/<run_id>/result \
  --s3-prefix s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W3/<run_id>
```

The outputs are a compact `metrics.json` and `manifest.json`; upload both create-only and download each exact returned version to verify its SHA-256. No HWM pointer is written.

## Result

The exact execution receipt and H86 artifact versions are recorded in `run_manifest.json` after the run. No Elo, utility, training, model-change or admission result is produced.
