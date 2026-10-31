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

Run `luna-bat02-h87-20261031T0717Z` (xbabe2, CPU, nice 19) against H86 run `luna-tel-h86-20261031T0709Z-01`. Compact outputs are committed here as `metrics.json` and `run_manifest.json`.

- **Integrity: PASS.** Archive fields exact, offsets valid, per-pool hashes and program-order digests match, **0 mask/pool status mismatches** over 1,404,287 DL-Train rows (1,219 contexts, 205 families).
- Status totals: 1,344,468 measured_pass, 58,876 measured_fail, 943 sentinel_shaped (366 contexts), 0 nonfinite, 0 anomalous. Arm-c: 522,210 clipped passing rows, 894 floor-bind contexts; arm-d: 1,403,344 measured log-ratio rows.
- **Verdict:** the exported H86 inference mask is internally consistent with the pinned pools. This does **not** supply recorded completion status, so W3 stays blocked per H88/H89 until source event records exist. No training, no model change, Elo N/A.
- **Red-team:** DL-Train only (no DL-Val/sealed/holdout reads), DOPE not loaded, no HWM pointer written, no metric direction involved.
- **Next:** none for H87; W3 unblock requires source completion-event provenance (H88 follow-up).
