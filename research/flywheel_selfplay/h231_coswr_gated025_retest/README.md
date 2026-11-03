# H231 — COSWR × GATED_025 stack retest

**INCOMPLETE; no efficacy verdict.** Terminal stop after 45/45 HistGBM fits, 2/15 cells with complete prediction/custody receipts and 120/600 durable updates. Unmeasured or partial scientific outcomes remain unknown, not zero. No retry.

## Hypothesis and setup

The hypothesis was that H136 COSWR adapter training and H217 GATED_025 selection
would combine to clear the strict +0.002 native and DL-Val plateau margins. The
four arms are R882 (frozen reference), COSWR, G025 and primary STACK. H230 failed
in raw-pool setup before any fit/update; H231 separately retested the hypothesis
after normalizing `program_sha256` into internal `pids`, matching H136. The raw
NPZ constructor regression failed on H230 and passed for byte/unicode IDs here,
including identical frozen sampler indices and pool hashes.

H230 seeds 230101–103, folds 230141–143 and all thresholds/recipes were retained.
Folds are identical to H230 (Jaccard 1), not an independent replication. The
pinned real-S3 packet supplies 205 DL-Train families/1,219 contexts and 91 DL-Val
contexts/19 families. A and B families are excluded from each 123-family FIT
training/selection closure; B is unused. This is a reused development cohort,
not sealed or unseen-family qualification. [PREREG.md](PREREG.md) fixes the
protocol; [fold_manifest.json](fold_manifest.json) and [provenance.json](provenance.json)
retain the input and split identities.

The planned inventory is 45 unique HistGBMs and 15 adapters × 40 updates. The
first 4 HistGBMs and first 40-update adapter form the non-scored TRAIN smoke and
are reused in that inventory. COSWR/STACK share each adapter; G025/STACK share
each HistGBM. Frozen H136 schedule/loss/sampler and H217 gate rules are retained.
FIT gate calibration uses inner-family OOF HistGBM predictions; COSWR's own FIT
reference is in-sample for the adapter, a disclosed threshold-optimism limitation.
No A/DL-Val outcome selects a model, update count, calibration or threshold.

## Outcome

The supervisor stopped the worker at **2026-11-03 01:20:49 UTC**, exit −15, because host available memory fell below the registered minimum. No partial-outcome scoring or efficacy selection was performed; gain, confidence interval, harm and Elo remain unknown.

The journal contains 120 updates. Three cells finished 40 updates each, but only two completed their prediction/custody receipts. The third saved its adapter checkpoint before prediction stopped. No subsequent cell started, so the saved update count is exact. The interrupted process's final in-memory state is unknown. All updates belonged to seed 230101; seeds 230102 and 230103 had zero optimizer updates. All three seeds' efficacy outcomes remain unknown.

The guard observed MemAvailable of **21,433,810,944 bytes**, below the 20 GiB minimum (21,474,836,480 bytes), while load5 was 73.378 and the worker was load-paused. Worker-tree RSS was 724,619,264 bytes; free disk was 744,176,377,856 bytes, with disk use at 90.5%. This was a host-memory guard stop, not evidence of a worker OOM or disk breach. The global scoring lock was never reached; no A-fold or treatment DL-Val utility was scored.

## Red-team audit

DOPE is pinned to c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814.
Native91 reproduced bit-exactly before any training, with mean 1.1377941898164525. Each completed cell resets a
copy of r882, trains only 47 non-anchor adapter tensors / 29,211 parameters, checks
positive adapter drift and equal non-trainable before/after hashes, and checks
DOPE unchanged. Completed adapter/prediction files and fsynced update inventory
are hash-verified in the terminal audit; source and review receipts remain pinned.

Both completed cells match their registered 123-family FIT sets (727 and 747 contexts), excluding A/B. The 205 TRAIN families and 19 DL-Val families are disjoint. Frozen source/input closure excludes sealed, holdout and meta-test payloads. DL-Val feature decoding and prediction with the fixed four revealed labels are permitted by the preregistration; they are distinct from using DL-Val outcomes for gradients or FIT threshold selection. The third cell lacks a final receipt, so no final in-memory DOPE/hash verification is claimed for it.

## Execution and custody

Preregistration PR #2058 merged execution `3978dc8ef0f3f1393980245f06c0b2c20a74df31` after 25 local
tests, Ruff, five records/pins/links and same-agent exact-head review. A1 explicitly
authorized admin squash despite inherited remote CI failures; those failures and
cancelled lanes are not reported as passing. Main-guard passed the merged source.
The single attempt ran on xbabe2 CPU, 4 cores [2,22,67,104], nice 19, no GPU. Hard RSS 4 GiB,
MemAvailable 20 GiB, free disk 300 GB, disk use 93.8%, output 2 GB and 03:30 UTC deadlines stayed
fixed. Load >60 pauses only its worker and resumes at <=55. No retry or affinity change.
Peak observed process-tree RSS was 3,895,635,968 bytes; minimum RAM
available 21,433,810,944 bytes and disk free 743,843,819,520 bytes.
Observed maximum disk use 90.5%; load actions and terminal exit are
in [metrics.json](metrics.json). Wall interval 15,527.5 s includes
pauses; it is a cost record, not a speed verdict. OOM/disk findings are based on
these samples plus the terminal receipt, not an assumption from missing outputs.

All durable worker S3 objects and the full runtime archive were recovered by
exact VersionId/SHA256; every archive member was checked. [manifest.json](manifest.json)
separates current documentation pins from immutable execution-source pins.
Custody: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H231/astra-t2040/closeout/manifest.json`; VersionId `ioP87KygfmWpAgwHOoO5GkQvEWsLP86i`;
SHA256 `e5783558d086b9c036076f97fc51fd3a95432d9d2917e3a415da1070ff524718`.

Next: retain the frozen references and this result; any new experiment requires a separate preregistration, with no tuning or retry on this readout.
