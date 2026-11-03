# H231 — COSWR × GATED_025 stack retest

**INCOMPLETE; no efficacy verdict.** Terminal stop after 45/45 HistGBM fits, 2/15 adapter cells and 120/600 durable updates. Unmeasured or partial scientific outcomes remain unknown, not zero. No retry.

## Hypothesis and setup

The hypothesis was that H136 COSWR adapter training and H217 GATED_025 selection
would combine to clear the strict+.002 native and DL-Val plateau margins. The
four arms are R882 (frozen reference), COSWR, G025 and primary STACK. H230 failed
in raw-pool setup before any fit/update; H231 separately retested the hypothesis
after normalizing `program_sha256` into internal `pids`, matching H136. The raw
NPZ constructor regression failed on H230 and passed for byte/unicode IDs here,
including identical frozen sampler indices and pool hashes.

H230 seeds230101–103, folds230141–143 and all thresholds/recipes were retained.
Folds are identical to H230 (Jaccard1), not an independent replication. The
pinned real-S3 packet supplies205 DL-Train families/1,219 contexts and91 DL-Val
contexts/19 families. A and B families are excluded from each123-family FIT
training/selection closure; B is unused. This is a reused development cohort,
not sealed or unseen-family qualification. [PREREG.md](PREREG.md) fixes the
protocol; [fold_manifest.json](fold_manifest.json) and [provenance.json](provenance.json)
retain the input and split identities.

The planned inventory is45 unique HistGBMs and15 adapters ×40 updates. The
first4HistGBMs and first40-update adapter form the non-scored TRAIN smoke and
are reused in that inventory. COSWR/STACK share each adapter; G025/STACK share
each HistGBM. Frozen H136 schedule/loss/sampler and H217 gate rules are retained.
FIT gate calibration uses inner-family OOF HistGBM predictions; COSWR's own FIT
reference is in-sample for the adapter, a disclosed threshold-optimism limitation.
No A/DL-Val outcome selects a model, update count, calibration or threshold.

## Outcome

Terminal error: `RuntimeError("resource stop: {'epoch': 1793668849.1330626, 'load5': 73.3779296875, 'free_bytes': 744176377856, 'disk_used_percent': 90.5, 'available_ram': 21433810944, 'swap_free': 40960, 'tree_rss': 724619264}")`. No partial-outcome scoring or efficacy selection was performed. Gain, CI, harm and Elo remain null/unreported. 120 fsynced updates are known. Three cells completed40updates each, but only two finished prediction/custody receipts; the third has an adapter checkpoint with predictions unfinished. Control-flow/artifact evidence places the stop after its last optimizer step; no subsequent adapter cell started. Completed cells' frozen-state checks are retained; the final in-memory state of an interrupted process is unknown.

The hard guard fired at MemAvailable21433810944bytes versus the20GiB minimum(21474836480bytes), while load5 was73.378 and the worker was load-paused. Its RSS was724619264bytes; disk free744176377856bytes/disk-used90.5%. This was a host-memory guard stop, not evidence of a worker OOM or disk breach. The global scoring lock was never reached; no A or treatment DL-Val utility was scored. All three seed outcome records remain unknown.

## Red-team audit

DOPE is pinned to c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814.
Native91 reproduced bit-exactly before any training. Each completed cell resets a
copy of r882, trains only47non-anchor adapter tensors/29,211parameters, checks
positive adapter drift and equal non-trainable before/after hashes, and checks
DOPE unchanged. Completed adapter/prediction files and fsynced update inventory
are hash-verified in the terminal audit; source and review receipts remain pinned.

## Execution and custody

Preregistration PR2058 merged execution `3978dc8ef0f3f1393980245f06c0b2c20a74df31` after25local
tests, Ruff, five records/pins/links and same-agent exact-head review. A1 explicitly
authorized admin squash despite inherited remoteCI failures; those failures and
cancelled lanes are not reported as passing. Main-guard passed the merged source.
The single attempt ran on xbabe2 CPU4[2,22,67,104], nice19, no GPU. Hard RSS4GiB,
MemAvailable20GiB, free300GB, disk-used93.8%, output2GB and03:30Z deadlines stayed
fixed. Load>60 pauses only its worker and resumes<=55. No retry or affinity change.
Peak observed process-tree RSS was3895635968bytes; minimum RAM
available21433810944bytes and disk free743843819520bytes.
Observed maxdisk used90.5%; load actions and terminal exit are
in [metrics.json](metrics.json). Wall interval15527.5s includes
pauses; it is a cost record, not a speed verdict. OOM/disk findings are based on
these samples plus the terminal receipt, not an assumption from missing outputs.

All durable worker S3 objects and the full runtime archive were recovered by
exact VersionId/SHA256; every archive member was checked. [manifest.json](manifest.json)
separates current documentation pins from immutable execution-source pins.
Custody: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H231/astra-t2040/closeout/manifest.json`; VersionId `ioP87KygfmWpAgwHOoO5GkQvEWsLP86i`;
SHA256 `e5783558d086b9c036076f97fc51fd3a95432d9d2917e3a415da1070ff524718`.

Next: retain the frozen references and this result; any new experiment requires a separate preregistration, with no tuning or retry on this readout.
