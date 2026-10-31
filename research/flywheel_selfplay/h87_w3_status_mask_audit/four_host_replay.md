# H87 — additional four-host integrity replay

**PASS: zero status mismatches across 1,404,287 rows, 1,219 DL-Train contexts
and 205 families.** This additional verification reproduces [H87](README.md)
and adds four-host receipts, corruption controls and durable S3 custody.
It establishes mask integrity only. [H89's completion-provenance gate](../h89_w3_completion_admission/README.md)
and W3's other admission requirements remain unchanged.

## What ran

Exact-version GET and SHA-256 checks bound all three [H86](../h86_w3_outcome_status_census/README.md)
objects: manifest, descriptive metrics and status mask. Every original H50
training pool was checked against the pinned training roster and mask pool hash.
Workers then checked ordered program hashes, aligned array lengths, gate counts,
all status bits and the independently recomputed fold totals. The mask schema,
status domain, sorted context identities and monotonic offsets also passed.

The original pools came from the admitted H50 local mirror; this report does
not invent per-pool S3 keys or source versions. Its S3 evidence archive preserves
the exact hash-bound training-only input shards used here. Only program IDs,
fitness, logratio and gate arrays were decoded. The selected roster reader stops
after the first train array, before decoding DL-Val metadata. No DL-Val, outer,
sealed or holdout payload was opened. No table, model or checkpoint was loaded.

| Host | H65 folds | Contexts | Worker CPU seconds |
|---|---|---:|---:|
| xbabe1 | 0 | 227 | 0.377908 |
| xbabe2 | 1 | 234 | 0.378437 |
| xbabe3 | 2 | 269 | 0.418589 |
| xbabe4 (Linux Docker/Colima) | 3, 4 | 489 | 0.540713 |

Each host ran one worker after a load check below the 85% ceiling, nice 19 on
xbabe1 and at least nice 10 elsewhere; xbabe1 had at least 150 GiB available. The Mac ran
only inside Linux Docker with one CPU, a 768 MiB limit, no network and read-only
container storage. Deterministic complete-pool auditing uses no random seed.
Total measured worker CPU was **1.715646 seconds**, within the claimed
300 CPU-second ceiling. Preparation and archive CPU are separately recorded;
transport, imports, controls and publication overhead are unmeasured, not zero.
There were no fits, optimizer updates, GPU work or HWM candidates.

## Results and limits

- Signature counts: 1,344,468 `measured_pass`, 58,876 `measured_fail`,
  943 `sentinel_shaped` in 366 contexts, zero nonfinite and zero anomalous.
  The names are H86's signature labels, not evidence of actual terminal status.
- All five fold counts match. Arm-c counts were independently computed per
  context: 1,219 top-16 paths, zero fallback/no-passer paths, 894 floor-bind
  contexts and 522,210 clipped passing rows. Arm-d's signature-measured count
  is 1,403,344; its quantiles were not rechecked.
- Fifteen synthetic checker controls passed, including changed pool bytes,
  program-order hash, status bit, foreign context, malformed offsets/status
  dtype and train-reader termination before a deliberately invalid val suffix.
  These are validation fixtures, not experimental observations.
- No 192-candidate training-slate analysis or W3 training ran. Elo, U4 and
  val_reg_inner are null; no performance or completion-provenance claim follows.

The adaptation credits bat02's checker at
`8b9c9601081e5df2cd2991aebb10948cbd546368` and preserves its source in custody.
The original checker compares the reported arm-c path counts to a literal
expected dictionary; this replay instead counts each pool's observed path.
It also uses the train-prefix reader instead of decoding the full selected
metadata file. The merged historical source and receipts remain unchanged.
PR #947 merged at 08:34:09 UTC, immediately before this claim at 08:34:10;
after refreshing main, the scope was corrected to additional verification,
not first closeout or a new experiment.

## Custody and reproduction

Manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W3/astra-h87-four-host-20261031T083936Z-f4ea73fb/manifest.json`

Version: `3DCqOIeEAhypX9qmdBOYD.DZnzRKVKhb`
SHA-256: `e46eee1ed044110aafe3e946d1a742f4c19090c418bc521b3cdc1dd2f37e37ac`

All four output objects passed exact-version GET and byte SHA-256 verification.
Every one of the 36 archive members was verified, including the four original
transport shards. [four_host_custody.json](four_host_custody.json) lists immutable
versions and hashes; [four_host_metrics.json](four_host_metrics.json) retains
input identities, fold counts, worker resources and control results.

Retrieve the listed `evidence.tar.gz` version, verify its SHA-256 and extract
outside a Git working tree. With Python and NumPy installed, run
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 python3 -B controls.py`.
For the original four-host topology, `nice -n 10 python3 -B dispatch.py` rechecks
host load/disk limits and replays the preserved shards; then run `combine.py`
under the same thread limits. An individual shard can be replayed with
`nice -n 10 python3 -B worker.py SHARD_SHA256 < xbabe2.tar`, substituting the
recorded shard name and SHA from `inputs.json`. These are standalone archived
checkers, not newly registered repository tests. Synthetic checks require no S3
access; real replay uses the preserved real input shards.

Owned local transport archives and download copies were removed only after
versioned custody and member-hash verification; small source/results/receipts
and the original H50 mirror remain. Remote workers consumed stdin and created
no persistent input files; the owned Docker container used `--rm`.
