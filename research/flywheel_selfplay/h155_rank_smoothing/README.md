# H155 — pairwise rank smoothing and margin calibration

Prospective executor for [PREREG.md](PREREG.md), merged in PR #1309.
No H155 scoring, calibration or adapter update has run at executor publication.

[runner.py](runner.py) uses immutable, SHA-bound H153 primitives from Git commit
ea6347d5cc642cd0d7ae847c4e95f309471a5e72; no foreign working files are imported.
[manifest.json](manifest.json) binds inputs, source assets, parent sources and
the executor. [fold_manifest.json](fold_manifest.json) fixes disjoint TRAIN
family fit/calibration/inner-evaluation sets.

Validation: nine CPU tests cover pairwise direction, effective smoothing,
ties, temperature fitting/argmin invariance, family isolation/equal weighting,
refusal to call a worse-than-r882 result promising, and reserved-block disk guards. Both runner and supervisor use the shared user-visible used/(used+bavail) accounting; 95% is a hard stop, with equality tests for 95% and40GB. Ruff check/format pass.
Zero-evaluation smoke verifies 47 trainable tensors/29,211 adapter parameters
and frozen DOPE c51a56d0, with zero CUDA initialization and no optimizer updates.
The exact native91-context baseline is still a runtime prerequisite.

Real-S3 custody: all1319 numerical inputs/pools/model assets match immutable H144 archive
members; runtime source code is separately hash-pinned (H144 excluded code); the archived source manifest was independently fetched at its pinned
S3 version and SHA checked. All276 dataset IDs join the authoritative real-S3
catalog; no TRAIN/Val family overlap, no sealed/holdout/meta-test input payload.
Original H50/H58 derivative S3 object versions remain unestablished; exact
used numeric closure has versioned H144 archive custody. No synthetic training.

Operational source export (not another checkout):
/home/ubuntu/jope_plan/evidence/H155. Owned data and output:
/home/ubuntu/jope-data/h155. CPU only, nice19, math threads1, at most4 OS
threads. Before scoring, launch.json must attest exact merged executor bytes
and manifest hash. A failed/partial attempt is preserved, never silently reused.
Model scoring and adapter/calibration updates occur only after executor merge
and source/provenance review. No speed gate or serving authorization.

Acceptance:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
python3 -m pytest test_h155.py -q -p no:cacheprovider
ruff check runner.py test_h155.py
ruff format --check runner.py test_h155.py

Full native just required / hosted CI was not run in another owner's checkout.
The central merger owns any generated inventory registration and exact-head
integration checks; no product/runtime code changes are included.
