# P6 bounded local execution and durable storage

The explicitly authorized one-shot ran once from source
`a9cb739732d94814228408f1bee3209456224080` as a fresh Rust process, with observed
`nice -n 15`, affinity 16–23, one math worker and a 60-second outer watchdog. It completed two
native evaluations on one cut, produced actual released Dope512 features with zero landmark
evidence, ingested one measured result, wrote a diagnostic archive/checkpoint and reloaded it
in a separate process. Both processes were reaped; no daemon remained. The independent artifact
test reused this exact saved run for readback, output-reuse and digest-tampering checks.

The source generates 80 synthetic TRAIN observations with three features and an empty outer-test
file. No existing real or protected dataset was opened. Physical FIT scope contains 63 rows;
the remaining generated rows form the inner validation cut. Both native evaluations succeeded.
The selected fit has RMSE 0.3479287778256347 on that generated validation cut. This is a fresh
execution measurement, not corpus retention or a model-quality qualification.

The local HWM ledger initialized one result with zero improvements. Its existing production
verifier re-derived the pipeline evidence, and `hwm::freeze` persisted one explicitly diagnostic,
non-teaching record. A stale pointer CAS was refused without changing it. The checkpoint raw
SHA256 is `978a4301a2b84952b3cb5796c5eb726907673191c813d7a8b66e31c4beafa5c8`;
this raw-byte namespace is distinct from the learning Snapshot envelope/domain identity.
The executable SHA256 is
`0f5baebea11d8191220c31f4cbc06f0f140695473d4103d0865cedab6e5d42a3`.
Exact process, input, result and log hashes are in the [execution record](optional-landmarks-p6.json).
Wall time is diagnostic test duration, not a latency benchmark.

A durable `StateSnapshotAuthority` storage port is also implemented. It uses fixed descriptor-relative
paths, nofollow reads, process locking, immutable candidate retention, fsynced root CAS, bounded
ancestry/pin validation and truthful local result states. Eight tests cover reopening, stale and
concurrent writers, retained losers, historical winner idempotency, incompatible pins, corruption,
symlinks and missing committed roots. It refuses unproved remote result states. This storage
implementation supplies no evidence verifier, issuer, adoption or serving permission.

The one-shot deliberately used the diagnostic HWM archive rather than publishing through a
fixture verifier. `SnapshotEvidenceVerifier` still needs authenticated input-root, lineage and
witness evidence from the actual controller boundary. None was fabricated or installed.
Production snapshot publication, global/remote updates, policy promotion, neural optimizer
steps, continuous-service health and the remaining scientific gates are not demonstrated here.
The existing multi-round shadow flywheel's minimum-round rule is unchanged.

Reproduce the bounded example only when another acquisition run is authorized:

```sh
cargo build -p jope-cli --features dope-encoding --example landmark_one_shot --locked
timeout --signal=TERM --kill-after=5s 60s nice -n 15 taskset -c 16-23 \
  target/debug/examples/landmark_one_shot --output NEW_DIRECTORY \
  --weights deploy/appliance/assets/dope-mlc-v3/model.safetensors \
  --source-commit FULL_SOURCE_SHA --device cpu
```

The saved immutable executable and run are retained under
`target/landmarks/p6-one-shot-execution/`. From the repository root, verify this existing run
without another acquisition:

```sh
env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only CARGO_BUILD_JOBS=4 \
  RUST_TEST_THREADS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  JOPE_ONE_SHOT_EXE="$PWD/target/landmarks/p6-one-shot-execution/bin/0f5baebea11d8191220c31f4cbc06f0f140695473d4103d0865cedab6e5d42a3/landmark_one_shot" \
  JOPE_ONE_SHOT_RUN="$PWD/target/landmarks/p6-one-shot-execution/run-001" \
  taskset -c 16-23 cargo test -p jope-cli --features dope-encoding --test offline_cli \
  landmark_one_shot::bounded_one_shot_measures_reloads_and_refuses_reuse_and_tampering \
  --locked -- --ignored --exact
```

Retained verification checks the original recorded source and executable hashes even after
later commits. It does not relabel the saved evidence with the current HEAD.
