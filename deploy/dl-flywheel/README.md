# JopeDime continual GP learning — v1.0 release candidate

This package joins real-data augmentation, encoder preparation, native GP measurement, immutable S3 results, training admission and measured neural updates. It fixes the operational distinction between original catalog coverage, new variants, unique optimizer exposure and repeated presentations.

This is a release candidate. The completed live qualification used 13 training and 3 validation contexts. Full-corpus admission and throughput changes are under development. No serving promotion, whole-corpus training claim or encoder-mask lift follows from this source import.

## Layout

- `jope_flywheel/`: portable contracts, lineage split, transformations, durable queues, native/Rust measurement, S3 library, score joins, batch journal, snapshot and admission.
- `pipelines/encoding/`: Kumo Small-only input replay and GPU qualification.
- `pipelines/publication/`: create-only dataset publication and exact-version verification.
- `pipelines/programs/`: source-pinned CPU GP evaluation campaign and role checks.
- `pipelines/neural/`: actual PolicyLM updates, decoder, fixed readouts, physical checkpoints and exact optimizer/RNG continuation.
- `pipelines/admission/`: evidence-backed atomic capacity release after actual train/validation exposure.
- `pipelines/archive/`: physical checkpoint and paper-evidence custody.
- `dashboard/`: read-only observer and live HTML.
- `tests/`: offline behavioral/integration tests.
- `docs/`: model, corpus inventory, experiments, evidence and operating plan.
- `SOURCE_MANIFEST.json`: source import map and hashes.

Native PolicyLM modules remain in the repository at `deploy/dl-trainonly/lane/src/dl_training_v1/dltrain`; native Rust evaluator/emitter are reused through pinned interfaces. They are not vendored or replaced here. Bulk data, checkpoints, binaries and credentials stay outside Git.

## Offline checks

Run from the repository root in the existing qualified Python environment. CI uses `run_offline.py --output <fresh-evidence-directory>` to require 226 original cases plus74 native/growth cases in an isolated640-token interpreter (300 total) and write a separate receipt:

```sh
rtk proxy env PYTHONPATH=deploy/dl-flywheel:deploy/dl-trainonly/lane/src/dl_training_v1 JOPE_POLICYLM_SOURCE=deploy/dl-trainonly/lane/src/dl_training_v1 python3 -B -m unittest discover -s deploy/dl-flywheel/tests -v
rtk proxy python3 -B deploy/dl-flywheel/check_source.py
rtk git diff --check
```

NumPy is required; native integration also needs the pinned Torch/safetensors runtime. A skipped native dependency is not a full integration pass. Tests never contact S3 or launch real jobs.

## Operational entrypoints

Pipeline entrypoints are qualified campaign source. They require explicit source/config/input/resource pins and owned runtime directories; copying a file here does not authorize executing an old manifest or resetting a deadline. Generate a fresh reviewed configuration for a new campaign; preserve the old runtime as evidence. Do not run campaign scripts blindly from their source directory. Frozen deployment controllers and all historical attempt receipts are preserved in the R4 custody location documented in `docs/release-status.md`.

The active roster is Kumo Small, 2,051 values. The expanded encoder/mask proposal is a separate experiment. A new architecture/input roster has its own model lineage and counters.

## Core invariants

1. Same source lineage stays on one side of validation, including every split and fuzzed descendant.
2. Every novel GP proposal has durable work identity; numeric quality requires actual exact-context native evaluation, emitted Rust parity and S3 readback.
3. Canonical GP cache identity includes input/target/split/program/evaluator/protocol.
4. Checkpoint, optimizer, RNG, source/corpus identity and batch journal advance together; unique examples and presentations remain separate.
5. Generator capacity releases only from verified supported training/validation exposure.
6. Terminal GP status is not a numeric reward, and a raw S3 object count is not a data-run count.
7. Mask searches use inner development evidence. No searched mask is called an out-of-sample improvement without independent outer evaluation.

Full-corpus and growth follow-up source lives in `pipelines/native/` and `pipelines/growth/`. Runtime pins are dated evidence, not automatic launch authorization. `native_variant_snapshot.py` prepares a new boundary snapshot; input-ready vectors do not imply measured GP labels or training admission. Unpromoted variant CE is disabled. The normalizer remains frozen on the original training subset. See [growth and inference](docs/growth-and-inference-v2.md).

HPC instrumentation and the original-context scored-program consumer live in `pipelines/hpc/`. Three additional offline modules add 26 tests (326 total), including immutable cache identity/corruption handling, native input masking, and log-chain/S3 archival. See [measured HPC findings](docs/hpc-bottlenecks-and-timing.md). Host GPU samples are distinct from CUDA execution timers. The cache result is a matched two-GP verification benchmark, not a full pipeline speedup.
