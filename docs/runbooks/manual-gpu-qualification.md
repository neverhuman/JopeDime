# Manual GPU qualification

CUDA and GPU qualification are local, explicitly requested operations. No GitHub
Actions workflow schedules GPU work, selects a GPU runner, or gates a merge on
CUDA hardware. CPU dependency fences that reject GPU libraries remain enforced.

For CUDA emitter host, compiler and device qualification, follow
[the CUDA contract](../cuda.md) and [hardware handoff](../cuda-handoff.md).
The local qualifier selects `feat-math/cuda-parity-tests`; normal Cargo tests do not.

The former weekly GPU workflow is removed. Its existing regression and restore
scripts remain available to an operator with the required pinned models, datasets,
AWS access, GPU capability and an explicitly approved compute budget:

```bash
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
# Prepare the pinned models and Golden42 inputs using the normal jope CLI.
jope models fetch --pinned
jope datasets fetch golden42
RECEIPTS_DIR="$PWD/target/regression" JOPE_REGISTRY="$PWD/datasets/by-id" \
  bash deploy/ci/regression-suite.sh all
RECEIPTS_DIR="$PWD/target/restore-drill" bash deploy/ci/restore-drill.sh
```

These commands are documentation, not evidence of execution. Missing hardware,
credentials, inputs or an accepted budget must fail explicitly; do not replace
them with synthetic success or describe CPU checks as a GPU qualification pass.
