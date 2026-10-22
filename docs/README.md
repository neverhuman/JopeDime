# JopeDime contributor map

Start with the [quick start](quickstart.md), [architecture](architecture.md), and
[contributor contract](../CONTRIBUTING.md). Use the table below to find the owning
implementation before changing a subsystem.

| Work | Implementation | Read first |
| --- | --- | --- |
| Saved fitted objects and versioned sessions | `crates/jope-runtime/src/model_object.rs`, `model_session.rs`, `crates/jope-appliance/src/model.rs` | [Model objects](model-objects.md), [format](model-object-format.md), [rollback](model-sessions.md) |
| Numerical operators, fitted models, code emission | `crates/feat-math/` | [Emitters](emitters.md), [parity matrix](parity/capability-matrix.md) |
| Bounded search, evaluation, caching, worker budgets | `crates/jope-pipeline/` | [Pipeline](pipeline.md) |
| Distributed execution and fitted-result authority | `crates/jope-engine/`, `crates/smartcluster/` | [Architecture](architecture.md), [runbooks](runbooks/) |
| Encoders, neural proposals, learning state | `crates/hyperion/`, `crates/jope-policy/`, `crates/jope-learning/` | [Learning](learning.md), [models](models.md) |
| Dataset custody and access | `crates/jope-datasets/`, `catalogs/`, `datasets/` | [Datasets](datasets.md) |
| Benchmarks and comparison evidence | `crates/jope-bench/`, `crates/jope-arena/`, `benchmarks/` | [Same-cut benchmarks](same-cut-bench.md) |
| Rust observatory and React/Vite dashboard | `crates/jope-observatory/`, `web/training-dashboard/` | [Dashboard](dashboard/README.md) |
| CLI and repository verification | `crates/jope-cli/`, `crates/jope-records/`, `crates/jope-ci/` | [Agent contract](../AGENTS.md) |
| CI, release, systemd and host maintenance | `deploy/ci/`, `deploy/release/`, `deploy/systemd/`, `deploy/maintenance/` | [Release runbook](release/README.md) |
| Experiment history and research execution | `docs/experiments/`, `docs/research/`, `deploy/dl-flywheel/`, `deploy/dl-trainonly/` | [Experiment index](experiments/README.md), [research workflow](runbooks/research-workflow.md) |

## Performance and audits

The [October 2026 performance and audit review](reviews/hpc-audit-20261021.md)
records the focused fixes, reproducible checks, and remaining work.

- `just required` runs the existing local contribution gate.
- `just audit` installs the SHA-256-pinned Jankurai release and runs its full audit.
  Reports and the repair queue go to `target/jankurai/`; a nonzero audit result
  remains a failed audit. The hosted job still requires the existing
  `JANKURAI_HOSTED_RUNNER=true` opt-in.
- Generated test/build output belongs in the lane's reused `target/` directory.
  Model weights, bulk datasets, and new run captures belong in the existing
  external stores described by the owning runbook.

Historical captures and pinned fixtures are evidence. Before moving or removing
one, establish its owner, all digest/path references, and a verified recovery
route under [the cleanup contract](../MASTER_CLEAN_UP.md). Source layout cleanup
must preserve those bindings.
