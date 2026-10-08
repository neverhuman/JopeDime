# 1.0.0 version-bump inventory

Prepared against main `a3aeb34fae3c7ff7723bdabf0b771104dfbf8bb0`. The existing unpublished release candidate
`release/v1.0.0` at `fbf39c3cd1f86ac5e68cd4d513c7942831291bbc` already contains the version bump and its lock/identity work.
ROOT retains that work. This inventory proposes no additional version, dependency, lock, evaluator-identity or snapshot edit.

All 36 tracked files named `Cargo.toml` or `pyproject.toml` are listed below. Package versions and workspace inheritance
come from parsed TOML; dependency version requirements are not package release versions.

## Direct product-version edits

| Manifest | Field | Main | Target |
| --- | --- | --- | --- |
| [Cargo.toml](../../Cargo.toml) | `workspace.package.version` | `0.2.0` | `1.0.0` |
| [jope-campaigns/Cargo.toml](../../crates/jope-campaigns/Cargo.toml) | `package.version` (`jope-dime-regressor`) | `0.2.0` | `1.0.0` |
| [jope-engine/Cargo.toml](../../crates/jope-engine/Cargo.toml) | `package.version` (`jope-parallel`) | `0.2.0` | `1.0.0` |
| [python/pyproject.toml](../../python/pyproject.toml) | `project.version` (`jope`) | `0.1.0` | `1.0.0` |

## Product crates inheriting the workspace version

These 17 manifests retain `version.workspace = true`; their effective package version changes from 0.2.0 to 1.0.0 with the root.
Together with the two explicit product crates above, the product workspace contains 19 release packages.

| Manifest | Package | Release action |
| --- | --- | --- |
| [crates/battle-gpu/Cargo.toml](../../crates/battle-gpu/Cargo.toml) | `battle-gpu` | Keep inheritance; effective `1.0.0` |
| [crates/feat-math/Cargo.toml](../../crates/feat-math/Cargo.toml) | `feat-math` | Keep inheritance; effective `1.0.0` |
| [crates/hyperion/Cargo.toml](../../crates/hyperion/Cargo.toml) | `hyperion` | Keep inheritance; effective `1.0.0` |
| [crates/jope-arena/Cargo.toml](../../crates/jope-arena/Cargo.toml) | `jope-arena` | Keep inheritance; effective `1.0.0` |
| [crates/jope-bench/Cargo.toml](../../crates/jope-bench/Cargo.toml) | `jope-bench` | Keep inheritance; effective `1.0.0` |
| [crates/jope-ci/Cargo.toml](../../crates/jope-ci/Cargo.toml) | `jope-ci` | Keep inheritance; effective `1.0.0` |
| [crates/jope-cli/Cargo.toml](../../crates/jope-cli/Cargo.toml) | `jope-cli` | Keep inheritance; effective `1.0.0` |
| [crates/jope-datasets/Cargo.toml](../../crates/jope-datasets/Cargo.toml) | `jope-datasets` | Keep inheritance; effective `1.0.0` |
| [crates/jope-encoders/Cargo.toml](../../crates/jope-encoders/Cargo.toml) | `starforge` | Keep inheritance; effective `1.0.0` |
| [crates/jope-learning/Cargo.toml](../../crates/jope-learning/Cargo.toml) | `jope-learning` | Keep inheritance; effective `1.0.0` |
| [crates/jope-models/Cargo.toml](../../crates/jope-models/Cargo.toml) | `jope-models` | Keep inheritance; effective `1.0.0` |
| [crates/jope-motifs/Cargo.toml](../../crates/jope-motifs/Cargo.toml) | `jope-motifs` | Keep inheritance; effective `1.0.0` |
| [crates/jope-paths/Cargo.toml](../../crates/jope-paths/Cargo.toml) | `jope-paths` | Keep inheritance; effective `1.0.0` |
| [crates/jope-pipeline/Cargo.toml](../../crates/jope-pipeline/Cargo.toml) | `jope-pipeline` | Keep inheritance; effective `1.0.0` |
| [crates/jope-policy/Cargo.toml](../../crates/jope-policy/Cargo.toml) | `jope-policy` | Keep inheritance; effective `1.0.0` |
| [crates/jope-records/Cargo.toml](../../crates/jope-records/Cargo.toml) | `jope-records` | Keep inheritance; effective `1.0.0` |
| [crates/numpy-pcg64/Cargo.toml](../../crates/numpy-pcg64/Cargo.toml) | `numpy-pcg64` | Keep inheritance; effective `1.0.0` |

## Independent and frozen package versions

Keep these versions. Vendored SmartCluster has its own release version; standalone Lambda imports and the legacy tool retain their source-history versions.
Changing the product’s version does not authorize a blanket bump of these packages.

| Manifest | Package | Main / retained version | Reason |
| --- | --- | --- | --- |
| [crates/smartcluster/jain-smartcluster-core/Cargo.toml](../../crates/smartcluster/jain-smartcluster-core/Cargo.toml) | `jain-smartcluster-core` | `8.0.0` | Vendored SmartCluster |
| [crates/smartcluster/jain-smartcluster-daemon/Cargo.toml](../../crates/smartcluster/jain-smartcluster-daemon/Cargo.toml) | `jain-smartcluster-daemon` | `8.0.0` | Vendored SmartCluster |
| [crates/smartcluster/jain-smartcluster-fabric/Cargo.toml](../../crates/smartcluster/jain-smartcluster-fabric/Cargo.toml) | `jain-smartcluster-fabric` | `8.0.0` | Vendored SmartCluster |
| [crates/smartcluster/jain-smartcluster/Cargo.toml](../../crates/smartcluster/jain-smartcluster/Cargo.toml) | `jain-smartcluster` | `8.0.0` | Vendored SmartCluster |
| [crates/smartcluster/jain-worker/Cargo.toml](../../crates/smartcluster/jain-worker/Cargo.toml) | `jain-worker` | `8.0.0` | Vendored SmartCluster |
| [crates/smartcluster/jainnode/Cargo.toml](../../crates/smartcluster/jainnode/Cargo.toml) | `jainnode` | `8.0.0` | Vendored SmartCluster |
| [crates/smartcluster/scq/Cargo.toml](../../crates/smartcluster/scq/Cargo.toml) | `scq` | `8.0.0` | Vendored SmartCluster |
| [deploy/lambda/csv/rust/Cargo.toml](../../deploy/lambda/csv/rust/Cargo.toml) | `jope-lambda-probe` | `0.1.0` | Frozen standalone Lambda workspace |
| [deploy/lambda/regression-ops/rust/Cargo.toml](../../deploy/lambda/regression-ops/rust/Cargo.toml) | `jope-regression-assessment` | `0.1.0` | Frozen standalone Lambda workspace |
| [deploy/lambda/regression/rust/Cargo.toml](../../deploy/lambda/regression/rust/Cargo.toml) | `jope-regression-lambda` | `0.1.0` | Frozen standalone Lambda workspace |
| [deploy/lambda/training/teacher-v2/rust/island-engine/Cargo.toml](../../deploy/lambda/training/teacher-v2/rust/island-engine/Cargo.toml) | `jope-island-engine` | `0.1.0` | Frozen standalone Lambda workspace |
| [deploy/lambda/training/teacher-v2/rust/lambda-worker/Cargo.toml](../../deploy/lambda/training/teacher-v2/rust/lambda-worker/Cargo.toml) | `jope-test-adaptive-teacher-lambda` | `0.1.0` | Frozen standalone Lambda workspace |
| [deploy/lambda/training/teacher-v2/rust/s1-proposal/Cargo.toml](../../deploy/lambda/training/teacher-v2/rust/s1-proposal/Cargo.toml) | `jope-s1-proposal-engine` | `0.1.0` | Frozen standalone Lambda workspace |
| [deploy/lambda/watch/Cargo.toml](../../deploy/lambda/watch/Cargo.toml) | `jope-lambda-watch` | `0.1.0` | Frozen standalone Lambda workspace |
| [xtools/registry-dump-legacy/Cargo.toml](../../xtools/registry-dump-legacy/Cargo.toml) | `jain-math-registry-dump` | `0.0.0` | Legacy non-product tool |

## Associated metadata and final checks

- Root `Cargo.lock`: only the 19 product package version fields follow the bump; preserve all third-party package records.
- `python/uv.lock`: align only the local `jope` package version and preserve the dependency resolution.
- The existing candidate includes owner-authorized Lambda working-lock updates and snapshot tree-change records. Preserve that work and the original frozen source hashes; do not repeat a blanket lock update. The main #262 libc-edge integration must survive the candidate’s final rebase.
- Preserve existing historical compatibility fixtures, evaluator lineage receipts and source-custody pins. Their owners must verify any release rebaseline; this list does not create another one.
- Run the maintained version-consistency, Python-lock, Lambda locked/offline metadata and snapshot checks on the integrated candidate. Recheck every listed manifest and lock delta at its exact head.
- Refresh the changelog/README and their source links against the merged PR inventory before release review. Main green alone does not discharge the other criteria in the major-release plan.

No manifest is modified by this document.
