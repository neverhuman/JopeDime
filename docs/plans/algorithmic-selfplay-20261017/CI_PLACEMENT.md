# CI placement and evidence contract

This document maps every declared CPU CI family to approved jobs, hosts, storage domains and limits at source C=`1a8a66c502ca2cede77c197b4f5b5bbc24ff398e`. It records supplied historical workflow outcomes and the newly executed offline tooling proof without turning configured placement into current physical-host evidence.

Prepared at 2026-10-18T03:27:07.293628+00:00; staged input/output hashes and raw offline test outputs are in `/home/ubuntu/jope-evidence/spec-integration-20261017/work-packets/staged-docs/i63-i64-staged-receipt.json`. Intended repository destination: `docs/plans/algorithmic-selfplay-20261017/CI_PLACEMENT.md`. I64 depends exactly on I01, I16 and I63. I01's accepted checkpoint is retained in PR_ACCEPTANCE.md; I16 acceptance and the later I63 lifecycle remain unverified here. This additive document and offline proof do not complete live I64 qualification.

## Approved physical allocation and trust domains

| Host / lane | Registered labels | Per-job hard ceiling | Writable domain / independent volumes |
| --- | --- | --- | --- |
| xbabe1 / ci | self-hosted, linux, x64, xbabe1-ci, jope-ci | 16 CPUs, 16 GiB RAM, swap 0 | pr-ci; own work, Cargo registry, target and uv volumes. |
| xbabe1 / ci2 | self-hosted, linux, x64, xbabe1-ci2, jope-ci | 16 CPUs, 16 GiB RAM, swap 0 | pr-ci2; independent of ci. |
| xbabe2 / full | self-hosted, linux, x64, xbabe2-full, jope-full-cpu, jope-ci | 16 CPUs, 48 GiB RAM, swap 0 | main-full on xbabe2; own persistent volumes. |
| xbabe3 / full | self-hosted, linux, x64, xbabe3-full, jope-full-cpu, jope-ci | 16 CPUs, 48 GiB RAM, swap 0 | main-full on xbabe3; separate physical volumes from xbabe2. |

Only these host/lane pairs are approved. The fixed role case is at `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/runner/common.sh:86`; the earlier packet's line-85 citation was a comment, corrected by ring3 I64-N1. Old ci3/ci4/GPU/kernel roles and the undeployed sizing proposal are not admitted. Each lane executes one job at a time; full slots retain their host-local exclusive CPU lock and populated-ASM-cgroup refusal.

`jope-ci` can schedule fast PR/manual jobs on either xbabe1 fast slot OR either full slot. A fast job on a full slot uses that slot's 16-CPU/48-GiB/no-swap ceiling and main-full volumes. Full-lane volumes can therefore see untrusted ci.yml and trusted-main jobs, as the runbook explicitly states. Ephemeral JIT registration does not cleanse persistent volumes or establish stronger trust separation. No new main-only secret/tool cache may be exposed by assuming otherwise.

Containers use uid 1001, dropped capabilities, no-new-privileges, read-only root, pids limit 4096 and 4-GiB temporary filesystems, as specified by the runbook. They have no GPU devices, host PAT, Docker socket, training/publishing/S3 credentials; SDK mounts are read-only. This source configuration is not a fresh inspection of running containers. Do not inspect credential bytes or Docker environment dumps.

## Source binding, persistent targets and capacity guards

Runner provenance binds installed reviewed scripts, entrypoint and provisioned image identity. For pull_request, deployed files bind the exact PR base; for push/workflow_dispatch/other events, they bind the exact checkout under test. Preserve separate source SHA, checkout SHA, base SHA, ref/event and deployed/image fingerprints. The action verifies actual cgroup CPU/memory/swap and admission-policy values. It does not hash every image file or prove independent process provenance from a hostname string.

`.github/actions/cargo-target/action.yml:10` selects the existing lane and exports `CARGO_TARGET_DIR=/var/cache/jope/target/$lane`. Preserve one reused target for each runner allocation, with run/attempt/group-scoped reports. No fresh target per family, concurrent writable sharing, global target override or build output under /tmp. Keep source/code cache trust bounds and full-lane compiler-cache evidence; its hits/misses are performance facts, not correctness or native qualification.

The installed runner startup threshold is **JOPE_RUNNER_MIN_FREE_GB=50** at `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/runner/common.sh:13`. `disk-guard.sh:13-15` reads available 1-KiB blocks using df -Pk and computes `free_gb = free_kb / 1024 / 1024` with integer division. Despite variable/log spelling “GB,” the startup comparison uses **whole GiB**. It checks the Docker root, runner state and existing /tmp filesystem. Less than 50 whole GiB refuses start; preserve the threshold and arithmetic.

A separate between-family reserve at `deploy/ci/full-lane.sh:9-10` compares byte availability against 50,000,000,000. That decimal-byte reserve is additional protection and is not the startup whole-GiB guard. Do not silently equate the units.

Low disk, an offline runner, unknown installed provenance, absent SDK, a busy/foreign allocation, failed cgroup/swap pin or a protected consumer remains BLOCKED with dated owner/evidence. Do not restart, reprovision, lower the guard, change timers/caps, delete/prune volumes/targets/evidence, or manufacture free-space measurements to make placement green. The offline mocks below prove refusal logic, not current free capacity on any host.

## Fast workflow placement

All ci.yml jobs request `[self-hosted, linux, x64, jope-ci]`. “Fast allocation” below means any one of the four approved lanes above, using its actual host-specific domain and cap; the label alone does not identify which host ran a job. The admission hook enforces the allowed workflow/event/domain combination. Every manual workflow_dispatch selects every fast lane; a pull_request uses the exact recorded path plan.

| Job / execution unit | Family and selection | Approved allocation / limits | Workflow wall timeout |
| --- | --- | --- | --- |
| changes | Declares eight-lane path applicability and immutable source/checkout/run/attempt identity. Static planning, no test-count claim. | Fast allocation: xbabe1 ci/ci2, 16 CPU/16 GiB/swap 0, or xbabe2/3 full, 16 CPU/48 GiB/swap 0. | 10 min |
| hygiene | Formatting, manifests, coverage/discovery, offline CI parser/runner tests and current static guards. Not feature-complete Rust proof. | Same fast allocation/domain rule. | 20 min |
| rust-check | Workspace all-target check plus systemd_unit_paths execution; preserve its nonzero Rust test counts. | Same fast allocation/domain rule. | 45 min |
| rust-test (1) | PR-scope nextest, explicit jope-cli/asm and jope-cli/qualified-training-context, partition slice:1/2. | Same fast allocation/domain rule; independent reused target of its admitted lane. | 45 min |
| rust-test (2) | Identical unsharded selection, partition slice:2/2; both shards must cover it exactly once. | Same fast allocation/domain rule; no shared writable target with concurrent shard. | 45 min |
| emitters | Strict Python/Rust/C++/JavaScript artifact parity, fitted C++ sanitizer and C++20 checks selected by the existing lane. | Same fast allocation/domain rule; actual compilers/SDK support must be recorded. | 30 min |
| inputs | Declared synthetic/input/catalog checks. Backend none does not imply tests executed. | Same fast allocation/domain rule. | 20 min |
| python | Explicit maintained pytest and offline research/integration subfamilies, Ruff, non-xbabe2 marker selection and their own retained subreceipts. | Same fast allocation/domain rule; no live campaign/credentials. | 25 min |
| e2e-smoke | Existing end-to-end synthetic pipeline/schema/emitted-artifact checks; not live serving or a campaign. | Same fast allocation/domain rule. | 25 min |
| aggregate | always(); validates eight lane dependencies plus both rust-test artifacts, hashes and exact unsharded coverage. | Same fast allocation/domain rule. | 20 min |

There are eight lane names plus aggregate, with rust-test expanded into two matrix legs: **10 jobs**, not nine. Matrix fail-fast is false; a cancelled or missing required shard cannot disappear from coverage. JOPE_SOURCE_SHA binds the PR head while GITHUB_SHA may be the synthetic merge checkout. Record both. Four Cargo build jobs at the fast workflow level are compilation concurrency, not a separate CPU isolation guarantee.

## Trusted-main CPU family placement

`full-main.yml` is named full-main-cpu. Its cpu matrix runs only when `github.ref == 'refs/heads/main'`, with cpu-1 and cpu-2 and fail-fast=false. Each group and the later aggregate requests `jope-full-cpu`, so only xbabe2/full or xbabe3/full is approved: 16 CPUs, 48 GiB RAM, swap 0, that host's main-full domain. Do not bind cpu-1 to xbabe2 or cpu-2 to xbabe3 by name; scheduler assignment and actual provenance determine the physical host.

The actual workflow at C sets **130 minutes per CPU group** and 20 minutes for aggregate. The runbook's older 120-minute wording is dated; it does not override the workflow. Family deadlines below remain independently bounded with TERM and 15-second kill grace; failures do not prevent later declared families from reporting. Full workflow Cargo build jobs are 8, with one-thread BLAS/OMP/Rayon.

| Family | Matrix group / approved allocation | Exact selection boundary | Family deadline / backend |
| --- | --- | --- | --- |
| full-build-reuse | cpu-1; xbabe2/full OR xbabe3/full; main-full; 16 CPU/48 GiB/swap 0 | Cargo metadata + jope-paths jope-units-render build/artifact selection; no test-count claim. | 300s + 15s kill grace; none; no invented test count |
| full-default | cpu-1; xbabe2/full OR xbabe3/full; main-full; 16 CPU/48 GiB/swap 0 | Workspace nextest; explicit jope-cli/asm,jope-cli/qualified-training-context; training-proof Rust validation. | 1500s + 15s kill grace; nextest; nonzero tests required |
| full-doctests | cpu-1; xbabe2/full OR xbabe3/full; main-full; 16 CPU/48 GiB/swap 0 | cargo test --workspace --doc --no-fail-fast --locked. | 600s + 15s kill grace; rust; nonzero tests required |
| full-e2e | cpu-1; xbabe2/full OR xbabe3/full; main-full; 16 CPU/48 GiB/swap 0 | fast-lane.sh e2e-smoke; synthetic integration/artifact checks. | 300s + 15s kill grace; none; no invented test count |
| full-emitters | cpu-1; xbabe2/full OR xbabe3/full; main-full; 16 CPU/48 GiB/swap 0 | JOPE_FULL_EMITTERS_PARALLEL=1 fast-lane.sh emitters; strict compiled parity. | 300s + 15s kill grace; rust; nonzero tests required |
| full-inputs | cpu-1; xbabe2/full OR xbabe3/full; main-full; 16 CPU/48 GiB/swap 0 | fast-lane.sh inputs; declared input verification. | 300s + 15s kill grace; none; no invented test count |
| full-neural | cpu-2; xbabe2/full OR xbabe3/full; main-full; 16 CPU/48 GiB/swap 0 | jope-parallel nextest --features neural; supported LibTorch CPU SDK; default-feature synthetic-child collection is separate. | 900s + 15s kill grace; nextest; nonzero tests required |
| full-policy | cpu-1; xbabe2/full OR xbabe3/full; main-full; 16 CPU/48 GiB/swap 0 | jope-policy nextest --features testing,context-producer; training-proof policy validation. | 600s + 15s kill grace; nextest; nonzero tests required |
| full-python | cpu-1; xbabe2/full OR xbabe3/full; main-full; 16 CPU/48 GiB/swap 0 | Coverage-selected maintained Python tests; locked dev/cpu environment; pytest -m not xbabe2; training-proof Python validation. | 600s + 15s kill grace; pytest; nonzero tests required |
| full-s3-offline | cpu-1; xbabe2/full OR xbabe3/full; main-full; 16 CPU/48 GiB/swap 0 | jope-arena nextest --features s3; offline fixtures, no live S3 credentials. | 300s + 15s kill grace; nextest; nonzero tests required |
| full-scratch | cpu-2; xbabe2/full OR xbabe3/full; main-full; 16 CPU/48 GiB/swap 0 | feat-math, jope-parallel, jope-dime-regressor nextest with each package gpr-scratch-reuse feature. | 1500s + 15s kill grace; nextest; nonzero tests required |

All 11 families must occur exactly once across the two groups. The table follows `deploy/ci/full-main.py:27-30` and the exact declared coverage rows. The trusted-main aggregate re-verifies each group's source/checkout/run/attempt/event, command, original evidence hashes and complete family partition. Missing, duplicate, failed, cancelled, unavailable, dirty or zero-required execution refuses the relevant result; no green mosaic across attempts.

Full-default is explicitly configured with the two jope-cli features shown above; “default” in the family name does not mean no Cargo feature flags. full-scratch's optimization feature is a separate variant. full-policy needs both testing and context-producer. full-neural first discovers the supported LibTorch CPU environment and emits neural-discovery.json; absent SDK exits unavailable. It also collects a same-source no-default-features synthetic child executable before the neural build; that helper's collection is not executed learning proof.

Neither a configured full-neural row nor a successful group API summary alone proves actual optimizer updates, committed journals/checkpoints, export/reload or next actor use. Require each promised stage's accepted I12/I13/I14/I31/I32 implementation and source/weight-bound receipts. Original native numerical/GPU dispatch remains held. No CUDA/BF16/multi-rank or private real-corpus qualification is inferred.

## Main guard, host-only exclusions and nonexecution

main-guard.yml's guard job requests jope-full-cpu with a five-minute timeout, using one approved full allocation and its normal caps/domain/provenance. It checks each pushed main commit's merged-PR association and fails closed on lookup errors. It detects after-the-fact violations; it is not proof of preventive branch protection or reviewer-process independence.

The host-real-corpus coverage disposition is a separate manual host lane against verified read-only custody, not a ci.yml/full-main job. Ignored/private host-input tests and GPU/kernel/retired runner roles stay excluded unless separately and concretely admitted. No role/credential provisioning or real-host probes occurred here.

At `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/ci/aggregate.py:166` a legitimately excluded PR lane is recorded **skipped-by-path**, with tests null. The path plan must match exact changed paths and declared filters; the job must be skipped with no execution receipt. A forged plan, executed artifact for an excluded lane, an excluded lane reported PASS, or a selected lane missing execution is rejected. Manual dispatch selects all fast lanes and cannot inherit PR-only path exemptions.

Receipt counts must be recomputed from original logs/JUnit/nextest collection. Required test families need nonzero collected and executed counts, no failures and complete known counts. “Unknown,” “not selected,” “unavailable,” “ignored,” “cancelled” and “zero tests” are distinct from passing. Backend-none static/build/input jobs retain their actual exit/result without inventing executed tests. Zero-test binaries/doctests never add coverage.

## Supplied observed workflow runs

| Run / event | Full source head | Attempt and terminal result | Observed job closure |
| --- | --- | --- | --- |
| `39087080072` / pull_request | `407c4248601d6e54145fec581093e0fce59ff084` | 1; completed/success | 6 success, 4 skipped; supplied local API readback |
| `39087091490` / workflow_dispatch | `407c4248601d6e54145fec581093e0fce59ff084` | 1; completed/success | 10 success; supplied local API readback |
| `39089266033` / receiving main-guard | `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e` | 1; completed/success | 1 success; supplied local API readback |
| `39089266014` / receiving full-main-cpu | `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e` | 1; completed/success | 3 success; supplied local API readback |

The supplied local readbacks contain these complete job outcomes:

| Run / attempt | Job ID | Job | Observed status / conclusion | Physical host, lane and measured caps |
| --- | --- | --- | --- | --- |
| 39087080072 / 1 | `117555937136` | hygiene | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39087080072 / 1 | `117555937336` | changes | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39087080072 / 1 | `117556043405` | rust-check | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39087080072 / 1 | `117556043472` | rust-test (1) | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39087080072 / 1 | `117556043473` | rust-test (2) | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39087080072 / 1 | `117556044229` | python | completed/skipped | UNKNOWN in supplied summary; require original runner provenance. |
| 39087080072 / 1 | `117556044511` | inputs | completed/skipped | UNKNOWN in supplied summary; require original runner provenance. |
| 39087080072 / 1 | `117556044602` | e2e-smoke | completed/skipped | UNKNOWN in supplied summary; require original runner provenance. |
| 39087080072 / 1 | `117556044722` | emitters | completed/skipped | UNKNOWN in supplied summary; require original runner provenance. |
| 39087080072 / 1 | `117559104342` | aggregate | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39087091490 / 1 | `117556194977` | hygiene | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39087091490 / 1 | `117556194985` | changes | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39087091490 / 1 | `117556491452` | python | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39087091490 / 1 | `117556491454` | e2e-smoke | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39087091490 / 1 | `117556491460` | emitters | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39087091490 / 1 | `117556491477` | inputs | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39087091490 / 1 | `117556491493` | rust-test (1) | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39087091490 / 1 | `117556491510` | rust-test (2) | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39087091490 / 1 | `117556491556` | rust-check | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39087091490 / 1 | `117561941170` | aggregate | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39089266033 / 1 | `117562284242` | guard | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39089266014 / 1 | `117562284182` | cpu (cpu-2) | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39089266014 / 1 | `117562284307` | cpu (cpu-1) | completed/success | UNKNOWN in supplied summary; require original runner provenance. |
| 39089266014 / 1 | `117566232049` | aggregate | completed/success | UNKNOWN in supplied summary; require original runner provenance. |

These are historical API-reported outcomes at the named source heads, not a new remote query. The JSON summaries contain no physical runner name/host/lane, cgroup provenance, per-family original command/log/count closure or independent artifact readback hash. Those observation fields remain **UNKNOWN**, even when the run and all jobs completed/success. The approved placement tables are source facts, not proof that each host executed every family. Read actual artifact/provenance and job/service receipts before making a physical placement or native coverage claim.

The I01 PR run's four path exclusions were emitters, inputs, python and e2e-smoke. Its manual exact-head run executed all 10 jobs. Receiving main-guard and both full CPU groups plus aggregate subsequently succeeded at C. These receipts close the documented I01 receiving checkpoint; they do not close later integration-head CI or current fleet admission. Preserve G1's squash exception and G2's OPEN/documented review-publication provenance from PR_ACCEPTANCE.md.

## Newly executed offline CI tooling proof

| Offline suite | Source-inventoried / unittest run | Passed / failed / skipped | Exit | Original log SHA-256 (stdout; stderr) |
| --- | --- | --- | --- | --- |
| test-aggregate.py | 18 / 18 | 18 / 0 / 0 | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; `46dbe77dcfb2b82669a348a2fa499376d63777e501a0cabcbf1e4f604c024b2a` |
| test-receipt.py | 6 / 6 | 6 / 0 / 0 | 0 | `601e0858a551a7592da0ed505d4a73e92ed6afbc6b5cf7bfc1c842434a52bc55`; `1a1e0050c39bfd8abe97ad48e912d840b692ed239cb6c1346ad7be6f5766ebef` |
| test-full-main.py | 22 / 22 | 22 / 0 / 0 | 0 | `c8fa13c9de43c9d5a3cc1414a5c2d7bee42bbfa01631bdf74f7898296589f75c`; `cf6814a167ec1c6ebf7316c9612bee85455ff0ab0f48bf2f12fe55b9502fe3d4` |
| test-runner.py | 25 / 25 | 25 / 0 / 0 | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; `84cd970f8f4f733cfeb19a4ce48f71b8bb2af3d6c2f5b08726388f5a5d0e4f35` |

The four exact requested commands used python3, their absolute worktree paths and -v, with PYTHONDONTWRITEBYTECODE=1. Before execution, the suite's source test names were inventoried; each suite's unittest count matched that inventory. Total: **71 tests passed, 0 failed, 0 skipped**. The staged receipt embeds stdout/stderr and their SHA-256, with separate raw log files, exact argv/cwd/env, UTC, elapsed time, source and observed Git heads.

The tests ran on `xbabe2` with recorded affinity `20,21`, using fake fixtures in the existing lane's target/tmp. That is local offline parser/mock evidence, not a workflow job or a new admitted fleet allocation. CI/runner/workflow bytes stayed unchanged even as agy advanced the integration branch from `0c06b14539a9a700091d046eef14f1c90f773482` to `0c06b14539a9a700091d046eef14f1c90f773482`. No Cargo build/test, actual Docker/systemd/runner change, network dispatch, IAM/S3 operation or GPU campaign was performed.

## Evidence fields, refusal and handoff

For every future promised executed family retain: full source/checkout/base/ref/event/workflow identity; RunId/Attempt/JobId; UTC start/end and conclusion; actual runner name AND independently bound physical host/lane/domain; requested labels and measured cgroup CPU/memory/swap/cpuset; installed/image/admission-policy hashes and external readback locator; disk-guard/SDK state; exact command/features/toolchain; unsharded/sharded collection, executed/pass/fail/ignored/filtered counts and all path/feature/runtime exclusions; original streams/JUnit; publication/aggregate inventories and SHA-256.

Keep “source configured,” “historically observed,” “current executed proof” and “proposed/held” separate. Preserve failure receipts. Missing/ambiguous provenance, stale source, changed root/lease, capacity refusal or a proposed infrastructure change stops acceptance and returns the exact blocker to agy. No blind restart, guard/cap change, storage cleanup, cache deletion or authority expansion is an I64 implementation step.

| Full-pin source anchor | Boundary |
| --- | --- |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:docs/runbooks/runner.md:8` | Approved host/slot/cap/domain allocation |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:docs/runbooks/runner.md:22` | Persistent domains and full-lane untrusted ci.yml exposure |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/runner/common.sh:13` | Fixed 50 whole-GiB startup guard |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/runner/common.sh:86` | Actual role case; correct ring3 anchor |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/runner/disk-guard.sh:13` | df -Pk available block read |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/runner/disk-guard.sh:14` | Whole-GiB integer arithmetic |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/runner/disk-guard.sh:15` | Startup refusal comparison |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:.github/actions/cargo-target/action.yml:10` | Reused lane target selection |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:.github/actions/runner-provenance/action.yml:17` | PR-base versus exact-checkout binding |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:.github/actions/runner-provenance/action.yml:45` | Actual admission/cgroup verification |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:.github/workflows/ci.yml:20` | Fast workflow jobs |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:.github/workflows/ci.yml:115` | Rust-test shard matrix |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:.github/workflows/ci.yml:260` | Fast aggregate |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:.github/workflows/full-main.yml:23` | Main-only CPU group jobs and 130-minute cap |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/ci/full-main.py:27` | Every declared family partitioned between cpu-1 and cpu-2 |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/ci/full-lane.sh:9` | Separate decimal-byte per-family disk reserve |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/ci/full-lane.sh:60` | Explicit default/scratch/policy/neural selections |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/ci/aggregate.py:124` | Nonzero required-test rejection |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/ci/aggregate.py:141` | Identity/path-plan/dependency validation |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/ci/aggregate.py:166` | skipped-by-path disposition |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/ci/coverage.json:9809` | Declared full-neural lane |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/ci/coverage.json:9854` | Declared full-policy lane |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-engine/src/task_contract.rs:313` | Original native numerical dispatch hold |

agy owns integration and current live evidence collection. The acceptance document remains partial until I16/I63 dependencies, actual promised placement/feature/runtime evidence and the new integration lifecycle are accepted. This staged file plus the four offline suites is the completed document/tooling portion only.
