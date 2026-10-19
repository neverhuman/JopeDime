# Validation and evidence matrix

This is an execution specification. Commands below are not evidence of having run them. Run in the claimed current checkout under existing resource controls. `W` is defined in WORKPLAN.md; on the user's Mac prefix shell invocations with `rtk`. On xbabe2 prepend the existing fleetcap wrapper to resource-heavy commands; preserve stdout/stderr, exit status, UTC interval and source identity.

## Baseline, ownership and current GitHub state

```sh
git -C "$W" status --short --branch
git -C "$W" rev-parse HEAD
git -C "$W" worktree list --porcelain
git -C "$W" log --all --not --remotes --format='%H %D %s'
/home/ubuntu/.local/bin/gh-role writer api repos/neverhuman/JopeDime/commits/main --jq .sha
/home/ubuntu/.local/bin/gh-role writer pr list -R neverhuman/JopeDime --state open --json number,headRefOid,baseRefName,reviewDecision,mergeStateStatus,url
/home/ubuntu/.local/bin/gh-role writer run view 39289511196 -R neverhuman/JopeDime --json headSha,status,conclusion,jobs,url
```

Expected: correct owned tree; no unexplained dirty assets; all refs inventoried; real remote head; original receiving-main run terminal success before C00 completion. Missing required job, queued/cancelled/failed/unknown or foreign head is not PASS. A dirty owned operational file is retained and disclosed; never reset it for a clean screenshot.

## Documentation packet — run now; no training or model download

```sh
cd "$W"
git diff --check
bash deploy/ci/ip-scrub.sh
cargo run --locked -p jope-records -- repo-layout
cargo run --locked -p jope-records -- readme-benchmarks --check
cargo run --locked -p jope-records -- paper-trace
cargo run --locked -p jope-models -- verify --manifests-only
```

Use only the existing reused target and `CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only`; no global `CARGO_TARGET_DIR`. Fleet wrapper example:

```sh
/home/ubuntu/xbabe2_kicker/fleetcap/run-capped --name jope-closure-docs -- \
  nice -n 10 env PATH=/home/ubuntu/.local/bin:/home/ubuntu/actions-runners/jarambit-xbabe2/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
  CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo run --locked -p jope-records -j4 -- paper-trace
```

Verify PATH/toolchain availability before command use; do not switch Rust version to get a pass. Preserve generated README arena/samecut bytes. Check all new local Markdown links resolve to tracked files and anchors where present; do not crawl arbitrary remote links or scan secrets. Independently compare every README command to actual CLI help/source. The current bundle verb is `verify`, not an invented `check` alias.

The existing full scope validator is `python3 docs/plans/major-release-20261018/validate_gp_scope.py`. Capture its real output; an MR09 receipt dependency/source failure remains a blocker for scope acceptance, not a docs-patch failure to be hidden. This validator proves mapping completeness, never implementation or science. Do not add tests that merely re-assert prose.

## P02 models and pathway qualification — after source exists

```sh
cargo test --locked -p jope-models --test release_set
cargo run --locked --bin jope -- models verify --manifests-only
cargo run --locked --bin jope -- models fetch hyperion-v7
cargo run --locked --bin jope -- models verify hyperion-v7
cargo run --locked --bin jope -- models path hyperion-v7
cargo test --locked -p jope-pipeline --test pathway_plumbing
cargo test --locked -p jope-pipeline --test model_build_flags
```

`release_set` is a proposed new target; absent target is unfinished C03. PR file-backed tests do not contact S3 or require credentials. The three online bundle commands require an admitted host/read credential and an empty separately scoped cache for cold-path evidence; they do not establish native Kumo serving compatibility. The release-set CLI spellings must be added from actual final `--help`, not copied as if already implemented.

Expected artifacts: release manifest/schema, exact-version transport/readback receipt, verified cache manifest and atomic install, legacy/native incompatibility refusals, actual encoder→policy→GP execution log, emitted fitted predictor parity. A CLI refusal test proves refusal only. Record license restrictions, CPU/CUDA support and runtime prerequisites explicitly.

## AutoML, ensemble, refit, and test-score — JA06 approved execution window

```sh
cargo test --locked -p jope-pipeline --lib model_build::
cargo test --locked -p jope-pipeline --test model_build_flags
cargo test --locked -p jope-pipeline --test model_build_automl
cargo test --locked -p jope-pipeline --test model_build_ensemble
cargo test --locked -p jope-pipeline --test model_build_persistence
```

The last three targets are proposed. Confirm actual names after owner handoff. Require nonzero collection and complete case map. Existing 30/11/39-test logs predate later corrections and cannot certify their new source. Include real numerical fitting, not only fake fitter callbacks; independent group/fold/label poisoning checks; stored member artifacts; fresh-process prediction, classification class/probability order; explicit unsupported roster entries; physical fit counters and global-budget proof. Run no paid model/GPU work implicitly from tests.

## Neural source and experiment contracts

Before importing a warm-resume adapter, run current `deploy/dl-flywheel/check_source.py --help`/source inventory entry points as implemented, and relevant existing `tests_native/test_neural_native_train.py`, batch-stream/admission/outbox tests under the owning runner's documented environment. Do not invent test entrypoint CLI options or call an external script's main on real parent state without admission. A header-only/schema/AST test is distinct from actual state restoration and forward/optimizer execution.

Expected: bound producer source and checkpoint sidecars, model/optimizer/RNG/stream consistency, no partial mutation on failed restore, actual parent parameter identity before any update, distinct fork if hyperparameters change, finite resource and artifact retention caps. Later live receipts report actual updates, changed parameters, train/dev curves, new measured labels and fresh-feedback count. Terminated/partial runs preserve terminal state, not completed status.

The scientific pilot must produce preregistration hash, authorized input/family/split/evaluator identity, all-arm actual cost and full outcomes, independent replay/export parity and family-cluster confidence intervals. Unknown, failed, zero-label, or reused-development outcomes do not become learning improvement.

## MR47 / original scope

Execute each task's `exact_acceptance_test` from the authoritative MR tracker after its dependency gates and source exist, under its owner/lease/resource envelope. `MR_TASKS.json` retains every original command and acceptance case for review. Verify the named `--exact` filter collected the intended case; exit0/zero tests is a failure of acceptance. Receipt case map must cover all clauses, including negative/failure/runtime/export/experimental witnesses. MRFV must separately cover all47 tasks and the original66 ledger, final source/build/main and all runtime/features; no schema edit can waive a case.

## UI, fast lanes and final full-main

Use actual package scripts in `web/training-dashboard/package.json` and `docs/dashboard/README.md`; `observatory-ui` is admitted inside ci.yml, not a new unadmitted workflow. `.github/workflows/ci.yml` has10 job definitions expanding to11 executions, including two rust-test shards; eight fast-lane receipts plus changes/UI/aggregate. All required UI tests need nonzero evidence. Preserve the existing runner admission and aggregate rules.

Run `just ci-pr` only when ready for the final declared local tier, then hosted exact-head required lanes through the single PR. Full release validation additionally runs the current `full-main` families, Python/emitter/numerical/feature/admission/custody/recovery tests and admitted host runtime cases; offline PR smoke is not that evidence. Do not rerun an unchanged expensive suite once qualified unless new changes/failure/unresolved concern warrants it. Do not cancel legitimate workflows or dispatch blindly for queue delay.

## Review / merge / main / release

Capture native PR, every commit's GitHub-associated author, exact-head reviews and all required checks. Independent reviewer posts native approval only on exact passing head. Normal merge command:

```sh
/home/ubuntu/.local/bin/gh-role writer pr merge <PR> -R neverhuman/JopeDime \
  --rebase --match-head-commit <reviewed-full-head>
```

No admin/force/reviewer impersonation. Read actual receiving-main SHA and newly created workflows, then validate source/tree, runtime and release-set cold path at that receiving SHA. Preserve timestamps for prior failed/cancelled attempts. Release dry-run/publish invocations come from `docs/release/README.md` and actual release.sh help; do not publish an invented version or imply source0.3 means a signed release exists.

## Artifact receipt requirements

For each validation: full source/tree/build, dirty state, owner, command/env/toolchain/hardware, allowed input/config/protocol/seed hashes, start/end/caps/actual cost, exit, collected/executed/pass/fail/skip/unknown counts, complete log/artifact digests and scope limits. Git wave records include exact S3 bucket/key/VersionId and compressed/decoded/member hashes plus independently restored contents. Keep old failures/source IDs. Hashes/paths copied from an order are not verified until read back.

Cleanup requires independent restoration and fresh consumer/retention closure in addition to merge inclusion. Validation failure never authorizes deleting assets. Root/paper/review trees and all symlinks remain protected until individually eligible.
