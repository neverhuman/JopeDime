# CI — lanes, budgets, env gates, receipts (as built by WP8)

Ground truth for what runs where. `ci.yml` (hosted, pull requests) is unchanged by WP8; this page
covers it for completeness and documents `main.yml`, `nightly.yml`, the self-hosted runner and the
receipts. Owner steps that remain are at the end.

## Lane matrix

### `ci.yml` — pull requests, hosted `ubuntu-24.04`, `permissions: contents: read`, no `id-token`

| Job | Runs | Budget |
|---|---|---|
| `changes` | path gate (`git diff --name-only` against the PR base) | 3 min |
| `hygiene` | `deploy/ci/check.sh` (cargo metadata, shellcheck, actionlint, gitleaks, family-remote and absolute-path scans, 5 MiB size cap), `cargo-deny check bans licenses sources` | 10 min |
| `rust-check` | `cargo check --workspace --all-targets --locked`, MSRV ≤ 1.97 | 45 min |
| `emitters` | `deploy/ci/emitters.sh` (Python/Rust/C++17/C++20 × g++/clang++/JavaScript, byte-identical sources and bit-identical predictions) + strict portable parity | 30 min |

### `main.yml` — `push: main`, `workflow_dispatch`, `workflow_call` (nightly); `runs-on: [self-hosted, linux, x64, xbabe2-gpu]` (`xbabe2-kernel` for the kernel build); `permissions: contents: read, id-token: write`; `concurrency: main-<ref>` (no cancel)

Every job exports `OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`
(the eval registry refuses to run otherwise) and `JOPE_PARITY_STRICT=1`.

| Job | Runs | Budget | Receipt |
|---|---|---|---|
| `full-rust` | `cargo fmt --check`; `cargo clippy --workspace --all-targets -D warnings`; `cargo nextest list -P main --workspace --message-format json`; `cargo nextest run -P main --workspace --locked -j 32` (2,318 tests listed today, 34 `#[ignore]`); **no-silent-skip** (below) | 90 min | junit + list + run log (artifact) |
| `neural-cuda` | `deploy/ci/neural-env.sh` → `deploy/ci/gpu-qualification.sh`: `cargo clippy -p jope-parallel --features neural --all-targets`; `cargo nextest run -P main -p jope-parallel --features neural` (549 tests incl. the 27 neural-gated); `cargo test -p battle-gpu -- --nocapture` (the log must show `battle-gpu test device:`); `cargo test -p hyperion --features hyperion-cuda` — **blocked** until WP3 adds the `hyperion` crate (`crates/hyperion/host` is excluded from the workspace and points at a source tree that is not in the repo) | 60 min | `jope.cuda-qualification/v1` (artifact + S3) |
| `encoder-parity` | OIDC (`JopeDimeCiRead`) → `cargo run -q -p jope-models --locked -- fetch --pinned --root $JOPE_MODELS_ROOT` (**lands with WP7**; today cargo fails with "package ID specification `jope-models` did not match") → weight gates exported from the bundle paths (every path must exist) → `cargo nextest run -P main -p starforge --features cuda` → no-silent-skip, lane `encoder-parity` | 45 min | junit + list (artifact) |
| `e2e-smoke` | `deploy/ci/e2e-smoke.sh`: toolchain → dataset digests → prepare → step-60000 proposals → `jope-parallel search --evaluations 40 --rng-seed 7 --qualification` → emit best genome → compile (g++/clang++ × C++17/20, rustc, node, python) → predict → bit-compare | 45 min | `jope.e2e-smoke/v1` `target/e2e/receipt.json` (artifact + S3) |
| `kernel-v1-build` | `deploy/ci/kernel-v1-product-build.sh --builder-image jope-kernel-builder:rust1.95-glibc2.35` (docker `--network none` through the socket proxy; `jope-build-receipt --max-glibc 2.36 --forbid-gpu-libraries`) | 40 min | `jope.product_evaluator_build.v1` + `jope.kernel-v1-source/v1` (artifact + S3) |

Receipts are uploaded as workflow artifacts (30 days) and copied to
`s3://veox-jopedime-learning-use2/ci-receipts/<sha>/` by assuming
`arn:aws:iam::714044927654:role/JopeDimeCiRead` (region `us-east-2`) — the only credential any lane
ever holds, and only for the duration of the upload step.

### `nightly.yml` — `0 6 * * *` + `workflow_dispatch`, self-hosted

| Job | Runs | Budget |
|---|---|---|
| `main-lanes` | `main.yml` via `workflow_call` (identical jobs, so a nightly red means the same as a main red) | as above |
| `campaign-smoke` | one Golden46 panel cell, 200 evaluations, seed 7, against `assets/campaigns/baseline.json`: `cargo run -p jope-campaigns -- campaign smoke --panel golden46 --cell 0 --evaluations 200 --seed 7 …`. **`continue-on-error: true` only because the `campaign smoke` subcommand does not exist yet** (the package is still `jope-dime-regressor`; WP5 renames it and adds the subcommand); the step writes a `blocked` receipt and prints a `::warning::` saying so. Remove `continue-on-error` when WP5 lands. | 60 min |
| `parity-matrix` | `JOPE_CXX_MARCH ∈ {x86-64-v3, znver2, native}` × (g++, clang++) × (C++17, C++20) through `deploy/ci/emitters.sh`; every emitted source and prediction bit pattern must equal the committed sha256 pins, so all three legs pin the same digests (`pins.txt` per leg) | 45 min × 3 |
| `audit` | `cargo audit --deny warnings`; `cargo deny --locked check bans licenses sources advisories` | 15 min |
| `sweep` | `cargo sweep --time 7` on the shared target volume (after the other jobs) | 15 min |

The plan's "winners replay sample from S3" is not wired: it needs the learning-store layout from
the data work packages and has no command in this tree yet.

## Test architecture

* Profiles in `.config/nextest.toml`: `pr` (hosted; excludes the campaigns crate and the slow
  deadline tests) and `main` (everything, `slow-timeout 600s`, junit at
  `target/nextest/main/junit-main.xml`, `success-output = "final"` so passing tests' captured output
  is printed after the summary for the skip-marker scan).
* Feature-gated tests keep their gates: `neural` (jope-parallel, 27 integration tests in 8 files +
  the neural `src` modules), `cuda` (starforge), `gpu` (jope-dime-regressor → battle-gpu),
  `hyperion-cuda` (WP3).
* Hosted PR lanes never see a self-hosted label, an `id-token`, a GPU or a model bundle.

### No silent skip

`deploy/ci/no-silent-skip.py` runs after every nextest lane on `main` with the lane's
`--lane` name, its `cargo nextest list … --message-format json`, its junit and its run log, and
the committed allowlist `deploy/ci/ignored-tests.txt`:

1. every test the list marks `ignored` must be an `ignored` line of the allowlist (34 in the
   non-neural workspace today, plus the neural-only ones);
2. every runnable test must appear in the junit and must not be `<skipped/>`;
3. a `gated` line names the one lane where that test runs for real: there its env variables must
   be set and a skip marker (`SKIP`, `skipped`, `skipping`, case-insensitive) in that test's captured
   output fails the lane; in any other lane the marker is allowed and counted as a note;
4. a skip marker in the output of any other test is a violation.

Run it locally with `just test-main && just no-silent-skip`.

## Env gates (what each one expects)

Plan B25 listed these as "gates none set in any lane". Reading the tests shows two different
kinds, and the correct treatment differs:

**Helper-process variables** — the *parent* test spawns `current_exe()` with `--ignored --exact
<child>` and sets the variable for the child. They must **not** be set globally: the child
fixtures are `#[ignore]` (allowlisted) and the parent is an ordinary test whose presence in the
junit proves the child ran. Setting them in `main.yml` would misroute the child branch.

| Variable | Set by (parent) | Value | Child (`#[ignore]`) |
|---|---|---|---|
| `JOPE_CORPUS_KILL_ROOT`, `JOPE_CORPUS_KILL_BOUNDARY` | `corpus_incremental::sigkill_at_each_cache_durable_boundary_keeps_only_complete_entries` | fixture root; one of five durable-boundary names | `cache_kill_child` |
| `JOPE_CACHE_TEST_ROOT`, `JOPE_CACHE_EXPECT_REJECTION` | `sealed_storage` shared-cache tests | fixture root; `1` on the restart leg | `shared_cache_child` |
| `JOPE_ENTRY_TEST_ROUTE` | `gpu_entry_guards` (neural-only) | `proposal`, `monitor`, `train`, `batch`, `encode`, `apply-contexts`, `binary-contexts`, `campaign`, `cycle`, `resident-submit` | `entry_guard_child` |
| `JOPE_SYNTHETIC_EXPORT_TEST` | `fit_boundary` denial tests | path to an export request JSON | `isolated_export_child_fixture` (not ignored: returns immediately when unset, which is the parent's leg) |
| `JOPE_SYNTHETIC_RESIDENT_TEST_POOL` | `resident_protocol::worker_death_after_acceptance_does_not_permit_reexecution` | temp pool dir | `accepted_child_fixture` (same pattern) |

**Mode and fixture variables**:

| Variable | Expects | Effect | In `main.yml`? |
|---|---|---|---|
| `JOPE_FITTED_CPP_SANITIZER` | `normal`, `address`, `undefined` or `address,undefined` | selects the sanitizer mode of `feat-math::advanced_v3_fitted_cpp` (default `normal`; the test's own sanitizer leg passes `address,undefined`) | not set: the test already covers both legs; `golden42_v3` sets `address`/`undefined` itself |
| `JOPE_PARITY_STRICT` | `1` | a missing `g++`/`clang++`/`node`/`python3` fails `feat-math::portable_parity` instead of printing `portable parity SKIP` | **set in every Rust lane** |
| `JOPE_CORPUS_BASELINE_BIN` | an *independently built, pre-cache* `jope-parallel` executable | `corpus_incremental::independent_legacy_binary_has_identical_artifact_bytes` (`#[ignore]`) | cannot be set: no such binary is published; the lineage binaries in the archive are the kernel-v1 product build, not the pre-cache engine. Documented in the allowlist. |
| `JOPE_PARALLEL_SNAPSHOT_ROOT`, `JOPE_PARALLEL_CAPTURE_OUTPUT` | a live campaign snapshot root with `runs/{hardware,gp-service,neural-service}.json`; an output dir | `dashboard::capture_live_dashboard_screens` (`#[ignore]`, "manual capture of live telemetry for review") | cannot be set: needs a running campaign's telemetry; a review capture, not a test |
| `STARFORGE_WEIGHTS`, `STARFORGE_GOLDEN_JSON`, `STARFORGE_REGRESSOR_WEIGHTS`, `STARFORGE_REGRESSOR_GOLDEN_JSON`, `TABICL_REGRESSOR_WEIGHTS`, `STARFORGE_CHIMERA_WEIGHTS`, `STARFORGE_CHIMERA_GOLDEN_JSON`, `STARFORGE_CHIMERA_CLASSIFIER_{EXPERT_A,TRUNK}_WEIGHTS`, `TABICL_CLASSIFIER_WEIGHTS`, `STARFORGE_CHIMERA_CLASSIFIER_GOLDEN_JSON`, `STARFORGE_CHIMERA_SMOKE=1`, `STARFORGE_CHIMERA_CLASSIFIER_SMOKE=1`, `*_SMOKE_DEVICE=cuda` | safetensors + golden JSON paths from the pinned bundles | the six `starforge::golden_parity` tests return early ("skipping …") when unset | set in `encoder-parity` from `JOPE_MODELS_ROOT` after `jope models fetch --pinned` (WP7); the exact bundle paths are the ones in `main.yml` and must match `models/manifest.json` when it lands |
| `NUMPY_PCG64_JOPE_DIME_GIT`, `NUMPY_PCG64_ARCHIVE_ROOT`, `NUMPY_PCG64_REQUIRE_SOURCES=1` | a clone of the family `jope-dime` repository and the `jain-datasets/.runtime` archive directory | `numpy-pcg64::sampling_golden::vendored_subset_matches_its_sources_when_available` prints `SKIP provenance …` per missing source | cannot be set on the runner: both sources are in the archive bucket (git bundles, node-archive-mirror); a maintainer with the mirror runs it with `REQUIRE_SOURCES=1`. Allowlisted (`archive-provenance`). |
| `NUMPY_PCG64_SAMPLING_FILES` | colon-separated committed `sampling.json` paths | `committed_sampling_files_regenerate_exactly` (`#[ignore]`) | run by hand when the recipes change |
| `JOPE_GPU_DEVICE` | informational (`device=0`, set by `deploy/runner/run.sh`) | none in tests; the hardware gates probe CUDA directly (`battle-gpu`, `jope-parallel::training::bf16_forward_observes_updated_master_weights`) and return early without a device — on xbabe2 a marker from them is a failure (`neural-cuda`) | set by the runner |

Neural SDK variables (`LIBTORCH`, `LIBTORCH_CXX11_ABI`, `CUDA_HOME`, `JOPE_NCCL_INCLUDE`,
`JOPE_CUDA_LIBRARY_PATH`, `JOPE_NCCL_SONAME`, `JOPE_CUDART_SONAME`, `CXX`) come from
`deploy/ci/neural-env.sh`, which mirrors `deploy/receipts/libtorch.json` (docs/runbooks/neural-sdk.md).

## Receipts and schemas

| Receipt | Schema | Written by |
|---|---|---|
| `jope.cuda-qualification/v1` | `schemas/jope.cuda-qualification.v1.schema.json` (validated by `deploy/ci/validate-receipt.py` before the script exits; `--dry-run` produces a receipt with every step `not-run`) | `deploy/ci/gpu-qualification.sh` |
| `jope.e2e-smoke/v1` | inline in `deploy/ci/e2e-smoke.sh` (`stages[]` with `passed/failed/blocked`, command, exit code, log sha256, dataset/prediction digests) | `deploy/ci/e2e-smoke.sh` |
| `jope.product_evaluator_build.v1` + `jope.kernel-v1-source/v1` | `crates/jope-engine/src/build_receipt.rs`; the source side-receipt names which tree was built (`git-tree` of `deploy/kernel-v1/source` or the archive object `sha256`) | `deploy/ci/kernel-v1-product-build.sh` |
| `jope.sdk-receipt/v1` | `deploy/receipts/libtorch.json` (WP8 precondition, B21) | by hand from real exit codes (runbook) |
| `jope.runner-image/v1` | `deploy/receipts/runner-image.json` — image ids and sizes of the locally built images | `docker build` on xbabe2 |

Committed evidence from this work package: `deploy/receipts/runner-image.json` (runner
`sha256:eed8e596…`, 6.09 GB; kernel builder `sha256:72769e40…`, 356 MB) and
`deploy/receipts/kernel-v1-{build,source}.json` (the lineage rebuilt with the checked-in builder
in 95 s: bounds passed, `rustc 1.95.0` recorded as an explicit override of the source's `1.97.1`
channel).

## Self-hosted runner (what runs where and why)

Heavy work (GPU, LibTorch, the 2,300-test suite, model bundles, docker) never reaches a hosted
runner or a pull request. `deploy/docker/runner.Dockerfile` is the image (base
`nvidia/cuda:12.8.1-devel-ubuntu24.04` — the receipted LibTorch is `cu126` and needs the CUDA 12
runtime; the 13.0.1 tag exists and is the base to move to when the SDK receipt is regenerated for
cu130), every tool pinned by sha256, uid 1001, no sudo, the SDK and models as volumes. Two labels:

* `xbabe2-gpu` — `--gpus device=0`, no docker socket: `full-rust`, `neural-cuda`,
  `encoder-parity`, `e2e-smoke`, nightly jobs;
* `xbabe2-kernel` — no GPU, docker through `jope-docker-proxy.service` (containers/images only):
  `kernel-v1-build`.

Each job is one container started by `jope-runner@<lane>.timer` → `run.sh` with a single-use JIT
config; the container is `--rm --user 1001 --cap-drop ALL --security-opt no-new-privileges
--pids-limit 4096 --memory 96g --cpus 64 --read-only --tmpfs /tmp:size=32g`. The disk guard refuses
to start under 50 GB free; `jope-runner-prune.timer` prunes nightly. Registration steps are in
`docs/runbooks/runner.md`; nothing in the tree registers a runner.

Budgets on xbabe2 (128 cores, 125 GB): the full suite was listed in 6 m 43 s of build; the
historical `just required` took 17.4 min for 319 tests, so `full-rust` is budgeted at 90 min until a
committed timing run (`docs/parity/test-timings.json`, WP1) says otherwise.

## What cannot run yet (precise)

| Step | Blocked by | What happens today |
|---|---|---|
| `encoder-parity` → `jope models fetch --pinned` | WP7 (`jope-models` crate, `models/PINNED`, `models/manifest.json`) | cargo: package ID `jope-models` did not match → job fails at that step (`continue-on-error: false` on purpose) |
| `e2e-smoke` → `prepare` | WP7 (`jope datasets prepare`) — or `JOPE_E2E_PREPARED=<dir with prepared.json>` | stage `blocked`, receipt written, exit 1 |
| `e2e-smoke` → `proposals` | WP7 (step-60000 checkpoint in the pinned bundles) + a `--features neural` binary — or `JOPE_E2E_PROPOSALS=<seed_population.json>` | stage `blocked` |
| `neural-cuda` → `hyperion-cuda` | WP3 (`hyperion` crate in the workspace) | step `blocked`, result `blocked`, exit 1 |
| `neural-cuda` → the 27 neural tests on the GPU | nothing in the tree; needs the runner registered and the SDK volume provisioned | first green run is the lane's own |
| `campaign-smoke` | WP5 (`campaign smoke` subcommand) | `continue-on-error`, warning, blocked receipt |
| every self-hosted lane | the owner registering the runner (runbook) | queued forever until a runner with the label exists |

## Owner steps that remain

1. **Runner registration** (`docs/runbooks/runner.md`): `provision.sh` as root, paste the
   fine-grained PAT (Administration: write, this repository only) into
   `/etc/jope-runner/github-pat`, enable the timers.
2. **Team upgrade** at go-live: branch protection with required checks `hygiene`, `rust-check`,
   `emitters` (+ `python` when WP6 lands), 1 review, conversation resolution; until then
   `main-guard.yml` files an issue for any main commit without a merged PR.
3. **Repository settings**: `sha_pinning_required: true`; `allowed_actions: selected` with
   `actions/*`, `Swatinem/rust-cache`, `astral-sh/setup-uv`, `aws-actions/configure-aws-credentials`,
   `EmbarkStudios/cargo-deny-action`, `taiki-e/install-action`, `anchore/sbom-action`,
   `rhysd/actionlint`; fork PR workflows off.
4. **IAM**: confirm the `JopeDimeCiRead` trust policy's `sub` covers
   `repo:veox-ai/JopeDime:ref:refs/heads/main` and `…:ref:refs/tags/v*` (the `nightly` schedule
   runs on `main`, so no extra claim is needed).
5. Remove `continue-on-error` from `campaign-smoke` when WP5 lands; fill the model paths of
   `encoder-parity` from `models/manifest.json` when WP7 lands.
