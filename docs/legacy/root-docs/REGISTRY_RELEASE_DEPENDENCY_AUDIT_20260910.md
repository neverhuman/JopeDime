# Registry release dependency and CI audit — 2026-09-10

This is a bounded source and evidence audit, not a release receipt. No tests, numerical fits, GPU tasks, service operations, or production artifact writes were performed for this audit. Cargo metadata was read with `--offline --locked --no-deps`; source and existing CI logs were inspected. The native authority service was still being integrated, so the release owner must regenerate the inventory and run the permitted checks against the final immutable snapshot.

The current release cannot claim complete CI. The captured root Python run has 207 passes and five external-fixture failures. The root Rust run was interrupted after an existing real-data test started fitting; that attempted work is permanently unavailable for automatic retry. Earlier scoped admission, registry, resident, and CLI receipts bind earlier source combinations, not the subsequent service integration.

## 1. Required source closure

Paths below are relative to the repository root. Preserve both Cargo lockfiles and the actual dirty/untracked source bytes through a content inventory; a Git commit alone does not identify this release. Exclude build outputs, `.git`, Python bytecode, and caches. A symlink needs explicit identity-bound resolution, not silent traversal into an unrelated checkout.

| Build or check | Files and directories required |
| --- | --- |
| Root library, 56 binaries, and root integration checks | `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, applicable `.cargo/`, all `src/`, `tests/`, `scripts/`, `manifests/`, `seeds/`, `systemd/`, and `vendor/feat-math/`. Root workspace metadata also names `vendor/battle-gpu/`; preserve it even for a CPU inventory. |
| Parallel CPU binary, library, and tests | `parallel/Cargo.toml`, `parallel/Cargo.lock`, `parallel/rust-toolchain.toml`, applicable `parallel/.cargo/`, `parallel/build.rs`, all `parallel/src/`, `parallel/tests/` including `support/`, `parallel/examples/`, `parallel/native/`, `parallel/tools/`, `parallel/assets/grammar/`, and `parallel/assets/source/jope-dime/vendor/feat-math/`. |
| Parallel runtime and service handoff | The preceding closure plus `parallel/systemd/` and the exact frozen task, controller, admission, grammar, family/context, authority scope, and execution manifests used by the deployment. Source packaging does not freeze mutable `runs/` pointers or authorize registry initialization. Retain the authority database, journal, high-water witness, exclusions, and immutable completions separately with their recovery coverage. |
| Parallel neural build | The same parallel closure, including `parallel/native/runtime.cpp`, with the `neural` feature and the externally bound PyTorch/CUDA/NCCL toolchain described below. Compile every required neural example explicitly; examples are not permission to run experiments. |
| Root optional GPU build | `vendor/battle-gpu/Cargo.toml`, `src/`, and `kernels/ridge_screen.ptx`, together with root feat-math. `src/lib.rs` embeds this PTX. This is a distinct optional feature from parallel `neural`. |
| Hyperion host extension, only if included in the release's required build scope | `parallel/native/hyperion-host/{Cargo.toml,Cargo.lock,build.rs,src/}` and all `parallel/assets/source/hyperion-jain/crates/feat-core/src/` plus `crates/feat-cli/src/bin/hyperion_vs.rs`, together with the parallel crate dependency. Preserve the full source tree and its metadata rather than a hand-maintained fragment. Production Hyperion remains off; compilation is not experiment qualification. |

Both feat-math copies require their entire `src/`, `cpp/`, `assets/`, `tests/`, `Cargo.toml`, and `build.rs`. Relevant embedded bytes include `assets/champions/8944cbdf.dsl`, `tests/portable/full_ref.cpp`, `tests/portable/full_ref.js`, and the native `.so` generated in `OUT_DIR`. The C++/Python/JavaScript parity fixtures and `tests/fixtures/advanced_v3/csv-runtime-v1.json` must survive packaging. Do not silently synchronize the two copies: verify their byte parity or record and review a difference.

Cargo metadata confirms that the mirrored feat-math crate is a member of the **parallel** workspace and inherits its package/dependency definitions from `parallel/Cargo.toml`. `parallel/assets/source/jope-dime/Cargo.toml` is therefore not required for this build path. The root and parallel workspaces are separate; compiling one does not compile the other.

Active root compile-time inputs include:

- `manifests/epoch6/JopeTaskSemantics.v1.json` and `JopeTaskHeadPolicy.v1.json`.
- `manifests/input-manifest.v1.json`, `binary42-test-adaptive.v1.json`, `dlgp-budget-first/protocol.v1.json`, and the referenced `manifests/selfplay/` policies.
- `seeds/generation-zero.txt`, `seeds/generation-zero-binary.txt`, and `systemd/jope-selfplay-trainer-watchdog.service`.
- `parallel/assets/grammar/featmath.json` and `parity.json` for parallel runtime/tests. Several paths are based on compile-time `CARGO_MANIFEST_DIR`; deployment must retain the expected assets or provide the qualified explicit paths.

`src/campaign.rs:2522` contains an external compile-time seed reference under `CARGO_MANIFEST_DIR/../../../prime/runs/exact-cv-inputs/`. The resolved `/prime/...` file is absent locally. No current module/path declaration reaches this dormant source file; this is **not** a demonstrated blocker for the present Cargo targets. If it is reintroduced, bind the original seed bytes or qualify a source change first.

## 2. Semantic identity and packaging gaps

At the audit boundary, `parallel/src/teaching_store.rs:237` hashes these source files: `corpus.rs`, `grammar.rs`, `teaching_store.rs`, `task_contract.rs`, `prepare.rs`, `encoding.rs`, `manifest.rs`, `label_admission.rs`, `eval_registry.rs`, and `eval_registry/authority.rs`.

The new `eval_registry/service.rs` is absent from that list. Its custody, original process completion, and committed-result verification semantics are part of the authority dependency closure. Bind it before release, or bind an equivalently complete immutable implementation/authority protocol digest. Merely adding `pub mod service` to the already hashed parent does not detect future edits to the child module.

The teaching and corpus scan hashes also do not directly include the vendored DSL/parser sources used by `grammar::parse_dsl` and native program canonicalization. The frozen admission evaluator digest constrains accepted fit implementation, but it does not by itself prove that an independently built packer has unchanged tokenization/parser behavior. Bind the actual packer executable or a complete semantic source/dependency manifest when deriving the store lineage. `task_contract.rs` is now present; the former omission is repaired. Preserve observations and create a new derived lineage for semantic changes.

Existing packaging tools have different scopes:

| Tool | Audit conclusion |
| --- | --- |
| `parallel/tools/freeze_registry_release.py` | Reproducible byte inventory, immutable copied files, tamper checking, and untracked-file capture are implemented. The first inspected roots omitted root `manifests/`, `seeds/`, and `systemd/`; the release owner added these during this audit. Parallel `systemd/` and the Hyperion source tree needed if its host is built remain absent from the last inspected defaults. Close the actual required scope before freezing. Its two unit tests exercise packaging mechanics, not complete Cargo dependency discovery. |
| `parallel/tools/kernel_release.py` | Carries the full parallel source/tool/grammar/native tree and Hyperion source when present, and checks root/mirror kernel parity. It does not package the full root library and all 56 root binaries. |
| `parallel/tools/hardening_release.py` | Uses an old fixed Python tool list and omits grammar assets and newer transitive Python consumers. Its legacy package is insufficient for this release. |
| `parallel/tools/build_runtime.py` | Builds CPU/neural artifacts from a kernel snapshot with separate target output. Its source archive does not itself bind compiler, resolved libraries, target ABI, or runtime checks. |
| `parallel/tools/build_flywheel_native.py` | Historical driver pins `cargo +1.90.0`, uses OMP/Rayon=2, old test selections and budget/lease locations, and includes broad `--lib` runtime execution. Current toolchain files pin 1.97.1 and the authority contract expects the approved all-1 fitting thread environment. Do not invoke this unchanged as a current release verifier. |

This table records inspected code, not a promise that concurrently edited files remain identical. Re-read and hash the final packager at freeze.

## 3. Python dependency closure

The safe default for the parallel package is all `parallel/tools/*.py`, not the older fixed list. Static AST import traversal from admission, collection, qualification, promotion, resident-pool, status, and source-freeze entrypoints and their scoped tests found this local module closure:

`binary_encode`, `binary_replay_v2`, `binary_reproduce`, `binary_verify`, `candidate_intake`, `continuous_learning`, `continuous_status`, `corpus_verification`, `experiment_runtime`, `experiment_scheduler`, `experiment_stages`, `freeze_registry_release`, `gpu_occupancy`, `kernel_release`, `label_admission`, `promotion_gate`, `publish_replay_candidate`, `qualification_budget`, `qualify_matched_v2`, `qualify_regression_gain`, `replay_policy`, `resident_gpu_pool`, `resume_expired_cycle`, `seed_experiment`, `teaching_manifest`, and `verify_experiment`, plus the selected `test_*.py` files. Resident and sealed-cache checks additionally need `shared_teaching_cache.py`, `qualify_resident_workers.py`, and their tests.

The inspected parallel control-plane import closure uses the Python standard library. Native subprocesses, dynamically loaded files, manifests, and training assets are still dependencies; an AST import scan cannot prove their absence. Include all tools so later local imports remain present, then bind the external executable and artifact paths declared by each selected task.

Root Python tests import local `scripts/` modules, including production modules whose filenames start with `test_exact_`. Compile/import collection is not permission to execute those scripts as CLIs. They also use external code and configuration:

- `tests/test_test_exact_campaign.py:249` imports `jope.policy.train_v3` from `/home/ubuntu/jope-prime/src`; `tests/test_durable_selfplay.py:458` imports `jope.policy.model_v3` from the same tree. Bind the complete imported Jope Python package, not only the two top-level files.
- `tests/test_test_exact_campaign.py:321` reads `/home/ubuntu/smart_cluster/etc/projects.toml`.
- Root fixture validation uses both repository-local `sotarena/` Git objects and `/home/ubuntu/sotarena`, the external fixture store, and the manifest referenced below.
- The root static import graph includes NumPy, PyTorch, pytest, safetensors, pandas, orjson, and conditional/external model packages such as AutoGluon, TabPFN, TabFM, and Chimera. Presence in the graph does not establish that every package is required for each selected synthetic test; classify the concrete runtime path before installing or running it. There is no complete Python lockfile for that full external graph in the audited root. `mcp/requirements.lock` pins historical Python MCP 1.28.1 and is not a lockfile for the current Rust MCP runtime or training environment.

## 4. Toolchain and external runtime binding

| Component | Required binding and locally observed state |
| --- | --- |
| Rust/Cargo | Both checked toolchain files pin 1.97.1. Local `rustc 1.97.1 (8bab26f4f68e0e26f0bb7960be334d5b520ea452)`, LLVM 22.1.6, host `x86_64-unknown-linux-gnu`; Cargo 1.97.1. Parallel minimum `rust-version` is 1.90, not the selected compiler identity. Bind compiler, target, lockfile, flags, build profile, and registry crate checksums/cache for offline builds. |
| Native feat-math runtime | `vendor/feat-math/build.rs` and its mirror directly invoke **g++**, compile C++17 with `-O3 -march=znver2 -ffp-contract=off -fno-fast-math -shared -fPIC -Wl,--build-id=none -ldl`, and embed the output `.so`. `CXX` does not override this invocation. Local g++ is Ubuntu 13.3.0; local glibc 2.39. Qualify the target CPU ISA and glibc/libstdc++ ABI; a local build is not a remote compatibility receipt. |
| SQLite | Root rusqlite links system SQLite (local pkg-config reports 3.45.1). Parallel enables bundled SQLite, requiring a C compiler and the locked libsqlite3-sys source. Bind the actual loaded numerical/runtime libraries separately from source metadata. |
| Neural C++ and libtorch | `parallel/build.rs` runs Python to locate torch and its C++ ABI, then builds `native/runtime.cpp` with C++17 using `CXX` or clang++. It requires torch headers, CUDA headers, and NCCL headers; links torch_cuda, c10_cuda, nccl, cudart, and torch libraries via tch. `tch/torch-sys=0.20.0` requires PyTorch **2.7.0**, verified from the locked crate's build script. Do not bypass this version check to manufacture compilation evidence. |
| Local neural library layout | Distribution metadata reports torch 2.7.0 under `/usr/lib/python3/dist-packages/torch`; required torch shared libraries exist. `/usr/local/cuda` paths in the build script are absent locally, but standard `/usr/include/{cuda_runtime.h,nccl.h}` and `/usr/lib/x86_64-linux-gnu/` libraries exist. Local symlink targets identify cudart 12.8.90, NCCL 2.26.2, and CUDA driver library 570.211.01. These are path observations, not device or driver qualification. Record actual compiler/link resolution and hashes on the target. |
| Runtime process/storage features | Linux `/proc` process birth and boot IDs, Unix sockets with peer credentials and descriptor passing, sealed memfd, flock, and durable local filesystem/SQLite behavior are required. Test failures on missing facilities are unavailable evidence, not optional passes. Storage latency and recovery tests must use the target storage class. |
| Python and cross-language checks | Locally Python 3.12.3, pytest 9.0.2, NumPy 1.26.4, safetensors 0.7.0, Node 26.1.0, clang++ 18.1.3. Vendor portable parity requires rustc, g++, Python, and Node. Its harness prints `SKIP` if a tool is absent while allowing the overall test to succeed; CI must treat a missing required backend as unavailable. |
| Neural/task assets | Immutable model/optimizer checkpoints, shard/context manifests, admission policy, vocabulary, training splits, and rank identities are separate content-bound inputs. Their existence cannot be inferred from compilation or resident startup. Eight-rank parameter/optimizer/step/corpus agreement and useful occupancy need a separately funded hardware manifest. |

## 5. Inventory and permitted verification

The initial inventory is `parallel/runs/registry-integration-20260910/ci/required-targets.json`. It records root library + 56 binaries + 11 integration targets, feat-math library/build script + 15 integration targets, battle-gpu library, and parallel library/binary/build script + **25** integration targets. Its Python list has 134 files and is an inventory, not a runnable-suite approval.

Read-only Cargo metadata during this audit reports **26** parallel integration targets, adding `fit_boundary`. The service also declares `eval_registry::service::tests`; its final tests were still being written. Regenerate the inventory after integration. List every target, feature set, runtime scope, required external dependencies, expected nonzero test count, result/skip status, source closure, toolchain, and artifact hashes.

Compile-only commands, run from an immutable source copy with a separate writable target directory, can cover the full Rust target set without executing tests:

```sh
rtk proxy cargo test --manifest-path Cargo.toml --workspace --all-targets --offline --locked --no-run
rtk proxy cargo test --manifest-path Cargo.toml --workspace --all-targets --features gpu --offline --locked --no-run
rtk proxy cargo test --manifest-path parallel/Cargo.toml --workspace --all-targets --offline --locked --no-run
rtk proxy env LIBTORCH_USE_PYTORCH=1 cargo test --manifest-path parallel/Cargo.toml --workspace --all-targets --features neural --offline --locked --no-run
```

Set the target-compatible compiler/linker environment explicitly. Bind `CARGO_TARGET_DIR` to the dedicated build output; do not mutate the source snapshot. A successful `--no-run` is compilation evidence only. Run formatting and lint checks on this same frozen closure; existing source formatting or warnings are failures to report, not authority to reformat other owners' source.

The following scoped runtime commands are concrete candidates for the final release's **no-numerical-fitting** task list. They were not run in this audit. Review the final service/test changes before executing; preserve evidence of how many tests actually ran. The authority fixture environment must match its approved single-thread numerical/runtime contract.

```sh
rtk proxy env OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 cargo test --manifest-path parallel/Cargo.toml --lib eval_registry --offline --locked -- --test-threads=4
rtk proxy env OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 cargo test --manifest-path parallel/Cargo.toml --lib teaching_store::admission_tests --offline --locked -- --test-threads=1
rtk proxy cargo test --manifest-path parallel/Cargo.toml --test corpus_incremental --test resident_protocol --test sealed_storage --test gp_admission --offline --locked
rtk proxy env PYTHONPATH=parallel/tools python3 -B -m unittest test_label_admission test_learning_inputs test_candidate_intake test_continuous_learning test_promotion_gate test_freeze_registry_release test_continuous_status
rtk proxy env PYTHONPATH=parallel/tools python3 -B -m unittest test_resident_gpu_pool test_shared_teaching_cache test_qualify_resident_workers
```

- Native registry tests use synthetic executors/original process-exit observation; their fake outcomes are not real model-quality evidence. Upcoming service tests must retain that property or be separated from this list.
- `teaching_store::admission_tests` uses an explicitly synthetic native committed receipt helper. It performs no numerical fitting.
- `gp_admission` now explicitly strips authority environment credentials and verifies that unavailable authority cannot publish an admissible quality label. Its pre-service compile was incomplete; it needs final verification.
- The scoped `fit_boundary` tests cover denial, missing DB/witness, forged tokens, and export fencing. Their current daemon adaptation remains in progress; include them after review. The associated `gp_contracts` positive test receipt records **two tiny synthetic numerical ridge fits**. Do not classify that positive target as zero-fitting fault qualification or rerun it without its own manifest/reservation.
- The neural `resident_worker` target has four startup/socket checks and accepts no numerical task. It needs a linked neural binary and dependencies. It is not eight-rank training, recovery, or GPU useful-work evidence.
- Vendor `portable_parity` uses hardcoded scalar fixtures and verifies that fixture source contains no dataset-loading markers. It can test Rust/C++/Python/JavaScript arithmetic under a separate bounded entry; inspect every backend result and fail required coverage on `SKIP`. Other vendor tests include fitting: `advanced_v3_fitted_cpp`, `golden`, `champion_parity`, `bench`, and the rest require explicit source-based scope classification, not blanket execution.

Intentional ignored cases need individual treatment:

| Fixture or ignored case | Classification |
| --- | --- |
| Native authority internal child, resident accepted-child, sealed-storage child, corpus `cache_kill_child` | Isolated subprocess fixtures; only their parent test invokes them. Do not run standalone or call the parent pass an unexplained skip. |
| Corpus `repeated_ledger_scan_diagnostic` | Optional explicitly bounded synthetic performance diagnostic; not part of ordinary parity. |
| Corpus `independent_legacy_binary_has_identical_artifact_bytes` | Requires independently built pre-cache baseline `JOPE_CORPUS_BASELINE_BIN`, currently unavailable. Source-reference parity passing does not satisfy this independent baseline comparison. |
| Root golden42 real-data near-winner parity | Permanently unavailable for legacy refitting; see incident below. Never use `--ignored` to run it. |

Root runtime coverage still needs reviewed filters across all 56 binaries and required integration tests. Do not rerun broad root Rust or broad experiment-script discovery just because the unsafe golden42 case was marked ignored. No broad suite execution was authorized by this audit.

Tuiwright and Playwright run only on **xbabe2**, outside ordinary CI. Jope TUI evidence must bind the actual release binary and pinned Jankurai Tuiwright harness and cover rendered navigation, scroll, resize, reconnect, compact/large layouts, stale/held states, registry and qualification statuses. No Jope browser surface was identified, so Jope Playwright is inapplicable. Bullet's actual browser receipts belong to its source owner and authenticated sessions; they are not Jope coverage.

## 6. Exact unavailable evidence

`parallel/runs/registry-integration-20260910/ci/root-python.log` reports `5 failed, 207 passed in 11.07s` for `python3 -B -m pytest --import-mode=importlib -q tests`:

| Failed test | Exact observed blocker |
| --- | --- |
| `tests/test_external_fixture_index.py::test_index_is_path_free_and_structurally_complete` | `git show -s --format=%T c1f108464d43d0c01417e048aa6044fa19ad2bcc` fails in `/home/ubuntu/sotarena`. |
| `tests/test_external_fixture_index.py::test_complete_verification_fails_closed_on_missing_binary_splits` | Same missing Git commit; the test expects the later explicit 46-unresolved-files failure and therefore fails its error match first. |
| `tests/test_external_fixture_index.py::test_workspace_profile_materializes_committed_read_only_bytes` | Same missing Git commit prevents identity-checked materialization. |
| `tests/test_external_fixture_index.py::test_qualified_local_recovery_imports_exact_46_into_temporary_cas` | `/home/ubuntu/sotarena_old` is absent. |
| `tests/test_golden46_special_execution.py::test_chimera_conversion_is_bound_to_frozen_source_identity` | `/home/ubuntu/TO_DELETE/jain_small/deployment/ops/starforge-artifacts.receipt.json` is absent. This is the first missing prerequisite, not proof that all later conversion/checkpoint assets exist. |

The checked fixture index is `manifests/epoch6/JopeExternalFixtureIndex.v1.json`, SHA256 `68a3194900946c6f41db1903fe0f2ff82eac8de62de9dbdd426560415ea8cd0e`. It contains 422 entries and 46 unresolved entries. Its repositories bind legacy-sotarena commit `e5db30a07c093661c8f6940d3e22192511f8388b` / tree `1ae3d687813f180c7668cf3d395fcfcd4a5866ee`, and sotarena-rust commit `c1f108464d43d0c01417e048aa6044fa19ad2bcc` / tree `0d31b7da2297fb71203d317a885bda1ba668673e`. Keep these unavailable statuses; do not substitute current repository contents or create synthetic receipts to obtain a pass.

`parallel/runs/registry-integration-20260910/ci/root-real-parity-incident.json` records PID 2447436, birth tick 516772748, command `target/debug/deps/jope_dime_regressor-993978fe55eebc78`, outcome `interrupted_unknown`, and `full_ci_pass=false`. The reached test is `golden42_v3::tests::target_campaign_v3_fit_preserves_frozen_near_winner_bits`; current source marks it ignored with the reason that original registered model artifacts must be reused and legacy refits are disallowed. Preserve the incident and permanent seen/unavailable exclusion. An interrupted process and unknown numerical work cannot be refunded or automatically retried as a CI repair.

## 7. Release-owner handoff

Before declaring the release reviewable, close packaging and semantic-hash omissions, finish daemon integration, then freeze once and regenerate metadata. Compile both workspaces and feature sets against the target-compatible toolchain. Execute only reviewed, funded runtime targets and retain source-bound logs with nonzero test counts and explicit skips. Attach the external fixture and hardware limitations above rather than converting them to passes. Preserve original model/reference and authority history independently of the code snapshot.

The audit establishes a concrete dependency checklist and runnable synthetic scope. It does not establish production migration, complete CI, representative 104-client registry latency, eight-rank recovery, fresh training exposure, qualification, or either required learning cycle.

## Integration follow-up

After this audit, root bound `eval_registry/service.rs` and `fit_boundary.rs` into teaching-store semantics. Both teaching-store and corpus scan identities now also bind `/proc/self/exe`, covering transitive vendored/parser/native code in the loaded packer; source-only nonnumerical changes conservatively force a new derived lineage. The snapshot defaults now include root manifests/seeds/systemd and parallel/systemd. The optional Hyperion source/build remains outside this registry release scope and unqualified. These source edits require final integrated tests.
