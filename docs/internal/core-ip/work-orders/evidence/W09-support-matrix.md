# W9-T01 fitted export support matrix and substitution trace

## 0. Status, scope and source identity

| Item | Value |
|---|---|
| Task | W9-T01 inventory, plus the fixture findings for W9-T02 to T06 that it relies on. See `docs/internal/core-ip/work-orders/W09-fitted-export-release.tasks.md`. |
| Evidence type | Source trace plus fixture tests in `crates/jope-bench`. **Fixture proof, not a released-package qualification.** No real selected subject was exported, and W9-T07 is out of scope. No campaign journal, S3, GPU or network was used. |
| Source | Worktree `/home/ubuntu/JopeDime/.agent/worktrees/honesty`, branch `rr/w9-export`, based on `feat/regression-remediation` `ce4aea8ec112ce49515190e4eaed4b55285bccc8`. |
| Toolchain observed | rustc 1.97.1, Python 3.12.3, Node v26.1.0, g++ at `/usr/bin/g++` (the default `$CXX`). |
| Run receipts | `/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/w9-*`. Each holds HEAD, the tracked diff, the command, exit code and the nextest log. From the commit-A runs on, each also holds an untracked-file patch. See section 9. |
| Findings routed | S01-G6, S01-K6, S07-G01..G10, S07-K01..K10. They stay **open** in `findings/register.json`. This document does not attest closure. |

Progress words follow `EXECUTION-CONTRACT.md`. Here, "fixture-proven" means only that the named tests pass on synthetic fitted subjects on this host.

### Path legend

| Tag | Path |
|---|---|
| FE | `crates/jope-bench/src/panel/fitted_export.rs` |
| FR | `crates/jope-bench/src/panel/fitted_export_run.rs` |
| INF | `crates/jope-bench/src/panel/inference.rs` |
| PKG | `crates/jope-bench/src/panel/inference_package.rs` (new) |
| CT | `crates/jope-bench/src/panel/inference_contract_tests.rs` (new, test-only) |
| IE | `crates/jope-bench/src/panel/inference_export.rs` |
| FP | `crates/feat-math/src/advanced_v3/fitted_portable.rs` (+ `fitted_portable/scalar.rs`) |
| PD | `crates/jope-pipeline/src/prepared_data.rs` |

## 1. States and entry points

These are distinct states. Success in one backend never stands in for another backend.

| State | Where it is decided | Recorded as |
|---|---|---|
| Parse | `FittedRustArtifact::validate` / `from_bytes` (exact-bytes digest), `FittedSynthesizedArtifact::validate`, `FittedInferenceView::{registered,synthesized}` | Error. Never a default. |
| Emission (source generated) | `inference::export_inference` returns `Emitted{PublicBundle}` or `Unsupported{reason}` | `PublicManifest.qualification = "emitted-unqualified"` |
| Runtime ready | The plan commands of `QualificationPlan` (compile plus first prediction) complete inside the bounded worker (FR `worker`) | FR `backend_report`: `failed` / `unavailable` / `unknown_started_without_finished` |
| Numerically qualified | `inference::verify_qualification` against native predictions frozen in the plan **before** lowering | FR status `qualified` + `QualificationResult` |
| Package loaded (new) | PKG `seal`/`install`/`load` | `LoadedPackage`; `assess` yields `complete` / `partial` / `blocked` with `release: "not-decided"` |
| Released | **No code path defines release.** | Open. Belongs to W9-T07 / W10. |

| Entry point | Code | Fits? | Output status |
|---|---|---|---|
| `jope-bench benchmark export-winner` (asm, linux) | FE `prepare` | No. Exports the existing fold-`k` fit (`--fold`, default 0). | `unattempted` or `unsupported` per backend |
| `jope-bench benchmark qualify-export` (asm, linux) | FR `run` → `ExecutionPool` → hidden `fitted-export-worker` (FR `worker`) | No. Runs only the emitted code with unlabeled rows. | FR statuses above |
| Campaign report | FE `report` → `PreparedExport::verify` + FR `recover` | No. It also runs no compiler or predictor. | Authenticated statuses only |
| Warm/ASM demos | `demo.rs:566-585`, `demo_asm.rs:307-327` | Their own explicit demo fits | `emitted-unqualified` only |
| Latency benchmark (asm, linux) | IE `fit_export` + `inference_prediction.py` | **Yes.** One explicit synthesized fit per timing subject. | Timing evidence. Not a selection, qualification-of-record or release route. |
| Neural checkpoint export | `crates/jope-engine/src/policy_export.rs` | n/a | Policy weights. Not a fitted predictor path. |

## 2. Supported fitted compositions per backend

Support is a predicate over the **fitted graph**, not over a recipe name or a language label. The shared admission check is FP `fitted_portable_unsupported`, lines 32-76. It returns the first refusal reason that applies, in this order:

1. `row_order`: non-IID schema or fitted row order.
2. `historical_state`: any historical fitted state.
3. `model_form`: a legacy (non-pipeline) predictor.
4. `target_mapping`: a fitted target transform.
5. `operations`: any fitted operation, e.g. `(standardize)`.
6. `calibration`: any non-`None` calibration.
7. `missing_aware`: missing-aware routing on a head other than TabBoost or ExtraTrees.
8. `categorical_mode`: a categorical mode other than onehot.
9. `head`: a head outside the portable set.

After that come the backend-specific checks. C++ refuses multiclass with `cpp_matrix`, because the scalar C ABI cannot return a probability matrix. It then applies `fitted_public_cpp_unsupported`. JavaScript applies `fitted_javascript_unsupported`. Invalid coefficients are **errors**, never `unsupported` (`invalid_fitted_coefficients_fail_instead_of_becoming_unsupported`).

| Fitted head (task) | python | rust | javascript | cpp17 | cpp20 |
|---|---|---|---|---|---|
| Ridge (regression) | yes | yes | yes | yes | yes |
| LogitRidge (binary) | yes | yes | yes | yes | yes |
| MulticlassRidge, MulticlassOvrLogistic, MulticlassDiagonalLda, MulticlassNearestCentroid (multiclass) | yes | yes | yes | `cpp_matrix` | `cpp_matrix` |
| TabBoost, ExtraTrees, GamBoost (regression, binary) | yes | yes | `head` | `head` | `head` |
| Pls (regression) | yes | yes | `head` | `head` | `head` |
| ClassificationPortfolio (binary) | yes | yes | `head` | `head` | `head` |
| Anything else: histgbm, histgbm2, ridge_gcv, elasticnet, ksmooth, ensemble, krr, gpr (both), hurdle2, stack, regression portfolio | `head` | `head` | `head` | `head` | `head` |

The input stage must be the fitted numeric fill, an optional missing indicator and a onehot vocabulary. Numeric-fill values other than the median (for example a generated `numeric=mean`) are exported from fitted state, never recomputed.

The table describes the 26 registered recipes (17 regression, 5 binary, 4 multiclass). Python and Rust support 14 of them, JavaScript 6, and C++ 2. **Tested registry:** CT `registered_inference_inventory_matches_actual_export_outcomes` fits each of the 26 recipes and checks that the actual export outcome for each of the 5 backends equals the claim in `registered_inference_inventory`. There were 0 mismatches. It also checks that an unsupported outcome leaves the fitted bytes unchanged.

### Per-backend contract

| Backend | Public files | Host boundary / runtime (serve) | Build (qualification) | Parity policy | Numerical policy |
|---|---|---|---|---|---|
| python | `model.py` | raw rows; Python ≥ 3.10 stdlib (`json`, `math`, `re`) | none | exact bits | `…platform-libm-v1` |
| rust | `model.rs` | raw rows; `predict_raw(&[Vec<Option<String>>]) -> Result<String,String>`; std only, compiled into the consumer | `rustc --edition=2021 -O` (harness) | exact bits | `…platform-libm-v1` |
| javascript | `model.js` | raw rows; ES2019 / Node ≥ 12 | none | abs/rel 1e-12 | `…ecmascript-math-es2019-v1` |
| cpp17 / cpp20 | `model.cpp`, `codec.json`, `boundary.py` | **Python ≥ 3.10 ctypes typed-table host** + `model.so` + host libm | `$CXX -std=c++17\|20 [-march=$JOPE_CXX_MARCH] -O2 -ffp-contract=off -fno-fast-math -shared -fPIC model.cpp -o model.so` | exact bits | `…platform-libm-v1` |

- There is **no Python-free native serving claim**. C++ keeps its declared Python typed-table host dependency.
- The qualification plan identity depends on `CXX` and `JOPE_CXX_MARCH` through the C++ build argv. A host with different values produces a different plan and needs requalification. `verify_plan_binding` checks the standard flag, the optional `-march=` and the fixed flag suffix exactly. The compiler program itself is bound by the pinned tool inventory in FR (`unavailable`, `derived`).
- The latency benchmark (`inference_prediction.py:35`) rebuilds with a hard-coded `g++` and without `-march`. Its receipts are timing evidence, with exact-bit and row-cardinality checks (line 50). It is not qualification of record.

### Numerical support evidence (fixture)

- INF `all_matrix_heads_and_scalar_boundaries_compile_and_replay_synthetic_fitted_state`: 38 exact/declared-policy replays over 14 heads.
- INF `generated_winners_preserve_raw_boundary_and_private_provenance_on_export`: 6 replays of synthesized winners.
- CT raw-input differential (section 4): each fixture is checked on every backend it exports to. Ridge and logit on 5 backends, multiclass on 3, and the two branch heads on 2.

## 3. Refit, default, substitution and input-adaptation trace

| # | Path | Verdict |
|---|---|---|
| 1 | FE `winner` / `prepare`: the winner is the minimum complete five-fold mean loss, with ties broken by method id. The existing fold fit is exported. | **No refit.** Refuses when there is no complete winner, or when the original fitted worker is missing or ambiguous. *Observation:* the exported subject is a fold fit. No authorized final whole-data refit identity exists, so W9-T02's "final refit identity" cannot be bound yet (W5/W10). |
| 2 | FE `winner` accepts `ConfirmedRecipe.diagnostic` recipes | *Observation for W5:* a diagnostic confirmation recipe can be the export winner if its complete five-fold mean loss is lowest. Selection semantics are not changed here. |
| 3 | External adapter winners (FE `prepare` else-branch, `verify`) | Every backend is `unsupported` / `external_library_state`. The complete native pickle is retained, and a standalone replacement is refused. |
| 4 | Unsupported fitted state (INF `export_inference`) | Returns `Entry::Unsupported{reason}`. The winner is unchanged (INF `unsupported_backend_keeps_the_fitted_winner_unchanged`, `extended_scalar_export_keeps_auxiliary_backends_explicitly_unsupported`). |
| 5 | Missing learned state and serde defaults | The fitted state path has no `#[serde(default)]`. The only one found is private C++ runtime `CodecMetadata.emitter_revision`, which the panel export does not reach. Missing `Option` fields decode as `None`. CT `missing_learned_state_is_refused_never_defaulted` deletes each single field of three fitted envelopes. The exact-bytes loader refuses every deletion. Under a fresh digest, every deletion either fails decode/validation or re-serializes to identical bytes (an absent `null`). **0 defaulted states.** |
| 6 | **Report path `PreparedExport::verify` (DEFECT, fixed on this branch)** | See section 5. Before the fix, it accepted a same-recipe state swap, executed-file swap, missing harness stage, corrupted constant, omitted operation and hidden support. Checksums, bundle identity and fitted checksum were all rebound in each case. |
| 7 | FE `verify_executable_plans` (before launching qualification) | Re-derives every entry, including native reference predictions, from the fitted artifact. Unchanged. |
| 8 | FR `worker` | Runs only plan files with structured argv (`Command::args`, no shell), inside the pool. Expected predictions and labels stay out (`export_child_closure_omits_reference_predictions_and_targets`). |
| 9 | Latency benchmark IE `fit_export` | **Fits** a given synthesized program for timing. The subject is bound by digest to the fit it just made. It must not be read as selection or release evidence. |
| 10 | Input adaptation | The manifest carries the artifact's own input policy (`median-onehot256-indicator/v1` or `jope.fitted-input-state/v1`). Fill and onehot vocabulary come from fitted state. CT covers an all-missing training column and the `max_onehot=256` boundary. The emitted vocabulary contains `c255` and excludes `c256`/`c299`, with bit-exact native parity. |

## 4. Raw-input behaviour (W9-T04). Verified safeguard, not a new defect

CT `emitted_backends_match_native_on_raw_edge_inputs_and_refuse_malformed_rows` runs 42 raw cases through the **actual emitted source** of each backend. It compares each result with the native fitted predictor on the same rows, under the backend parity policy. The fixtures are ridge, logit-ridge, multiclass-ridge, binary tab_boost and regression extra_trees. Each is fitted on training data that contains the hostile category strings.

- **Valid cases (9)** must match native exactly: all-missing rows and columns; unseen categories; present `"NA"`, `"null"`, `""`, `"na"` kept distinct from JSON null; 18 numeric lexical forms (Unicode spacing, sign/exponent forms, subnormal, max finite, underflow to 0); hostile present categories; a 100 000-character category; a 50-row cardinality batch; empty input (every backend returns zero rows, as native does).
- **Malformed cases (33)** must be refused, with no partial output: 18 numeric forms (`NaN`, `inf`, `1e400`, empty/blank, `1_000`, `0x10`, `1,5`, a non-ASCII digit, dangling exponent, `U+001C` / `U+200B` padding, …); narrow, extra-column and empty rows; one bad row inside a good batch; and 11 JSON type-confusion cases (numbers, booleans, nested cells, named record rows, string rows, object/null batches, null rows, a lone UTF-16 surrogate).
- **Tallies are asserted** per backend. For example, Rust sees the 31 typed cases, accepting 9 and refusing 22. The dynamic backends see all 42, accepting 9 and refusing 33.
- **Result: 0 divergences on all backends.** Row-count and order checks rule out zip truncation both in the boundary and in `verify_qualification`. CT `raw_input_comparator_detects_changed_constants_and_truncated_outputs` is the positive control. It shows the comparator rejects a truncated output, a reordered output and a changed intercept.
- **Named-column mode is not offered.** The interface is positional, and the manifest declares column names and kinds. A reordered pair of columns is refused only when the kinds or values make it invalid. For example, two categorical columns swapped with each other would be accepted as data. This is a documented limit, not a canonicalization rule.

## 5. Defect found and fixed: report-path capsule binding (W9-T02, S07-G01/G02/K01/K10)

**Reproduction.** Receipt `w9-t02-repro-20260926T062146Z`, exit 100. FE `same_recipe_state_swaps_and_missing_stages_fail_even_with_equal_predictions` builds two fitted states **of one registered recipe**. B differs from A only in the learned weight of the numeric column. Every qualification row sets that column to its fitted mean, so both states predict **bit-identical** values. The test shows that B's emitted Python passes A's qualification plan by parity alone. Before the fix, the entry checks accepted 6 of 9 tampered entries: `state-swap`, `executed-file-swap`, `missing-harness-stage`, `corrupted-constant`, `omitted-operation` and `hidden-support`. These checks were extracted from `verify()` unchanged into `verify_entries`, and the reproduction ran against them. The 3 already refused were `receipt-swap`, `missing-wrapper` (bundle allowlist) and `poisoned-reference`.

**Fix.**
- `verify_entries` re-emits public source from the authenticated fitted output and compares bundle identity.
- It requires the fitted view digest to equal the selected fitted checksum.
- It requires every `Unsupported` reason to equal the reason the fitted state yields, so support can be neither hidden nor invented.
- It calls the new `inference::verify_plan_binding`, which checks that the executed plan files are exactly the shipped files plus the deterministic harness and unlabeled input, and that the commands are the fixed harness commands.
- `qualification_plan_for_view` now self-checks with the same function. The harness text moved into one function unchanged.
- The report path still never compiles or predicts. It now re-emits source text, which is pure string generation.

**After the fix:** receipt `w9-t02-fixed-20260926T062434Z`, 19/19 fitted_export and inference tests pass. The 9 tampered entries are refused, the intact entry verifies, and both fitted subjects and the receipt are byte-identical afterwards.

**Compatibility note.** A retained `prepared.json` from an older binary whose emitter output differs now fails `verify()`. That was already the case for template or emitter version changes (`PublicBundle::validate`). This makes the fail-closed property explicit for any lowering change.

## 6. Package manifest and install/load (W9-T05). New capability, fixture-proven

The **gap was reproduced** by PKG `public_manifest_alone_cannot_see_a_substituted_native_library`. The existing `PublicManifest` binds only source files. When another state's qualified `model.so` is placed beside intact sources, `PublicBundle::validate` still passes and the host silently serves different predictions.

PKG adds the following. It **reuses** `PublicManifest`/`PublicBundle`, `QualificationPlan` and `QualificationResult`, and introduces no parallel authority.
- `PackageManifest` (`jope.fitted-inference-package/v1`). It binds backend, required/optional, fitted digest, bundle identity, public manifest, the native reference digest (the plan's frozen `expected`), plan identity, `QualificationResult`, the dependency profile (build tools and exact build argv, serve tools, each pinned by absolute path and digest) and the digest of every shipping file. `model.so` is bound to the digest recorded by the qualification execution.
- `seal` refuses a result from another plan, bundle or state. It also refuses a profile that is not the executed one, unqualified or missing native bytes, and a wrong-backend receipt.
- `load(dir, identity)` re-hashes the bytes actually installed. It refuses a stale or substituted reference; changed, missing, extra or linked files; a substituted executable; a missing wrapper or codec stage; a rebound wrong-backend or other-state receipt; and a changed serving dependency.
- `assess(request, outcomes)` works as follows. A required backend that failed or is absent returns `blocked`. An optional failure returns `partial`. A package sealed as optional cannot fill a required slot. The result is always `release: "not-decided"`.

Tests:
- `intact_packages_load_and_tampered_installations_refuse_before_serving` covers Python, Rust and C++17. It refuses 21 installed mutations. The rebound-receipt mutations are presented with their own new digest, so the refusal comes from the internal bindings, not from the reference. It also covers stale references and seal-time refusals. The intact package still loads afterwards. The installed Python and C++17 bytes serve exactly the qualified prediction digest; the Rust package is a source library and is not served here.
- `changed_serving_dependency_refuses_and_required_failures_block`.
- `public_manifest_alone_cannot_see_a_substituted_native_library`, the gap reproduction above.

**Trust limit.** `load` trusts the `identity` it is given (the digest of `package.json`). A package whose internal bindings were all rewritten consistently to another fitted state cannot be told apart without that reference, or without re-emitting from the fitted state as FE `verify_entries` does. The reference must come from an authenticated release record (W9-T07/W12).

## 7. Hostile strings, determinism and structural checks (W9-T03/T06)

- CT `hostile_fitted_strings_never_reach_command_arguments`: every plan's argv is the fixed harness (`verify_plan_binding`). No category, class or column string appears in any argument. The raw differential run confirms that no `PWNED` canary file was created by strings carrying shell, Python, JS and Rust syntax. Template-marker look-alikes are handled by the existing `render`.
- CT `re_emission_of_an_identical_capsule_is_source_deterministic`: re-emitting a capsule directly, or after an exact-bytes reload, gives byte-identical bundles and identical plan identities for 10 backend/head pairs. **This is source determinism only.** No reproducible-binary claim is made. No clean-process, locale or directory variation was run.
- Structural checks are bounded to equality with the deterministic lowering of the fitted capsule: re-emission identity plus plan binding. Tests exercise an omitted operation, a corrupted constant, a missing harness stage and a missing wrapper (section 5; PKG for a missing codec stage). Reordered stages fall to the same byte-identity comparison, but no test targets them specifically. These checks are **not** a semantic program-equivalence check. A new lowering does not inherit support merely because it compiles.

## 8. Open items and risks

1. **Release is undefined.** No code joins the selected final-fit identity, package qualification, a release decision and installed bytes (W9-T07).
2. **Production wiring.** PKG is fixture-proven only. FR `run`/`recover` do not yet seal packages from `qualified` receipts, and there is no install CLI.
3. There is no final whole-data refit identity. The exported subject is a fold fit (item 1 in section 3).
4. A diagnostic recipe may win the export (item 2 in section 3). This is for the W5 owner.
5. There is no named-column input mode (section 4).
6. Bounded builds are only as bounded as FR's pool. The credential-free, network-denied and private-canary builder checks of W9-T03 were not run here.
7. The C++ plan identity is host-environment dependent (`CXX`, `JOPE_CXX_MARCH`).

## 9. Run receipts

Directories are under `/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`. Every run used `cargo nextest run --locked --no-fail-fast --test-threads 4 -p jope-bench`, adding `--features jope-bench/asm` for asm runs, with the lane's own `target/`.

| Run | Directory | Result |
|---|---|---|
| Baseline, default features | `w9-baseline-default-20260926T060609Z` | 81/81 pass |
| Baseline, asm | `w9-baseline-asm-20260926T060644Z` | 117/117 pass |
| T02 reproduction (superseded fixture) | `w9-t02-repro-20260926T062110Z` | exit 100. The "states differ elsewhere" row saturated the ±60 clamp. The fixture row was corrected and nothing else changed. |
| T02 reproduction | `w9-t02-repro-20260926T062146Z` | exit 100. 6 tampered entries accepted (section 5). |
| T02 fixed (focused: fitted_export, inference) | `w9-t02-fixed-20260926T062434Z` | 19/19 pass |
| Commit 1 state, default / asm | `w9-commitA-default-20260926T063215Z`, `w9-commitA-asm-20260926T063236Z` | 81/81, 118/118 |
| Commit 2 state, default / asm | `w9-commitB-default-20260926T063328Z`, `w9-commitB-asm-20260926T063354Z` | 88/88, 125/125 |
| Commit 3 state before a test-only clippy type alias, default / asm | `w9-commitC-default-20260926T063442Z`, `w9-commitC-asm-20260926T063509Z` | 91/91, 128/128 |
| Commit 3 as committed, default / asm | `w9-commitC2-default-20260926T063720Z`, `w9-commitC2-asm-20260926T063747Z` | 91/91, 128/128 |
| Commit 4 (clippy fix; also covers this document), default / asm | `w9-commitE-default-20260926T063905Z`, `w9-commitE-asm-20260926T063937Z` | 91/91, 128/128 |

At every commit, the pass-name sets were compared with the baseline, and no previously passing test was lost. `python3 deploy/ci/coverage.py validate` reports 0 unassigned or invalid items. The new tests live in the existing `jope-bench` lib target, so no coverage key was needed.

Clippy: `cargo clippy -p jope-bench --no-deps --all-targets --locked`, default and with `--features jope-bench/asm`, reports nothing in `fitted_export.rs`, `inference_contract_tests.rs` or `inference_package.rs`. The remaining jope-bench findings are pre-existing and on unchanged lines: six `needless_borrow` in `inference.rs`; under asm also `fitted_export_run.rs:116` (`needless_lifetimes`), `demo_asm.rs`, `experimental_proposal.rs` and `optimize.rs`. The workspace `just clippy` (`-D warnings`) already stops on pre-existing lints in `jope-pipeline` (`search_plan.rs:200`), `hyperion` and `jope-encoders` with the pinned 1.97.1 toolchain.
