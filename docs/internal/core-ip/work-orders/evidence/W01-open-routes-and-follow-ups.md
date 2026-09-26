# W1 round 3 — open consumer-map routes, outer-release CLI, projected export input, planner closure check, harness extension

Branch `rr/w1-inputs-3`, base `248ded85f` (`feat/regression-remediation`). Owner: Claude
implementation subagent, lane `.agent/worktrees/demo`. Inputs:
- the committed [consumer map](W01-consumer-map.md);
- its appended route status;
- [W01-label-release-counterfactuals-staging.md](W01-label-release-counterfactuals-staging.md).

Every state below is local only and not CI-verified. Most is fixture scope. Two sections
also name local evidence runs on real inputs: 2d replays retained custody evidence, and
the 2e encoder proof runs on the accepted encoder assets. No register entry is closed.

Evidence lives under
`~/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/w1b-r3-*`,
`w1b-r4-*`, `w1b-r5-*` and `w1b-c<N>-*`. Each directory holds `HEAD`, `diff.patch`, `untracked.patch`, `status.txt`,
`source.txt`, `nextest.log` and `summary.txt`.

## 2a — Consumer-map routes

Commits: `b613823a1` (engine: R1, R8, R11; R9 proof) and `4abfb0818` (datasets: R4).

Pre-commit runs, each covering the W1 regression set plus the fence suites, jope-datasets,
workspace all-targets check, coverage validate, verify-test-inventory, fmt and diff-check:

| Run | Engine | jope-datasets | Coverage |
|---|---|---|---|
| `w1b-c1-*` | 157/157 | 63/63 | 0 unassigned |
| `w1b-c2-*` | 157/157 | 68/68 | 0 unassigned |

### Enforced, each with a failing reproduction first

**R1 — missing recorded backing path re-rooted by the environment.** Enforced.
- Reproductions:
  - `w1b-r3-2a-prefix-roots-bytes-*`: the first attempt carried a fixture copy bug; it is
    retained.
  - `w1b-r3-2a-prefix-roots-rerun-*`: the clean reproduction.
- In both, `JOPE_REGISTRY` alone selected which allowlisted backing object preparation read.
  The two env values gave two different `storage.source_sha256`.
- Fix (`prepare.rs`):
  - The recorded path is read when it exists.
  - Otherwise the source is read only from a location the explicit backing allowlist names
    as a `source` with the same `hyper/data`-relative identity. Exactly one such location
    is allowed.
  - `jope_paths::quant_root` is no longer consulted.
  - The reconstruction receipt records `source_resolution`.
  - The source is read once; the digest describes the parsed bytes.
- Tests (`receiver_roots`):
  - `a_missing_recorded_backing_path_is_never_re_rooted_by_the_environment`: env-independent;
    the allowlisted relocation is used; ambiguous and absent relocations are refused and
    leave no output.
  - `an_existing_recorded_backing_path_is_read_as_recorded` (positive control).

**R11 — env-derived roots deciding authority.** Enforced at the preparation receiver.
- Reproduction (same runs): the same audit and inputs gave `family_partition_frozen`
  true or false depending on `JOPE_REGISTRY`.
- Fix:
  - Without a frozen family registry, the partition is frozen only for an explicitly
    declared `authoritative_inventory_root`: a new `run_with_admission` parameter and the
    `prepare --authoritative-inventory-root` flag.
  - With neither, it is not frozen.
  - `reconstruction.json` records `family_partition_frozen_authority`.
- Test: `the_frozen_family_partition_is_never_decided_by_the_environment`, covering the
  positive control with the declared root and a negative declared root.
- Not changed: `jope_paths::jope_home`'s current-directory fallback is workspace-wide
  ADR-0003 behavior. After this change no W1 preparation decision reads it.

**R8 — hash, then reopen by path.** Enforced at the preparation and GP receivers.
- New `manifest::read_verified` reads once, hashes those bytes and returns them.
- Receivers converted:
  - `load_raw_view_in_root` (backing table);
  - `load_view_with_root` (preprocessing);
  - `run_with_admission` (metadata; historical training split, verified against the audited
    transfer entry when the audit bound it);
  - reconstruction (backing source);
  - `gp::fit_view`, `gp::evaluate_view` (which previously re-read preprocessing
    unverified) and `gp::search_inner` (preprocessing).
- Reproductions:
  - Unit tests `prepare::verified_bytes_tests::{backing_bytes_replaced_after_verification_are_never_consumed, preprocessing_replaced_after_verification_is_never_consumed}`.
    They use a `cfg(test)`-only instrumentation point (`manifest::after_verification`) that
    replaces the file between verification and consumption. Before the fix, the receiver
    returned the replacement's rows or refused on it.
  - `receiver_roots::a_training_split_changed_after_its_audit_is_refused`: the training split
    was prepared after changing post-audit.
- The GP sites use the same helper. They have no separate race reproduction, because a
  fit-level reproduction needs the fixture fit authority.
- Cost: the native input cache no longer short-circuits the digest of these view objects.
  They are hashed from the bytes already being read, one SHA-256 pass.

**R4 — cache-first manifest, LATEST resolution, cached-row fallback.** (`jope-datasets`)
- Reproduction `w1b-r3-2a-prefix-r4-*`: with the store reachable, an altered cached
  `by-id/<id>/manifest.json` was served. The dataset's `train.csv` was materialized from
  another blob.
- Fixes:
  - Online, the store's manifest is authoritative. A stale cached manifest's consumer copies
    are withdrawn and replaced from the resolved row.
  - New pinned resolution (a new capability, so there is no pre-existing reproduction):
    `CatalogPin`, `Fetcher::{pinned_catalog, fetch_pinned}`, and
    `fetch --catalog-version v<N> --catalog-sha256 <sha>`.
  - Pinned resolution never reads `LATEST`, never substitutes cached rows or manifests, and
    serves a cached catalog only when its bytes match the pin. Offline it refuses otherwise;
    online it re-verifies the store copy against the pin.
- Tests (`pinned_catalog`, 5): pin versus advancing LATEST (online and offline); tampered
  catalog refused offline and re-verified online; wrong pin refused with nothing
  materialized; stale manifest replaced even offline; selector limits.
- Not changed: unpinned offline fetch still serves a cached manifest (it is all that
  exists) and the no-LATEST cached-row listing that `link` relies on. Both are documented
  in the test; use the pin to exclude them.

**R9 — worker input cache reuse.** Existing safeguard, now proven at the receiver.
- New unit test
  `native_input_cache::tests::receiver_hit_miss_and_resume_across_scopes_never_serve_a_replaced_input`,
  through `manifest::file_hash` (the call in the worker's verification loop,
  `smartcluster_worker.rs:497`).
- It covers: a miss is a full hash; an unchanged identity is a hit; a resumed scope reuses;
  an input replaced between scopes is re-hashed and refused; a re-bound replacement is
  accepted.
- It passes on the existing code: this is a safeguard demonstration, not a defect.

### Recorded, not source-provable in this lane

| Route | Why it stops here |
|---|---|
| R2 (ancestor-walk root in `load_view`/`load_raw_view`) | The no-root API is used by the pinned `eval_registry/native_scq.rs:451`, which may not be edited, and by the neural-gated `encoding.rs`, `components.rs` and `hyperion.rs` and `hyperion-batch-worker`. The fence suites also use it (`view_contracts` calls `load_view` and `expand_cuts`). The bytes are digest-verified; the location is unbounded. The export worker's own view load is removed by 2c. The remaining callers need their owners. |
| R5 (network fetch on cache miss) | Acquisition-component behavior, digest-checked. `--offline`/`JOPE_OFFLINE` disable it, and a pinned fetch never substitutes. Denying network to the managed learning worker is W7 (live). |
| R6 (`prepare.py` frozen-mode fetch) | Python, and `benchmarks/prepare.py`'s SHA-256 is bound into the committed regression release, so editing it changes that identity. Acquisition-only and digest-checked. No Python may be added or changed in this lane. |
| R10 (`JOPE_DATASETS_URL`, credential chain) | Only `jope-datasets` (`store.rs`, `cli.rs`, `lib.rs`) reads them: the acquisition component. No engine or bench receiver does. Removing broad credentials from the managed worker is W7 (live). |
| R12 (self-certified context-producer training identity) | Binding needs two things outside this lane. (a) A per-cut fitting-row alignment digest in the regression release: a release v2, so a new release identity. (b) The context producer's `TrainingIdentity` to carry and verify it: the jope-policy W2 schema. W2-T03 added request-issued admission (`prepared_sha256` binding), not a release binding. |
| R8 remainder | `label_admission.rs:477-481` (teacher; outside this lane's partition). Pinned `eval_registry` sites. |

## Round 4 — branch `rr/w1-inputs-4`, base `be687d8d5`

Evidence: `runs/w1b-r4-*` (reproductions and post-fix runs) and `runs/w1b-c<N>-*`
(pre-commit sets).

The integrated head `be687d8d5` was verified before any change was built on it, on a
clean tree:
- engine 157/157 and jope-datasets 68/68;
- workspace all-targets check, coverage validate (0 unassigned), verify-test-inventory,
  fmt and diff-check.

`manifest::read_verified` and the W6 `atomic_write` do not interact. The writer publishes
by rename from a unique temporary; the reader reads one opened object once and hashes
those bytes. A rename between the reader's open and read cannot mix the two objects.

## 2b — `outer-release` CLI

This is a new capability, so it has no defect reproduction. The prefix run
`w1b-r4-2b-prefix-cli-*` records that the shipped binary had no such verb (`unrecognized
subcommand 'outer-release'`).

Commit `4f43295d6`. Verbs:
- `jope-parallel outer-release init|grant|score|release|events`;
- each call goes to the one `gp::outer_release::Authority` (grant, withheld score, single
  release, hash-chained ledger).

The shipped binary keeps the fit fence. A grant for a predictor that no authority-bound
export produced is refused and recorded.

Tests:
- `tests/outer_release_cli.rs`, against the shipped binary:
  - refused re-init;
  - refused grant;
  - refused release without a withheld score;
  - the event sequence;
  - `--help` listing.
- `gp::fit_contract_tests::the_outer_release_cli_grants_scores_and_releases_once`, under
  the fixture fit authority:
  - grant, score and release once;
  - a second grant, score and release are each refused;
  - an unknown verb or a missing flag is a parse refusal.

Pre-commit `w1b-c4-2b-*`: engine 159/159 and datasets 68/68. The parity matrix, coverage
(0 unassigned), inventory, fmt and diff-check all exit 0. The capability matrix has a new
`outer-release` row, and the counts are recomputed.

## 2c — the export worker receives its projected input (R2 at the export receiver)

**Defect.** `gp::handoff::export_sealed` spawned `fit-view --prepared <path>`. The worker
reloaded the fitting view itself through `load_view`, whose ancestor walk picks the
nearest directory holding `manifests/` and `Cargo.toml`, not the search's declared root.

**Reproduction.** `w1b-r4-2c-prefix-20260926T102350Z`. The setup:
- the declared root is relocated without workspace markers;
- an ancestor is a workspace holding a byte-identical backing table;
- the root's own backing is moved away.

The export succeeded by reading the ancestor's copy ("the export read its fitting view from
outside the declared root").

**Fix.**
- The parent projects the cut's fitting rows once, under the declared root
  (`load_view_in_root`). It reads the preparation state with `read_verified`.
- The worker receives only a sealed memfd holding those rows and that state
  (`gp::fit_memory_input`: write, grow, shrink and seal seals).
- The new hidden verb `fit-memory` reads only that descriptor. It refuses:
  - an unsealed descriptor;
  - a training digest that differs from the cut's;
  - a preparation digest that differs from the cut's;
  - trailing bytes.

  It then fits through the same `fit_boundary` authority (`gp::fit_projected`, which the
  in-process `fit_view` also uses).
- The export receipt records `artifact_root`, `view_resolution`, `worker_input`,
  `worker_argv` and `worker_opens_prepared: false`.
- A projection that cannot be made under the root is a failed receipt, and no worker
  starts.

**Tests.**
- `gp::fit_contract_tests::the_export_worker_receives_its_projected_input_and_never_resolves_the_view_itself`:
  - negative: the backing exists only outside the root, so the export fails, the receipt
    fails and no worker argv is recorded;
  - positive control: the backing is restored inside the root and the decoy ancestor
    stays a workspace. The export succeeds through `sealed_memory_projection`, and no argv
    element names the prepared path.
- `tests/projected_export.rs`, against the shipped binary: `fit-memory` is fenced
  (`UNVERIFIED_ACTIVATION`). It refuses a mismatched training identity and an unsealed
  regular file, with zero fit starts and zero worker spawns.

**First post-fix run.** `w1b-r4-2c-postfix-*` had 2 fence failures (47/49). The legacy
`fit_view_isolated` resolved its workspace before creating its receipt. On the fence
suites' marker-less fixture views, a fenced export therefore left no failed receipt. The
legacy wrapper now projects through `load_view` inside the receipted closure. The receipt
records `view_resolution: legacy_view_resolution`.

**Results.**
- Post-fix `w1b-r4-2c-postfix2-20260926T102732Z`: 49/49, fence suites included.
- Pre-commit `w1b-c5-2c-*`: engine 161/161 and datasets 68/68. The parity matrix,
  coverage (0 unassigned, 253 targets), inventory, fmt and diff-check all exit 0.
- The capability matrix has a new `fit-memory` row (hidden from `--help`), and the counts
  are recomputed.

**Still open.**
- R2 for the public in-process `fit-view` verb and the legacy `fit_view_isolated` wrapper:
  both still resolve through `load_view`, with digest-verified bytes and an unbounded
  location.
- The remaining `load_view` callers listed above.
- The inherited memfd is sealed read-only and carries fitting rows only. It is created
  without close-on-exec so the worker can inherit it, so a child spawned concurrently by
  another thread of the same parent could inherit it too. The coordinator accepted this
  as disclosed.

2b and 2c were integrated as `7835e79ff` and `1217bac14`. 2d and 2e are on
`rr/w1-inputs-5`, cut from the integration head `a606bad2e`.

## 2d — the W1-T07 replay verifier at `jope benchmark inference-plan`

**Defect.** The v3 planner checked only the shape of the fitting-input manifest's
`projection_source_sha` (40 hex). It re-read the manifest after verifying it. It never
consulted the committed fitting-only staging protocol. Plan admission (`read`: `run`,
`cold` and the v3 `report`) did not consult it either.

**Reproduction.** `w1b-r5-2d-prefix-20260926T103830Z`, on the base's own build
(`JOPE_SOURCE_SHA=a606bad2e`, asm), replaying retained custody evidence:
- the retained v3 planner command on the retained historical closure
  (`study-final/retained-fitting-inputs.json`, `e0256090…`): exit 0, plan written;
- report admission of the retained local v3 plan built on that closure
  (`repaired-planner-local/plan.json`, `acd1df4b…`): exit 0;
- the retained manifest re-attributed to the current source with every object unchanged
  (`39e1a8f8…`): exit 0, plan written.

The same result on `21d7073e7` is in `w1b-r4-2d-prefix-*`.

**Fix.**
- `jope_datasets::staging_disposition::committed()` opens the committed disposition and
  protocol, embedded at build time, with every identity check.
- `refuse_historical_objects` refuses pinned digests of any staging product.
- In `inference_inputs::load`, the manifest is read once as a regular file, and the
  consumed bytes must be the pin. `verify_closure` admits the closure before any split
  membership or projected input is opened. The planner's output records the closure
  receipt (`staging_closure`, with `staging_protocol_id`).
- `inference_benchmark::read` refuses a plan that pins the historical manifest, a
  historical projected input or a historical inner-fold input, whichever planner built
  it.
- No `inference_report*` file is changed.

**Post-fix.** `w1b-r5-2d-postfix-20260926T104151Z`, the same three replays:
- planner, exit 1: "the historical input closure (manifest e0256090…) cannot be replayed
  under the corrected fitting-only protocol";
- report, exit 1: "fitting-input manifest e0256090… was produced by the historical
  staging path";
- re-attributed closure, exit 1: "197_cpu_act: projected fitting input 64595358… was
  produced by the historical staging path".

**Tests.**
- `panel::inference_benchmark::inputs::tests` (jope-bench, asm):
  - `the_planner_admits_only_closures_outside_the_historical_staging_path`. Positive
    control: a fresh closure is admitted from exactly the pinned bytes, with the receipt.
    Refusals: the historical projection source, a historical projected input and a
    historical inner-fold input (each before any named object is opened); bytes replaced
    after pinning; a symlinked manifest.
  - `a_plan_pinning_any_historical_object_is_never_admitted`. Positive control, the
    retained historical manifest, and a fresh manifest re-pinning one historical input.
- jope-datasets `staging_disposition::the_embedded_pair_is_the_committed_release_and_refuses_only_staging_products`.

**Limits.**
- The check is by digest. A re-encoded copy of a historical projection is a new object.
- The producer's projection attestation (`verify_corrected_run`) is not required by the
  planner, because no producer emits a corrected run document yet.
- Report admission of an archival v2 plan is unchanged.

**Commit and pre-commit.** Commit `2ae401b9a`. Pre-commit `w1b-c6-2d-*`: engine W1 set plus
fence suites 160/160, jope-datasets 69/69, jope-bench (asm) 163/163. The workspace
all-targets check and the asm (full-default feature) check, parity-matrix, coverage
(0 unassigned), inventory, fmt and diff-check all exit 0.

## 2e — the counterfactual harness extended to the context producer, encoders and teacher

This is a new verification capability, not a defect fix. No production code changes. Each
proof freezes the permitted inputs, perturbs one thing, and compares SHA-256 digests with
zero tolerance. Authorized controls show that each receiver is sensitive to its permitted
inputs, so a pass is not vacuous. One receipt per mutation goes to
`$JOPE_W1_RECEIPTS/counterfactual/`.

**Context producer** (CI; jope-bench, asm):
`origin_proof::tests::the_context_producer_receives_identical_bytes_under_every_outer_mutation`.
- Receiver: `optimize::build_pair_input` builds the `TrainingInput`, then
  `TrainingInput::prepare`. Those are the bytes the CPU encoders consume.
- The fixture is the synthetic regression table with numeric and categorical columns.
- Forbidden: cut 0's outer scoring partition (fold 0). Its labels changed or shuffled,
  values changed or missing, categories changed, and rows reordered all leave the
  producer input, preparation state, prepared rows and context identical.
- The pair input is identical too, except under reordering: the outer request binds the
  hold rows' identities and order, never their labels, values or categories.
- Authorized controls: a fitting-row label, value or category changes the input, context
  and prepared rows. The category control also changes the preparation state.
- Run `w1b-r5-2e-harness-20260926T104720Z`: 9 receipts. Its receipts showed that only
  reordering moves the pair input, so the assertion was then tightened. The tightened
  test is re-run, with receipts, in the pre-commit bench step `w1b-c8-2e-bench-*`.

**CPU encoders** (local evidence; `#[ignore]`, because the accepted assets are not in
the repository):
`origin_proof::tests::the_cpu_encoders_produce_identical_context_under_every_outer_mutation`.
- It runs `CpuProducer::produce` (preparation, Hyperion, foundation and target encoders,
  frozen assembly) on each mutation's producer input. Run it with
  `JOPE_W1_ENCODER_FILES=<encoder-files.json>` and `--run-ignored only`.
- Determinism control: the same input twice gives byte-identical artifacts.
- All 6 forbidden mutations give a byte-identical serialized context artifact.
- Every authorized control changes the final context. The receipts record which products
  moved:
  - feature value or category: Hyperion and foundation;
  - label: foundation and target.

  Each encoder is therefore shown to respond to its own authorized inputs.
- Run `w1b-r5-2e-encoders-20260926T104735Z`: 183 s, 9 receipts, with the accepted asm-demo
  encoder files.
- Re-run on the final pre-committed tree `w1b-r5-2e-encoders-final-20260926T110038Z`: 191 s,
  9 receipts, identical outcomes.

**Teacher input admission** (CI; engine, default features):
`tests/teacher_counterfactuals.rs`.
- Receiver: `label_admission::verify_inputs`, which binds an admitted teacher label to the
  cut's training identity.
- Fixture: the W1-T06 legacy registry fixture, unchanged. A single-family audit always
  places its one family in `meta_test`, which the teacher never admits. The proofs
  therefore add a distinct, approved, never-prepared companion dataset of a family that
  ranks after the example's, which puts the example's family in `meta_train`.
- The tests:
  - `sealed_split_mutations_never_reach_the_teachers_admitted_inputs`: sealed labels
    changed or shuffled, rows reordered, categories changed. Every cut's admitted
    identity is identical, while `prepared.json` differs, which shows each mutation is
    real.
  - `the_teachers_input_admission_never_opens_the_sealed_split`: identical with the
    sealed split moved away (denied read). The default single-family fixture
    (`meta_test`) is refused ("admission sealed").
  - `a_cuts_held_out_rows_never_reach_its_admitted_training_identity`: cut 0's held-out
    labels leave cut 0's identity unchanged. Authorized: they change cuts 1–4.
  - `a_training_label_changes_the_admitted_identity_and_stale_bytes_are_refused`:
    authorized control, plus refusal of a teacher label that names the earlier
    preparation bytes.
- Run `w1b-r5-2e-teacher3-20260926T105307Z`: 4/4, 7 receipts. The first two runs
  (`teacher`, `teacher2`) stopped at the fixture's `meta_test` family. They are kept.
  `teacher2` renamed the family, which cannot help, because the audit assigns partitions
  by rank among families.

**Recorded, not source-provable here.**

| Receiver | Why |
|---|---|
| Trainer | `jope-learning` and the engine's `training*` and `neural*` modules are outside this lane's partition. Training needs LibTorch/CUDA (W6/W7). |
| Teacher: parse versus hash | `native_genome.rs:347-349` parses `prepared.json` with `read_json`. `verify_inputs` then hashes the file separately and checks provenance (partition, family, clean evaluation) against that caller-supplied parse, not against the hashed bytes. This is the same R8 pattern as `label_admission.rs:477-481`, and as `teaching_store.rs:551,572`, where `search.json` is hashed, then re-read by path. **Closed in round 6 (below).** |
| Encoders in CI | The accepted encoder assets are outside the repository, so the encoder proof is an ignored local-evidence test. |

## Round 6 — R8 at the teacher's receivers (branch `rr/w1-inputs-6`, base `07bc7e60b`)

2d and 2e were integrated as `4c3c7c806` and `07bc7e60b`. The teacher partition's owners
have finished, so the R8 sites recorded above are closed here at their receivers. No
authority state, schema or digest binding changes. The refusal messages keep their
previous prefixes.

**Reproduction.** `w1b-r6-prefix3-20260926T111518Z`: 6/6 FAIL. Test-only instrumentation
(`manifest::after_verification`) was placed right after each pre-fix digest check. It
replaces the file between the check and its consumption.

| Test | Pre-fix outcome |
|---|---|
| `teaching_store::read_once_tests::a_forged_scale_is_refused_even_if_the_state_is_replaced_after_its_check` | A forged target scale, with its own committed completion, was **admitted**. |
| `…::an_honest_scale_is_admitted_against_the_verified_state_even_if_replaced_after_its_check` | The honest row was refused on the replacement's scale. |
| `…::the_teachers_input_admission_binds_the_callers_preparation_to_the_hashed_bytes` | A non-clean bound preparation was **accepted** on a caller's parse that said clean. |
| `…::a_preparation_replaced_after_its_check_never_decides_the_admitted_views` | The admitted views were resolved from the replacement. |
| `…::a_search_receipt_is_validated_from_the_bytes_its_ledger_commits` | **9 labels admitted** while the ledger committed a receipt with `final_tests_opened: true`. |
| `native_genome::tests::the_candidate_boundary_is_decided_on_the_verified_preparation` | A non-clean bound preparation passed the scientific boundary on its replacement. |

Earlier attempts are kept:
- `w1b-r6-prefix-*`: the store fixture lacked the two extra programs per family, so no
  family reached a shard. The other three tests reproduced.
- `w1b-r6-prefix2-*`: INTERRUPTED by the operator during compilation; no test ran.

**Fixes.**
- `label_admission::validate_preverified` (was `:475-481`): the target scale is parsed
  from `read_verified` bytes of the preparation state.
- `label_admission::verify_inputs` (was `:240-266`), the receiver behind
  `native_genome.rs:347-349` and the teaching store's preparation map:
  - `prepared.json` is read once against the context's digest;
  - the caller's parse must equal it ("admission prepared parse differs from the bound
    preparation bytes");
  - the views are resolved from it through the new `prepare::load_bound_view_in_root`,
    which does not reopen `prepared.json`.

  `load_view` and `load_view_in_root` are unchanged; they parse, then share the same view
  code (`view_of`).
- `native_genome::verify_result` (was `:347-349`) and `candidate()` (was `:195-200`, the
  same object) read the bound preparation once.
- Teaching-store ingestion (was `:551,572`): `search.json` is read once through the new
  `manifest::read_hashed`, so the committed digest and the validated receipt are the same
  bytes. The preparation map (was `:466-472`) holds `read_verified` bytes.

Each test has a positive and a negative control. The store fixture reuses
`admission_tests`' `Fixture` (made `pub(super)`; no existing test changed).

**Results.**
- Post-fix `w1b-r6-postfix-20260926T111709Z`: 25/25. This covers the six new tests plus the
  existing `admission_tests`, `label_admission`, `teaching_store`, `native_genome`,
  `preparation_overlay` and `prepare::verified_bytes_tests` unit tests.
- Pre-commit `w1b-c9-r6-*`:
  - engine: the W1 set, the fence suites and the teacher suites, 281/281. The teacher
    suites are `gp_admission`, `corpus_incremental`, `corpus_projection`,
    `frozen_inputs`, `request_integrity`, `bounded_run`, `teacher_counterfactuals` and the
    `teaching_store`/`label_admission`/`native_genome`/`preparation_overlay`/
    `native_graph`/`corpus*`/`champions`/`manifest` unit modules;
  - jope-datasets 69/69;
  - workspace all-targets check, parity-matrix, coverage (0 unassigned, 260 targets),
    verify-test-inventory, fmt and diff-check all exit 0;
  - no warnings in the changed files.
- jope-learning is not touched. No coverage target or inventory entry is added: the tests
  live in the engine library target.

**Still open (recorded, not changed).**
- The pinned `eval_registry` sites (`authority.rs`, `witness.rs`, `service.rs`) and
  `native_scq*`.
- Further hash-then-read sites found while closing these, not part of this assignment,
  each needing its own reproduction. All are addressed in round 7 (below); the
  `provenance.json` entry turned out not to be an R8 site:
  - `native_genome::required_inputs` `:479-481`: an input is hashed, then re-read to
    discover nested references.
  - `native_genome::required_inputs` `:498`: the storage source comes from an unverified
    parse of `prepared.json`, although its digest is recorded at `:494`.
  - `teaching_store::verify_shard_metadata` `:103/106`: `shard.json`.
  - `teaching_store::load_impl` `:1142-1145`: `provenance.json` is read unbound.
  - `teaching_store::teacher_programs` `:1243-1246`: `index.json` is parsed, then hashed
    separately.
- The packer's `semantic_dependencies_sha256` covers `manifest.rs`, `prepare.rs`,
  `label_admission.rs` and `teaching_store.rs`. Any changed implementation gets a new store
  lineage, as designed.

## Round 7 — the remaining R8 sites (branch `rr/w1-inputs-7`, base `fe4cdce08`)

Round 6 (`fb4b46ee2`) was integrated as `fe4cdce08`. This round closes the further sites
listed above.

**`load_resident_verified` checked first.** `load_impl`, `load` and
`load_resident_verified` already read `index.json` once (W6 follow-up `b5052f8d6`), and
the index they return is parsed from the hashed bytes. They are unaffected.
`teacher_programs` is a separate receiver. It re-read `index.json` by path, and used the
caller's verified `Index` only to select records.

**Step 0: a behavior-preserving extraction.** In `native_genome::required_inputs`, the
reference loop and the prepared-directory walk moved into `artifact_closure` and
`prepared_closure`, with the same statements in the same order. Each is then testable
without a full native-graph fixture.

**Reproduction.** `w1b-r7-prefix-20260926T113012Z`: 7/7 FAIL. Test-only instrumentation
(`manifest::after_verification`) was placed after each pre-fix digest check. For
`teacher_programs` it sat between the parse read and the digest read.

| Test | Pre-fix outcome |
|---|---|
| `native_genome::tests::the_input_closure_follows_only_the_verified_bytes` | The closure followed the replacement's reference (`y.txt`) and lost the bound one (`x.txt`). |
| `native_genome::tests::the_prepared_closure_is_the_preparation_the_candidate_bound` | After the candidate's check passed, the closure recorded the replacement `prepared.json` digest and its storage source (`s2.csv`). |
| `teaching_store::read_once_tests::an_honest_shard_manifest_replaced_after_its_check_is_still_accepted` | Refused, "shard binding changed". |
| `…::a_shard_manifest_disagreeing_with_its_binding_is_refused_even_if_replaced` | A bound `shard.json` that disagrees with its binding was **accepted**. |
| `…::teacher_labels_come_from_the_named_index_even_if_it_is_replaced_after_its_read` | Refused, "label index changed". |
| `…::teacher_labels_of_an_unnamed_index_are_refused_even_if_it_is_replaced_after_its_read` | The **forged teacher labels were returned** for an index the provenance does not name. |
| `…::teacher_labels_are_those_of_the_callers_verified_index` | Labels were returned for an index other than the caller's verified one. |

**Fixes.** Each receiver reads once and keeps its error wording.
- `native_genome::artifact_closure` (was `:484-490`): a file within the 16 MiB parse bound
  is read once, hashed, and its nested references are taken from those bytes. Larger files
  are hashed and never parsed, as before.
- `native_genome::prepared_closure` (was `:493-504`):
  - `prepared.json` is read once against the candidate's bound digest, and the storage
    source comes from those bytes;
  - the walk's recorded digest for `prepared.json` must be the bound one;
  - failures read "native genome input changed".
- `teaching_store::verify_shard_metadata` (was `:103-106`): the binding is compared with
  the parse of the `read_verified` bytes ("shard manifest changed…").
- `teaching_store::teacher_programs` (was `:1243-1248`):
  - `index.json` is read once (`read_hashed`); the provenance check and the label sources
    use those bytes;
  - its records identity must equal the caller's verified `Index` ("label index changed").

**Not an R8 site: `load_impl` `:1142-1145`, `provenance.json`.**
- There is no hash-then-read. `provenance.json` is read once, and only its
  `index_sha256` back-reference is consumed. It is compared with the digest of the single
  index read, so no swap window exists and there is nothing to reproduce. The earlier
  record grouped it wrongly.
- The real property is of a different class and is recorded, not changed: `provenance.json`
  is bound by nothing but its own back-reference (the index cannot carry its digest).
  `teacher_programs`' legacy (`parallel.teaching_corpus.v1`) branch takes the expected
  `labels_sha256` from it. A legacy corpus's labels are therefore authenticated only by
  an unbound file.

**Results.**
- Post-fix `w1b-r7-postfix-20260926T113144Z`: 42/42. It covers the new tests plus
  `admission_tests`, `teaching_store`, `native_genome`, the prepare byte tests,
  `corpus_incremental` and `frozen_inputs`. The one warning was the now-unused
  `read_json` import in `native_genome`, removed before the pre-commit.
- Pre-commit `w1b-c10-r7-*`:
  - engine: the W1 set, the fence suites and the teacher suites, 288/288;
  - jope-datasets 69/69;
  - workspace all-targets check, parity-matrix, coverage (0 unassigned, 260 targets),
    verify-test-inventory, fmt and diff-check all exit 0.
- jope-learning is not touched. `native_loop/materialize.rs` and the request schema are
  not touched.

**Still open (recorded).**
- The pinned `eval_registry` sites and `native_scq*`.
- The unbound legacy `provenance.json` above.
- `vocabulary.json` and `groups.bin`: `load_impl` hashes them but does not consume them.
  Readers that open the corpus vocabulary by path after `load` are outside this lane or
  neural-gated, and were not examined for this pattern: `pipeline.rs:457`,
  `monitoring.rs:52` and `native_loop.rs:1130`.

## Review of #111 — the outer release authority reads each bound object once (W1-T04)

The independent review (neverhumanbot, CHANGES_REQUESTED at `0b303b293`) found finding
[P1] at `gp/outer_release.rs:284-289`. `grant_body()` hashed `prepared.json` against the
sealed selection, then reopened it with `read_json`. A preparation substituted between
the two reads supplied the grant's `original_test_sha256`, `clean_evaluation` and
dataset/family metadata, while the grant still recorded the original preparation's
digest. The scorer then checked the named label file against that substituted digest.

The rest of `outer_release.rs` and `outer_release_cli.rs` was then checked for the same
pattern. Branch `rr/w1-review-fix`, base `0b303b293`.

**Reproduction.** `w1b-review-outer-pre-20260926T143705Z`: 6/6 FAIL.

The tests are in `gp::fit_contract_tests`, over the fixture fit authority's exported
selected subject. Test-only instrumentation (`manifest::after_verification`) was placed
right after each check, or each authority write, before consumption. Each swap happens
at the last reported verification of its path; an identical dry run counts those
verifications first, and doubles as the positive control.

| Test | Pre-fix outcome |
|---|---|
| `an_outer_grant_takes_its_labels_and_metadata_from_the_verified_preparation` | **[P1]** The grant bound the substituted preparation's label digest (`33c25f…`, another label file) and its metadata. |
| `an_outer_grant_is_bound_to_the_seal_it_checked` | A grant was issued for a handoff whose recorded seal is not the replayed selection: the replay seal was swapped in after the handoff digest check. |
| `an_outer_grant_is_for_the_predictor_it_verified` | A grant was issued for a subject (declaring another training scope) never verified as the final refit: the verified export was swapped in after the subject check. |
| `an_outer_grant_records_the_handoff_it_checked` | The grant recorded the digest of a handoff other than the one whose content it checked and copied. |
| `an_outer_grant_consumes_the_fit_completion_it_verified` | `lineage::verify_final_refit` hashed the final fit completion, then re-read it; an honest grant was refused on a replacement. |
| `the_release_ledger_binds_the_bytes_the_authority_wrote` | The ledger recorded the digest of a grant reread after writing. A grant replaced after its write, naming another label file, was accepted by `score`, which **scored the other labels** (`33c25f…`). |

**Fix.**
- `grant_body`:
  - the handoff, predictor, seal and `prepared.json` are each read once;
  - each check, each recorded digest and every grant field come from those bytes:
    - the seal through the new `handoff::verify_seal_bytes`;
    - the predictor through the new `lineage::verify_final_refit_bytes`;
    - `prepared.json` through the new `manifest::read_bound`.
- `lineage::verify_final_refit` reads the fit completion once (`read_bound`), and
  `predictor_sha256` is the digest of the verified bytes.
- `grant()` and `score()` record the digest of the exact bytes they write (the grant and
  the withheld result), never a reread.
- The refusal wording is unchanged:
  - "the preparation changed after its sealed selection";
  - "the seal changed after its handoff";
  - "the predictor is not the one its handoff exported";
  - "final refit completion changed";
  - "an outer score needs an exported handoff".
- `manifest::read_bound` is `read_verified` with a JSON-valued binding and the receiver's
  own refusal text.

**Checked, unchanged.**
- `score` already read the grant, the model and the labels once.
- `release` already read the withheld score once.
- `events` only reads the ledger, which is verified as a chain.
- `outer_release_cli` rereads the grant only to print its `grant_id`. That is display;
  the authority decides nothing from it.

**Results.**
- Post-fix `w1b-review-outer-post-20260926T143909Z`: 22/22. It covers the six new tests,
  `outer_release_cli`, `projected_export` and the export, handoff and lineage tests. No
  warning is reported in a changed file.
- Pre-commit `w1b-review-outer-c-*`:
  - engine: the W1 set, the fence suites, `outer_release_cli`, `projected_export` and
    `teacher_counterfactuals`, 171/171;
  - jope-datasets 69/69;
  - workspace all-targets check, parity-matrix, coverage (0 unassigned, 261 targets),
    verify-test-inventory, fmt and diff-check all exit 0.
- The pinned `eval_registry` files, jope-learning, `native_loop/materialize.rs` and the
  request schema are not touched.

**Different class, recorded, not changed.** `lineage::verify_final_refit` binds the
exported predictor's recipe, training scope and fit-completion reference. It does not
bind the predictor's own `model` parameters to the verified completion's `model_sha256`.
A predictor with the same recipe and references but other model parameters would pass
the refit check. This is not reproduced here; it needs its own owner decision.

### Second commit — the exported predictor's model parameters are bound to the verified final refit

The gap recorded above ("Different class") is closed on the same branch, after `74ffdf9df`.

**Finding.**
- `lineage::verify_final_refit` bound the exported predictor's recipe, training scope and
  fit-completion reference, and checked that completion against the selected fold fit.
- It never bound the predictor's own `model` member to that completion's `model_sha256`.
  A predictor with the same recipe and references but other model parameters passed the
  check, and `grant_body` issued a grant for it.
- `grant_body` also copied the grant's `subject.model_sha256` from the handoff's
  unverified `final_refit` record rather than from the refit it had just verified.

**Reproduction.** `w1b-review-refit-pre-20260926T144858Z`: 2/2 FAIL. The positive
controls held in both tests before the negative assertions failed.

| Test | Pre-fix outcome |
|---|---|
| `an_exported_predictors_model_parameters_are_those_of_its_verified_final_refit` | One model parameter was changed; the recipe, training scope and completion reference were not, and the handoff recorded the changed predictor. `verify_final_refit` returned `Ok` with the honest `model_sha256` (`236ff2…`), and **a grant was issued** naming that `model_sha256` for the changed predictor. |
| `an_outer_grant_names_the_model_its_final_refit_verified` | A handoff misrecorded `final_refit.model_sha256` as `eee…e`. The grant named `eee…e`, not the verified `236ff2…`. |

**Definition used.** The fit authority records a successful fit's model as follows
(`eval_registry::service`, pinned, unchanged):
- it parses the worker output as a JSON value;
- it takes the value's `model` member and encodes it with `serde_json::to_vec`;
- it stores those bytes, and `model_sha256` is their SHA-256.

That is a canonical encoding. The check therefore applies exactly that definition to the
exact verified predictor bytes: parse, take `model`, `serde_json::to_vec`, SHA-256. It does
not re-serialize the typed `Model`. The positive control shows that the honest predictor's
`model`, encoded this way, is byte-identical to the retained model object in the
authority's store.

**Fix.**
- `lineage::verify_final_refit_bytes` checks the predictor's model parameters. This runs
  after the fold-fit check, so the existing refusals and their order are unchanged. The
  model digest must equal the verified completion's `model_sha256`. Otherwise the check
  refuses with "final refit model parameters are not the verified completion's model"
  (new; no refusal existed for this).
- `handoff::export_sealed` and its resume path share this check, through
  `verify_final_refit`.
- `grant_body` records `subject.model_sha256` from the verified refit, never from the
  handoff.
- Every existing error message is unchanged.

**Controls.**
- Positive: the honest final refit passes `verify_final_refit`, and its model is
  byte-identical to the retained model object. A grant is issued, and it names the
  verified model identity.
- Negative: one model parameter is changed; the recipe, training scope and completion
  reference are not. The handoff records the changed predictor. `verify_final_refit`
  refuses it, the grant is refused, and no grant file is written.

**Results.**
- Post-fix `w1b-review-refit-post-20260926T144954Z`: 18/18. It covers the two new tests,
  the outer-release swap tests of `74ffdf9df`, `outer_release_cli`, the export and
  handoff tests, and the lineage unit tests. No warning is reported in a changed file.
- Pre-commit `w1b-review-refit-c-*`:
  - engine: the W1 set, the fence suites, `outer_release_cli`, `projected_export` and
    `teacher_counterfactuals`, 173/173. This includes the existing final-refit refusals
    in `generic_bootstrap_candidates_retain_their_original_proposal_and_one_adaptation`;
  - jope-datasets 69/69;
  - workspace all-targets check, parity-matrix, coverage (0 unassigned, 261 targets),
    verify-test-inventory, fmt and diff-check all exit 0.
- Not touched: the pinned `eval_registry` files (including the `service.rs` definition
  used above), `fit_boundary`, jope-learning, `native_loop/materialize.rs` and the
  request schema.

## Round 8 — R2 at the public `fit-view` verb and the legacy isolated export (branch `rr2/integration-c`, base `e4f1b6c62`)

Evidence: `runs/w1c-r2-*` under the claude-w0-w12 custody root.

**Defect.** Two fitted-view entry points still resolved a pointer view through
`prepare::load_view`, whose ancestor walk picks the nearest directory holding `manifests/`
and `Cargo.toml`:
- the public `fit-view` verb (`gp::fit_view`), which ignored the CLI's `--root`;
- the legacy `gp::fit_view_isolated` wrapper, whose parent projected through that walk
  before handing the worker its sealed memfd.

**Reproduction.** `w1c-r2-pre-20260926T193941Z`: 3/3 FAIL (the fourth test in the run,
`shipped_fit_memory_is_fenced…`, is an unchanged control and passed). The setup is the one
of round 4's 2c: the declared root carries no workspace markers, an ancestor is a workspace
holding a byte-identical backing table, and the root's own backing is moved away.

| Test | Pre-fix outcome |
|---|---|
| `gp::fit_contract_tests::the_public_fit_view_projects_under_its_declared_root_and_never_resolves_an_ancestor` | `fit-view --root <declared>` **fitted** the ancestor's copy. |
| `gp::fit_contract_tests::the_legacy_isolated_export_never_resolves_a_pointer_view_from_its_ancestors` | The root-less legacy export **exported** the pointer view from the ancestor's copy. |
| `projected_export::shipped_fit_view_projects_under_its_declared_root_before_the_fence` | The shipped binary projected the ancestor's copy and reached the activation fence (`UNVERIFIED_ACTIVATION`) instead of refusing at the projection. |

**Fix.**
- `gp::fit_view_in_root` (new) is the public verb: the cut's fitting rows are projected once
  under the declared root, from the preparation it read (`prepare::load_bound_view_in_root`,
  so `prepared.json` is not reopened), and only that projection and the verified preparation
  state reach `fit_projected`. `main.rs` passes the CLI's `--root`.
- Without a root (`gp::fit_view`, `gp::fit_view_isolated`) only a contained view is
  projected: the new `prepare::load_bound_contained_view` resolves the cut's own tables and
  refuses a pointer view ("pointer view needs its declared data root: its backing table is
  never resolved from the view's ancestors").
- The isolated export projects from the preparation bytes it already read (was a second
  read of `prepared.json`). Its receipt's `view_resolution` is `declared_root` or
  `contained_view`; the value `legacy_view_resolution` (the ancestor walk) is no longer
  produced.
- `examples/champion_export.rs` passes its declared `--root`.

**Callers updated (assertions unchanged).** The fence suite `view_contracts` projects its
pointer-view fixture under the fixture's root: `unbound_fitted_view_export_is_fenced_before_output_and_receipted`
calls the `_in_root` variants and `configured_authority_refuses_fit_view_without_verified_activation`
passes `--root`. `fit_boundary`'s fixture is a contained view and is unchanged. In
`gp::fit_contract_tests`, three positive tests of pointer views pass `--root`.

**Controls.**
- Negative: the backing table only outside the declared root is refused before any fit
  start (fixture authority and shipped binary); the root-less pointer export leaves a
  failed receipt with no worker argv and `view_resolution: contained_view`.
- Positive: the backing inside the declared root fits (fixture authority) or reaches only
  the activation fence (shipped); the pointer view under its root and a contained view with
  no root both export through `sealed_memory_projection`.

**Results.**
- Post-fix `w1c-r2-post-20260926T194142Z`: 41/41 (the four tests, the three updated
  positives, the 2c test, and the `view_contracts` and `fit_boundary` suites).
- Pre-commit `w1c-r2-c-20260926T194403Z`: fence suites plus `gp::fit_contract_tests` and
  `projected_export`, 84/84; workspace all-targets check, coverage (0 unassigned, 261
  targets), verify-test-inventory, fmt and diff-check exit 0; no warning in a changed file.

**Still open (recorded).** The pinned `eval_registry/native_scq.rs:451` still calls the
root-less `load_view` (pinned, not touched). The other `load_view` callers (neural-gated
`encoding.rs`, `components.rs`, `hyperion.rs`, `hyperion-batch-worker`, examples and test
fixtures) are unchanged; `load_view` itself keeps its behavior.

## Round 8 — `vocabulary.json` readers after `load` (same branch)

Evidence: `runs/w1c-vocab-*`.

**Defect.** Readers that hold a corpus index reopened `vocabulary.json` by path and parsed it
with no binding:
- `native_loop::candidates` (was `:1130-1140`): `teaching_store::read_index`, then
  `read_json(vocabulary.json)`. For a sharded store, `load` had hashed the file by path first
  (hash, then reread); for a legacy `parallel.teaching_corpus.v1` index, `load` never opens
  the vocabulary at all, so any vocabulary was consumed.
- `pipeline::run` (was `:457`, neural): `read_json(vocabulary.json)` after an unhashed index
  read.
- `monitoring::evaluate` (was `:52`, neural): the vocabulary parsed before
  `corpus_publication::verify` hashed the file by path; the index likewise parsed, then hashed
  separately twice.
- Same class, found while closing these (neural): `proposals::rank_impl` hashed the vocabulary
  against the checkpoint's `vocabulary_sha256`, then reread it; `ranked_proposals` (two sites)
  parsed it, then hashed it separately.

**Step 0.** The native loop's index and vocabulary reads moved, unchanged, into
`teaching_store::read_index_and_vocabulary`, so the receiver is testable without a native-graph
fixture. For the reproduction only, test-only instrumentation (`manifest::after_verification`)
sat between `read_index` and the vocabulary read, as in rounds 6 and 7.

**Reproduction.** `w1c-vocab-pre-20260926T194802Z`: 2/2 FAIL.

| Test | Pre-fix outcome |
|---|---|
| `teaching_store::read_once_tests::the_corpus_vocabulary_is_the_bytes_its_index_binds_even_if_replaced_after_its_check` | A sharded store's vocabulary replaced after its check (two tokens swapped) was **consumed**. |
| `…::a_legacy_corpus_vocabulary_other_than_its_index_binding_is_refused` | A legacy index's substituted vocabulary was **consumed** (no check at all). |

**Fix.** New `corpus::read_bound_vocabulary(corpus, vocabulary_sha256)`: `read_verified`
against the digest the index (or checkpoint) binds, then parse those bytes ("corpus vocabulary
changed: its bytes differ from the bound identity").
- `teaching_store::read_index_and_vocabulary` (the native loop), `pipeline::run` and
  `monitoring::evaluate` use it with the index's `vocabulary_sha256`; `proposals::rank_impl` with
  the checkpoint's.
- `monitoring::evaluate` reads `index.json` once (`read_hashed`); the panel's `corpus_sha256`
  check and the recorded binding use that digest.
- `ranked_proposals` (both sites) reads the vocabulary once (`read_hashed`); its recorded
  `vocab_sha` is the digest of the parsed bytes.

**Controls.** Positive: the honest vocabulary is returned for both a sharded store and a
legacy index. Negative: a replacement after the check is never consumed; a legacy index's
substituted vocabulary is refused.

**Results.**
- Post-fix `w1c-vocab-post-20260926T194935Z`: 54/54 (the two tests plus every
  `teaching_store`, `native_genome`, `corpus` and `native_loop` unit test and
  `corpus_incremental`).
- `w1c-vocab-neural-check-20260926T195150Z`: `cargo check -p jope-parallel --features neural
  --all-targets` (CPU, `neural-env.sh`), exit 0. `pipeline`, `monitoring`, `proposals` and
  `ranked_proposals` are compile-checked only: they need a GPU grant to run.
- Pre-commit `w1c-vocab-c-20260926T195243Z`: fence suites, `gp::fit_contract_tests`, `teaching_store` and `corpus` unit tests, 98/98; workspace all-targets check, coverage (0 unassigned, 261 targets), verify-test-inventory, fmt and diff-check exit 0.

## Round 8 — the legacy `provenance.json` is no longer trusted for labels (same branch)

Evidence: `runs/w1c-provenance-*`.

**Defect (recorded in round 7, not changed there).** A legacy (`parallel.teaching_corpus.v1`)
corpus's `provenance.json` is bound only by its own back-reference to the index (the index
cannot carry the provenance's digest). `teaching_store::teacher_programs`' legacy branch took
the expected `labels_sha256` from that file, so a legacy corpus's teacher labels were
authenticated only by an unbound file.

**Reproduction.** `w1c-provenance-pre-20260926T195610Z`: 2/2 FAIL. Both tests export a published
store to the legacy layout (`teaching_store::export`).

| Test | Pre-fix outcome |
|---|---|
| `teaching_store::read_once_tests::legacy_teacher_labels_are_authenticated_by_the_verified_index_never_by_provenance` | Forged `labels.jsonl` plus a `provenance.json` naming its digest (back-reference intact): the **forged programs were returned**. |
| `…::legacy_teacher_labels_without_an_index_binding_are_unbound_and_refused` | A legacy index with no labels binding: its labels were **trusted** on the provenance's word. |

**Fix.** The binding is a digest the index records (the index is what callers verify: the
checkpoint's `corpus_sha256`, `read_index`, `load_resident_verified`).
- `corpus::Index` gains `labels_sha256: Option<String>` (`serde(default)`, omitted when absent,
  so every index written before this change parses and re-serializes to the same bytes).
- Writers that produce a legacy index with labels bind them: `corpus::build_observed*` and
  `teaching_store::export`. Sharded stores bind labels per shard (`None` here).
- `teacher_programs`' legacy branch requires the index's binding (and that the bytes it read
  bind the same as the caller's index). Without one it refuses: "legacy teaching corpus labels
  are unbound: the index binds no labels digest and provenance.json is bound only by its
  back-reference to the index". A provenance naming another digest is refused ("teacher labels
  changed"). The provenance is still read, and its back-reference still checked.
- Legacy meaning is kept where the binding is absent: such a corpus still opens (`read_index`,
  `load`, training, projection); only its labels are refused as teacher labels.
- `corpus_publication::verify` requires a bound index's digest to agree with its provenance.
  `resident_protocol` stages `labels.jsonl` under the index's binding when there is one, and
  under the provenance's for an unbound legacy index (whose labels `teacher_programs` refuses).

**Deliberately unchanged.** `continuation_corpus::render` still writes an index without the
binding: its rendering is re-derived byte for byte by native Graph verification, so binding
there would make every existing continuation corpus fail verification. Its labels stay unbound
for `teacher_programs`; continuation corpora are training inputs, and no product path reads
their teacher programs. Every struct-literal `Index` in tests and `auxiliary_training` gains
`labels_sha256: None` (unchanged bytes).

**Controls.** Positive: an exported (bound) legacy corpus returns its honest programs, and
`corpus_incremental::sharded_export_matches_reference_and_only_changed_groups_are_rebuilt` still
finds the built reference and the export byte-identical, with equal teacher programs.
Negative: forged labels named by the provenance, and an unbound index, are refused.

**Results.**
- `w1c-provenance-neural-check-20260926T200058Z`: `cargo check -p jope-parallel --features neural --all-targets`, exit 0 (the
  neural test literals are compile-checked only).
- Post-fix and pre-commit `w1c-provenance-c-20260926T200127Z`: the two tests, the fence suites, `gp::fit_contract_tests`, every `teaching_store`, `corpus*` and `campaign` unit test, and `corpus_incremental`, `native_loop_continuation`, `corpus_projection` and `resident_protocol`: 157/157. Workspace all-targets check, coverage (0 unassigned, 261 targets), verify-test-inventory, fmt and diff-check exit 0. (A first attempt, `w1c-provenance-c-20260926T195908Z`, stopped at compilation: two `cfg(test)` literals in `campaign.rs` lacked the new field.)

## Round 8 — W1-T01: the ASM development loader and the context-producer input consume the frozen release; W1-T07: the planner requires the producer's corrected-run attestation (same branch)

Evidence: `runs/w1c-t01-t07-*`.

### W1-T01

**Defect.** Nothing consumed the committed ten-regression release
(`crates/jope-datasets/releases/regression-10-v1/`, `release_id 859ce689…`). The ASM
development loader (`jope_pipeline::prepared_data::{load_development, parse_development}`,
re-exported to jope-bench) checked a partition only against the `PanelDataset` its caller
handed it. An ad-hoc partition with a self-consistent panel entry under a release member's
name was admitted as that dataset. The context-producer input (`optimize::build_pair_input`)
and the inference planner's input admission (`inference_inputs::load`) took their identities
from the panel entry alone.

**Reproduction.** `w1c-t01-t07-pre-20260926T200458Z`:
`jope-pipeline prepared_data::tests::a_release_members_name_never_admits_other_development_bytes`
FAIL: a synthetic regression partition renamed `197_cpu_act`, with its panel entry's digest
set to the synthetic bytes, was **admitted**.

**Fix.**
- `jope_datasets::regression_release::committed()` (new): the committed release embedded at
  build time (`include_bytes!`), opened with every identity check, pinned to
  `COMMITTED_RELEASE_ID`, and required to bind the committed family registry. Receivers cannot
  be pointed at another release document.
- `parse_development` (and so `load_development` and every caller: jope-bench campaign,
  origin-proof, demo, teaching, fitted export, CLI input verification; jope-cli train/corpus;
  jope-learning eligibility) admits a dataset the release names only as that member, before any
  row is interpreted: task, raw source, split, development and sealed digests, row and fold
  counts, group columns and feature schema must be the member's
  ("<id>: panel entry differs from its member of regression release <id>"), and
  `RegressionRelease::verify_development` must accept the bytes (exact digest and row/target
  alignment). `DevelopmentData.release` records the admission (`release_id`, source and
  development digests, alignment digest); it is `None` for a dataset outside the release
  (classification members of the panel, fixtures).
- `build_pair_input` takes the context-producer input's `dataset_sha256`/`prepared_sha256`
  from that admission for a release member (the values equal the panel's; they now come from
  the verified release).
- `inference_inputs::load` requires the fitting inputs' panel to be the release's panel, the
  input inventory to be the release's members, and each projected context-producer input and
  its split membership to name the member's identities.

**Scope and limits.** The panel and the release agree for every member (the local-evidence
test checks all ten against the real partitions), so no retained valid input is newly refused.
The release is a v1 release: it binds whole-partition alignment, not a per-cut fitting-row
alignment digest, so route R12 (the context producer's self-certified training identity) still
needs a release v2 and the W2 `TrainingIdentity` schema. The `TrainingInput` schema is
unchanged (no release id in the producer input).

### W1-T07

**Defect.** The v3 planner admitted any fitting-input closure outside the historical staging
path. It never required the producer's projection attestation (`verify_corrected_run`), and no
component could emit one.

**Reproduction.** Same run: `jope-bench panel::inference_benchmark::inputs::tests::the_planner_requires_the_producers_corrected_run_attestation`
FAIL: a fresh closure was **admitted** with no attestation.

**Fix.**
- Producer side: `staging_disposition::attest_corrected_run(protocol, manifest_bytes,
  attestation, window)` (new) checks the closure against the protocol, then emits the
  `jope.fitting-only-staging-run/v1` document for exactly those bytes and that manifest's
  projection source. New verb `jope-datasets staging-disposition attest --closure <manifest>
  --output <run.json> [--window <sha256>]` emits it under the committed protocol, declaring a
  fitting-only projection, and writes it once. No Rust component in this tree writes a
  fitting-input manifest today (the historical v2 planner that did is gone), so the verb is the
  emission point for whichever producer projects the inputs. The declaration is recorded and
  required, not independently proven.
- Planner side: `jope benchmark inference-plan` takes `--corrected-run` and
  `--corrected-run-sha256` (required). `admit_closure` still runs `verify_closure` first (the
  historical refusals and their wording are unchanged), then requires the pinned run and
  `verify_corrected_run` for exactly the manifest bytes it consumed; the receipt records the run
  decision under `staging_closure.corrected_run`. The plan pins the run (`Plan.corrected_run`,
  serialized only when present), and plan admission (`read`: run, cold, v3 report) refuses a plan
  without one.

**Controls.** Negative: no run; a run for other manifest bytes; a run declaring parsed outer
labels; a pinned run replaced after pinning; a historical closure cannot be attested (library
and CLI). Positive: the producer's attestation of exactly the closure is admitted, and
`staging-disposition verify --run` accepts the CLI's run.

**Limits.** A corrected run still has no execution window (W11 admission) and no real
projected closure exists; plans built before this change carry no attestation and are refused
by plan admission.

### Runs

- Post-fix `w1c-t01-t07-post-20260926T200807Z`: jope-datasets 72/72 (with `the_embedded_release_is_the_committed_document`, `the_producer_attests_exactly_its_fitting_only_closure` and `the_attest_verb_emits_a_run_the_verifier_accepts`); jope-pipeline 84/84; the local-evidence test `the_real_development_partitions_are_admitted_as_their_release_members` 1/1 on the prepared panel (`~/.cache/jope/asm-private-30-v1`); jope-bench (asm) 146/146; jope-learning 152/152; the full-default workspace check (`jope-cli/asm,jope-cli/qualified-training-context`) exit 0.
- Pre-commit `w1c-t01-t07-c-20260926T201132Z`: fence suites and `gp::fit_contract_tests` 82/82; workspace all-targets check, parity-matrix, coverage (0 unassigned, 261 targets), verify-test-inventory, fmt and diff-check exit 0. The capability matrix's jope-datasets row names the new `staging-disposition attest` verb (a binary-level row; counts unchanged).
