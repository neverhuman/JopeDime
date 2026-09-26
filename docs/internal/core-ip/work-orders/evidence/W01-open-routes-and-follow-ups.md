# W1 round 3 — open consumer-map routes, outer-release CLI, projected export input, planner closure check, harness extension

Branch `rr/w1-inputs-3`, base `248ded85f` (`feat/regression-remediation`). Owner: Claude
implementation subagent, lane `.agent/worktrees/demo`. Inputs:
- the committed [consumer map](W01-consumer-map.md);
- its appended route status;
- [W01-label-release-counterfactuals-staging.md](W01-label-release-counterfactuals-staging.md).

Every state below is fixture scope, local only and not CI-verified. No register entry is
closed.

Evidence lives under
`~/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/w1b-r3-*` and
`w1b-c<N>-*`. Each directory holds `HEAD`, `diff.patch`, `untracked.patch`, `status.txt`,
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
