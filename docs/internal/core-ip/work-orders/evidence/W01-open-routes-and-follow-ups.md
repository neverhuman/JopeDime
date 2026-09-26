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
