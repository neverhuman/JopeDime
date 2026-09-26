# W1-T02, W1-T03 and W1-T05 (fixture scope) — release, preparation semantics, staging

Branch `rr/w1-inputs`, base `7216cad21`. Owner: Claude implementation subagent, lane
`.agent/worktrees/honesty`. The route inventory these tasks rely on is
[W01-consumer-map.md](W01-consumer-map.md).

Observed states, each scoped:

| Task | State | Not claimed |
|---|---|---|
| W1-T02 release and registry | implemented; fixture-proven locally; the real release was derived and re-verified from custody bytes | No receiver consumes the release yet; not CI-verified |
| W1-T03 preparation semantics | reproduced (11 failing fixtures); implemented; fixture-proven locally | Only the jope-engine pointer-view receiver and producer are covered |
| W1-T05 staging closure | one gap reproduced (stale same-size cache entry served); implemented; fixture-proven locally | No live worker isolation, credential removal or network denial (W7) |

Evidence lives outside the tree, under
`~/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`.
Each `w1-*` directory holds:
- `HEAD`, `diff.patch`, `status.txt`;
- `source.txt` (argv, toolchain, environment, exit code);
- `nextest.log` and `summary.txt`.

## W1-T02 — Immutable ten-regression release and family registry

**Contract.** `crates/jope-datasets/src/regression_release.rs`:
- `RegressionRelease` (`open` `:307`, `validate` `:334`, `verify_staged` `:413`,
  `verify_development` `:427`);
- `derive` `:932`, `check_against_panel` `:1024`;
- `load_staged` `:1128` with `read_exact_object` `:1075`;
- `FamilyRegistry` (`validate` `:1315`, `never_seen_claim_blockers` `:1409`).

**Canonical encoding** (`canonical_json` `:60`):
- object keys in byte order, compact, one trailing newline;
- strings via JSON escaping; integers only (a float is an error).

Schemas are exact strings: `jope.regression-release/v1` and
`jope.regression-family-registry/v1`. `open` enforces four things:
- every struct is `deny_unknown_fields`;
- the schema string must match exactly;
- the input bytes must equal the canonical re-encoding, which rejects pretty-printed
  input, duplicate keys and a missing final newline;
- `release_id` must equal sha256 of the canonical body without `release_id`.

The registry's sha256 is bound into the release.

**What a member binds:**

| Field | Binding |
|---|---|
| `raw` | Acquired object sha256 and size, source metadata sha256. URLs are provenance only and never fetched. |
| `target` | Column `target`, kind `original_numeric_column`, original label roles `development` / `sealed_final` |
| `schema` | Ordered features (name, kind) plus a canonical digest; the target may not be a feature |
| `rows` | Total, development and sealed counts; digest of ordered row ids (source order, both partitions); digest of development `[id, source_index, target]` (row/target alignment) |
| `groups` | Declared group columns; digest of `[id, group]` |
| `folds` | Count, per-fold rows, digest of `[id, fold or null]` |
| `prepared` | Split, development and sealed partition sha256; sealed is hashed, never parsed |
| release level | Panel sha256 and census; membership rule; preparation protocol, seed, folds; `benchmarks/prepare.py` sha256; twenty exclusions with reasons |

**Registry rules** (`FamilyRegistry::validate`):
- Every member has exactly one family and exactly one exposure record.
- An alias or source key cannot belong to two families.
- A **resolved** relation of any kind (alias, alternate target, resampled, content
  duplicate, shared entity, same source, derived variant) must stay inside its family's
  exclusion unit.
- Exposure is recorded separately for catalog availability, encoder, teacher and checkpoint.
  - `ExposureState::{Unknown, Unseen, Exposed}` are distinct variants.
  - `unseen` and `exposed` require evidence; missing evidence means `unknown`.
- A never-seen-family claim is blocked unless all three components are `unseen` and no
  unresolved relation touches the family.

**Real derivation.** The committed artifacts are
`crates/jope-datasets/releases/regression-10-v1/{release.json,family-registry.json}`.
Argv, from the lane root:

```sh
target/debug/jope-datasets regression-release derive \
  --panel benchmarks/panel.json --preparation-source benchmarks/prepare.py \
  --registry <run>/family-registry.json \
  --staged-root /home/ubuntu/.cache/jope/asm-private-30-v1 --output <run>/release.json
target/debug/jope-datasets regression-release verify --release <run>/release.json \
  --registry <run>/family-registry.json --panel benchmarks/panel.json \
  --preparation-source benchmarks/prepare.py \
  --staged-root /home/ubuntu/.cache/jope/asm-private-30-v1
```

Results:
- `release_id` `859ce6895f6ca2b016103465f44e4f84a3dd54dcce8b2c832ba15ec536454d4a`.
- Release bytes sha256 `34c94de630c32177794bf441d3ed22f4887bd972a63c34000b387772a8e5c4e3`.
- Registry sha256 `757ee4d20089a4cbd34d650904fcc8d9e05b55bf6c0d491353c478f680568e70`.
- `verify` reported `members_verified: 10`, `rederived_bytes_equal: true`,
  `sealed_partitions_parsed: false` and `network_access: false`, with 3–6 never-seen
  blockers per member.
- Receipt: `runs/w1-release-derive-20260926T054743Z/`. The pre-commit source is
  `7216cad21` plus the recorded `diff.patch`.
- Re-verification of the committed artifacts at head `5b7e591c1` is in
  `runs/w1-release-verify-20260926T060255Z/`. The release module is unchanged since
  `3714257c4`. It reported the same `release_id` and `rederived_bytes_equal: true`.

**Acceptance tests.** `cargo nextest run -p jope-datasets --test regression_release`
runs eight tests:
- `derived_release_reopens_to_identical_bytes_and_identity`: exact bytes, deterministic
  re-derivation, file round trip, panel order, census, exclusions.
- `changed_source_target_rows_schema_groups_folds_or_family_mapping_is_a_new_identity_or_rejected`.
  For each of 12 input mutations it checks three things:
  - the stale panel declaration is refused;
  - the old release rejects the changed bytes;
  - the consistently re-declared input yields a new `release_id`.

  It also covers:
  - group-column declaration → new identity;
  - split rows out of source order → rejected;
  - registry relation, exposure or source-key changes → new identity;
  - preparation source change → new identity;
  - missing, substituted or surplus objects → rejected.

  Decisions are written to `mutation-decisions.json` in the run directory.
- `development_receiver_rejects_changed_bytes_and_realigned_labels`.
- `unknown_fields_versions_and_noncanonical_encodings_are_rejected`: top-level and nested
  unknown fields, v2 schema, forged identity, census inconsistency, target alias,
  label-role change, float, pretty-printed input, missing newline, duplicate key; the
  registry equivalents.
- `membership_is_every_regression_entry_of_the_frozen_thirty_entry_panel`: nine
  regression entries, duplicate id, 29 entries, unknown panel field, other raw version,
  mixed protocols.
- `family_registry_keeps_unknown_distinct_from_unseen_and_blocks_never_seen_claims`: the
  positive control permits the claim only when fully `unseen` with evidence.
- `staged_loader_reads_only_exact_regular_objects_without_fallback`: symlinked leaf,
  symlinked directory, `latest`-named copy, different version under the exact name.
- `committed_ten_regression_release_matches_the_frozen_panel`: the committed bytes
  reopen exactly, the registry is bound, the frozen panel and `prepare.py` match, all
  never-seen claims are blocked.

This is a new contract, so there was no failing pre-fix receiver to reproduce. The
tests are its acceptance and regression protection.

**Limitations:**
- No receiver consumes the release yet: the jope-pipeline development loader and the
  context producer are outside this lane's partition.
- The raw→prepared derivation (`prepare.py` under sklearn 1.7.2) was not re-executed.
  It is bound by source sha256 and protocol string, not by a Python runtime identity.
- Historical exposure stays `unknown` and cannot be upgraded by this release.

## W1-T03 — Preparation semantics at the receiving path

Declared semantics for jope-engine pointer views, as enforced at `prepare.rs` *(post)*:

- **Prediction-time features.**
  - Exactly `ViewSource.feature_columns`: distinct, non-empty, target excluded
    (`validate_view_source` `:1172`).
  - Each column name must occur exactly once in the backing header
    (`load_raw_view_in_root` `:1365`).
- **Join keys and cardinality.**
  - Cut index → `original_training_row_map` → backing row.
  - Each membership list is strictly increasing: each original row once, in original order.
  - Validation is disjoint from training by index and by backing-row pointer
    (`check_cut_membership` `:1231`).
- **Target aliases.** A feature that equals the target on every fitting row is refused
  when the target has at least three distinct values (`reject_target_copies` `:1198`).
- **Learned state is bound to fitting membership.** The stored `preprocessing.json` must
  equal `Preprocessing::fit` on exactly the cut's training rows
  (`ensure_fitted_on_membership` `:1268`). This rejects:
  - recoded or reordered categories;
  - imputation or scale taken from held rows;
  - state fitted on training plus validation rows.
- **Missing values (unchanged, declared).**
  - Numeric: `missing()` tokens and non-finite values are imputed with the training median.
  - Categorical: missing tokens are excluded from categories; an unseen category encodes
    as −1.
  - Held-side-only category or missingness changes do not alter training preparation
    (positive control).
- **Groups (unchanged).** `folds` keeps entity groups and duplicate feature rows inside
  one fold.

Where the checks run:
- The receiver `load_view_with_root` `:1301` runs them for both partitions before
  preparation state is applied or any row is returned.
- The producers run the same checks before writing: `run_with_admission` `:1044-1072`,
  `expand_cuts` `:1442-1477` and `index_existing` `:1552`.

**Tests** (`cargo nextest run -p jope-parallel --test preparation_semantics`). There are
11 reproductions. Before the fix each *accepted* the bad view; after the fix each is
refused:
- `target_column_declared_as_a_feature_is_rejected`
- `renamed_exact_target_copy_is_rejected`
- `ambiguous_backing_header_is_rejected`
- `duplicated_feature_names_are_rejected`
- `duplicate_expanding_training_membership_is_rejected`
- `reordered_training_membership_is_rejected`
- `validation_rows_inside_training_are_rejected`
- `one_backing_row_on_both_sides_of_a_cut_is_rejected`
- `preparation_state_fitted_beyond_training_membership_is_rejected`
- `recoded_categories_are_rejected`
- `producer_refuses_target_feature_before_writing_any_view`. After the fix the producer
  refuses and **no output directory exists**: no authority-bearing state changed.

There are two positive controls, which pass both before and after:
- `valid_pointer_view_round_trips_original_labels`: exact raw label strings, and the
  de-standardized targets equal the originals.
- `heldout_only_category_and_missingness_changes_do_not_alter_training_preparation`
  (S02-I8).

Runs:
- `runs/w1-t03-prefix-20260926T054952Z`: 11 FAIL, 2 PASS. Every failure is
  "accepted … view"; the producer test failed at `unwrap_err` on `Ok`.
- `runs/w1-t03-postfix-20260926T055311Z`: 35/35 pass (preparation_semantics,
  data_contracts, families, sealed_storage, binary_preparation).
- `runs/w1-t03-regression-20260926T055348Z`: `--lib`, view_contracts, gp_contracts,
  gp_admission, fit_boundary, experiment_isolation. There are 83 failures, and
  `fail-set.txt` equals `baseline-fail-set.txt` from `runs/w1-baseline-engine-*`. They
  are the known fenced-fit-authority, view_contracts and environmental
  `eval_registry` lib failures. The view_contracts panics are at the same source lines
  before and after.

**Not covered by T03:**
- Legacy numeric-CSV views (no `storage`) are unchanged.
- The ASM receiver `jope-pipeline/src/prepared_data.rs` already enforces exact bytes,
  declared kinds, ids, groups and folds. It is outside this partition and was not modified.
- The context producer's self-certified identity is W2.
- Held-side numeric cells that do not parse are still imputed silently. That is an
  undeclared coercion and remains open.
- Hash-then-reopen remains. It is mitigated by the final view hash and the refit check.

**Behavior-change risks** (fail-closed; none observed in the fixtures or the regression
comparison):
- An existing on-disk pointer view whose `preprocessing.json` is not byte-equivalent to
  a refit on its training rows is now refused. This covers state produced by older
  `Preprocessing::fit` code and `index_existing` copies of legacy state. It needs a new
  prepared identity, not an override.
- A cut whose training and validation rows resolve to one backing row is now refused.
  Historical reconstruction can map identical source rows to one pointer. Grouped
  regression folds keep those rows together; time-series cuts (deferred) can split them.
- A feature that exactly equals a target with at least three distinct values is refused.
  This includes a genuine identity task (`y = x`), which would need a separately
  declared transformation that does not exist yet.
- A validation-view load now also reads the training rows (one pass over the backing
  table), so validation loads do more I/O.

## W1-T05 (fixture scope) — Staging loader closure

**Gap reproduced.** `BlobDir::ensure` (`crates/jope-datasets/src/blobs.rs`) served any
cached blob of the right *size* without re-hashing it. A stale or corrupted same-size
entry was therefore served:
- online, reported as "cached" and never re-downloaded;
- offline, served with no error.

The `by-id/<hash>/<file>` consumer copy is a hard link to the blob, so it carried the
stale bytes.

**Fix.**
- A cache hit is served only when its bytes re-hash to the named sha256.
- A stale entry is removed. Online, it is re-downloaded through the verified path;
  offline, the request fails ("no substitute is served").
- `Fetcher::fetch` (`cache.rs`) no longer materializes a row with any unverified object.
  It removes that object's consumer copy so it cannot stay servable, and reports the
  error; the CLI then exits non-zero.
- `jope-models` shares `BlobDir::ensure` and gains the same hit verification. Its
  `bundles/` hard links are not cleaned (outside this partition; open).

**Tests** (`cargo nextest run -p jope-datasets --test input_closure`). All go through the
real `Fetcher` after an online cache-miss fetch:
- `stale_same_size_cache_entry_is_replaced_online_and_never_served`: FAIL → PASS.
- `stale_cache_entry_offline_stops_the_request_and_is_not_left_servable`: FAIL → PASS.
- Safeguard controls, passing before and after:
  - `verified_cache_hit_is_served_offline_without_network` (authorized pre-staged
    object, no store);
  - `missing_required_object_offline_stops_the_request`;
  - `corrupt_resume_partial_is_never_promoted` (resume path).

Runs:
- `runs/w1-t05-prefix-20260926T060016Z`: 2 FAIL, 3 PASS.
- `runs/w1-t05-postfix-final-20260926T060134Z`: jope-datasets and jope-models, 64/64 pass.

Related receiver closure already in place:
- The release loader refuses symlink components, absent objects and `latest`-named
  substitutes (`staged_loader_reads_only_exact_regular_objects_without_fallback`).
- The engine view receiver refuses symlinks and alternate roots (`prepare.rs` unit test
  `explicit_backing_root_preserves_original_rows_and_refuses_fallback`).

**Open (W1-T05 proper and W7):**
- Remove broad credentials, home access and network fallback from the managed learning
  worker; live denial receipts come from W7.
- Engine routes R2 (upward root walk), R8 (hash-then-reopen) and R11 (`jope_home` CWD
  fallback) from the consumer map.
- `store_file` still returns early for an existing destination without verifying it
  (seed path).
- Hardlink/replacement race tests beyond the existing `native_input_cache` unit tests.

## Evidence index

| Run directory | Purpose |
|---|---|
| `w1-baseline-engine-20260926T053533Z` | Pre-change: data_contracts, families, sealed_storage, view_contracts, binary_preparation |
| `w1-baseline-engine-consumers-20260926T053714Z` | Pre-change: `--lib`, gp_contracts, gp_admission, fit_boundary, experiment_isolation |
| `w1-baseline-datasets-20260926T053635Z` | Pre-change: jope-datasets, 42/42 |
| `w1-release-derive-20260926T054743Z` | Real release derivation and verification receipts |
| `w1-t02-release-fmt-20260926T055138Z` | T02 tests, 52/52, with mutation decisions |
| `w1-t03-prefix-20260926T054952Z` / `w1-t03-postfix-20260926T055311Z` / `w1-t03-regression-20260926T055348Z` | T03 reproduction, fix and regression comparison |
| `w1-t05-prefix-20260926T060016Z` / `w1-t05-postfix-final-20260926T060134Z` | T05 reproduction and fix |
| `w1-release-verify-20260926T060255Z` | Committed release re-verified from custody at `5b7e591c1` |
| `w1-final-focused-20260926T060300Z` | Final run at `5b7e591c1`: jope-parallel preparation_semantics, data_contracts, families, sealed_storage, binary_preparation (35 pass, 1 skipped as at baseline) |
| `w1-final-datasets-20260926T060356Z` | Final run at `5b7e591c1`: jope-datasets and jope-models, 64/64 |

---

## Appended 2026-09-26 — receiving-path extensions from `rr/w1-inputs-2` (W1-T06)

*Appended section.* The T03 record above is unchanged. The W1-T06 counterfactual harness
(`crates/jope-engine/tests/counterfactual_boundaries.rs`) found three gaps in the receiving
path. Each was reproduced before it was fixed (`w1b-t06-prefix-20260926T092507Z`), and all
three are fixed in commit `bb186aaf8`.

1. **Duplicate join onto one backing row.** Two rows of one cut membership that resolve to
   one backing row were not refused. `check_cut_membership` now refuses them as a
   "duplicate-expanding join", in both the receiver and the producer.
2. **Genuine duplicate rows collapsed.** Equal historical rows were all mapped to the first
   equal backing row. Reconstruction now maps them to distinct equal backing rows in
   source order, so genuine duplicates in the source remain preparable.
3. **Symlinked audited files.** Preparation followed a symlinked `meta.json` or `train.csv`.
   It now refuses them. The audit likewise refuses a symlinked split and records the
   refusal as an issue.

**New behavior-change risk (fail-closed):** an existing pointer view whose membership repeats
a backing pointer is refused at load and needs re-preparation under a new identity. The full
record is [W01-label-release-counterfactuals-staging.md](W01-label-release-counterfactuals-staging.md).
