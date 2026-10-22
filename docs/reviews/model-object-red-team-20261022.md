# Model object and session red-team review — 2026-10-22

Scope: PR #413's language-neutral object envelope, known fitted-state readers, default
fit persistence, import/export, disk inference, CLI/MCP and session selection. The review
compares the initial implementation at `8f9942171ba73c790066d7814479449fd42cbafa` with the
hardening changes in the same PR. It is an implementing-agent review, not an independent
auditor certification. Hosted CI remains separate evidence and is required before merge.

## Reproduced findings and corrections

The first five adversarial tests were added before their fixes: the focused object suite
ran 12 tests, with 7 passing and **5 failing** on the initial implementation. The same
five tests passed after the fixes. Their IDs remain in the source and test inventory.

| Finding | Failure and correction | Regression proof |
| --- | --- | --- |
| P1: future learned state silently dropped | Serde accepted unknown nested fields in a known decoder. Reload could change semantics while reporting support. Known state now uses `serde_ignored` and refuses any unconsumed field, for both bound objects and native fitted files. | `new_gp_fields_cannot_be_silently_dropped_by_an_older_decoder` adds a required field to a ridge head under the existing decoder ID. |
| P1: cross-parser key disagreement | Repeated keys inside metadata used last-value-wins parsing. A streaming visitor now rejects duplicate decoded keys at every depth, including escaped aliases, before typed decoding. MCP shares this validation. | `duplicate_json_metadata_is_refused_before_cross_language_disagreement`; `escaped_duplicate_keys_nested_arrays_and_trailing_values_are_refused`. |
| P1: invalid known model published before failure | A checksum-valid envelope could contain invalid native state; import wrote it before inspection failed. Save now performs registered decoder/binding validation before creating the root. Opaque unknown codecs remain retainable. | `malformed_known_state_is_refused_before_creating_a_store`. |
| P2: save interruption prevented retry | Existing uncommitted directories were read as complete objects and blocked retries. A publisher now verifies retained declared parts, adds missing ones and atomically commits the manifest. Conflicting state is never overwritten. | `an_interrupted_publication_resumes_without_overwriting_retained_parts`; concurrent publication/conflicting-recovery test. |
| P2: OS-dependent filenames | Windows device names, trailing periods and case aliases could collide or fail on a different filesystem. Rust, JSON Schema and the Node writer now reject them; paths are case-insensitively unique. | `portable_parts_refuse_device_names_trailing_dots_and_case_collisions`. |
| P2: relative executable changed after cwd | The Node object pinned its model path but left `./jopedime` relative, so moving cwd broke reload or could select another binary. Paths containing separators are now resolved at construction. | Node acceptance changes cwd before verification and inference. Bare PATH lookup remains explicit. |
| P2: object `.` failed identity validation | Identity used the lexical path basename. Reader now checks the canonical final directory name after rejecting a direct directory symlink. | Packaged CLI test runs inspection inside the object with `--model .`. |
| P2: state exports inherited broad file permissions | New part/export temporary files used the process umask. They now request mode 0600 on Unix; new object/session directories request 0700. | Unix assertions in concurrent-publication test. Existing files/ancestors are not chmodded. |
| P2: fitted export could normalize bytes | Decode/re-encode could change valid noncanonical body JSON. Native import now retains the original fitted bytes, and fitted export returns those bytes after validation. | Existing packaged exact-byte export assertion, plus the retained-byte APIs. |
| P2: inspection reused the manifest schema name | Consumers could confuse a report with an on-disk manifest. Reports now use `jope.model-object-inspection/v1`; adapter types and docs match. | Packaged inspection and schema checks. |

## Continuing work and rollback

An earlier “best model” must not be overwritten when a later candidate is saved. Immutable
content IDs already retain all candidates. The added `jope.model-session/v1` history gives
those candidates a named sequence and an active selection. Record and rollback append
create-only hash-chained events, requiring the previously observed head. A stale writer
or concurrent publisher cannot overwrite a selection. A rollback itself is a version;
continuing after rollback retains the abandoned candidate and the selection history.

Tests cover A → B → rollback A → C, four simultaneous competing publishers, stale-head
refusal, bad rollback references, path traversal/device names, corrupted history, and CLI
selection. MCP exposes the same shared implementation; notification requests cannot append.
The Node acceptance example compares inference bytes before and after rollback.
See the [session contract](../model-sessions.md) for crash reconciliation, backup closure,
valid-prefix truncation limits and the distinction between inference and training resume.

## Program-state risks checked

- The current 26 native appliance recipes cover regression, binary and multiclass output.
  The existing roster test now exercises envelope save/load in addition to native file
  roundtrip, comparing predictions bit for bit and preserving class order. Kernel/support
  methods retain their state and training-context notice; no lossy graph lowering occurs.
- Additional disk tests cover a symbolic bound program, target inverse transforms,
  categorical statistics, ordered category encoding, standardization, missing cells and
  an unseen category. These are representative admitted forms, not proof for all future
  programs or every combination of operators.
- A recipe/program alone is insufficient fitted state. Unknown variants retain exact
  binary parts but are explicitly nonexecutable without a registered decoder. Unknown
  nested state in an existing decoder fails; new semantics must not hide in metadata.
- Source export retains the existing emitter coverage gate. Export does not establish
  every language executor's parity. Encoded raw cells, feature order and class-label
  mapping still need sidecars when distributing generated source. ONNX is explicitly
  unavailable; a custom ONNX operator would still need its own compatible runtime.
- JSON floats and arbitrary large integers must not be normalized by a foreign language.
  The envelope hashes original bytes; the known native reader enables float roundtrip.
  Arbitrary binary buffers need a versioned type/shape/endian contract in their decoder.

## Remaining limits requiring attention

| Limit | Consequence / required follow-up |
| --- | --- |
| Existing binding validators are not exhaustive numeric-state validators | Checksums plus a known decoder do not certify every head dimension, tree invariant or allocation size. The store accepts trusted fitted state. Before exposing arbitrary foreign known-codec uploads, add exhaustive per-family validation, numerical allocation/work limits and fuzzing in a resource-limited process. `executable` is decoder acceptance, not that certification. |
| No training-worker resume decoder in this PR | Exact continuation needs population/archive, optimizer, random state, budgets/deadline, progress, data/cut/evaluator/vocabulary/protocol identities and often worker state. The envelope can retain those bytes, but cannot reconstruct missing state. Do not label inference objects “resumable checkpoints.” |
| Trusted local filesystem only | Direct symlinks/traversal are refused, but hostile ancestor changes/hard links and network filesystem consistency are not sandboxed. Qualify Windows/macOS ACL, hard-link and crash behavior on actual runners before claiming those platforms. Linux filename-rule tests are not OS execution tests. |
| No automatic garbage collection | Session history and external receipts can reference superseded objects. Backup/retention must preserve that reference closure; rollback does not delete newer candidates. |
| Hash integrity is not authenticity | A valid replacement manifest, complete history rewrite or suffix truncation needs an external retained ID/head/signature to detect. Preserve receipt identities and apply application access controls. |
| Source export and ONNX remain partial | Unsupported forms return a backend error without output. No fork is needed to retain state; decoder qualification and operator runtimes remain work for each new target. |

The newly added dependency is `serde_ignored` 0.1.14, registry checksum pinned in `Cargo.lock`.
It is a pure Rust serde adapter with no build script, native library, network or file I/O;
source inspection found no unsafe block. Its dependency closure adds only existing serde
components. Unrelated lockfile resolution drift was removed. No workflow, runner, credentials,
branch-protection setting or numerical recipe was changed for this feature.

## Acceptance commands

Run from the repository root with the repository's required Cargo environment:

```sh
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only CARGO_BUILD_JOBS=4
cargo test --locked -p jope-runtime -p jope-appliance --lib \
  --test method_roster --test offline_fit_predict --test single_file
cargo clippy --locked -p jope-runtime -p jope-appliance --all-targets -- -D warnings
cargo fmt -p jope-runtime -p jope-appliance -- --check
node examples/model-objects/check.mjs /absolute/path/to/authenticated/jopedime
```

The local command receipts distinguish failures corrected during development from the final
successful runs. The committed test inventory is a listing, not an execution certificate.
PR review and hosted/manual CI must bind to the final exact head after publication. Queued,
cancelled, skipped-by-path or unrun lanes do not establish a passing full check.
