# Campaign policy audit after the workspace split

Observation date: 2026-09-25 UTC. This follow-up does not replace the monograph’s
pinned evidence cutoff or its failed GPU-study record.

Full CPU run [36182992040, attempt 1](https://github.com/veox-ai/JopeDime/actions/runs/36182992040)
at `a84cba5417321287652cfb6b4c9e53467cfbcb77` failed. Its default family recorded
2,979 passing and 21 failing executions, with 38 skipped tests. The neural and
scratch families also failed; repairing this checker is not full CPU qualification.

The campaign resource-policy scanner still resolved its root to the campaign crate,
searched the old `src/` and `scripts/` layout, and read the Python policy from a
missing path. A focused local reproduction produced four failures and seven passes.
Correcting the root, scan directories and registry exposed a second defect: the
Python JSON copy retained the old runtime-bundle path, while the Rust policy had
migrated to `.local/state/jope/legacy`. The intermediate result was one failure and
ten passes. Regenerating the JSON initially corrected the local copy, but inspection exposed
another portability defect: JSON emission resolved paths against the generator
host’s state directory. Version2 now retains typed literal/root-relative arguments;
Rust and Python resolve those arguments at render time using the same state-root
precedence. The generated JSON is invariant under `JOPE_STATE` changes. Resource
reservations are unchanged, and capability probes are not executed.

The corrected local library harness passes all 13 tests, with 201 other tests
filtered out. A new filesystem fixture verifies discovery and rejection of new
Rust and Python renderers in the migrated directories. A cross-language test compares every family’s resource and placement blocks,
including runtime path resolution; malformed relative bindings are refused.
Existing negative cases
cover disagreement, forbidden placement forms, and conflicting RAM units. The CLI
also succeeds from outside the checkout: 28 registered sites, no drift findings.
The `inputs` CI lane now runs only this library harness, retains a nested Rust
receipt with collection/execution counts, and rejects zero executed tests. The
remaining campaign tests retain their prior selection and status. Path classification
selects this lane for either migrated renderer tree and its policy files; a regression
test verifies those paths and unrelated-path exclusions.

Reproduce without campaign submission or GPU execution:

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo test -j4 --locked -p jope-dime-regressor --lib spec_drift::tests -- --test-threads=1
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo run -j4 --locked -p jope-dime-regressor --bin jope_spec_drift -- --list
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo run -j4 --locked -p jope-dime-regressor --bin jope_spec_drift
```

Local evidence, original full-run artifacts and reviewed lifecycle receipts are
retained under `/home/ubuntu/jope-rescue-20260921/core-ip-cpu-followup-20260925/`.
The original monograph remains a dated publication. Hosted verification and merge
status belong to the exact-head lifecycle receipt, not to these local observations.
