# PY-P1 — Python removal, phase P1

Branch `rr2/python-p1`, base `8a55c4fef`. Owner ruling 2026-09-26: the stack is
Rust/Vite/TypeScript/React; Python remains only for GP-emitted code and its harness. Live tools
are ported with differential tests, unused code is archived. Evidence runs live under
`/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/py-p1-*`
(identity, diff, command and log per run).

Observed states: every port below is **differentially tested on this host** against the Python it
replaces, run before the Python was deleted. Nothing here is CI-verified or live-qualified, and
no installed host file was changed.

## U8 — grown-family policy (`crates/jope-engine/policy/family_growth_policy.py`)

- Before: `families::freeze_growth` hash-checked `<root>/tools/family_growth_policy.py` against a
  compiled-in copy and ran it with `/usr/bin/python3 -I -B … --verify <receipt>`.
- After: `crates/jope-engine/src/family_growth.rs` is the native `assign_new_components` /
  `verify_assignment`; `verified_growth` hashes and parses the receipt once and recomputes it in
  process. No helper in the workspace is consulted, so a planted one can neither authorize a
  forged receipt nor block a genuine one (`component_growth_refuses_forged_summaries_changed_helper_and_stale_parent`).
  Every refusal keeps the Python message.
- Differential proof: 142 scenarios — the 14 cases of the retired module's own tests (same
  fixture, same mutations) and 128 fixed-seed random worlds (seed 20260926: random parents,
  merges, alias faults, 38 decision faults, receipt tampering) — each verified by the Python
  policy restored from `8a55c4fef` and by the native verifier, in both reference forms (relative
  and absolute paths). Result: 0 disagreements; 43 accepted with byte-identical output, 87
  identical `ValueError` refusals, 12 other Python exceptions matched as malformed input.
  Capture run `py-p1-u8-capture-20260927T022535Z` (`capture/report.json`).
- The relative-form Python outputs are committed as
  `crates/jope-engine/tests/fixtures/family-growth/python-verify-outputs.jsonl` and replayed by
  `growth_policy_matches_the_captured_python_outputs`; `capture_growth_policy_outputs_from_python`
  (ignored) regenerates them from the archived file.
- Deliberate fail-closed differences: JSON Python reads but RFC 8259 does not (`NaN`, UTF-16/32,
  BOM); non-ASCII reviewer names; `//`-rooted path spellings.
- Not ported, no consumer outside archived Python: `decide_unresolved`, `is_synthetic`,
  `refuse_live_index`, `v5_canary_contract`, and the `task` / `provenance_resolved` /
  `component_ids` refusals, which only Python callers could reach.
- Runs: pre `py-p1-u8-pre-20260927T021255Z` (8 engine tests with the Python helper; the Python
  module's 11 unittests); post `py-p1-u8-post-20260927T022644Z` (`families` 9 passed + the
  ignored capture; `family_growth` unit tests 4; fence suites; workspace check; fmt; coverage,
  test-inventory, parity and static-Python guards).
- Records: `deploy/ci/coverage.json` and `findings/test-inventory.json` drop
  `python:python/src/jope/tools/test_family_growth_policy.py`; the capability-matrix rows for the
  policy and its test now name the Rust port and its parity test.

## U2 — Cargo artifact selection (`deploy/ci/cargo-artifact.py`)

- Home: new crate `crates/jope-ci` (binary `jope-ci`, subcommand `cargo-artifact
  build|select|target-dir|nextest-config`, same flags, output and refusal messages). A new crate
  rather than `jope-records`: this is build tooling, not a core-IP record, and it is where the P2
  ports of the other `deploy/ci` helpers (receipts, aggregate, plan) are planned. It depends only
  on workspace crates already in the lock (`clap`, `nix`, `serde_json`, `toml`); `Cargo.lock`
  gains only the `jope-ci` entry. It is in the `rust-test` product scope.
- Consumers now run `cargo run -q --locked [--manifest-path <repo>/Cargo.toml] -p jope-ci --
  cargo-artifact …`: `fast-lane.sh` (hygiene protocol tests, rust-test target and nextest config),
  `retain-nextest.sh`, `emitters.sh`, `e2e-smoke.sh`, `full-lane.sh`, `regression-suite.sh` (3),
  `restore-drill.sh`, `kernel-v1-product-build.sh`, `deploy/systemd/render.sh`,
  `release.yml:73`, `weekly.yml:44`, and the one argv of `deploy/install-private.py:284` (a U6 P2
  tool; no other line of it changed). `deploy/ci/ci-plan.py` (U1, P2) still names the old path in
  its image-path pattern; the pattern is inert for a deleted file and is left for the U1 port.
- Differential run `py-p1-u2-diff-20260927T023243Z` (`comparison.txt`): Python and Rust on the
  real tree — `target-dir`, `nextest-config` (byte-identical TOML), `build` of a bin and of an
  example, a failing build (both exit 101, no stdout), and `select` on a captured real Cargo
  message stream plus identity and failing path maps, unfinished, failed, wrong-package,
  wrong-kind and doubled streams, and a usage error. Every exit status, stdout and
  `cargo-artifact:` line is identical except the OS wording of "file exists".
- Tests: `crates/jope-ci/tests/cargo_artifact.rs` ports the 11 Python cases (the fake Cargo is a
  shell script now); pre `py-p1-u2-pre-20260927T023312Z` (the Python suite, 11 OK); post
  `py-p1-u2-post-20260927T023758Z` (jope-ci 3 + 11, scope partition 17 + 9 = 26 packages,
  `deploy/ci/check.sh` with the host's history-scanning gitleaks shimmed — it reports 244
  pre-existing historical findings on this host — fence suites, workspace check and guards).
- Records: coverage.json items for the `jope-ci` package and its three targets (hygiene lane
  added to the test-bearing ones; the hygiene `checks` list names `cargo test -q --locked -p
  jope-ci`), test-inventory roles `infrastructure`, one `new-bin` capability row.

## U3 — hygiene guards (`coverage.py`, `rust-test-scope-check.py`, `parity-census.py`, `readme-benchmarks.py`)

- Home: `jope-records` subcommands `coverage discover|validate`, `rust-scope-check`,
  `parity-check` and `readme-benchmarks --check|--write` (modules `coverage`, `scope_check`,
  `parity`, `readme`), reusing the crate's Python-compatible JSON, CSV, `repr`, `float()` and
  `splitlines` layer so the output is the Python output. `check.sh`, `parity-matrix.sh` and the
  hygiene lane of `fast-lane.sh` call them; the hygiene lane runs their unit tests
  (`cargo test -q --locked -p jope-records --lib -- coverage:: scope_check:: parity:: readme::`)
  where it ran `test-coverage.py`.
- Not ported: `parity-census.py census`, a one-off generator that needs the read-only pre-port
  checkout and has no consumer (archived; restore it from `8a55c4fef` to regenerate the matrix).
- Differential run `py-p1-u3-diff-20260927T024147Z` (`comparison.txt`), Python and Rust on the
  same inputs: `coverage validate` and `discover` on the real tree (3,020-line discovery,
  byte-identical) and on nine mutated mappings (unassigned, stale, schema, lane fields,
  dispositions, unknown lane/disposition/scope, unassigned item, required features, changed
  discovery); `rust-scope-check` on the real tree; `parity-check` on the real matrix with and
  without the built engine's `--help`, and on eight mutated matrices (counts, tally, preamble,
  invalid status, pending range, dropped archive, missing target, missing row);
  `readme-benchmarks --check` on the real and a stale README, and `--write` from the stale copy.
  Every exit status and stdout is identical, the two `--write` results are byte-identical and
  equal the committed README, and the only differences are the two remediation hints that named
  the deleted scripts (the preamble refusal and the stale-README notice now name the Rust
  command).
- Tests: `test-coverage.py`'s nine cases are unit tests in `coverage.rs` (the test-inventory
  collection-failure case still drives `deploy/ci/test-inventory.sh` with a fake `uv`), plus
  unit tests for the scope reader, the matrix parser and tally, and Python's number formats.
  Pre `py-p1-u3-pre-20260927T024301Z` (the Python suite 9 OK and all four guards green); post
  `py-p1-u3-post-20260927T024409Z` (jope-records suites, the four guards, `check.sh`, fence
  suites, workspace check and the record guards).
- Records: coverage.json drops `python:deploy/ci/test-coverage.py` and
  `benchmark:deploy/ci/readme-benchmarks.py` and adds the hygiene lane to the jope-records library
  target; test-inventory.json drops the Python family. Prose that told readers to run the
  deleted scripts (README, quickstart, same-cut and regression SOURCES, coverage.md, the matrix
  preamble) names the Rust commands.

## U5 — build-directory report (`deploy/maintenance/jope_report.py`)

- Home: `jope-ci build-report scan|report` (module `build_report`), using jope-records'
  Python-compatible JSON so the scan documents and receipts keep their shape and key order.
  SIGTERM/SIGINT still become a failed receipt ("report interrupted by signal N") through
  `signal-hook` (already in the lock). The wrappers `cargo-build-dirs.sh` and
  `jope-build-report.sh` exec the `jope-ci` installed beside them; the report wrapper passes its
  own directory so the receipt still fingerprints the wrappers with the core.
- One deliberate move: the Python's `APPROVED_ROOTS` constant (two absolute host paths) would
  fail `just lint-paths` (ADR-0003) inside `crates/`, so the literal stays in
  `deploy/maintenance/`, now in `jope-build-report.sh`, which passes it as two required
  `--approved-root` arguments. The fence is unchanged: the report refuses unless exactly two
  roots are given and the configuration names exactly those, and the wrapper holding them is
  checksummed in every receipt, as `jope_report.py` was. The fixture runner reads the roots from
  the wrapper and adds two refusal cases (a single approved root; a configuration naming one).
- Out of scope, unchanged: the installed `~/lane-tools` copy (still the Python core) and the user
  units. `deploy/maintenance/README.md` now says how to build and install the Rust core and how
  to run its fixture runners, and that the superseded installed `jope_report.py` is removed only
  after its bytes are preserved.
- Differential run `py-p1-u5-diff-20260927T025057Z` (`comparison.txt`): Python and Rust `scan`
  on the retired test's fixture tree (with and without `--stale`, a nested root, missing and
  symlinked roots, `--delete` before and after root validation, bad, negative and missing ages, and
  an unreadable `.git` subtree the mtime walk must not suppress), `scan` of this lane's real
  30 GB build tree, and `report` with malformed, key-missing and wrong-root configurations and
  `/tmp` and working-tree custody. Every exit status matches; documents are identical apart from
  `completed_at`; receipts are identical apart from timestamps, custody paths and the core's own
  checksum; failure lines match except argument-parser wording. The first lane scan differed by
  one 4 KiB block measured moments after a build; the back-to-back re-measure Python, Rust,
  Python (`scan-lane-rerun.txt`) is identical.
- Tests: `crates/jope-ci/tests/build_report.rs` ports both fixture runners (ignored by default,
  `JOPE_MAINTENANCE_SCRATCH` outside `/tmp` and working trees, like the Python `--scratch`); the
  wrapper runner drives the library with injected scans and measurements, and a real SIGTERM
  against the binary scanning the approved roots. Pre `py-p1-u5-pre-20260927T025026Z` (both
  Python runners PASS, 15 + 15 scenarios). Post `py-p1-u5-post-20260927T030252Z`: the full gate
  plus `cargo test -p jope-ci`, both fixture runners (scratch
  `py-p1-u5-post-scratch-20260927T030252Z`, including the real SIGTERM run and the two new
  approved-root refusals), ShellCheck on the wrappers and `deploy/ci/check.sh` all green. Earlier
  post attempts failed and were fixed, not waived: restored fixture modes (the runner now
  restores the original modes), `just lint-paths` flagging the approved-root literal (moved to the
  wrapper, above), and one missed call site.
- Records: coverage.json and test-inventory.json replace the two Python fixture families with
  `target:jope-ci/test/build_report` (disposition `maintenance`, role `infrastructure`); the
  jope-ci capability row names both ports.
