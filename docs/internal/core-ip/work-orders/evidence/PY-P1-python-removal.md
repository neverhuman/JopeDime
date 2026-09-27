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
