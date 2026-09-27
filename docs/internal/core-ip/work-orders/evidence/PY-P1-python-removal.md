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
