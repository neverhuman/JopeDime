# PY-P1 — Python removal, phase P1

Written on branch `rr2/python-p1` (base `8a55c4fef`) and landed as `rr2/python-p1-landing`, a
replay onto main at `cee1fc8bd`. The commits named here are the landed ones; the commit map at the
end gives each one's original. Owner ruling 2026-09-26: the stack is Rust/Vite/TypeScript/React;
Python remains only for GP-emitted code and its harness. Live tools are ported with differential
tests, unused code is archived. Evidence runs live under
`/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/py-p1-*`
and, for the landing, `p1-landing-*` (identity, diff, command and log per run).

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

## Archive — every A file, U12–U14 with their tests, kernel-v1 Python, 9 unit templates

- Removed with `git rm`: 703 files (694 `.py` + 9 systemd templates that only ran archived U12/U13
  code). By category: 625 A, 37 L and 32 T (the U12–U14 units), 9 templates. By unit: 103
  kernel-v1 (one unit), U12 17, U13 33, U14 28, the rest unowned A. By directory:
  `python/src/jope/tools` 285, `python/src/jope/research` 87, `python/legacy` 82,
  `python/src/jope/chimera` 44, `python/tests` 29, top-level `python/src/jope` 14,
  `python/src/jope/zyal` 12, `python/src/jope/genome_perf` 10, `python/src/jope/fitness_corrector` 6,
  `python/src/jope/target_maker` 5, `python/src/jope/broad_shadow` 3, `python/src/jope/data` 1,
  `python/src/jope/dime_score` 1, `deploy/kernel-v1` 103, `deploy/systemd/engine` 7,
  `deploy/systemd/selfplay` 2, `docs/legacy/root-docs/audits/worldclass-redteam-20260911` 11,
  `crates/jope-campaigns` 1. `git ls-files '*.py'` now lists 216 files.
- Manifest: `docs/archive/python-archive-manifest.json`, 715 rows (the 703 above plus the 12
  Python files the U8/U2/U3/U5 commits ported), each with path, blob, mode, category, unit,
  disposition, Rust replacement for ported rows, source commit `8a55c4fef407db01…` and
  `git checkout 8a55c4fef -- <path>`. Every blob equals the base tree (checked by joining
  `git ls-tree -r 8a55c4fef`) and no archived file changed between the base and the archive commit.
- Custody: `/home/ubuntu/jope-evidence/python-archive-20260926/python-archive-20260926.tar.gz`
  (`git archive 8a55c4fef -- <715 paths> | gzip -n -9`, 1,856,667 bytes, sha256
  `41d1247a9628efb3133289f5b72bee451ad655a7725b5c25cc1e58159a5c1205`) with the manifest, the
  TSV and the file list; `SHA256SUMS` lists every file there except itself and verifies.
  Verification (`VERIFICATION.txt`): extracted into a fresh directory, all 715 paths present,
  `git hash-object --no-filters` and the executable bit equal the manifest for every row, no other
  file or link; RESULT PASS, then the verified extraction was removed.
- Superseded first build, kept unchanged under `superseded-718-rows-20260927T030130Z/` with its own
  `SHA256SUMS` and a `NOTE.txt`: it also held `python/src/jope/elo_replica.py`,
  `python/src/jope/chimera/elo_finetune/elo_meter.py` and
  `python/src/jope/research/render_elo_ranking.py`. Those three are digest-checked `retained`
  inputs of `docs/internal/core-ip/elo-history/histories.json` (`jope-records elo-history
  --require-all`, hygiene lane), so they have a live consumer and stay. Its first verification
  report failed only because the checker read the TSV with `read`, which collapses empty tab
  fields; it is kept as `VERIFICATION-20260927T030130Z-parser-bug.txt`, and the rerun on the same
  tarball passed.
- Kept although classified A (live consumer found while archiving): the 19 research modules and
  `deploy_selfplay_bundle.py` imported or read by kept tests, `test_exact_canary.py` (launched by
  path from kept tests, so its `spec_drift` entry stays live), the three elo-history files above,
  and from U13 `shared_teaching_cache.py`, `resume_expired_cycle.py` and `tools/__init__.py`
  (`crates/jope-engine/tests/sealed_storage.rs` spawns the cache server). The cache's own
  never-collected test imports the archived `resident_gpu_pool.py` and went with U13.
- Guards, none weakened:
  - `jope-records coverage validate`: 186 stale items removed from `deploy/ci/coverage.json`
    (605 → 419 items; key order kept); 0 unassigned or invalid.
  - `verify-test-inventory`: the same 170 test-bearing keys removed from
    `docs/internal/core-ip/findings/test-inventory.json` (473 → 303).
  - `deploy/ci/test-inventory.sh --static-python`: 276 → 172 static test functions (111 in
    archived files removed; 7 existing `test_policy_training_contract.py` tests that were never
    recorded are now recorded). The floor goes from 214 to exactly 172 in both the committed
    inventory and `FLOORS`, with the reason and this manifest's path in a comment.
  - Capability matrix: the 380 rows whose target was archived (script-py 83, script-test 4,
    systemd-unit 9, tool-py 183, tool-test 101) become `dropped` with the reason, the all-refs
    bundle key and the `8a55c4fef` restore command; Counts recomputed from the rows (moved
    735 → 355, dropped 20 → 400; no-test tally 657 → 300); the legend names the P1 drops. The
    shared-cache row's note now says why the module stays.
  - `spec_drift` registry: the entries for `train_only_redteam_scq.py` and
    `genome_boost_cluster_campaign.py` are removed (both archived) with a comment on how to
    restore and re-register them; `test_exact_backlog.py`, `test_exact_canary.py`,
    `test_exact_ops.py`, `test_exact_train.py` and `smart_v3_cpu_scq.py` stay registered.
  - `python/pyproject.toml`: the `src/jope/chimera/**` per-file ignore is removed (ruff passes on
    the one chimera file left without it).
  - Provenance: `assets/regression/SOURCES.md` cites
    `8a55c4fef:python/src/jope/research/index_combined_regression_evolution.py:183` and explains
    that the SHA-pinned corrections file (and the README block rendered from it) keeps its original
    paths; `crates/jope-learning/src/metric.rs` cites `8a55c4fef:…/promotion_gate.py:22-27`;
    `jope_selfplay_bundle.rs` says why the workspace-sync unit names stay excluded.
- Left in place and reported: the SHA-pinned `protocol-corrections-tabular-regression-v2.json`
  citations; `crates/jope-engine/manifests/opportunistic-proposals-v44/v53` and
  `crates/jope-campaigns/manifests/workspace-sync*.json` (historical manifests naming archived
  files); `jope-parallel compare-seeds`, which still execs `<state root>/tools/seed_experiment.py`
  (a state-root copy, not this checkout); `deploy_selfplay_bundle.py`'s reference to the archived
  `tests/test_realtime_selfplay.py` (an xbabe2-only path); the `.gitleaks.toml` allowlist entry
  for `test_workspace_sync.py` (history is still scanned); manual and training-support prose that
  cites archived paths (`docs/internal/asm-manual`, `docs/internal/training-support`,
  `docs/internal/core-ip/sections/06.md`).
- Python lane before the archive: `py-p1-archive-pre-20260927T030540Z` (uv sync, ruff check and
  format, pytest 343 passed / 15 deselected, training-proof's required cases: PASS).
- After: `py-p1-archive-post-20260927T031513Z` — the full gate (fences, `cargo check --workspace
  --all-targets`, fmt, `git diff --check`, coverage validate, `verify-test-inventory`, the matrix
  check, the static-Python inventory) plus the `spec_drift` tests (13 passed), `elo-history
  --require-all`, `jope-arena`'s `elo_regression` (4 passed), `cargo test -p jope-records`,
  ShellCheck on `test-inventory.sh`, `deploy/ci/check.sh` and the Python lane (uv sync, ruff check
  and format, pytest 245 passed / 2 deselected, training-proof's required cases: PASS). Its one
  red step named the wrong package for the `sealed_storage` test (`jope-engine` instead of
  `jope-parallel`, so nothing ran); `py-p1-archive-post-20260927T032003Z` ran it under the right
  name: the test that spawns `shared_teaching_cache.py` as the cache server passes.

## Docs — ADR-0006 and the contributor-facing pages

- `docs/adr/ADR-0006-rust-only-stack.md` records the owner ruling of 2026-09-26 in the four ADR
  sections: it supersedes owner decision 2 of the plan and, for future work, the Python-training
  parts of ADR-0005; names the GP-emitted-code exception, the always-keep list (the eight G files,
  `python3` in the runner image, the sha-pinned grammar sources until P3 decides) and the P1, P2
  and P3 scopes. It is numbered ADR-0006, the next number after ADR-0005 as `CONTRIBUTING.md`
  requires. It was first committed as ADR-0008 (the brief's name, `4ed0e2096`) and renumbered in a
  follow-up commit (`ee24eae50`); every reference moved with it, including the `ruling` string of
  the archive manifest. The custody copy of that manifest was replaced by the committed one (only
  `.ruling` differs; `.files`, `.counts` and `.custody` are identical), with the previous copy
  kept under `superseded-manifest-adr-0008-20260927/` and a `NOTE-20260927-adr-renumber.txt`;
  `SHA256SUMS` was regenerated and verifies. Renumber gate `py-p1-adr-post-20260927T033239Z`: the
  full gate, the relative-link check and `deploy/ci/check.sh`, all green.
- `CONTRIBUTING.md` (toolchain, `just ci-pr`, required checks, dependencies and the former "Rust is
  authoritative" rule, now "Rust only"), `docs/architecture.md` (the summary, the Python package
  section and the contracts note), `README.md` (the ruff note and the layout row) and
  `docs/asm.md` (the training row, dataset preparation, the external adapters and the panel
  runtime preparer) now say that Python remains only on the listed live paths until P2/P3 and
  that new Python is not accepted except emitted model code and its harness. The first docs gate
  (`py-p1-docs-post-20260927T032141Z`) failed `deploy/ci/ip-scrub.sh`: the public prose used the
  ruling's internal term for that exception. The public pages now say "the Python the emitters
  generate"; the ADR body (a historical record the scrub does not scan) keeps the ruling's
  wording, and its title does not. Rerun `py-p1-docs-post-20260927T032336Z`: the full gate plus a
  relative-link check of the five edited pages (no link checker is installed here) and
  `deploy/ci/check.sh`, all green. One wording fix followed (architecture.md: "694 files, plus 9
  unit templates"), rechecked with `ip-scrub.sh`, the link check and `git diff --check`.

## Follow-up — captured Python outputs as committed fixtures (U2, U5, U3)

U8 already replayed its captured Python outputs (`python-verify-outputs.jsonl`). U2, U3 and U5 had
differential runs but no committed fixtures; these tests close that gap. Each fixture holds the
Python outputs of the unit's differential run, normalised the same way in the fixture and in the
Rust output under test (the no-home-paths lint scans `crates/*/tests`), and the inputs they were
produced from. Where a committed input is reduced or normalised, the archived Python (restored
from `8a55c4fef`, run from a scratch directory, never extended) was re-run on exactly the
committed inputs and reproduced every captured output. The intended differences are asserted
explicitly, case by case, as the only allowed ones. Each test was also shown to fail when one
captured output is altered (negative control).

- U2 — `crates/jope-ci/tests/cargo_artifact_python_parity.rs`, fixture
  `crates/jope-ci/tests/fixtures/python-parity/cargo-artifact/`: 11 cases from
  `py-p1-u2-diff-20260927T023243Z` (select, select with a path map, a bad path map, unfinished,
  failed, wrong-package, example-kind and ambiguous streams, usage, nextest-config and an existing
  nextest output). Inputs: the captured `cargo build --message-format=json-render-diagnostics`
  stream of `jope-paths`/`jope-units-render`, that workspace's `cargo metadata` reduced to the
  fields `select` reads (package names and ids, members, target directory, root), and
  `.config/nextest.toml`; the lane root is `@ROOT@`. The test runs the real binary with those
  inputs under a fresh directory standing for `@ROOT@` and compares exit status, stdout, stderr
  and the relocated nextest config byte for byte. Allowed differences: argument-parser wording
  (`usage`) and the OS wording for an existing output (`nextest-exists`). Faithfulness re-run:
  `py-p1-fixtures-u2-20260927T033630Z/verification.txt` (11 of 11 identical).
- U5 — `crates/jope-ci/tests/build_report_python_parity.rs`, fixture
  `crates/jope-ci/tests/fixtures/python-parity/build-report/`: from
  `py-p1-u5-diff-20260927T025057Z`, 11 `scan` cases on the synthetic build tree that run created
  (stale, all, fingerprint root, missing and symlinked roots, `--delete` twice, three bad
  `--stale` values, a traversal failure), 2 custody refusals (`/tmp`, a Git working tree) and 3
  failure receipts (malformed, key-missing and wrong-root configurations). The test rebuilds the
  same tree with the same fixed epoch; tree root, run directory, `completed_at`, allocated bytes
  (filesystem-dependent: checked positive and summing instead) and fresh mtimes (checked to be the
  time the test touched them) are placeholders. Scans and custody refusals run by default; the
  receipt replay is ignored by default because custody must be outside `/tmp` and Git working
  trees, and ran green with `JOPE_MAINTENANCE_SCRATCH`. The traversal-failure case is skipped only
  for a user that ignores permissions. Allowed differences: argument-parser wording for the three
  bad `--stale` values. Not replayed: the differential run's scan of this lane's own 30 GB build
  tree. Faithfulness re-run: `py-p1-fixtures-u5-20260927T033842Z/verification.txt` (16 of 16
  identical).
- Negative control for both: `py-p1-fixtures-mutation-ci.txt` beside the verification files
  (four altered outputs, four failures, fixtures restored byte-identically).
- Records: `target:jope-ci/test/cargo_artifact_python_parity` and
  `target:jope-ci/test/build_report_python_parity` in `deploy/ci/coverage.json` (the hygiene lane
  runs them with `cargo test -p jope-ci`; the second carries the `maintenance` disposition for its
  ignored receipt replay) and in the regression test inventory (role `infrastructure`).
- Gate `py-p1-fixtures-ci-post-20260927T034300Z`: the full gate plus `cargo test -p jope-ci`
  (the new targets 1 + 2 passed, 1 ignored), both ignored fixture runners with
  `JOPE_MAINTENANCE_SCRATCH` (receipt replay and the build-report runners) and
  `deploy/ci/check.sh`, all green.
- U3 — `crates/jope-records/tests/python_parity.rs`, fixture
  `crates/jope-records/tests/fixtures/python-parity/`: all 25 cases of
  `py-p1-u3-diff-20260927T024147Z`. The inputs are those of that run's tree (the U2 commit as
  first written on `rr2/python-p1`, which the run's `identity.txt` names and the commit map below
  pairs with its landed replay `6b95c9208`, plus the run's working changes: the jope-records
  `lib.rs`/`main.rs` edits and four new, untracked modules), recovered from an export of it
  (`git archive`, the run's `diff.patch`, the four modules; not a clone or worktree). The landed
  `6b95c9208` does not hold that tree (its `deploy/ci/coverage.json` also has the entries main
  gained up to `cee1fc8bd`), so the committed fixture is the record of the inputs:
  - coverage `validate` (10 cases): the coverage map of the run (`mapping.json`, blob `5e78ee4ba`,
    the run's `deploy/ci/coverage.json`) and its nine malformed variants as `ed` scripts, each
    with the SHA-256 of the variant file the Python read (regenerated from the base with the
    run's own `jq` edits and compared byte for byte first); every case is checked against
    `discovered.json`, the Python's discovery output of that tree.
  - coverage `discover` (1 case): `cargo metadata` of the export reduced to the fields discovery
    reads (root as `@ROOT@`), the file list `git ls-files --cached --others --exclude-standard`
    gave (the tree's files plus the four untracked modules) and the workflow texts; the Rust
    discovery of those inputs must print `discovered.json` byte for byte.
  - `rust-scope-check` (1 case): the workspace package names, what `rust-test-scope.sh` emitted
    for both scopes and the script text of that tree.
  - `parity-check` (10 cases): the capability matrix of the run and its eight variants as `ed`
    scripts with SHA-256s (again regenerated with the run's own `sed`/`grep` first); the seven
    lines whose notes cite `/home/<user>/` read `~/` in the fixture, which the no-home-paths lint
    requires and which touches no checked cell. `tree.json` records what the check reads from the
    tree (89 binaries, 19 examples, 92 clap subcommands, 430 Python tools, `jope-py` commands, 0
    systemd units, and whether each row target existed), captured from the export by the ignored
    `capture_parity_tree` test.
  - `readme-benchmarks` (3 cases): every render input at that tree (byte-identical to the
    committed evidence, so git stores them once; `docs/same-cut-bench.md` is blob `8df62628d`, the
    text at `8a55c4fef` and at `cee1fc8bd`, which the renderer only names), the run's stale README,
    and the README its `--write` produced; `--check` on the stale copy, `--write`, and `--check` on
    the result.

  To replay recorded inputs the guards gained seams with no behaviour change: `coverage::
  discover_from`, `validate_text` and `discover_document`; `parity::Tree`, `scan_tree`,
  `evaluate`, `parse_matrix` and `target_heads`; `scope_check::evaluate`. The live commands
  gather the same inputs and call them. Allowed differences, asserted as exact substitutions: the
  two remediation hints that name the Rust source or command instead of the retired script
  (`parity-preamble`, `readme-stale`). Build and verification of the inputs:
  `py-p1-fixtures-u3-20260927T034740Z`.
- Negative control for U3: `py-p1-fixtures-u3-20260927T034740Z/mutation.txt` (nine alterations:
  a validate output, a variant SHA-256, the discover output, a discover input file, the scope
  output, a parity output, a recorded target existence, the written README and a README render
  input; nine failures, fixtures restored byte-identically). A first attempt
  (`mutation-first-attempt.txt`) altered the bench baseline's `schema`, which the renderer does
  not read, and so rightly went undetected; the rerun alters a rendered value.
- Records: `target:jope-records/test/python_parity` in `deploy/ci/coverage.json` (rust-check,
  rust-test, full-default) and in the regression test inventory (role `infrastructure`).
- Gate `py-p1-fixtures-u3-post-20260927T040129Z`: the full gate plus `cargo test -p jope-records`
  (the parity target 5 passed, 1 ignored helper), `clippy -p jope-records -p jope-ci
  --all-targets -D warnings` and the live guards on this tree (`coverage discover`,
  `rust-scope-check`, `readme-benchmarks --check`), all green except `deploy/ci/check.sh`, whose
  size-cap step hit a stale index entry: the first gate (`py-p1-fixtures-u3-post-20260927T035846Z`)
  failed `git diff --check` on the trailing spaces of the captured `--help` text, which is now
  kept as a JSON string (`parity/help.json`, with its SHA-256) and the old path's intent-to-add
  entry was dropped. `py-p1-fixtures-u3-post-20260927T040335Z` then ran `deploy/ci/check.sh`, the
  parity tests and `git diff --check`: green.

## Follow-up — clippy on the P1 code

`just ci-pr` runs `cargo clippy --workspace --all-targets --locked -- -D warnings`; the per-commit
gate above did not. Run on the crates this branch touched, it flagged code the branch wrote: an
import `families.rs` stopped using in the U8 port (also a plain rustc warning), U8's
eight-argument `authenticate` (it mirrors the Python function; allowed as `run_report` is), U5's
walk visitor type (now the named alias `Visit`), decimal `0` modes in the U5 tests (`0o000`), and
U3's `is_multiple_of` and `sort_by_key` idioms in the README renderer and the scope check. All are
fixed without behaviour change. `jope-ci` and `jope-records` are clean with `-D warnings`;
`jope-parallel` and `jope-dime-regressor` still carry findings that predate this branch (in
`dashboard.rs`, `grammar.rs`, `service.rs`, `spec_drift.rs` outside the comment this branch
edited, and others), none in code this branch wrote. Gate
`py-p1-clippy-post-20260927T035020Z` (full gate, `cargo test -p jope-ci` and both ignored runners,
the families tests, `readme::` unit tests, no clippy finding in `families.rs`, `family_growth.rs`
or `readme.rs`, `deploy/ci/check.sh`) was green except `clippy -p jope-ci -D warnings`, which
also lints the `jope-records` dependency and stopped at the scope check's `% 2`. With that fixed,
`py-p1-clippy-post-20260927T035412Z` passed `clippy -p jope-ci -p jope-records --all-targets
-D warnings` and the scope-check unit tests; its last chained step ran `deploy/ci/check.sh`
without this host's gitleaks shim, which stops on the 244 historical findings, so that run exits
with status 1. `py-p1-clippy-post-20260927T035547Z` then ran fmt, `git diff --check` and
`deploy/ci/check.sh` with the shim: green.

## Follow-up — stale references to archived Python: PY-P2 items

References that still name archived Python after P1, beyond the archive records themselves (the
manifest, the matrix's dropped rows, this record). Each item gives the consumer (file:line at
this branch), what it points at, and the proposed disposition for P2. No CI lane needs the
archived files for any of them.

- **PY-P2-1 `jope-parallel compare-seeds`.** `crates/jope-engine/src/main.rs:1348-1358`: the verb
  execs `python3 <state root>/tools/seed_experiment.py` (a copy in the state root, not this
  checkout); `crates/jope-engine/tests/experiment_cli.rs:24` writes a stand-in script there. The
  source is archived (`8a55c4fef:python/src/jope/tools/seed_experiment.py`, U13) and no unit has
  run it since the flywheel stopped on 2026-09-16. Proposed: retire the verb, or port the seed
  comparison to Rust if the owner wants it back.
- **PY-P2-2 dashboard service list.** `crates/jope-engine/src/dashboard.rs:402` (`live_operations`)
  asks systemd for `jope-parallel-console`, whose template
  (`deploy/systemd/engine/jope-parallel-console.service.in`) was archived with U13. Until then the
  dashboard simply reports that unit as absent. Proposed: drop the name when the dashboard is next
  changed.
- **PY-P2-3 opportunistic-proposal manifests.**
  `crates/jope-engine/manifests/opportunistic-proposals-v44.sha256:4` and
  `opportunistic-proposals-v53.sha256:5` pin the archived unit files
  `systemd/jope-parallel-proposal-library-v44.service` and
  `systemd/jope-parallel-pure-proposal-library-v53.service`; the `.json` specs were read by the
  archived `opportunistic_proposals.py` / `_v2.py`. Nothing in the tree reads these copies; the
  only units that name such manifests are the frozen kernel-v1 ones,
  `deploy/kernel-v1/source/systemd/jope-parallel-proposal-library-v44.service:9-10` and
  `jope-parallel-pure-proposal-library-v53.service:9-10`, which point at the old `parallel/`
  layout and are not installed. Proposed: keep the pins as history (they record what ran).
- **PY-P2-4 workspace-sync manifests.** `crates/jope-campaigns/manifests/workspace-sync.v2.json:35-38`,
  `workspace-sync.references.v2.json:34-37` and `workspace-sync.canonical-raw.v2.json:33-36` pin
  the archived U12 sources (`scripts/workspace_{inventory,store,sync}.py`,
  `parallel/tools/kernel_remote_state.py`). No consumer in the tree (`git grep`). Proposed: keep
  as history.
- **PY-P2-5 source bindings.** `python/src/jope/tools/flywheel-source-binding.json:42,47,52` (and
  its frozen twin `deploy/kernel-v1/source/tools/flywheel-source-binding.json:42,47,52`) bind the
  archived `kernel_remote_state.py`, `campaign_bound_jobs.py` and `diagnose_val_vs_corpus.py`;
  `python/src/jope/tools/research-readouts-binding.json:8-16` binds nine archived readout tools. No
  consumer in the tree. Proposed: keep the kernel-v1 twin as history; retire the two
  `python/src/jope/tools/` copies with that directory (P2/P3).
- **PY-P2-6 durable self-play deploy.** `python/src/jope/research/deploy_selfplay_bundle.py:130`
  names `tests/test_realtime_selfplay.py`, archived in P1; the module is kept because
  `python/tests/test_durable_selfplay.py:441` reads it by path, and line 130 runs only in the
  xbabe2 activation step. Proposed: decide together with that test (a `?` unit): port the
  activation to Rust or retire both.

Left unchanged on purpose:

- `assets/regression/protocol-corrections-tabular-regression-v2.json:38,43` cite
  `index_combined_regression_evolution.py:183` and `train_only_redteam_scq.py:1527-1536`; the file
  is SHA-256-pinned, and `README.md:64` quotes the first citation from it in the block
  `jope-records readme-benchmarks` renders. `assets/regression/SOURCES.md` maps both to
  `8a55c4fef`.
- `docs/internal/training-support/audit.md:13,44` and `audit.html:17,94`: the generation receipt
  `audit.receipt.json` binds both files (with the PDF) by SHA-256.
- `docs/internal/asm-manual/sections/deep-stack.md:248` and `sections/training.md:43` (listed in
  the manual's `render-receipt.json`) and the rendered `MANUAL.md:917,1926`, and
  `docs/internal/core-ip/sections/06.md:181`, which repeats the manual's sentence: historical
  citations of what a snapshot contained. They go with the documentation tooling (U16, P2).
- `docs/coordination/AGENT_CHAT-20260923-pr68-to-pr72.md:29`: an archived coordination record.

Fixed: `.gitleaks.toml` keeps the allowlist entry for
`python/src/jope/research/test_workspace_sync.py`: `deploy/ci/check.sh` runs `gitleaks detect`
over git history, so on any full clone (as contributors run it) the file's synthetic key fixture
is still scanned in the commits before the archive. A comment now says the file is archived at
`8a55c4fef` and why the entry stays; removing it would only make full-history scans fail.
Check `py-p1-stale-post-20260927T040410Z`: `git diff --check`, `gitleaks detect` loading the
edited configuration (no leaks), and `deploy/ci/check.sh`: green.

## Commit map — `rr2/python-p1` to the landing

The work above was committed on `rr2/python-p1` and landed as `rr2/python-p1-landing`: its eleven
commits were replayed in order onto main at `cee1fc8bd` (after #114 and #115), followed by two
landing-only commits. Every replayed commit has a new SHA, and the originals are not in the
landed history. Where this record, the tests or the fixtures cite a commit for P1's own
provenance, it is a landed commit, `8a55c4fef` or `cee1fc8bd`, or they cite the blob id of a
committed file; fixture copies of main's files keep main's own citations unchanged. An original
appears only in this table and in landed commit messages, which cannot change: the four
`(cherry picked from commit …)` trailers (on `6b95c9208`, `7382a4275`, `6cd7672ec` and
`6e6bf5ad2`) and `bc3ba2ca6`, which names the U3 run's tree.

| Landed | Original | Patch-id | Subject (short) |
| --- | --- | --- | --- |
| `364c06fd9` | `29f4d1077` | differs | jope-engine: grow families natively (P1 U8) |
| `6b95c9208` | `5136986fb` | differs | jope-ci: port cargo-artifact to Rust (P1 U2) |
| `7382a4275` | `1f1791025` | differs | jope-records: port the four hygiene guards (P1 U3) |
| `6cd7672ec` | `6ce8fcfbe` | differs | jope-ci: port the build-directory report to Rust (P1 U5) |
| `6e6bf5ad2` | `389898805` | differs | python: archive 703 files with no live consumer (P1 archive) |
| `4ed0e2096` | `e9511a2fc` | same | docs: ADR-0008 Rust-only stack (P1 docs) |
| `ee24eae50` | `9b810cd80` | same | docs: renumber the Rust-only ADR to ADR-0006 |
| `037b8f885` | `9e8aacedc` | same | jope-ci: replay the captured Python outputs (U2, U5) |
| `0b213a36e` | `4b4018dff` | same | jope-ci, jope-engine, jope-records: clear clippy findings |
| `bc3ba2ca6` | `df9c78ae3` | same | jope-records: replay the captured Python outputs (U3) |
| `575c55ea6` | `2200bc872` | same | docs, gitleaks: stale references to archived Python (PY-P2) |

Landing-only: `f7a19392c` (the W01 and W09 records name the Rust coverage validator) and
`f0b1402ec` (the capability matrix recounts the jope-records tests).

"Patch-id" compares `git patch-id --stable` of the two commits. Each pair changes the same paths
with the same status. The tree proof (`p1-landing-post-20260927T170033Z`, `step-treeproof.log`)
finds that 8 of the 829 paths the eleven commits touch differ between the original tip and
`f0b1402ec`, all of them paths main also changed: 5 equal the clean three-way merge, and the
jope-records `lib.rs` and `main.rs` and the capability matrix were resolved by hand.

The originals stay on `rr2/python-p1`, locally and on `origin`. Independent of any ref, all
eleven are in the custody bundle `p1-landing-preserve-20260927T172032Z/rr2-python-p1.bundle`
(406474 bytes, SHA-256 `750c4ce1abf56023380028b32717b321a4c3b674464b59c242d24a3040211cdd`,
prerequisite `8a55c4fef`, which main contains). Restored into a bare clone of the landing branch
alone, the state once that branch is deleted and gc has run, every original came back with its
tree, the four trailers resolved, `git fsck --connectivity-only` passed, and `git show` of the
U2 original's `deploy/ci/coverage.json` gave blob `5e78ee4ba`, the fixture's `mapping.json`
(`log.txt` beside the bundle). To restore into any clone that has main:

```
git fetch <bundle> refs/heads/rr2/python-p1:refs/preserve/rr2-python-p1
```
