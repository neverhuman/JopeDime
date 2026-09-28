# PY-P1 — Python removal, phase P1

Written on branch `rr2/python-p1` (base `8a55c4fef`) and landed through `rr2/python-p1-landing`,
a replay onto main at `cee1fc8bd`, then rebased onto main at `f46a9048f` (after #116). This
repository merges pull requests only by rebase or squash, and either gives every commit a new SHA
on main, so this record names P1's own commits by role (the U2 commit, the ADR-0008 commit, …).
The commit map at the end gives each one's exact subject, which a rebase merge keeps and a squash
merge lists in its message, so `git log -F --grep='<subject>' main` finds it after either; the map
also gives each one's original and its SHA before the rebase. Owner ruling 2026-09-26: the stack
is Rust/Vite/TypeScript/React; Python remains only for GP-emitted code and its harness. Live tools
are ported with differential tests, unused code is archived. Evidence runs live under
`/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/py-p1-*`
and, for the landing, `p1-landing-*` and `p1-rebase-*` (identity, diff, command and log per run).

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

- Removed with `git rm`: 703 files (694 `.py` + 9 systemd templates that only ran archived U12–U14
  code; per-template decisions in "Follow-up — the nine archived unit templates"). By category: 625 A, 37 L and 32 T (the U12–U14 units), 9 templates. By unit: 103
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
  requires. It was first committed as ADR-0008 (the brief's name, in the ADR-0008 commit) and
  renumbered in a follow-up commit (the renumbering commit; the commit map gives both subjects);
  every reference moved with it, including the `ruling` string of the archive manifest. The
  custody copy of that manifest was replaced by the committed one (only
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
  pairs with the landed U2 commit, plus the run's working changes: the jope-records
  `lib.rs`/`main.rs` edits and four new, untracked modules), recovered from an export of it
  (`git archive`, the run's `diff.patch`, the four modules; not a clone or worktree). The landed
  U2 commit does not hold that tree (its `deploy/ci/coverage.json` also has the entries main
  gained up to `f46a9048f`), so the committed fixture is the record of the inputs:
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
  changed. **Resolved in P1** (see "Follow-up — the nine archived unit templates" below): the
  name is dropped, and "reports as absent" was wrong — `systemctl show` answers for a missing
  unit with `ActiveState=inactive`, `SubState=dead` and exit status 0, a stopped service.
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
- **PY-P2-7 dashboard receipts with no writer.** `crates/jope-engine/src/dashboard.rs`
  (`campaign`) reads receipts whose only writers were Python. `git grep` of each path finds no
  writer on the branch.
  - Three were written by programs of the archived templates: `runs/report-progress.json`
    (`watch_campaign_reports.py`, `jope-parallel-reports`), `runs/proposal-library.json`
    (`opportunistic_proposals.py`, v44) and `runs/pure-proposal-library.json`
    (`opportunistic_proposals_v2.py`, v53). The archived `snapshot_dashboard.py` only copied them
    into a new snapshot directory given by `--output`. **Resolved in P1** (see "Follow-up — the nine
    archived unit templates"): a saved receipt is shown as history with its saved status, and
    with no receipt no row is shown. The "Matched-budget reports" row used to be shown always and
    said "Completed-cycle reporting receipt pending".
  - The reports watcher also published each cycle's `report-v1/` (it ran a state-root
    `tools/report_cycle.py`; the repository copy is archived, U13). Saved reports stay readable.
    Without one, the reports list falls back to the cycle's `comparisons.json`, which Rust writes
    (`pipeline.rs`), so no view waits on it.
  - Not written by any of the nine templates: `runs/proposal-library-admission.json`
    (`deferred_proposal_admission.py`, archived, category A; no template ran it), and
    `runs/continuous-progress.json` and `runs/frozen-monitoring.json` (`continuous_learning.py`,
    U14; the only template naming it, `controller-next`, named a state-root copy and was held).
    `learning_history.rs:230` also reads `continuous-progress.json`.
  - Proposed: P2 decides, receipt by receipt, whether the dashboard keeps showing it as history
    or drops it. If the owner wants a producer back, it is ported to Rust.

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

## Follow-up — the nine archived unit templates (PR #117 `rust-check`)

Hosted CI failed `rust-check` on PR #117: `cargo test -p jope-dime-regressor --test
systemd_unit_paths`, `every_tracked_unit_is_a_template`, "expected the 39 + 22 + 1 units as
templates, found 53" (the guard was `count >= 60`). Reproduced locally in
`p1-units-pre-20260927T205438Z` (same assertion, 3 passed, 1 failed). The archive commit removed
9 of the 62 templates: `engine` 22 → 15, `selfplay` 39 → 37, `pending` 1 → 1.

Each template was decided on two kinds of consumer: what renders, installs, enables, starts or
copies the template, and what reads the files its program wrote. For the first kind, the ones
common to all nine: `deploy/systemd/render.sh`, `jope-units-render` and `just render-units`
render whatever `*.in` files exist and name no unit; `jope_selfplay_bundle` renders only
`deploy/systemd/selfplay/`; no installer, workflow under `.github/`, `justfile` recipe or runbook
names any of the nine (`git grep` of each name over the branch). The remaining mentions are
archive records (the manifest, the capability matrix's dropped rows, the parity fixtures), the
frozen kernel-v1 unit files (`deploy/kernel-v1/source/systemd/*.service`, separate files, not
templates, not installed) and `docs/legacy`. For the second kind, every path each program writes
was searched with `git grep` over the branch. The first version of this follow-up searched
unit names only. It missed that the live dashboard reads what the reports, v44 and v53 programs
wrote, and it said those templates had no consumer, or only a history pin. A review found this.
`p1-units-pre-20260927T212741Z` (`receipt-writers.log`, `dashboard-reads.log`) shows the reads
and that nothing on the branch writes those files. The table below is corrected.

| Template (`deploy/systemd/…`) | Archive unit | What `ExecStart` ran | Consumers on the branch | Decision |
| --- | --- | --- | --- | --- |
| `engine/jope-parallel-console.service.in` | U13 | `python3 @JOPE_HOME@/python/src/jope/tools/tui-session.py` (archived, U13), which kept an observer binary running in a tmux session and wrote no files | `dashboard.rs` `live_operations` asked `systemctl show` for it; no view read that row | stays archived; name dropped from the dashboard |
| `engine/jope-parallel-controller-next.service.in` | U14 | nothing: `ExecCondition=/usr/bin/false` (held); would run a state-root copy of `continuous_learning.py` (repo source archived, U14) | none of the template: held, it never wrote anything. Its program's receipts (`runs/continuous-progress.json`, `runs/frozen-monitoring.json`) are read by `dashboard.rs` and have no writer on the branch (PY-P2-7) | stays archived |
| `engine/jope-parallel-health.service.in`, `.timer.in` | U14 | a state-root copy of `continuous_status.py` (repo source archived, U14); the timer only starts that service | none: its program wrote only `runs/continuous-health-20260909/status/`, which nothing on the branch reads | stays archived |
| `engine/jope-parallel-proposal-library-v44.service.in` | U13 | `python3 @JOPE_HOME@/python/src/jope/tools/opportunistic_proposals.py` (archived, U13) | `dashboard.rs` reads its receipt `runs/proposal-library.json` (the "Proposal seed library" row, and the GPU line when nothing else explains it); `crates/jope-engine/manifests/opportunistic-proposals-v44.sha256:4` pins the old `systemd/` file as history, and nothing checks it (PY-P2-3) | stays archived; a saved receipt shows as history (PY-P2-7) |
| `engine/jope-parallel-pure-proposal-library-v53.service.in` | U13 | `opportunistic_proposals_v2.py` (archived, U13) | `dashboard.rs` reads `runs/pure-proposal-library.json` (the "Pure policy predictions" row, and the GPU line); `opportunistic-proposals-v53.sha256:5` as above | stays archived; a saved receipt shows as history (PY-P2-7) |
| `engine/jope-parallel-reports.service.in` | U13 | `python3 @JOPE_HOME@/python/src/jope/tools/watch_campaign_reports.py` (archived, U13) | `dashboard.rs` reads `runs/report-progress.json` (the "Matched-budget reports" row) and each cycle's `report-v1/report.json` (the reports list, which falls back to the Rust-written `comparisons.json`) | stays archived; the row is shown only for a saved receipt (PY-P2-7) |
| `selfplay/jope-workspace-sync.service.in`, `.timer.in` | U12 | `python3 @JOPE_STATE@/jope-dime-mirror/controller/current/scripts/workspace_sync.py sync`, a mirror copy outside the repository (repo source and its installer archived, U12) | `jope_selfplay_bundle.rs` `HOST_INSTALLED_UNITS` removes them from the rendered bundle units: a stop-list that tolerates their absence. The program wrote the mirror; `git grep` finds `jope-dime-mirror` only in `crates/jope-campaigns/manifests/workspace-sync.v1.json` | stays archived; the exclusion is kept and now tested |

No template has a live consumer that renders, installs, enables, starts or copies it, so none
is restored, and `docs/archive/python-archive-manifest.json` and the custody tarball keep all
nine (the gate below re-checks that each `systemd-template` row's blob equals `8a55c4fef`'s,
that the path is absent from the branch, and the custody `SHA256SUMS`). Restoring the reports,
v44 or v53 template would not bring back a writer for the dashboard's receipts either, because
each one ran archived Python. So the dashboard now treats those receipts as history (Changes,
below).

Host state was read on xbabe2 only, the host this lane runs on, and xbabe2 is not an engine
host. `systemctl show` there gives `LoadState=not-found` for the seven archived engine names, in
the system and the user manager (`host-installed-units.log` in `p1-units-pre-20260927T205438Z`).
It gives the same for the tracked live engine units `jope-parallel-campaign`, `jope-parallel-gp`
and `jope-parallel-monitor`. `systemctl list-unit-files 'jope-*'` lists no `jope-parallel-*`
unit in either manager (`host-engine-units.log` in `p1-units-pre-20260927T212741Z`). So the
absence of the archived engine units on xbabe2 says nothing about an engine host.

**Not checked: whether any engine host still has them installed.** No engine host was checked.
This lane has no access to one: it is CPU only and has no SSH. The repository does not name the
engine host either. An engine host is one that runs `jope-parallel-campaign` and
`jope-parallel-gp`, and it may still have rendered copies of `jope-parallel-health.timer`,
`jope-parallel-reports.service` or the other five installed. The check for its operator is
read-only: in the system and user managers, for each of the seven names, run
`systemctl show <name>.service -p LoadState -p UnitFileState -p ActiveState -p FragmentPath`
(and `<name>.timer` for `health`), plus `systemctl list-unit-files 'jope-parallel-*'`. A copy
found there runs archived Python, or for `controller-next` nothing. Disabling and removing it
is the owner's call.

On xbabe2 the two workspace-sync units are still installed as system units by the archived
installer (`/etc/systemd/system`, timer `enabled`), but they are dead: the timer is `inactive`,
last triggered 2026-09-14 14:40:00 UTC with no next elapse, and the service last exited with
status 2 (`workspace-sync-installed.log`). They run the mirror's copy, not a repository file, so
the archive does not change them. Retiring them needs root and is left to the owner.

Changes:

- `crates/jope-campaigns/tests/systemd_unit_paths.rs`: `every_tracked_unit_is_a_template` now
  checks a floor per directory, `engine` 15 + `selfplay` 37 + `pending` 1 = 53, and a total
  floor of 53, and its message prints the found breakdown. Per directory, so a template added in
  one cannot hide one lost in another. The stray-rendered-unit check is unchanged.
- `crates/jope-campaigns/src/bin/jope_selfplay_bundle.rs`: new test
  `host_installed_units_are_optional_and_never_rendered_into_the_bundle` renders a fixture
  checkout without the workspace-sync templates (renders), then with restored copies (still
  excluded). It characterises existing behaviour, so it has no failing reproduction.
- `crates/jope-engine/src/dashboard.rs`: the observed units are `OBSERVED_SERVICES`
  (`jope-parallel-campaign`, `jope-parallel-gp`); `jope-parallel-console` is dropped, since for a
  unit that is not installed `systemctl show` returns `ActiveState=inactive`, `SubState=dead`
  and exit status 0 (`systemctl-absent-unit.log`), which reads as a stopped service. The new
  unit test `every_observed_service_has_a_tracked_unit_template` requires a tracked
  `deploy/systemd/engine/<name>.service.in` for every observed name, with a negative control.
  Reproduction `p1-units-pre-20260927T205650Z`: with the old list the test fails on
  `["jope-parallel-console"]`.
- `crates/jope-engine/src/dashboard.rs`, after the review: `campaign` says that nothing in the
  workspace writes `runs/report-progress.json`, `runs/proposal-library.json` or
  `runs/pure-proposal-library.json`. The "Matched-budget reports" row is now shown only when
  its receipt exists, as the two library rows already were. With no receipt it used to show
  `QUEUED · awaiting manifest · Completed-cycle reporting receipt pending`, a wait that no
  writer can end. The new test `receipts_of_retired_producers_show_history_not_pending_work`
  in `tests/dashboard.rs` runs `campaign` on a real state root. With no receipt, none of the
  three rows is shown and nothing says a receipt is pending. With the receipts a stopped
  producer leaves (a day old, the libraries still `running`), each row is `STALE` with its
  saved counts, never `RUNNING`, and neither "Building saved proposal seed library on eight
  GPUs" nor "Saving pure policy predictions on eight GPUs" is shown. Finished receipts read
  `DONE`, `SAVED` and `STOPPED`. Reproduction `p1-units-pre-20260927T212741Z`
  (`dashboard-repro.log`): against the old row the test fails on "Matched-budget reports
  shown without a receipt", and the row reads `QUEUED  awaiting manifest  Completed-cycle
  reporting receipt pending`. The existing tests for these rows with live receipts are
  unchanged.

Runs: `p1-units-pre-20260927T205438Z` (the CI failure reproduced, template counts at main and
here, the host's unit state), `p1-units-pre-20260927T205650Z` (the dashboard reproduction),
`p1-units-post-20260927T205744Z` (the three tests pass; negative control: with
`engine/jope-parallel-gp.service.in` removed the guard fails with "found engine 14 + selfplay
37 + pending 1 = 52", and the file was restored with its blob unchanged) and
`p1-units-post-gate-20260927T210016Z`, which runs the hosted `rust-check` and `rust-test` lanes
locally exactly as `.github/workflows/ci.yml` does (`deploy/ci/receipt.py` around
`deploy/ci/fast-lane.sh <lane>`) together with the landing gate. It ran on the branch before
this paragraph's results were added (the amendment changes only this file, checked by
`p1-units-post-docs-*`), and every step passed: the `rust-check` receipt (`cargo check
--workspace --all-targets --locked`, then `systemd_unit_paths` 4 of 4, then the MSRV check);
the `rust-test` receipt (the 17 product crates under nextest: 1593 collected, 1570 run and
passed, 23 skipped, 0 failed); `cargo test -p jope-dime-regressor` over all its targets (997
passed, 11 ignored, 0 failed); the dashboard tests (the new unit test, `tests/dashboard.rs` 26
passed and 1 ignored, `tests/learning_history.rs` 16 passed); the fence suites; `jope-ci` (21
passed, 3 ignored) and `jope-records` (113 passed, 1 ignored), both Python-parity suites
included; fmt, `git diff --check`, `coverage validate` (0 unassigned or invalid),
`verify-test-inventory` and the other records checks, `parity-matrix.sh --no-build`,
`deploy/ci/check.sh` (gitleaks shimmed) and the unshimmed `secret-scan.sh`; and the nine
manifest rows and the custody `SHA256SUMS` above.

Runs for the review follow-up:
- `p1-units-pre-20260927T212741Z`: which files write the dashboard's receipts, on the branch
  and at `8a55c4fef`; the dashboard's reads; xbabe2's engine units; and the dashboard
  reproduction.
- `p1-units-post-20260927T212904Z`: fmt, after `rustfmt` of the new test; the new test;
  `tests/dashboard.rs`, 27 passed and 1 ignored; `tests/learning_history.rs`, 16 passed; and the
  observed-service unit test.
- `p1-units-post-gate-20260927T213133Z`: the same gate script, with its scope step set to these
  two commits. It ran on the branch before this entry was added; the amendment changes only
  this file, which `p1-units-post-docs-*` checks. Every step passed:
  - the hosted lanes run locally as `.github/workflows/ci.yml` runs them. The `rust-check`
    receipt passed 4 of 4. The `rust-test` receipt collected 1593, ran and passed 1570, skipped
    23 and failed 0. The new test is not in that lane's package set
    (`deploy/ci/rust-test-scope.sh`), so the count is unchanged, and the gate ran it directly;
  - `cargo test -p jope-dime-regressor`: 997 passed, 11 ignored;
  - `tests/dashboard.rs` with `tests/learning_history.rs`: 43 passed, 1 ignored, the new test
    included;
  - the fence suites;
  - `jope-ci`: 21 passed, 3 ignored. `jope-records`: 113 passed, 1 ignored;
  - fmt, `git diff --check`, `coverage validate` (0 unassigned or invalid),
    `verify-test-inventory`, the records checks, `parity-matrix.sh --no-build`,
    `deploy/ci/check.sh` (gitleaks shimmed) and the unshimmed `secret-scan.sh`;
  - the archive manifest and custody checks.

## Follow-up — fail-closed malformed inputs (PR #117 review)

A review of PR #117 found that the P1 ports could pass inputs the retired Python failed on. The
case it gave: `coverage validate` dropped any lane or disposition that was not a string, where
`coverage.py` printed `unknown lane None for <key>` and exited 1. Reproduction
`p1-failclosed-pre-20260927T221449Z` ran the original `coverage.py` (git bytes of `f46a9048f`)
and the port at `a997ab825` on the same live tree and the same 12 mutated maps. The Python exited
1 on all 12. The port exited 0 on 8 of them (`comparison.txt`).

**Contract.** It covers every malformed-input class of every P1 port: `coverage validate` and
`discover`, `rust-scope-check`, `parity-check`, `readme-benchmarks`, `jope-ci cargo-artifact`
(`select`, `target-dir`, `nextest-config`, `build`) and `jope-ci build-report report`.

- **Never more permissive.** Wherever the Python exited non-zero, by a refusal or a crash, the
  port exits non-zero and prints no success report.
- **Explicit messages.** Where the Python printed one, the port prints the identical line. This
  covers a report line such as `unknown lane None for <key>`, a `SystemExit` message, a caught
  `cargo-artifact: ...` line and a failure receipt. `jope-records` prints its refusals after
  `jope-records: `.
- **Crashes.** Where the Python only crashed (`TypeError`, `KeyError`, `AttributeError`,
  `IndexError`, or a library `ValueError`), the port refuses with an error naming the field and
  the offending type. It does not reproduce CPython's exception text, and no comment says it
  does. The wording is modelled on the exception so a reader can recognise the case
  (`jope_records::py::exception`).
- **Valid input.** Output stays byte-identical. The existing parity fixtures are unchanged and
  still pass.

These bullets stated the goal; the round-one fixtures did not establish all of it. The
differential review of round two (next section) found inputs where the port was still more
permissive, explicit lines that differed, and valid inputs whose output differed. What holds
after round two, with every remaining difference named, is under **Contract after round two**
there.

**Findings and what changed.** The previous agent's uncommitted edits were kept and completed:

1. Nothing was committed: no negative fixtures, and doc comments described fixtures that did not
   exist. The fixtures and tests below now exist, and the doc comments describe them.
2. `cargo-artifact select` compared ids with serde_json's `==`, which is stricter than Python's
   `==` in one direction and looser in the other:
   - `1` and `1.0` compared unequal;
   - `2^64` and `2^64 + 1` compared equal, because both become the same `f64`.

   So `select` could find one package or executable where the Python found two, or one where it
   found none. The metadata, `select`'s files and every message line are now read with
   `jope_records::pyjson`: Python's JSON reading, and its decoder messages for malformed JSON.
   Membership and `package_id` use Python's `==` (`1 == 1.0 == True`, `[1] == [1.0]`). `success`
   must be JSON `true` (`is True`).
3. `cargo fmt --all --check` failed. The code is now formatted, and clippy passes with
   `-D warnings`.
4. Refusals claimed to end with CPython's exception line. The claims now say what the messages
   are. `Path(value)` on a value that is not a string now refuses with
   `exception::path_argument`, and `os.path` with `exception::fspath`.
5. `parity::bins_from_metadata` gave a falsy `src_path` that was not a string (`0`, `false`, `[]`)
   the type error. `os.path.relpath` refused any falsy value first (`no path specified`), and the
   port now does the same, before the type check.
6. `coverage::discover_from` gave every failed `workspace_members` membership one generic message.
   `py::contains_or_raise` and `exception::membership` now name which operand had the wrong type:
   a dict with an unhashable id, a string with an id that is not a string, or a container that is
   neither.

Nits:
- `cargo-artifact build` replaced bytes that are not UTF-8. It now refuses them, as
  `text=True` did.
- `coverage` gives `.` for the root itself and drops repeated separators and `.` components, as
  `str(Path.relative_to)` did.
- A `target_directory` that is not a string refuses, naming the field and its type (it used to
  be reported as a missing key).
- The README renderer's `value[key]` on a value that is not a dict names the value's type (it
  used to say "missing"). Its number and CSV refusals name the field.

**Negative fixtures.** They were captured in `p1-failclosed-capture-20260928T020533Z` by
`capture.sh`, kept in that run. The script is bash and jq only.
- The original tools were run from `f46a9048f`'s git bytes, unmodified and never committed, in
  scratch roots. They ran on variants of the committed parity inputs. `cargo` and `git` shims
  served the committed metadata and file list.
- A variant is an `ed` script against the committed base, with the SHA-256 of the file the
  Python read. The small cargo-artifact inputs are raw text instead.
- A case records the exit status, stdout, and either the explicit message or the last traceback
  line. Each fixture names the run and the command (`source`, `capture`).
- Controls:
  - the shimmed `coverage.py discover` reproduces `discovered.json` byte for byte, and its
    `validate` the committed `cov-validate` output;
  - the scope check reproduces the committed `scope` output;
  - `--write` from `README.stale.md` writes the committed `README.written.md`;
  - the parity and cargo-artifact bases pass.
- Two earlier captures are superseded: `p1-failclosed-capture-dbg*` (debugging) and
  `p1-failclosed-capture-20260928T020349Z`, whose build-report fixture still held the approved
  roots as home paths. The Python outputs of all of them are identical.

| Fixture | Cases | Python exit 0 | Explicit refusal or failing report | Crash |
| --- | ---: | ---: | ---: | ---: |
| `jope-records/…/coverage/negative-validate.json` | 55 | 5 | 20 | 30 |
| `jope-records/…/coverage/negative-discover.json` | 59 | 18 | 0 | 41 |
| `jope-records/…/scope/negative.json` (+ `scope/metadata.json`) | 17 | 1 | 2 | 14 |
| `jope-records/…/parity/negative.json` (+ `negative-metadata.json`, `negative-matrix.md`) | 32 | 7 | 1 | 24 |
| `jope-records/…/readme/negative.json` | 85 | 23 | 19 | 43 |
| `jope-ci/…/cargo-artifact/negative.json` | 50 | 13 | 19 | 18 |
| `jope-ci/…/build-report/negative.json` | 12 | 0 | 12 | 0 |
| **total** | **310** | **67** | **73** | **170** |

**Tests.** They are in existing targets only, so `deploy/ci/coverage.json` and the test inventory
are unchanged.
- `crates/jope-records/tests/python_parity.rs` has five replays:
  - `coverage_validate_refuses_every_map_the_python_refused`;
  - `coverage_discover_…`;
  - `rust_scope_check_…`;
  - `parity_check_…`;
  - `readme_benchmarks_refuses_every_evidence_the_python_refused`, which also checks the README
    bytes `--write` left.

  It also has the negative control `negative_replay_detects_a_changed_outcome`. One `compare`
  function holds every case to the contract.
- `crates/jope-ci/tests/cargo_artifact_python_parity.rs`:
  `cargo_artifact_refuses_every_input_the_python_refused` runs the real binary behind a `cargo`
  shim. It also checks what a refused `nextest-config` leaves behind; since round two that is
  what the Python left (an empty output where its crash came after `open("x")`). Its `judge`
  function has its own negative control.
- `crates/jope-ci/tests/build_report_python_parity.rs`: `config_refusals_match_the_captured_python_outputs`
  compares the exit status, stdout, the receipt and `failure.txt`. Like the existing receipt
  replay, it is ignored by default because it needs `JOPE_MAINTENANCE_SCRATCH`, and the gate runs
  it. The approved roots are `@APPROVED_ROOT_<n>@` in the fixture, because the no-home-paths lint
  scans `crates/*/tests`.
- With `JOPE_NEGATIVE_REPORT` set, the replays write one line per case: the Python's outcome and
  the port's. The gate keeps this table as `negative-report.tsv`.

**Allowed differences.** Each one is listed by name in the tests.

Stricter than the Python (it exited 0 and the port refuses):
- coverage `scope-bigint`: the port's JSON reader refuses integers beyond 64 bits;
- coverage discover `feature-deps-int` and `required-int-member`: the Python sorted a list of
  numbers;
- parity `package-name-int` and `target-name-int`: the Python sorted and printed a name that is
  not a string;
- readme `valid-rankable-float`: `format(8718.0, ",")` printed `8,718.0`;
- cargo-artifact `target-dir-int` and `target-dir-null`: the Python printed `5` and `None`.

Both refuse, with different words (both identical since round two):
- cargo-artifact `ids-beyond-64-bits`: the port refused at parse; the Python failed the package
  lookup (`found 0`);
- cargo-artifact `build-non-utf8`: Rust's decoder wording, not Python's codec wording.

**Audit table.** The table lists the classes the review and this audit found. "Before" is the
port at `a997ab825`, before the uncommitted edits. "Now" is this follow-up.

| Port | Input class | Python | Before | Now |
| --- | --- | --- | --- | --- |
| coverage validate | lane or disposition list with a null, number or bool member | `unknown lane None/123/1.5/True for <key>`, exit 1 | member dropped, **exit 0** | identical line, exit 1 |
| coverage validate | assignment that is a string or a dict | each character or key looked up | empty, **exit 0** | identical lines |
| coverage validate | assignment that is null or a number, or has a list or dict member | crash (not iterable, unhashable) | **exit 0** | refuses |
| coverage validate | `items`, `lanes`, `dispositions`, a lane, disposition or item, or the whole map not an object | crash (`AttributeError`/`TypeError`) | errors or defaults | refuses |
| coverage validate | target: default lane as a dict key, dispositions as a substring string | `non-default required features …` | **exit 0** | identical line |
| coverage validate | target: dispositions null or a number | crash (`in`) | varied | refuses |
| coverage discover | any indexed field missing or of the wrong type (`packages`, `id`, `workspace_members`, `name`, `manifest_path`, `features`, `targets`, `kind`, `src_path`, `test`, `doctest`, `required-features`) | crash | **skipped or defaulted** | refuses, naming the field and type |
| coverage discover | id `7` with member `7.0`, `True` for `1`, `[1]` for `[1.0]`; members as dict keys or a substring string | membership by Python `==`, hashing or substring | not a member | identical output |
| coverage discover | dependency or required-feature list of numbers | sorted, exit 0 | filtered | refuses (stricter) |
| rust-scope-check | `packages` or a package missing, null, a string or a list; a name missing, unhashable or not a string | crash | package **skipped** | refuses |
| parity-check | `packages`, `targets`, `kind[0]` (empty, missing, dict, null), `src_path` (missing, falsy, not a string), names missing | crash | **skipped or defaulted, exit 0** | refuses |
| parity-check | package or target name not a string | sorted and printed, exit 0 | skipped | refuses (stricter) |
| readme | track, snapshots, corrections, bench baseline, CSV or reports of the wrong shape (43 crash classes: not an object, not iterable, unhashable, missing key, `+` of `int` and `str`, `.upper`/`.split` on a number, `[:9]` on a number or dict, format of a string or `None`, unsortable ids, unsafe command tokens) | crash | many **rendered** (`screen None`, digits, `name=`) | refuses |
| readme | a hidden `.bench-report.json` | counted, so the report count disagrees | **ignored** | counted, identical refusal |
| cargo-artifact select | ids equal only under Python `==`, ids beyond 64 bits | found 2, 1 or 0 packages or executables | serde `==`, so the count differs | Python `==` (identical); beyond 64 bits refused at parse |
| cargo-artifact select | a package without a name or id; `workspace_members` missing with no name match | caught `KeyError` (`'name'`, `'id'`) or `found 0` | skipped, success | identical line |
| cargo-artifact select | message `target` or `profile` null or a list; `kind` null or a number; `executable` not a string | crash | **skipped, success** | refuses |
| cargo-artifact select | `kind` a dict or a string containing `bin` | counted | not counted | counted |
| cargo-artifact build | Cargo output that is not UTF-8 | caught `UnicodeDecodeError`, exit 1 | replaced, **continued** | refuses |
| cargo-artifact target-dir | `target_directory` not a string | printed `str(value)`, exit 0 | refused | refuses (stricter), naming the type |
| build-report report | configuration not an object, `stale_hours` or `cargo_roots` wrong, `measurement_roots` not a list | failure receipt, exit 1 | the same | the same (now replayed) |

**Runs.**
- `p1-failclosed-pre-20260927T221449Z`: the reproduction above.
- `p1-failclosed-capture-20260928T020533Z`: the fixtures.
- `p1-failclosed-commits-20260928T021000Z`: the four commits. Each commit's own tree was checked
  with later changes stashed: fmt, `git diff --check`, and the tests of the crates it changes.
  The first commit also ran the workspace check, the second clippy, and the third the ignored
  build-report replay.
- `p1-failclosed-post-20260928T021102Z`: `pre.sh` rerun on the committed head (`468f9ff35`).
  All 14 cases match: the same exit status and the same stdout as the Python. Before the fix,
  8 exited 0 where the Python exited 1.
- `p1-failclosed-gate-20260928T021114Z`: the full gate on `468f9ff35`, with every step green.
  - The hosted lanes, run locally as `.github/workflows/ci.yml` runs them:
    - `rust-check` receipt: 4 of 4 passed;
    - `rust-test` receipt: 1601 collected, 1577 run and passed, 24 skipped, 0 failed. That is
      seven more than before: the new replays in the product scope.
  - fmt, `git diff --check` over `f46a9048f..HEAD` and over the tree, and `jq` on every new
    fixture.
  - `cargo check --workspace --all-targets --locked`, and clippy on `jope-ci` and `jope-records`
    with `-D warnings`.
  - `jope-records`: 119 passed and 1 ignored.
  - `jope-ci` with `--include-ignored`: 26 passed. The build-report runners and both receipt
    replays ran with custody scratch.
  - The fence suites: 41 passed; `gp::fit_contract_tests` 44 passed and 3 ignored.
  - `coverage validate` (0 unassigned or invalid), `verify-test-inventory` and the other
    records checks.
  - The live guards on this tree: `rust-scope-check`, `readme-benchmarks --check` and
    `coverage discover`.
  - `parity-matrix.sh --no-build`, `deploy/ci/check.sh` (gitleaks shimmed), the unshimmed
    `secret-scan.sh` (no leaks) and a clean final status.
  - `negative-report.tsv` has one line for each of the 298 replayed `jope-records` and
    cargo-artifact cases.

  This paragraph was added after that gate. It changes only this file, which
  `p1-failclosed-post-docs-*` checks.

## Follow-up — differential review, round two (PR #117)

A differential checker ran the original tools (git bytes of `f46a9048f`) and the ports at
`ef8d31c0b` side by side on 140 crafted inputs. Its table is kept in
`p1-failclosed-r2-pre-20260928T024938Z` (`results.tsv`, with the harness): 30 cases where the port
exited 0 and the Python did not, 6 with different output on success, 14 stricter, 71 failing on
both sides and 19 identical. It raised 15 findings.

**Dispositions.** Every "Case" below is a captured fixture case replayed by a test.

| # | Finding | Disposition | Change | Cases |
| ---: | --- | --- | --- | --- |
| 1 | `cargo metadata` stdout with a BOM or in UTF-16 accepted (coverage, scope, parity) | fixed | Cargo's stdout is decoded as `text=True` decoded it (strict UTF-8, universal newlines) and read as a `str`, which refuses a BOM (`coverage::metadata_from_stdout`, `scope_check::check`, `parity::cargo_bins`) | `cov-*-meta-utf8-bom`, `cov-discover-meta-utf16`, `cov-discover-meta-not-utf8`, `scope-meta-*`, `parity-meta-*` |
| 2 | stderr that is not UTF-8 ignored | fixed | `py::text_streams` decodes stdout, then stderr, before the exit status is read (the script runs and both Cargo calls); the script's stdout is no longer decoded lossily | `scope-script-stderr-not-utf8`, `scope-cargo-*stderr-not-utf8`, `parity-cargo-*stderr-not-utf8` |
| 3 | Rust `\s`/`\w` are not Python's | fixed | `py::regex` compiles each pattern with Python 3.12's classes: `\s` adds U+001C..U+001F, `\w` is `L*`, `N*` or `_`, `\d` is `Nd`, all restricted to Unicode 15.0 (`Age=V15_0`) | 18 `cov-*-wf-*`, `parity-help-name-*`, `parity-clap-variant-unit-sep`, `parity-clap-name-attr-unit-sep`, `parity-jope-py-*` |
| 4 | `NaN in [NaN]` false | fixed | `pyjson::eq_or_identical` (identity first, as `PyObject_RichCompareBool`) for `in` and list/dict element equality; a top-level `NaN == NaN` stays false | `cov-*-meta-id-nan`, cargo-artifact `id-nan`, readme `report-rmse-nan-tie` |
| 5 | README: `NaN` for `nan`, Rust's sort and `max` | fixed | `fixed()` prints `nan`/`inf`/`-inf`; `py_sorted` reproduces CPython 3.12's `list.sort` for fewer than 64 values holding a NaN (one `count_run`, then `binarysort`) and refuses 64 or more; `py_max` keeps the first value nothing exceeds; `py_sum` is 3.12's `sum` (exact integers, Neumaier-compensated floats, integers after a float added plainly) | readme `bench-*-nan`, `bench-mean-inf`, `report-produce-*` (7), `csv-average-rank-nan` |
| 6 | lossy file names collapse (parity) | fixed; one listed difference | a Python tool or systemd unit whose path is not UTF-8 is refused, naming it; two names can no longer collapse into one | `parity-tool-name-not-utf8-*`, `parity-unit-name-not-utf8` |
| 7 | `nextest-config` accepts a leading BOM | fixed | refused with `tomllib`'s `Invalid statement (at line 1, column 1)` | `toml-utf8-bom` |
| 8 | `build-report scan` fails on names that are not UTF-8 | fixed | `walk` keeps names as bytes, `du` gets the path's bytes and its output is decoded strictly (CPython's codec message), an `OSError` names the path surrogate-escaped | `file-name-not-utf8-in-candidate`, `dir-name-not-utf8-outside-candidate`, `candidate-path-not-utf8` |
| 9 | `pending-WP<N>` and the tally overflow or refuse Unicode digits | fixed | `py::decimal_digits`: `int()` of Unicode decimal digits, unbounded | `parity-pending-wp-*` (3), `parity-tally-*` (3) |
| 10 | steps run in a different order | fixed | `a == b == c` short-circuits; `parity::check` reads in `cmd_check`'s order (matrix, row-target `Path.exists`, which raises on errors other than a missing path, clap enum, `--help`, the rest); `nextest-config` reads, parses, refuses a store, creates the output, then reads the target, leaving the empty file the Python left | readme `scores-sha-short-circuit`, `parity-enum-missing-help-missing`, `parity-target-name-too-long`, `store-and-*-target`, `int-target*` |
| 11 | `repr` escapes too little | fixed | `pyjson::str_repr` uses CPython 3.12's `isprintable` (`C*` and `Z*` of Unicode 15.0 escaped); `jope_ci::py::repr` delegates to it | `package-*` (5), `scope-emitted-repr-*`, readme `snapshot-*` (3) |
| 12 | more explicit lines differ | (1), (2), (4) fixed; (3) rejected | (1) `--path-map` names `str(PurePath(container))`; (2) `returncode` is minus the signal (`exit -9`; `target-dir` exits 247 as `sys.exit(-9)` did); (4) cargo-artifact reads integers of any size (`pyjson::parse_*_big`, `Value::BigInt`) | `path-map-*`, `scope-script-killed`, `metadata-killed`, `build-killed`, `ids-beyond-64-bits*` |
| 13 | caught library messages differ (should-fix) | fixed except `tomllib` | `[Errno N] <strerror>: '<str(Path)>'`; CPython's UTF-8 codec messages; JSON positions after universal newlines, a `\uXXXX` error at the `u`, a bad escape after a high surrogate, a lone surrogate read on; `realpath(strict=True)` naming the first missing component | `meta-*`, `messages-*`, `source-*`, `output-exists`, `msg-*` (5), `root-missing-nested`, `root-under-a-file`, `config-not-utf8`, `config-truncated-utf8` |
| 14 | the contract bullets overstate (should-fix) | fixed | the round-one bullets now say they were the goal; **Contract after round two** below states what holds | — |
| 15 | a CRLF `main.rs` fails (should-fix) | fixed | `main.rs`, `cli.py` and the fallback's `Cargo.toml` are read as `read_text(errors="replace")` read them (`py::read_text_replace`, universal newlines) | `parity-clap-crlf` |

Rejected, with the reason:
- **12 (3), a configuration holding a lone surrogate** (UTF-8 bytes `ED A0 80`, or `\ud800`). The
  Python decoded it (`surrogatepass`), wrote it into the receipt as `\ud800` and then refused the
  roots. The port's JSON strings cannot hold an unpaired surrogate, so it refuses the
  configuration when it reads it: a failure receipt and exit status 1 on both sides, with
  different words. Matching the Python would need surrogate-capable strings in every JSON value
  and key the port handles.
- **13, `tomllib` messages.** The port reads TOML with the `toml` crate and prints its message.
  Reproducing `tomllib`'s wording would mean porting its parser. Both refuse with exit status 1.

Found here as well and fixed: `target-dir` prints the newline of an empty directory
(`target-dir-empty`); `build` decodes Cargo's stream before looking at its exit status
(`build-non-utf8-and-failed`); `git ls-files` output is decoded strictly (`cov-discover-git-not-utf8`);
the README's shell-safe token check matches before a final newline, as `$` did
(`report-adapter-token-newline`); the product scope is checked before the deferred one runs
(`scope-product-bad-deferred-fails`); and the fallback warning prints `CalledProcessError`'s text,
signals included (`parity-cargo-fails`, `parity-cargo-killed`).

**Contract after round two.** It covers the same ports as round one.
- **Never more permissive.** Wherever the Python exited non-zero, the port exits non-zero and
  prints no success report. No case of the reviewer's harness, rerun unchanged on the fixed
  ports, is more permissive, and none prints different output on success.
- **Explicit messages.** Where the Python printed a line of its own (a report line, a
  `SystemExit` message, a `cargo-artifact:` refusal, a failure receipt), or a library message it
  caught and printed (`OSError`, `UnicodeDecodeError`, `JSONDecodeError`), the port prints the
  identical line, `jope-records` after `jope-records: `. The exceptions are listed below.
- **Crashes.** Unchanged from round one: the port refuses with an error naming the field and the
  offending type. It does not reproduce CPython's exception text, and nothing here claims it
  does.
- **Valid input.** Byte-identical on every fixture, old and new, apart from the stricter cases
  listed below.

Remaining differences, each named in the tests:
- Stricter (the Python exited 0): round one's list; a JSON nesting deeper than 512 (Python
  allowed about 1000); a lone surrogate escape in JSON; a TOML integer beyond 64 bits (the `toml`
  crate); a README render over 64 or more reports holding a NaN; `rankable_rows` of `true`
  (formatted as `1` by the Python; the same class as round one's `valid-rankable-float`).
- Both fail, with different words: `tomllib`'s messages (`toml-invalid-value`); a configuration
  with a lone surrogate (`config-utf8-encoded-surrogate`, `config-surrogate-escape`); a parity
  report naming a path that is not UTF-8, which the Python printed with its raw bytes under the
  capture's `C.UTF-8` locale (whose stdout handler is `surrogateescape`) and crashed on under
  other UTF-8 locales (`parity-tool-name-not-utf8-decoy`, `parity-unit-name-not-utf8`); argparse
  versus clap usage text for a bad `--stale`.
- Python 3.12 runs on Unicode 15.0; the `regex` crate has 16.0 tables. The Python classes and
  `isprintable` are built from them restricted to `Age=V15_0`, so the two agree on every code
  point.

**Fixtures.** Captured in `p1-failclosed-r2-capture-20260928T031831Z` by `cap2.sh`, kept in that
run: bash and jq only, the original tools from `f46a9048f`'s git bytes in scratch roots. Each
case holds its complete input: a tree, the text or bytes a `cargo` (and `git`) shim prints, or
the files and arguments of a cargo-artifact call. The debugging runs
`p1-failclosed-r2-capture-dbg1` and `-dbg2` recorded identical cases.

| Fixture | Cases | Python exit 0 | Explicit refusal or failing report | Crash |
| --- | ---: | ---: | ---: | ---: |
| `jope-records/…/process.json` | 65 | 18 | 22 | 25 |
| `jope-records/…/readme/negative-r2.json` | 18 | 14 | 4 | 0 |
| `jope-ci/…/cargo-artifact/negative-r2.json` | 40 | 5 | 34 | 1 |
| `jope-ci/…/build-report/negative-r2.json` | 10 | 3 | 7 | 0 |
| **total** | **133** | **40** | **67** | **26** |

**Tests.** New test functions in existing targets, so `deploy/ci/coverage.json` and the test
inventory are unchanged.
- `jope-records` `python_parity.rs`: `process_cases_match_the_captured_python` runs the binary on
  each `process.json` case, with a negative control; `readme_benchmarks_matches_round_two`
  replays the README cases through the round-one `compare`.
- `jope-ci` `cargo_artifact_python_parity.rs`: `cargo_artifact_matches_round_two`, which also
  compares the output file each case left. The round-one replays now require the Python's exact
  words for `nextest-exists`, `ids-beyond-64-bits` and `build-non-utf8`, and the output file the
  Python left.
- `jope-ci` `build_report_python_parity.rs`: `scans_match_round_two`, and
  `config_bytes_match_round_two`, ignored by default like the other receipt replays (it needs
  `JOPE_MAINTENANCE_SCRATCH`).
- Unit tests for the new helpers in `py.rs`, `pyjson.rs`, `parity.rs` and `jope_ci::py`.

**Runs.**
- `p1-failclosed-r2-pre-20260928T024938Z`: the reviewer's table at `ef8d31c0b`.
- `p1-failclosed-r2-capture-20260928T031831Z`: the fixtures.
- `p1-failclosed-r2-commits-20260928T032104Z`: the two code commits. The first commit's own tree
  was checked with the second's changes stashed: fmt, `git diff --check`, and the `jope-records`
  and `jope-ci` tests.
- `p1-failclosed-r2-gate-20260928T032600Z`: the full gate on the head (this file's commit): the
  hosted `rust-check` and `rust-test` lanes run locally as `.github/workflows/ci.yml` runs them,
  fmt, whitespace, the workspace check, clippy with `-D warnings`, both crates' tests (the
  ignored receipt replays included), the fences, the records checks, the parity matrix,
  `deploy/ci/check.sh`, the secret scan, and the reviewer's harness rerun unchanged except for
  its scratch paths (`results.tsv`, `summary.txt`): no case more permissive and none different
  on success; its stricter and both-failing rows are the ones listed above.

## Follow-up — review round three

Independent review 5333747872 of PR #117 (REQUEST_CHANGES on `e8640b7b0`, by `neverhumanbot`)
ran the original tools (git bytes of `f46a9048f`) and the ports side by side on 17 more invalid
inputs. Both refused 13; four were accepted only by the port, from two root causes. Its receipts
are in `independent-review-pr117-e8640b7b0` (`spot-results.json`, `guard-spot-results.json`,
`review-notes.md`). That refutes round two's "never more permissive" for these inputs; what holds
now is under **Contract after round three** below.

| # | Finding | Root cause | Disposition | Change | Cases |
| ---: | --- | --- | --- | --- | --- |
| 1 | `select --path-map /r /r` printed `/r/demo` for an executable `//r/demo`; the Python refused (`'//r/demo' is not in the subpath of '/r'`) | pathlib, like POSIX, keeps exactly two leading slashes as a root of their own (three or more are `/`) and compares paths as spelled after parsing; `jope_ci::py::pure_path` collapsed `//` through Rust's `Path` components | fixed, with the class | `jope_records::py::PosixPath::relative_to` (Python 3.12, `walk_up=False`), `joinpath` and `parent`; `select` maps `Path(host) / path.relative_to(container)` on them and prints `str()` of the path; `pure_path` is removed | cargo-artifact `negative-r3-paths.json` (15) |
| 2 | a 4301-digit integer in an unused metadata field, and a negative one in a Cargo message, accepted by `select` | Python 3.12 converts at most `sys.get_int_max_str_digits()` = 4300 decimal digits (`ValueError: Exceeds the limit (4300 digits) for integer string conversion: value has N digits; ...`), and `json.loads` converts every integer literal as it scans it | fixed, with the class | `py::check_int_digits` (the sign is not a digit, leading zeros are); `pyjson` checks every integer literal, in the 64-bit and the big-integer readers, with Python's message and no position | cargo-artifact `negative-r3-digits.json` (14) |
| 3 | the covering-test tally prefixed with 4301 zeros accepted by `parity-check` | the same limit, on `int(m.group(1))` | fixed, with the class | `py::decimal_int` (each Unicode digit counts once, leading zeros included) for the tally and for `pending-WP<n>`; `parity::evaluate` returns a `Result`; classifying a status converts nothing, as `PENDING_RE.match` did not | `process-r3-digits.json` (15) |

**The two classes, swept.** Every place a P1 port parses an integer from text:
- `pyjson` (all JSON the ports read: cargo-artifact's metadata, `select`'s files, every Cargo
  message line and `target-dir`'s metadata; build-report's configuration; coverage's metadata and
  map; scope's and parity's metadata; the README evidence): fixed centrally. The 64-bit reader
  already refused every integer outside 64 bits (all of 20 digits or more); past 4300 digits it
  now refuses in Python's words.
- `py::decimal_int` (parity's tally and pending number): fixed.
- `py::int` (README CSV cells, and the non-P1 records readers), build-report's `du` field: 64-bit,
  so they refuse long before 4300 digits.
- `nextest-config`'s TOML: the `toml` crate reads no integer beyond 64 bits, so a 4301-digit one is
  refused on both sides (`toml-int-4301-digits`, in the crate's words; `tomllib` raised the limit's
  `ValueError`).
- The native family-growth verifier (U8): `serde_json` reads no integer a double cannot hold
  (about 310 digits and up); a unit test pins the 4301-digit refusal, and the module lists the 310
  to 4300 digit range among its deliberate, stricter differences.

Every place a P1 port compares or normalises a POSIX path:
- cargo-artifact `select`: finding 1. Also fixed: a container equal to the executable maps to the
  host itself (the port had appended a separator and refused it), and the refusal for a missing
  executable prints `//...` as `str()` did.
- build-report: `abspath` keeps `//` as `posixpath.normpath` does, and `walk` compares the
  normalised spellings, so `scan //r` is refused as `root.resolve(strict=True) != root` refused it
  (`///r` and `r/./` are scanned). A measurement root is compared as `Path(text)` with its resolved
  path (`explicit_root`), and its refusals and the receipt's `custody_argument` print
  `str(Path(...))`.
- coverage discovery: the root is resolved, as `ROOT` was, and Cargo's manifest and source paths
  are made relative with `relative_to`, so a `//` spelling is refused as the Python's `ValueError`
  crashed it. The listed files' name, suffix and parent are read as pathlib reads them, so
  `./.github/workflows/ci.yml` is a workflow, as it was.
- parity-check: the entry points it lists are made relative with `relative_to`, so a `[[bin]]`
  path of the src/bin fallback normalises (`./src/x.rs` is `crates/<c>/src/x.rs`) and one outside
  the tree, either root, refuses as the Python crashed. `os.path.relpath` of Cargo's `src_path`
  splits on separators and has no `//` root; the port already matched it, and a control case and a
  unit test pin that.
- Not a P1 comparison: the README renderer joins only constant paths; the other `PosixPath` users
  in `jope-records` predate P1.

The measurement-root check is pinned by a unit test, not a capture: the Python reached it only
after scanning its two hard-coded host roots. Its predicate is the scan root's
(`resolve(strict=True) != path`), which the captured scans pin.

**Reproduction first.** `p1-r3-pre-20260928T040118Z`, at `e8640b7b0` (only additive, unused
helpers in the tree): the reviewer's four inputs, rerun with its recorded Rust command lines, exit
0 where its recorded Python exited 1, and its valid control exits 0 on both. The round-three
fixtures replayed there: 18 cases exited 0 where the Python did not (cargo-artifact 6 integer and 4
path, build-report 2, parity 3 integer, coverage 3 path), two refused where the Python succeeded
(`map-equals-executable`, `parity-fallback-bin-paths-normalised`), two printed a report where the
Python crashed (`parity-fallback-bin-path-absolute`, `-double-slash`), one discovery differed
(`cov-discover-workflow-dot-prefix`) and two refusals printed a different path.

**After.** `p1-r3-post-20260928T041107Z`, at the paths commit: the reviewer's four inputs exit 1,
the three cargo-artifact refusals byte-identical to its recorded Python stderr
(`stderr-comparison.txt`), the tally with CPython's `ValueError` text; the valid control exits 0;
every replay passes.

**Fixtures.** Captured in `p1-r3-capture-20260928T035917Z` by `cap3.sh`, kept in that run: bash and
jq only, the original tools from `f46a9048f`'s git bytes in scratch roots, the layouts of round two.

| Fixture | Cases | Python exit 0 | Explicit refusal | Crash |
| --- | ---: | ---: | ---: | ---: |
| `jope-records/…/process-r3-digits.json` | 15 | 7 | 0 | 8 |
| `jope-records/…/process-r3-paths.json` | 15 | 10 | 0 | 5 |
| `jope-records/…/readme/negative-r3.json` | 3 | 1 | 0 | 2 |
| `jope-ci/…/cargo-artifact/negative-r3-digits.json` | 14 | 7 | 7 | 0 |
| `jope-ci/…/cargo-artifact/negative-r3-paths.json` | 15 | 9 | 6 | 0 |
| `jope-ci/…/build-report/negative-r3.json` | 6 | 3 | 3 | 0 |
| **total** | **68** | **37** | **16** | **15** |

**Tests.** New test functions in existing targets, so `deploy/ci/coverage.json` and the test
inventory are unchanged; the capability matrix's `jope-records::*` count is now 135 (from
`cargo test -q --locked -p jope-records -- --list`).
- `jope-records` `python_parity.rs`: `process_cases_match_round_three_digits` and `_paths` (the
  round-two replay is factored into `replay_process`), `readme_benchmarks_matches_round_three`.
- `jope-ci` `cargo_artifact_python_parity.rs`: `cargo_artifact_matches_round_three_digits` and
  `_paths` (`replay_round`; the TOML case is the one allowed wording difference);
  `build_report_python_parity.rs`: `scans_match_round_three` (`replay_scans`).
- Unit tests in `py.rs`, `pyjson.rs`, `parity.rs` and `coverage.rs` (`jope-records`),
  `build_report.rs` (`jope-ci`) and `family_growth.rs` (`jope-engine`).

**Contract after round three.** As after round two, for the same ports, with these corrections:
round two's finding 9 (`int()` of Unicode digits, "unbounded") and 12 (4) (cargo-artifact reads
"integers of any size") now stop at 4300 digits, as Python 3.12 did; and paths are compared as
pathlib compares them, `//` included. One case joins round two's `tomllib` wording difference:
a 4301-digit TOML integer, refused on both sides in different words (`toml-int-4301-digits`). The
64-bit readers' strictness from 20 digits up was already listed; the family-growth verifier's
from about 310 digits is now listed in its module.

**Commits.**
- `jope-records, jope-ci: refuse integers of more than 4300 digits, as Python 3.12 did (P1 review 3)`
- `jope-records, jope-ci: keep POSIX's distinct '//' root where the ports compare paths (P1 review 3)`
- `docs: record review round three of the P1 ports (P1 review 3)` (this section and the matrix
  count)

The first commit's own tree was tested with the path-class tests and fixtures set aside (fmt,
`git diff --check`, clippy, the `jope-records` and `jope-ci` tests, the family-growth unit tests).

**Gate.** `p1-r3-gate-20260928T041354Z`, on the head (this file's commit): the hosted `rust-check` and
`rust-test` lanes run locally as `.github/workflows/ci.yml` runs them, fmt, whitespace, the
workspace check, clippy with `-D warnings` on `jope-ci` and `jope-records`, both crates' tests (the
ignored receipt replays included), the family-growth unit tests, the fences, `coverage validate`,
`verify-test-inventory`, the records checks, `parity-matrix.sh --no-build`, `deploy/ci/check.sh`,
the secret scan, the reviewer's four inputs, and round two's differential harness rerun unchanged
except for its scratch paths.

### Round three, addendum

A verifier reran the original tools (git bytes of `f46a9048f`) beside the ports at `ed49fbc3e` on
132 new inputs of the two classes (`check-r3`). Four were accepted only by the port, from one
root cause, and they refute the sweep's `py::int` bullet above: `py::int` and build-report's `du`
parse stripped the whitespace, the sign and the underscores and parsed with `i64`, which reads
any number of leading zeros. So a README CSV cell (`int(r["datasets_scored"])`; `elo_rank` is read
the same way) of 4299 zeros and `45`, plain, with `+`, between spaces or with `0_` separators,
rendered where Python 3.12 raised the limit's `ValueError`.

- **Fix.** `py::int` returns `Result<i64, py::IntError>`. As CPython did, once the literal is
  well-formed and before it is converted, it counts the digits (leading zeros, not the sign,
  whitespace or underscores) and calls `check_int_digits`: `IntError::Limit` carries CPython's
  message. Everything it refused before is `IntError::Invalid`, worded by each caller as before:
  the README's `int_cell`, and in the non-P1 readers `task_status::int_key` and the finding
  number of `findings`.
  build-report's `py_int` makes the same check. Every other text-to-integer parse in the two
  crates is bounded before it (`--stale`, eight characters), parses `decimal_int`'s output
  (parity), or is `pyjson`'s.
- **Reproduction first.** `p1-r3b-pre-20260928T044721Z`, at `ed49fbc3e` with only the fixture,
  the replay count and the new assertions: the verifier's harness, rerun unchanged but for its
  scratch paths, reports the four (Python 1, port 0); the README replay fails on the same four
  and on the 4301-digit control's message; the `du` unit test fails.
- **Fixtures.** Nine cases appended to `jope-records/…/readme/negative-r3.json` (now 12: 5 exit 0,
  7 crash), captured in `p1-r3b-capture-20260928T044551Z` by `cap3b.sh` (bash and jq, the
  original tool unmodified in a scratch root) and naming that run in their own `source` and
  `capture`: 4300 digits with leading zeros, plain, with `+`, between spaces and with `0_`
  separators render (README SHA-256 recorded); the same at 4301 digits, and 4301 ones, raise the
  limit. The round-three total is 77 cases.
- **Tests.** `readme_benchmarks_matches_round_three` replays 12 cases and now requires a refusal
  to carry CPython's text wherever the Python raised that limit; `py.rs`
  `int_of_text_counts_digits_as_python_did` (new), `int_conversion_limit_is_pythons` and
  `python_int_and_strip` (the new type); `build_report.rs`
  `stale_hours_and_du_parsing_follow_the_python_rules` (extended). No new target; the capability
  matrix's `jope-records::*` count is now 136.
- **After.** `p1-r3b-post-20260928T045128Z`, on the fix before it was committed (its tree): the
  verifier's harness reports none of its 132 cases, integer or path, refused by the Python and
  accepted by the port (the four are refused, with CPython's text); the README replay and the
  `du` unit test pass.
- **Gate.** `p1-r3b-gate-20260928T045221Z`, on the head (this file's commit): the round-three
  gate's steps, the listed `jope-records` test count against the matrix, and the verifier's
  harness rerun unchanged except for its scratch paths.
- **Commits.** `jope-records, jope-ci: count leading zeros toward the 4300-digit limit in int() of
  text (P1 review 3 addendum)` and `docs: record round three's addendum of the P1 ports (P1 review 3
  addendum)` (this section and the matrix count).

## Follow-up — review round four: strict grammars

Independent review 5334300806 of PR #117 (REQUEST_CHANGES on `bc22bc6ca`, by `neverhumanbot`)
found two more classes where a port accepted what the original tool (git bytes of `f46a9048f`)
refused. Its receipts are in `independent-review-pr117-bc22bc6ca` (`review-notes.md`,
`report-spotcheck-results.json`, `numeric-spotcheck-results.json`).

| # | Finding | Root cause | Change | Cases |
| ---: | --- | --- | --- | --- |
| 1 | custody `outer/link/../receipts`, with `outer/link -> guarded/sub` and a `.git` in `guarded`: the Python resolved `guarded/receipts` and refused; build-report resolved `outer/receipts` and wrote a successful report there | `resolve_lenient` applied `os.path.abspath` (lexical: `link/..` removed first) and then canonicalised the existing prefix; `Path.resolve()` resolves each symlink before a later `..` | `py::realpath` and `py::resolve`, CPython 3.12's `posixpath._joinrealpath` and `Path.resolve()`, used for custody, scan roots and measurement roots | build-report `negative-r4.json` (22) |
| 2 | a README CSV cell `\x1f45` (a published `datasets_scored`) or `\x1f21.511` (a benchmark-valid `average_rank`): the Python's `int()` and `float()` refused; the port rendered | `py::int` and `py::float` trimmed with `str.strip()`'s whitespace, which includes U+001C..U+001F; `int()` and `float()` strip only `Py_ISSPACE` (ASCII space, `\t\n\r\x0b\x0c`) after mapping Unicode whitespace at or above U+0080 to spaces | the strict number grammar below | readme `negative-r4.json` (37) |

Each of the four rounds has found another Python conversion or path quirk the ports copied
imperfectly. This round converges by construction instead of by case: a port may be stricter than
the Python, never more permissive, so text is read as a number by one grammar that accepts a
subset of what Python read, and a path is resolved for a safety decision only as `realpath`
resolves it.

**Numbers: one strict grammar** (`jope_records::py`). `decimal` accepts optional surrounding
whitespace from ASCII space, `\t`, `\n`, `\r`, `\x0b` and `\x0c` only (`is_number_space`), an
optional ASCII sign and one or more ASCII digits, at most 4300 of them (leading zeros count, as
they did for CPython 3.12, which refused more with the message `check_int_digits` words); the text
is validated before its digits are counted. `int` also needs the value to fit `i64`.
`decimal_int` reads what a pattern's `\d+` captured through the same grammar. `float` takes the same
whitespace and sign, then `digits [. [digits]]` or `. digits`, then an optional `e`/`E`, sign and
digits, correctly rounded (`1e999` is an infinity, as in Python). Every conversion of text to a
number in the P1 ports goes through them:

| Where | Python | Now |
| --- | --- | --- |
| README CSV cells: `datasets_scored`, `elo_rank` (`readme::int_cell`) | `int(cell)` | `py::int` |
| README CSV cells: `average_rank` (`readme::float_cell`) | `float(cell)` | `py::float` |
| parity: `pending-WP<n>` and the covering-test tally | `int(m.group(1))` of `\d+` | `py::decimal_int`; classifying a status only matches (`py::is_decimal_run`) and converts nothing |
| build-report: the `du` field (`du_bytes`) | `int(stdout.split("\t", 1)[0])` | `py::int`, not negative |
| build-report: `--stale` (`whole_hours`) | `int(value)` after `isascii()`, `isdecimal()`, 8 characters | the same checks, then `py::int` |
| `pyjson`: every JSON integer and float literal the ports read (cargo-artifact's metadata and messages, build-report's configuration, coverage's map and metadata, scope's and parity's metadata, the README evidence) | `json.loads` | the JSON grammar chooses the literal; `py::decimal` and `py::float`, of which every JSON number is a sentence, convert it |
| records readers outside P1: finding numbers (`findings`), sort keys (`task_status`) | `int(text)` | `py::int` (their committed inputs are ASCII digits; `verify-test-inventory` and the records tests are unchanged) |

Not converted by the grammar, and why: `nextest-config` parses its TOML source with the `toml`
crate only to look for a `store` key, and TOML's number grammar is ASCII-only (44 TOML number
literals in the round-four harness find none that `tomllib` refused and the crate read; the crate
refuses four that `tomllib` read: two 64-bit overflows, `0xFFFFFFFFFFFFFFFF` and `1e999`); the
family-growth verifier (U8) reads JSON with `serde_json` (JSON's ASCII grammar, no `NaN` or
`Infinity`) and converts no other text (its one conversion is the hexadecimal prefix of its own
SHA-256); `readme.rs`, `pyjson.rs` and `py::round` re-read exponents and digits of their own
float formatting. `nan`, `inf` and `infinity`: no committed ladder writes them in a cell the ports
read with `float()` (every committed leaderboard and same-cut CSV was searched), so the grammar
refuses them.

**Paths: symlinks before `..`.** `py::realpath(path, strict)` ports `posixpath._joinrealpath`
byte for byte: each name is `lstat`ed and a symlink replaced by its target, resolved in turn,
before a later `..` removes the last component of the path resolved so far; in strict mode a
missing component refuses naming the path up to it, otherwise it is kept as spelled; a loop
refuses in strict mode and, as Python did, appends the rest unresolved otherwise; the result is
`abspath` of it. `py::resolve` adds `Path.resolve()`'s `stat` of a non-strict result, so a loop
refuses as pathlib's `RuntimeError("Symlink loop from ...")`. Every place a P1 port resolves a
path for a safety decision:
- build-report custody (`custody_output`): `py::resolve(custody, strict=False)`, then `/tmp` among
  the output and its parents, then a `.git` from the output upwards with `Path.exists()`'s
  semantics (`py::path_exists`): a missing path is absent, any other error refuses in the
  `OSError`'s words. The port's `exists()` had read every error as absence, so under a directory
  that may not be searched it went on to create the custody and failed there in other words
  (`custody-permission-denied` now prints the Python's refusal).
- build-report scan roots (`walk`): `abspath` and then `py::resolve(strict=True)`, compared with
  the spelling. The Python applied `os.path.abspath` first here too, so only here is `..` removed
  before symlinks are resolved; `realpath_strict` is replaced by the shared port.
- build-report measurement roots (`explicit_root`): `py::resolve(strict=True)` instead of
  `canonicalize` (the same resolution; a missing component is now named as Python named it).
- coverage discovery and parity-check: the repository or `--root` is `canonicalize`d (`realpath(3)`,
  strict: the Python's non-strict `resolve()` of an existing path gives the same path, and a missing
  root refuses, which is stricter); `relative_to` is pure (`PosixPath`), as pathlib's was.
- family growth (U8): evidence is `canonicalize`d and checked with `starts_with`
  (`resolve(strict=True)` and `is_relative_to`); the policy and reviewer-key checks `canonicalize`
  and, where that fails and Python's non-strict `resolve()` went on, the read that follows refuses
  (unchanged; `jope-engine` does not depend on `jope-records`).
- cargo-artifact: `relative_to` is pure; the final `resolve()` of the executable is output, after
  `is_file()` and `X_OK` of the unresolved path (unchanged).

The choice is realpath semantics rather than refusing `..` or symlinks in operator paths: the
installed wrapper passes an absolute custody with neither, but refusing them would reject
spellings the Python accepted and resolved to a safe destination (`outer/realdir/../receipts`,
`outer/link/../../outer/receipts`), and faithful resolution gives the Python's destination, or its
refusal, for every spelling. `abspath` moves to `py` (build-report re-exports it).

**Stricter than the Python, each listed with its reason in the tests.** `RUST_STRICTER`:
`readme/negative-r2.json` `csv-average-rank-nan`, `readme/negative-r3.json`
`csv-int-underscores-4300-digits`, and 14 cases of `readme/negative-r4.json` (Unicode whitespace
around an integer or a float, `_` separators, Arabic-Indic and fullwidth digits, `nan`, `inf`,
`-Infinity`). `REFUSED_BEFORE_THE_LIMIT`: `csv-int-underscores-4301-digits` (the Python raised the
4300-digit limit; the grammar refuses the `_` first, so CPython's message is not required).
`PROCESS_STRICTER`: `parity-tally-arabic-equal`, `parity-tally-4300-arabic-indic-digits`.
`PROCESS_REFUSES_REPORT`: `parity-pending-wp-arabic-digit`, `parity-tally-arabic-differs` (the
Python printed a failing report; the port refuses before reporting). Outside the fixtures: the
`du` field (du prints ASCII digits) and every JSON number are read by the same grammar.

**Reproduction first.** `p1-strict-pre-20260928T053625Z`: the binaries of `bc22bc6ca` (built from
the clean tree). A new differential harness, `check-r4.sh` (bash and jq; the original tools from
`f46a9048f`'s git bytes in scratch roots), 244 cases: README `int()` and `float()` cells of each
character class in three ladders, 32 JSON and 44 TOML number literals, parity numbers in other
digits, 22 custody spellings through a synthetic layout (links into a tree whose `.git` is an empty
directory or a file, a loop, a dangling link, an unsearchable directory), and 6 scan roots. 19 were
refused by the Python and accepted by the port: 12 numeric (the reviewer's two and ten like them)
and 7 custody spellings where the Python refused and the port wrote a receipt. With the pre-fix
sources and the new tests (`negative-control/`), `readme_benchmarks_matches_round_four` fails on
17 cases (the six U+001F cells, the reviewer's two among them, and 11 the grammar now refuses)
and `custody_matches_round_four` on 9 custody cases; the sources were restored byte for byte.

**After.** `p1-strict-post-20260928T054638Z`, on the fix before it was committed (its tree and
diff are kept): the harness finds no case refused by the Python and accepted by the port (0 of
244), no output or receipt difference, 42 stricter cases, all of the listed classes; every custody
refusal prints the Python's text (the symlink loop as its `RuntimeError`), and every accepted
custody writes the Python's receipt, `custody_resolved` included. Round two's harness (137 cases)
and round three's (132), rerun unchanged except for their scratch paths: none accepted where the Python
refused; round two gains 2 stricter cases (`tally-arabic-equal`, `csv-average-rank-nan`), round
three 5 (four parity numbers in other digits, the 4300 `0_` digits), and round three's one receipt
difference (`br-int-report-config-4300`: both exit 1 with a failure receipt, the port's 64-bit JSON
reader refusing the integer the Python kept) is unchanged. The reviewer's 18 probes that write
nothing, rerun from its receipts' command lines: each outcome as recorded (6 accepted with the same
output, 12 refused); its custody and README probes, which write, are rebuilt in scratch
(`custody-rv-symlink-parent`, `readme-rv-*`: refused, with the Python's text). The live guards on
the real tree give identical exit status, stdout and stderr before and after (coverage `validate`
and `discover`, `rust-scope-check`, `readme-benchmarks --check`, `parity-check`,
`verify-test-inventory`, cargo-artifact `target-dir`, `nextest-config` and its file, `build`, and
`select` on real Cargo output with and without a path map, build-report `scan` of the lane's Cargo
target and of `crates/`); the one difference is Cargo's own timing line in `build`'s stderr.

**Fixtures.** Captured in `p1-strict-capture-20260928T054323Z` by `cap4.sh`, kept in that run:
bash and jq only, the original tools from `f46a9048f`'s git bytes in scratch roots.

| Fixture | Cases | Python exit 0 | Explicit refusal | Crash |
| --- | ---: | ---: | ---: | ---: |
| `jope-records/…/readme/negative-r4.json` | 37 | 24 | 2 | 11 |
| `jope-ci/…/build-report/negative-r4.json` | 22 | 0 | 21 (12 of the custody; 9 of the configuration, after accepting the custody and writing the failure receipt) | 1 |
| **total** | **59** | **24** | **23** | **12** |

The custody fixture records its synthetic layout, which the test rebuilds per case, and runs
`report` with a configuration naming wrong Cargo roots, so nothing is scanned and a custody the
Python accepted ends in the same failure receipt; it compares the exit status, stdout, stderr, the
receipt (`custody_resolved` included), `failure.txt` and every path the run created.

**Tests.** New test functions in existing targets, so `deploy/ci/coverage.json` and the test
inventory are unchanged; the capability matrix's `jope-records::*` count is now 142 (from
`cargo test -q --locked -p jope-records -- --list`).
- `jope-records` `py.rs`: `strict_int_reads_ascii_decimal`,
  `strict_int_refuses_what_python_refused_and_more`,
  `strict_float_reads_ascii_decimal_and_exponent_forms`,
  `strict_float_refuses_what_python_refused_and_more`,
  `realpath_resolves_symlinks_before_parent_components`;
  `python_strip` (from `python_int_and_strip`), `int_conversion_limit_is_pythons`,
  `int_of_text_counts_digits_as_python_did` and `regex_classes_are_pythons` corrected (the test at
  the old lines 1103–1105 expected U+001F around an integer to be read); `parity.rs`
  `status_classes_and_pending_range`; `python_parity.rs` `readme_benchmarks_matches_round_four`.
- `jope-ci` `build_report.rs`: `custody_is_resolved_symlinks_first`,
  `custody_under_tmp_or_a_working_tree_is_refused` and
  `stale_hours_and_du_parsing_follow_the_python_rules`;
  `build_report_python_parity.rs`: `custody_matches_round_four` (ignored by default like the other
  receipt replays; it needs `JOPE_MAINTENANCE_SCRATCH` outside `/tmp` and any working tree).

**Contract after round four.** As after round three, for the same ports, with these corrections:
text is read as a number only by the strict grammar, so the forms Python read beyond it (Unicode
whitespace, `_` separators, other Unicode digits, `nan` and the infinities in a CSV cell) are
refused and listed; and a path the ports resolve for a safety decision is resolved as `realpath`
resolves it, symlinks before `..`, with `Path.exists()`'s errors.

**Commits.**
- `jope-records, jope-ci: read every number from text through one strict grammar (P1 review 4)`
- `jope-records, jope-ci: resolve custody symlinks before '..', as Path.resolve() did (P1 review 4)`
- `docs: record review round four of the P1 ports: strict grammars (P1 review 4)` (this section,
  the commit map and the matrix count)

The first commit's own tree was tested with the path-class change set aside (fmt,
`git diff --check`, clippy with `-D warnings`, the `jope-records` and `jope-ci` tests, the ignored
receipt replays).

**Gate.** `p1-strict-gate-20260928T055524Z`, on the head (this file's commit): the hosted
`rust-check` and `rust-test` lanes run locally as `.github/workflows/ci.yml` runs them, fmt,
whitespace, the workspace check, clippy with `-D warnings` on `jope-ci` and `jope-records`, both
crates' tests (the ignored receipt replays included), the fences, `coverage validate`,
`verify-test-inventory`, `parity-matrix.sh --no-build`, `deploy/ci/check.sh`, the listed
`jope-records` test count against the matrix, the round-four harness, round two's and round
three's harnesses, and the reviewer's probes.

## Follow-up — review round five: fail-closed filesystem and value sweep

An independent verifier of round four (`check-strict.sh`, 327 cases on `5f3de344e`, the originals
from `f46a9048f`'s git bytes in scratch roots; report and cases kept in the pre-fix run below)
found three more classes where a port accepted what the original tool refused:

| # | Finding | Root cause | Change | Cases |
| ---: | --- | --- | --- | --- |
| 1 | `cargo-artifact nextest-config` wrote a configuration from `x = 07:32:60`, `1979-05-27T23:59:60Z`, `0000-01-01` and two more year-0000 datetimes; `tomllib` refused all five | the `toml` crate (0.8) reads a seconds field of 60 and the year 0000; the port parsed with it and read any value type | the nextest source may hold only the value types the committed `.config/nextest.toml` uses (tables, inline tables, strings, integers, booleans); a date, time or datetime of any form, a float or an array refuses, naming the key, after the `store` check and before any output exists (an exhaustive `match` on `toml::Value`, no catch-all) | cargo-artifact `negative-r5.json` (31) |
| 2 | `parity-check` passed with `crates/jope-engine/src` or `python/src/jope` of mode 000, or `python/src/jope/tools` or `deploy/systemd` of mode 644 (listable, not searchable); the Python raised `PermissionError` | `main_rs.exists()`, `cli.exists()` and the tool and unit walks (`is_dir()`, `is_file()`) read every error as "absent" | `py::path_exists`, `path_is_dir`, `path_is_file`, `path_is_symlink`: `False` only where Python 3.12's pathlib answered `False` (`ENOENT`, `ENOTDIR`, `EBADF`, `ELOOP`, a NUL byte), any other error returned and refused; the walks list entries as `rglob("*")` listed them and ask `is_file()` of every one, as the Python did | `process-r5-sweep.json` (28) |
| 3 | family growth read the pinned policy through `manifests -> <a directory outside the root whose resolved name exceeds PATH_MAX>`; the Python refused "pinned provenance policy escaped workspace custody" | `escapes()` took a `canonicalize()` error for "does not escape" (policy and reviewer key) | `escapes()` returns an outcome: `lstat` error malformed, a symlink escapes, a resolved path outside the root escapes, a name too long to resolve escapes (Python's non-strict `resolve()` built it and found it outside), any other resolution error malformed | family-growth `python-custody-r5.jsonl` (7) |

Round four recorded the family-growth custody checks as "unchanged" on the claim that the read
after a failed `canonicalize()` refuses where Python's `resolve()` went on. That was wrong: the read
goes through the root's short path, which the kernel resolves, so it succeeded.

**The sweep.** Every product line of the P1 ports and the helpers they call
(`jope-records` `coverage`, `scope_check`, `parity`, `readme`, `py`, `pyjson`, `pycsv`, `error`;
`jope-ci` `build_report`, `cargo_artifact`, `main`, `py`; `jope-engine` `family_growth`) was grepped
for `.exists()`, `.is_dir()`, `.is_file()`, `is_symlink`, `try_exists`, `metadata(`,
`canonicalize`, `read_dir`, `read_link`, `file_type()`, `filter_map(Result::ok)`, `.flatten()`,
`.ok()`, `is_ok`/`is_err`, `unwrap_or*`, `if let Ok`/`let Ok(`, `Err(_)`, `current_dir`,
`current_exe`, `env::var` and catch-all `_ =>` arms (`audit.sh`, kept in the post-fix run). The
rule for each hit: an error other than "not there" refuses; an error inside a safety or custody
check refuses; a value type the committed inputs do not hold refuses. The sweep found seven more
places, fixed here with the verifier's three:

| # | Where | Found | Now | Cases |
| ---: | --- | --- | --- | --- |
| 4 | parity-check's `src/bin` fallback (when `cargo metadata` fails) | `root.glob("crates/**/Cargo.toml")`, `glob("*.rs")` list entries of every kind by name; the port kept only regular files with an `rs` extension, so a directory, a dangling link or a bare `.rs` named like a source got no required row (the Python reported it missing), and a `Cargo.toml` that is a directory or a dangling link was skipped (the Python's `read_text` raised) | the globs listed by name (`fallback_bins`), each manifest read as the Python read it, `(crate/"src"/"main.rs").exists()` and `(crate/"src"/"bin").is_dir()` refusing as they raised | `process-r5-sweep.json` fallback cases |
| 5 | README `reports` glob | an entry the listing could not read was skipped (`.flatten()`); a report whose name is not UTF-8 was skipped, so one extra report changed the Python's count (a refusal) but not the port's (a render) | entry errors refuse; the glob matches names as bytes; a matching name that is not UTF-8 refuses (stricter) | readme `negative-r5.json` (9) |
| 6 | build-report `walk` | a subdirectory was typed by `stat` through its path, so in a directory that may be listed but not searched it was taken for a file and silently not entered; `os.walk` typed it by `d_type`, entered it and raised | subdirectories typed by the entry (`d_type`, else `lstat`); a link is listed and never entered (where `os.walk` listed it is not observable: both lists are `lstat`ed and nothing else is asked of a link); the `islink` check before entering refuses on error | build-report `negative-r5.json` scans (8) |
| 7 | build-report `report`, unit files | `path.is_file()` read an error as "missing" and went on (to a successful report, given a configuration the port accepts); `HOME` unset was read as `""` and the path kept its doubled separators | `py::path_is_file` refuses where it raised; the unit directory is `Path.home() / ".config/systemd/user"` as pathlib spells it (trailing separators dropped, `/` for an empty `HOME`); an unset or undecodable `HOME` refuses (stricter) | build-report `negative-r5.json` units (7, a receipt replay) |
| 8 | `os.path.abspath` (`py::abspath`, parity's `abspath`) | a relative path from a deleted working directory was taken relative to `/` (`current_dir().unwrap_or_default()`): `build-report scan <relative>` scanned the directory the relative path names from `/`; `os.getcwd()` raised | `abspath` returns the error; `scan`, custody resolution (`ResolveError::Cwd`) and parity's `src_path` refuse with `[Errno 2] No such file or directory` | build-report `scan-r5-relative-from-deleted-cwd`; unit test |
| 9 | cargo-artifact `select` | `is_file()` of an executable under a directory that may not be searched: both refused, the port in its own words | the Python's caught `OSError` (`cargo-artifact: [Errno 13] Permission denied: '<path>'`) | cargo-artifact `select-r5-*` (3) |
| 10 | build-report `report` start; `jope-ci` tools directory | an `INVOCATION_ID` that is not UTF-8 was written as `null`; the default tools directory fell back to the working directory when the executable could not be located | both refuse (the first stricter: the Python wrote the surrogate-escaped value) | unit tests |

**The audit table.** Every hit of `audit.sh` in product code (doc comments and unit tests
excluded), at the lines of this round's fixed tree, and the calls to the new fail-closed
predicates, which the pattern does not match. Decisions: **fixed** (this round; the row says to
what); **refuses** (the error is returned, `?` or `map_err`); **Python** (the Python's own
non-error semantics: a `dict.get` default, a `Counter`, a comparison of unlike types, a lexical
path step); **accessor** (a typed view returning `None`/`false` for another type, which every
caller refuses or which is Python's own answer); **unreachable** (the default can never be
taken, and why); **text** (a message, an exit status or a timestamp, no decision); **not a
lookup** (the pattern matched a name).

| file:line | pattern | decision |
| --- | --- | --- |
| `jope-records/src/coverage.rs:52` | `unwrap_or` | unreachable (`split` yields at least one part) |
| `coverage.rs:141` | `.flatten()` | Python (`features.get(feature, [])`) |
| `coverage.rs:154` | `current_dir` | not a lookup (the child's working directory) |
| `coverage.rs:162` | `unwrap_or(-1)` | text (a refusal's exit status) |
| `coverage.rs:192` | `_ =>` | Python (`Path.stem`) |
| `coverage.rs:242` | `canonicalize` | refuses (stricter than a non-strict `resolve()` only for a missing root) |
| `coverage.rs:527`, `:528` | `unwrap_or` | Python (`item.get("lanes", [])`, `item.get("dispositions", [])`) |
| `coverage.rs:532` | `unwrap_or_default` | unreachable (`split`) |
| `scope_check.rs:36`, `:118` | `current_dir` | not a lookup |
| `parity.rs:177` | `unwrap_or_default` | Python (`parser_cmd.get(parent, "")`) |
| `parity.rs:217` | `current_dir` | not a lookup |
| `parity.rs:227`, `:817` | `canonicalize` | refuses |
| `parity.rs:228`, `:270` | `metadata(` | not a lookup (`bins_from_metadata`) |
| `parity.rs:246` | `current_dir` | fixed: `abspath` refuses a working directory it cannot read (`os.getcwd()` raised) instead of `/` |
| `parity.rs:255` | `_ =>` | Python (lexical `abspath`) |
| `parity.rs:306` | `_ =>` | Python (`kind in ("bin", "example")`) |
| `parity.rs:369` | `read_dir` | fixed: a directory the walks cannot list refuses (pathlib skipped it: stricter), one that vanished is empty |
| `parity.rs:380` | `file_type()` | fixed: refuses (was `.flatten()`, which skipped the entry) |
| `parity.rs:397` | `.is_dir()` | fixed: descend by the entry's own type, as `Path.walk(follow_symlinks=False)` (was `path.is_dir()`, `false` on error) |
| `parity.rs:352`, `:357`, `:362` | `path_exists` / `path_is_dir` / `path_is_file` | fixed: the fail-closed predicates behind every lookup below |
| `parity.rs:461`, `:462` | `is_dir`, walk | fixed: `root.glob("crates/**/Cargo.toml")` lists every entry named `Cargo.toml`, of any kind; each read refuses as `read_text` raised |
| `parity.rs:473` | `exists` | fixed (was `.exists()`): `src/main.rs` |
| `parity.rs:483`, `:486`, `:487` | `is_dir`, list, name | fixed: `glob("*.rs")` by name, any kind (was `extension()` and `is_file()`) |
| `parity.rs:521`, `:524`, `:525` | `is_dir`, walk, `is_file` | fixed (was `walk_files`: `read_dir` errors and `is_dir()`/`is_file()` errors read as absent) |
| `parity.rs:543`, `:547`, `:548` | `is_dir`, walk, `is_file` | fixed (systemd units, as above) |
| `parity.rs:752` | `exists` | fixed (was `main_rs.exists()`) |
| `parity.rs:768` | `exists` | fixed (was `cli.exists()`) |
| `parity.rs:821` | `path_exists` | refuses (round four) |
| `parity.rs:431` | `unwrap_or` | text (a refusal's path) |
| `parity.rs:494` | `unwrap_or_default` | unreachable (a listed entry has a name) |
| `parity.rs:639` | `_ =>` | refuses by report (an unknown status is a problem that fails the check) |
| `parity.rs:665`, `:671` | `unwrap_or(0)` | Python (`Counter`) |
| `parity.rs:685` | `unwrap_or_else` | Python (`cells[0] if cells else line`) |
| `parity.rs:724`, `:851` | `unwrap_or` | unreachable (`split`; a constant) |
| `parity.rs:869` | `is_ok_and` | Python (a number that does not fit is outside WP4..WP12 and is reported) |
| `parity.rs:892` | `_ =>` | Python (the `moved`/`new` branch) |
| `parity.rs:927` | `.flatten()` | Python (`jope-py` commands only when `cli.py` exists) |
| `readme.rs:115` | `unwrap_or(&NULL)` | accessor (`.get(key)`; each call follows a `dict_get`/`at` that refused a non-dict) |
| `readme.rs:241`, `:245`, `:247`, `:306` | `_ =>` | accessor (numbers; a non-number refuses) |
| `readme.rs:676` | `unwrap_or(0)` | Python (`scored.get(method, 0)`) |
| `readme.rs:748`, `:749`, `:753` | `unwrap_or` | Python (`.get("split", {})`, …) |
| `readme.rs:807` | `unwrap_or_default` | unreachable (`split`) |
| `readme.rs:1000` | `read_dir` | fixed: entry errors refuse (was `.flatten()`), names matched as bytes, a matching name that is not UTF-8 refuses (stricter) |
| `readme.rs:1071` | `kind_of(m)?` | fixed: `next(... if m["method"]["kind"] == "search")` reads each method until the first search one (was `position(... is_ok_and ...)`, which skipped an unreadable one; unreachable after `load_samecut`, fixed anyway) |
| `readme.rs:1490` | `_ =>` | refuses |
| `readme.rs:1560` | `path_exists` | fixed (was `.exists()`): a required input under a directory that may not be searched refuses as `p.exists()` raised |
| `py.rs:83`, `:84` | `.ok()` | accessor (`to_i64`; `None` refuses) |
| `py.rs:102` | `_ =>` | Python (no sign) |
| `py.rs:157`, `:169`, `:188` | `_ =>` | accessor (`Key::as_str`, `iterate`, `contains`: `None` where Python raised `TypeError`, which callers raise) |
| `py.rs:430`, `:435`, `:440`, `:444`–`:445` | `metadata(` | fixed: `path_exists`, `path_is_dir`, `path_is_file`, `path_is_symlink`, `Ok(false)` only for `ENOENT`, `ENOTDIR`, `EBADF`, `ELOOP` and a NUL byte (pathlib's `_ignore_error`), any other error returned |
| `py.rs:456`, `:535` | `unwrap_or` | text (`strerror`) |
| `py.rs:519` | `_ =>` | text (signal names) |
| `py.rs:587`, `:682`, `:986`, `:1099`, `:1151` | `_ =>`/`unwrap_or` | Python (regex translation, lexical parts, `suffix`, sign, `splitlines`) |
| `py.rs:610` | `_ =>` | refuses (`decimal_int`) |
| `py.rs:648`–`:649` | `is_symlink` | not used by P1 code any more (was parity's walk); kept, documented, for the older record verifiers |
| `py.rs:663` | `current_dir` | fixed: `abspath` returns the error (`ResolveError::Cwd` in `realpath`) |
| `py.rs:799`–`:807` | `symlink_metadata`, `Err(_)` | Python (`_joinrealpath` in non-strict mode reads an `lstat` error as "not a link"; strict mode refuses; the non-strict result feeds only custody's `path_exists` checks, which refuse) |
| `py.rs:827` | `read_link` | refuses |
| `py.rs:859` | `metadata(` | Python (`Path.resolve()`'s `stat` of a non-strict result raises only for a loop; same consumers) |
| `py.rs:1086` | `unwrap_or` | unreachable (re-parsing its own formatting) |
| `py.rs:1131` | `.ok()` | accessor (`float`; `None` refuses) |
| `pyjson.rs:172` | `unwrap_or_default` | unreachable (one side is an integer in that arm) |
| `pyjson.rs:181` | `_ =>` | Python (unlike types are unequal) |
| `pyjson.rs:212`–`:263` | `_ =>` | accessor (`integer`, `get`, `as_str`, `as_int`, `as_list`, `as_list_mut`, `as_dict`, `as_dict_mut`) |
| `pyjson.rs:332` | `unwrap_or` | unreachable (a count beyond `i64`) |
| `pyjson.rs:371`, `:372` | `unwrap_or` | unreachable (re-reading `{:e}`) |
| `pyjson.rs:815`, `:856`, `:982`, `:1024`, `:1048` | `_ =>` | refuses (the JSON grammar's errors, or its number branch) |
| `pyjson.rs:908` | `.ok()` | refuses (an invalid `\u` escape) |
| `pycsv.rs:37` | `_ =>` | Python (not a line end) |
| `error.rs:71` | `_ =>` | not a lookup (`Error::source`) |
| `jope-ci/src/build_report.rs:69` | `unwrap_or` | text |
| `build_report.rs:96`, `:182`, `:328` | `absolute` | fixed: a relative scan root from a deleted working directory refuses (was resolved against `/`) |
| `build_report.rs:122`, `:355` | `unwrap_or_default` | text (a clock before 1970) |
| `build_report.rs:153` | `symlink_metadata` | refuses |
| `build_report.rs:188`, `:580` | `.is_dir()` | refuses (`false` refuses the root; `stat` after a strict resolution) |
| `build_report.rs:197` | `read_dir` | refuses |
| `build_report.rs:202`–`:203` | `file_type()` | fixed: a subdirectory is typed by the entry, as `os.walk` did (was `stat` through the path, `false` on error, so a subdirectory of a directory that may not be searched was taken for a file and not entered) |
| `build_report.rs:215` | `lstat` | fixed: `islink` before entering refuses on error (was `is_ok_and`) |
| `build_report.rs:293` | `_ =>` | refuses (`du_bytes`) |
| `build_report.rs:318` | `unwrap_or` | unreachable (`split`) |
| `build_report.rs:334`–`:345` | `lstat` | refuses |
| `build_report.rs:438`, `:439` | `.ok()` | refuses (`--stale`) |
| `build_report.rs:477` | `path_exists` | refuses (round four) |
| `build_report.rs:501` | `env::var` | fixed: an `INVOCATION_ID` that is not UTF-8 refuses (stricter; was `null`) |
| `build_report.rs:567` | `is_ok` | text (exit status) |
| `build_report.rs:594`, `:644` | `env::var` | fixed: `Path.home()` spelled as pathlib spells it; unset or undecodable `HOME` refuses (stricter; was `""`) |
| `build_report.rs:650` | `path_is_file` | fixed (was `path.is_file()`, an error recorded as a missing unit) |
| `build_report.rs:636`–`:637`, `:664` | `current_exe`, `canonicalize` | refuses |
| `build_report.rs:723` | `unwrap_or_default` | text (the diagnostic) |
| `cargo_artifact.rs:41`, `:73` | `unwrap_or` | unreachable (a status that is neither an exit nor a signal; a return code outside `i32`) |
| `cargo_artifact.rs:48`, `:51` | `env::var_os` | Python (`env.setdefault`) |
| `cargo_artifact.rs:61`, `:349`, `:410` | `metadata(` | not a lookup |
| `cargo_artifact.rs:197`, `:206`, `:215` | `unwrap_or` | Python (`message.get(...)`, `.get("kind", [])`) |
| `cargo_artifact.rs:246` | `path_is_file` | fixed: the Python's caught `OSError` (was the port's own words) |
| `cargo_artifact.rs:247` | `is_err` | refuses (`os.access` answered `False`) |
| `cargo_artifact.rs:252` | `canonicalize` | refuses (stricter than a non-strict `resolve()` only in a race) |
| `cargo_artifact.rs:265`–`:315` | `toml::Value` | fixed: the committed configuration's value types only, an exhaustive `match` (no catch-all) |
| `main.rs:96`–`:97` | `current_exe` | fixed: refuses (was the working directory) |
| `main.rs:281` | `unwrap_or` | unreachable (`& 0xff`) |
| `jope-ci/src/py.rs:68` | `unwrap_or` | Python (an incomplete sequence at the end is the rest of the bytes) |
| `family_growth.rs:90` | `_ =>` | unreachable (a JSON number has an `f64`) |
| `family_growth.rs:120`, `:158` | `_ =>` | accessor (`number`, `key`) |
| `family_growth.rs:135`, `:137`, `:204`, `:254`, `:709` | `_ =>`/`unwrap_or` | Python (`==` of unlike types, `dict.get`, `compare_digest`'s length, `memberships.get(cid, set())`) |
| `family_growth.rs:172`, `:182`, `:196`, `:198`, `:378`, `:417`, `:431`, `:474`, `:488`, `:504` | `_ =>` | refuses |
| `family_growth.rs:286` | `canonicalize` | refuses (`resolve(strict=True)`) |
| `family_growth.rs:306`–`:316`, `:507`, `:629` | `symlink_metadata`, `canonicalize` | fixed: `escapes` refuses every error (was `is_ok_and`: an error was "not a link" and "does not escape") |

**Stricter than the Python, each listed with its reason in the tests.** `PROCESS_STRICTER`:
`parity-r5-tools-sub-unlistable`, `parity-r5-fallback-bin-unlistable`,
`parity-r5-fallback-crate-unlistable` (a directory the walks cannot list; pathlib's globs skipped
it). cargo-artifact `R5_STRICTER`: 17 nextest sources holding a date, time or datetime `tomllib`
read (10), a float (3) or an array (4). Where both refuse in other words: cargo-artifact
`R5_MESSAGE_DIFFERS` (the verifier's five dates and times and three the crate refuses too, in
`tomllib`'s words and the port's), `README_MESSAGE_DIFFERS` (`readme-r5-report-name-not-utf8`,
`readme-r5-reports-unlistable`). Outside the fixtures: an unset or undecodable `HOME` and an
undecodable `INVOCATION_ID` refuse the report; a family-growth policy or key whose resolved name is
too long refuses as an escape even inside the root; a README report whose name is not UTF-8
refuses. On family growth's seven custody cases the refusal classes agree with the Python's: the
escapes word for word (`ValueError`), a directory that may not be searched as malformed input
(`PermissionError` there).

**Reproduction first.** `p1-sweep5-pre-20260928T064944Z`: the binaries of `5f3de344e` (the
verifier's build of the clean tree, SHA-256s in its identity), the verifier's report and summary,
and a new differential harness, `sweep5.sh` (bash and jq; the original tools from `f46a9048f`'s
git bytes in scratch roots), 83 cases: parity-check under modes 000 and 644 at every level the
walks reach, the `src/bin` fallback's globs, the README reports glob and inputs, 28 nextest
sources and three `select` modes, eight scans and a deleted working directory, and six reports
with unit directories under modes and `HOME` spellings. 25 were refused by the Python and accepted
by the port: the verifier's ten and fifteen more of the same classes (two deeper permission cases
in the tool and unit walks, eight in the `src/bin` fallback, one README report name, two
unsearchable scan directories, the deleted working directory, and a report that succeeded where
the Python refused its unit files), and three receipts differed (the unit files' refusal and
`HOME`'s spelling). The negative control (`negative-control/`): with the pre-fix sources
(`HEAD`'s bytes) and the new tests and fixtures, every round-five replay fails
(`process_cases_match_round_five`, `readme_benchmarks_matches_round_five`,
`cargo_artifact_matches_round_five`, `scans_match_round_five`,
`scan_from_a_deleted_working_directory_matches_round_five`, `units_match_round_five`,
`custody_that_cannot_be_resolved_is_refused_as_python_refused_it`); the fixed sources were restored
byte for byte (SHA-256 checked).

**After.** `p1-sweep5-post-20260928T070302Z`, on the fix before it was committed (its diff is
kept): the sweep finds no case refused by the Python and accepted by the port (0 of 83), no output
or receipt difference, and 20 stricter cases, all of the listed classes. The verifier's
`check-strict.sh`, rerun unchanged except for its scratch paths: 0 of 327 (its ten blocking cases
refuse), no difference, 45 stricter (its 29 and 16 nextest sources holding a float or a datetime).
Round four's harness (244 cases): none accepted where the Python refused, no difference, 54
stricter (its 42 and 12 TOML float literals, now refused as a type the committed configuration
does not use). Round two's (137) and round three's (132), rerun unchanged except for their scratch
paths: none accepted where the Python refused, their stricter cases unchanged (9 and 17), and round
three's one receipt difference (`br-int-report-config-4300`) unchanged. The 14 live guards on the
real tree (`guards.sh`: coverage `validate` and `discover`, `rust-scope-check`,
`readme-benchmarks --check`, `parity-check`, `verify-test-inventory`, cargo-artifact `target-dir`,
`nextest-config` and its file, `build`, `select` with and without a path map, build-report `scan`
of the lane's Cargo target and of `crates/`) give byte-identical exit status, stdout, stderr and
files with the pre-fix binaries of `5f3de344e` and the post-fix ones (44 of 44 files).

**Fixtures.** Captured in `p1-sweep5-capture-20260928T065333Z` by `cap5.sh` (every case as data;
bash and jq only; the originals from `f46a9048f`'s git bytes in scratch roots), and for family
growth by `capture_growth_custody_outputs_from_python` (`/usr/bin/python3 -I -B <policy> --verify`,
the policy from the same commit), both kept in that run.

| Fixture | Cases | Python exit 0 | Explicit refusal | Crash |
| --- | ---: | ---: | ---: | ---: |
| `jope-records/…/process-r5-sweep.json` | 28 | 11 | 5 (failing reports) | 12 |
| `jope-records/…/readme/negative-r5.json` | 9 | 2 | 3 | 4 |
| `jope-ci/…/cargo-artifact/negative-r5.json` | 31 | 20 | 11 | 0 |
| `jope-ci/…/build-report/negative-r5.json` | 16 | 4 | 12 (5 `report failed`, 7 failure receipts) | 0 |
| `jope-engine/…/family-growth/python-custody-r5.jsonl` | 7 | 1 | 4 (`ValueError`) | 2 (`PermissionError`) |
| **total** | **91** | **38** | **35** | **18** |

The process, README and scan replays rebuild each case's tree (directories, links, copies, a name
given in hex) and apply its modes for the run only; a case that relies on a refused search is
skipped with a note when the user may search anything (root).

**Tests.** New test functions in existing targets, so `deploy/ci/coverage.json` and the test
inventory are unchanged; the capability matrix's `jope-records::*` count is now 146.
- `jope-records` `py.rs`: `predicates_refuse_what_pathlib_raised` (`realpath_resolves_symlinks_…`
  adapted to `abspath`'s result); `parity.rs`: `walks_list_as_pathlib_listed_and_refuse_where_it_raised`;
  `python_parity.rs`: `process_cases_match_round_five`, `readme_benchmarks_matches_round_five`.
- `jope-ci` `build_report.rs`: `a_deleted_working_directory_refuses_relative_paths`,
  `unit_files_are_located_as_path_home_located_them` (both in a child process, which owns its
  working directory and environment); `abspath_is_lexical` adapted;
  `build_report_python_parity.rs`: `scans_match_round_five`,
  `scan_from_a_deleted_working_directory_matches_round_five`, `units_match_round_five` (ignored like
  the other receipt replays); `cargo_artifact_python_parity.rs`: `cargo_artifact_matches_round_five`.
- `jope-engine` `tests/families.rs`: `custody_that_cannot_be_resolved_is_refused_as_python_refused_it`
  and the ignored `capture_growth_custody_outputs_from_python`, which drive `verify_assignment` with
  the policy or reviewer-key directory moved out of the root behind a link whose resolved name
  exceeds `PATH_MAX`, behind a short link, or made unsearchable.

**Contract after round five.** As after round four, with these corrections: a filesystem question
the ports ask is answered "absent" only where Python 3.12's pathlib answered `False`; every other
error refuses, and inside a custody check any error refuses; the directory walks list what
pathlib's globs listed and enter what `os.walk` or `Path.walk` entered, refusing a directory they
cannot list; and the nextest source may hold only the value types of the committed configuration.

**Commits.**
- `jope-records, jope-ci: refuse filesystem errors pathlib raised and list what its globs listed (P1 review 5)`
- `jope-ci: read only the committed nextest config's TOML types; print select's OSError (P1 review 5)`
- `jope-engine: refuse family-growth custody that cannot be resolved (P1 review 5)`
- `docs: record review round five of the P1 ports: fail-closed filesystem and value sweep (P1 review 5)`
  (this section, the commit map and the matrix count)

**Gate.** `p1-sweep5-gate2-20260928T072000Z`, on the head (this file's commit): the hosted `rust-check`
and `rust-test` lanes run locally as `.github/workflows/ci.yml` runs them, fmt, whitespace, the
workspace check, clippy with `-D warnings` on `jope-ci` and `jope-records` (and on `jope-parallel`,
whose library already fails `-D warnings` with 84 findings in files this round does not touch, no
finding in `family_growth.rs` or `tests/families.rs`), both crates' tests (the ignored receipt
replays included), the family-growth tests, the fences, `coverage validate`,
`verify-test-inventory`, `parity-matrix.sh --no-build`, `deploy/ci/check.sh`, the listed
`jope-records` test count against the matrix, the round-five sweep, the verifier's harness, round
four's, two's and three's harnesses, and round four's reviewer probes.

A first run on the same code, `p1-sweep5-gate-20260928T071200Z`, passed every step but one,
`engine-families`: the negative control had put the fixed sources back with their earlier
modification times (`cp -p`), older than the `jope-parallel` library it had just built from the
pre-fix `family_growth.rs`, so Cargo reused that library and
`custody_that_cannot_be_resolved_is_refused_as_python_refused_it` failed as it fails on the pre-fix
source. The other crates had been rebuilt after a later edit (the post-fix harness runs, which found
no case accepted where the Python refused, used binaries built after it). The restored sources were
touched, the family-growth tests passed on the rebuilt library, the negative-control script now
restores with a fresh time, and the gate was run again.

## Follow-up — review round six: lossy path conversion and environment values

An independent verifier of round five (`check-sweep5.sh`, 192 cases on `3c31229ec`; report and
harness kept in the scratch of that check and summarised in the pre-fix run below) found two
classes of input that the original tools refused and the ports accepted. Neither was introduced by
round five. Both are closed by construction here: every conversion of a path, name, argument or
environment value to text in the P1 port code was enumerated (the table below), and each one whose
text is printed, written, recorded or compared now refuses a value that is not UTF-8; none is ever
converted lossily (U+FFFD for an undecodable byte) outside a message.

| # | Finding | Root cause | Change | Cases |
| ---: | --- | --- | --- | --- |
| 1 | `cargo-artifact select` and `build` printed an executable whose resolved name is not UTF-8 (a link to a directory named with the byte `0xff`) as U+FFFD, exit 0; the Python, under a strict UTF-8 stdout (`en_US.UTF-8`), raised `UnicodeEncodeError` printing it (exit 1), and under C or `C.UTF-8` printed the raw bytes | `path.display().to_string()` of the resolved path in `main.rs` | `executable()` returns text: the resolved name, or the refusal `cargo-artifact: 'utf-8' codec can't encode character '\udcff' in position N: surrogates not allowed` (CPython's text: the position in characters, a run of undecodable bytes one range), in every locale; the `--path-map` roots are compared and joined only as text (a lossy container root matched a message path spelled with U+FFFD, a lossy host root named another directory) | cargo-artifact `negative-r6.json` (29) |
| 2 | `build-report report` recorded `custody_resolved`, `custody_argument` and the checksum keys lossily, and printed the receipt path lossily (exit 0); the Python wrote the receipt and then failed printing its path under `en_US.UTF-8` (`report failed: 'utf-8' codec can't encode ...`, exit 1) | `to_string_lossy()` / `display()` at the receipt's fields and the final print | a resolved custody that is not UTF-8 refuses before anything is created, with the Python's message; a custody argument that is not UTF-8 refuses; a checksum key that is not UTF-8 (this executable, the tools directory, the configuration reached through such a name) fails the run with a failure receipt; a scan root refused for a link component is named as the Python's stderr printed it (`\udcff`) | build-report `negative-r6.json` (25) |
| 3 | `build-report report` with `HOME` beginning with `~` (`~`, `~/x`, `~nobody/x`): Python 3.12's `Path.home()` raised `RuntimeError("Could not determine home directory.")` and the receipt failed; the port read a relative directory and succeeded | `units_dir` read `HOME` as any UTF-8 text | `HOME` is read only when set, UTF-8 and absolute; one beginning with `~` (whatever follows, even a byte that is not UTF-8) fails the receipt with the Python's text at the same step; unset, not UTF-8, empty or relative refuses (stricter) | build-report `negative-r6.json` units (12, a receipt replay) |
| 4 | `parity-check` on a root whose resolved name is not UTF-8 compared entry points with the root converted lossily: a `[[bin]]` path spelled with U+FFFD in the byte's place was found under the root and the check passed where the Python's `relative_to` raised | `rel()` compared with `root.to_string_lossy()`; `relpath()` returned a lossy row name | `check` refuses such a root before reading anything (stricter: the Python ran the check); `rel()` refuses a root that is not UTF-8 and `relpath()` returns a name that is not UTF-8 for its caller to refuse | `process-r6.json` (4) |
| 5 | `jope-ci` panicked (exit 101) on an argument that is not UTF-8, before its `--delete` check | `std::env::args()` | `args_os()`: `--delete` is refused first (exit 2, the Python's text), then any argument that is not UTF-8 (`jope-ci: argument N '<repr>' is not UTF-8; the ports read their arguments as text`, exit 2; stricter: the Python took it surrogate-escaped) | the argument cases in both fixtures |
| 6 | the repository root found without `--repo` (`git rev-parse --show-toplevel`) was read lossily, so every guard run from a checkout whose path is not UTF-8 failed (stricter, not permissive) | `String::from_utf8_lossy` of git's output | the bytes git printed | `layout.rs` unit test |

Messages that name a path now render it as the Python did — `repr` (`'/r/\udcff'`) or, where the
Python wrote a `str` to stderr, `\udcff` (its `backslashreplace` handler) — through
`jope_records::py::{fsdecode, os_repr, os_display, os_error_path, utf8_encode_error}`; the few
message-only conversions that stay lossy are listed as such below.

**The audit table.** Every path/name-to-text conversion and every environment read in the product
lines of the P1 ports and the helpers they call (`jope-ci` `main`, `build_report`, `cargo_artifact`,
`py`; `jope-records` `coverage`, `parity`, `readme`, `scope_check`, `py`, `pyjson`, `findings`,
`task_status`, `main`, `layout`, `error`; `jope-engine` `family_growth`), by the patterns
`display()`, `to_string_lossy`, `from_utf8_lossy`, `to_str()`, `into_string()`, `{x:?}`, U+FFFD,
`env::var`, `var_os`, `args`/`args_os`, `current_dir`, `current_exe`, `Command::new`, `.env(`
(`audit6.sh`, kept in the post-fix run; unit-test modules excluded), at the lines of this round's
tree. Decisions: **refuses** (a value that is not UTF-8 refuses, and where); **bytes** (the value
stays an `OsStr`/`PathBuf` and reaches only the OS); **message** (the text of a refusal, rendered as
the Python printed it, never U+FFFD); **message, lossy** (a refusal's text only, left lossy);
**Python** (the Python's own semantics); **outside P1** (a verifier this PR does not port).

| file:line | conversion or read | decision |
| --- | --- | --- |
| `jope-ci/src/main.rs:97` | `current_exe()`, `canonicalize()` (default tools directory) | bytes; its names reach the receipt only as checksum keys, which refuse (`build_report.rs:692`) |
| `main.rs:272`–`:280`, `:288`, `:293`, `:297` | `args_os()`, `to_str()` of each argument, `os_repr` | fixed: `--delete` refused first; an argument that is not UTF-8 refuses (exit 2, named as `repr`); `Cli::parse_from` then reads only UTF-8 arguments (was `std::env::args()`, a panic) |
| `build_report.rs:64` | `os_error` | fixed: `py::os_error_path`, the `OSError` text with the path as `repr` (was `display()` for an error without a code) |
| `build_report.rs:71`–`:74` | `receipt_text` | fixed: the text of a path the receipt records, or a refusal naming it as `repr` |
| `build_report.rs:87` | `os_repr` (symlink loop) | message |
| `build_report.rs:192` | `os_display` (root with a link component) | fixed: message as the Python's stderr printed it, `\udcff` (was `display()`, U+FFFD, which also reached `scan-N.stderr` and the receipt's `error`) |
| `build_report.rs:236`, `:241` | `os_repr` (the command in `CalledProcessError`); `Command::new(command[0])` | message; bytes (`/usr/bin/du`, `/usr/bin/sudo`, `df` looked up on `PATH` as `subprocess` did) |
| `build_report.rs:340` | `file_name().to_str()` | Python (compared with `debug`/`release`: a name that is not UTF-8 is neither) |
| `build_report.rs:386`–`:389`, `:403`–`:406` | candidate, scan root | refuses (round three) |
| `build_report.rs:504`–`:506` | resolved custody | fixed: refuses before anything is created, with the `UnicodeEncodeError` the Python's `print(receipt_path)` raised under `en_US.UTF-8` (was `custody_resolved` lossy and a lossy print) |
| `build_report.rs:512`, `:643`–`:644` | custody argument | fixed: refuses (was `to_string_lossy`) |
| `build_report.rs:516` | `env::var("INVOCATION_ID")` | refuses a value that is not UTF-8 (round five) |
| `build_report.rs:581` | `receipt_path.to_str()` (print) | cannot fail: a UTF-8 custody (`:504`), an ASCII stamp and file name (was `display()`) |
| `build_report.rs:595` | `resolved.to_str()` (measurement root) | Python (a UTF-8 spelling never equals a resolved name that is not UTF-8: refused as the Python refused it) |
| `build_report.rs:615`–`:635`, `:674` | `var_os("HOME")`, `to_str()` | fixed: see the environment table (was `env::var`, any UTF-8 text) |
| `build_report.rs:666`, `:692`, `:702` | `current_exe()`; checksum keys | fixed: a key that is not UTF-8 fails the run (was `to_string_lossy`) |
| `cargo_artifact.rs:47`–`:55` | `Command::new("cargo")`, `var_os`, `.env(` | bytes / presence only (see the environment table) |
| `cargo_artifact.rs:96`–`:98` | `os_error_at` (`read_text`, the output file) | fixed: message, pathlib's spelling or `repr` (was `to_string_lossy` for the name) |
| `cargo_artifact.rs:259`–`:262` | `--path-map` roots | fixed: refuses a root that is not UTF-8 (was `to_string_lossy` in the comparison and the join) |
| `cargo_artifact.rs:285`–`:290` | resolved executable | fixed: text, or the Python's `UnicodeEncodeError` (was `display()` in `main.rs`); `canonicalize()` refuses a name over `PATH_MAX` (stricter, now listed) |
| `jope-ci/src/py.rs:50`–`:51` | `os_repr` | delegates to `jope_records::py::os_repr` |
| `jope-records/src/coverage.rs:152`–`:154` | `Command::new`, `current_dir(root)` | bytes |
| `coverage.rs:176`–`:181` | `root.to_str()` | refuses (no metadata path is under a root that is not UTF-8); fixed: the message names it as `repr` (was the lossy root) |
| `parity.rs:96` | `os_display` (enum not found) | fixed: message (was `display()`) |
| `parity.rs:207`, `:336` | `relpath` result | fixed: a relative `src_path` that is not UTF-8 refuses (was `to_string_lossy`, a row name with U+FFFD) |
| `parity.rs:221`–`:223`, `:252` | `Command::new("cargo")`, `current_dir(root)`; `env::current_dir()` | bytes; the joined path is refused by `relpath` where it is not UTF-8 |
| `parity.rs:355`, `:854` | `os_error_path` | fixed: message (was `to_string_lossy`) |
| `parity.rs:443`–`:446` | `rel()`: root | fixed: refuses a root that is not UTF-8 (was compared lossily, finding a `[[bin]]` path spelled with U+FFFD under it) |
| `parity.rs:449`–`:453` | `rel()`: path | refuses (round two); fixed: named with `os_display` (was lossy) |
| `parity.rs:514` | `file_stem().to_str()` | cannot fail: `rel()` refused a name that is not UTF-8 (was `to_string_lossy`) |
| `parity.rs:842`–`:845` | `check`: resolved root | fixed: refuses before anything is read (stricter) |
| `readme.rs:1006`–`:1009`, `:1563` | report name; missing input | refuses (round five); fixed: messages with `os_display` (were lossy) |
| `scope_check.rs:33`–`:36`, `:116`–`:118` | `Command::new`, `current_dir(root)` | bytes: nothing about the root is converted (the check matches the Python's on a root that is not UTF-8) |
| `jope-records/src/py.rs:379` | `from_utf8_lossy` (`read_text_replace`) | Python (`read_text(errors="replace")` of file contents, as the Python read them) |
| `py.rs:579`–`:621` | `os_repr`, `os_error_path`, `os_display`, `utf8_encode_error` (with `fsdecode`) | new: the renderers every message above uses |
| `py.rs:757` | `current_dir()` (`abspath`) | bytes |
| `py.rs:802`–`:805` | `ResolveError` display | fixed: message with `os_error_path`/`os_repr` (was `to_string_lossy`) |
| `pyjson.rs:978` | U+FFFD in place of an unpaired surrogate escape | refuses: the document is refused when parsed (the placeholder is never output) |
| `findings.rs:117` | `{name:?}` | not a path (a record key) |
| `task_status.rs:88`, `:105` | `to_str()` of a work-order name | outside P1 (the W12-T03 verifier; this PR changed only its `int_key`): a name that is not UTF-8 is skipped where `glob("W*.tasks.md")` read it; reported, unchanged |
| `jope-records/src/main.rs:179`, `:336` | `current_dir()`; `Cli::parse()` | bytes; clap keeps a `PathBuf` argument as bytes and refuses a text argument that is not UTF-8 (exit 2) |
| `layout.rs:41`–`:51` | `git rev-parse --show-toplevel` | fixed: the output's bytes (was `from_utf8_lossy`, a directory that does not exist); the refusal names the directory with `os_display`; git's own stderr in it: message, lossy |
| `error.rs:61`–`:62` | `display()` of `Error::Parse`/`Error::Io` | message, lossy (shared by every verifier; the refusal is the decision) |
| `family_growth.rs:216`, `:782` | `{name:?}` | not a path (JSON keys) |
| `family_growth.rs:287`, `:305`, `:321` | `display()` in `malformed` | message, lossy (the refusal is the decision) |
| `family_growth.rs:625` | `root.to_str()` | refuses (`root is not UTF-8`, stricter since the port) |

**Environment.** Every value the ports read from the environment, and why it is safe:

| Value | Where | Decision |
| --- | --- | --- |
| `HOME` | `build_report.rs:615`–`:635` | read only when set, UTF-8 and absolute; `~...` fails as `Path.home()` raised (`Could not determine home directory.`); unset, not UTF-8, empty or relative refuses (stricter); never an empty, default or relative directory |
| `INVOCATION_ID` | `build_report.rs:516` | an identifier, not a path: recorded as the text it is, `null` when unset (`os.environ.get`); not UTF-8 refuses before anything is created (round five, stricter) |
| `CARGO_INCREMENTAL`, `CARGO_PROFILE_DEV_DEBUG` | `cargo_artifact.rs:48`–`:52` | presence only (`env.setdefault`); the value is never read, and Cargo inherits it as the bytes it is |
| `CARGO_TARGET_DIR` | `cargo_artifact.rs:55` | written for Cargo from `--target-dir` (an argument, UTF-8); never read |
| `PATH` | every `Command::new` of a bare name (`cargo`, `git`, `bash`, `df`) | the executable is looked up by the OS from `PATH` as `subprocess` looked it up; never read as text |
| arguments | `jope-ci` `main.rs:288`; `jope-records` `main.rs:336` | `jope-ci`: bytes first, `--delete` refused, then any argument that is not UTF-8 (exit 2); `jope-records`: clap keeps a `PathBuf` argument as bytes (each use is in the table) and refuses one that is not UTF-8 where it reads text (exit 2) |
| working directory | `py.rs:757`, `parity.rs:252`, `jope-records` `main.rs:179` | bytes (`abspath`, `git -C`); a relative name joined to it is refused where it would be printed (`relpath`, the scan root, the custody) |
| this executable | `jope-ci` `main.rs:97`, `build_report.rs:666` | bytes; recorded only as a checksum key, which refuses a name that is not UTF-8 (`:692`) |
| `LANG`, `LC_ALL`, `LC_CTYPE` | not read | the ports are locale-independent: where the Python's outcome depended on the locale (a strict UTF-8 stdout raised, C and `C.UTF-8` printed the raw bytes), the ports refuse in every locale |
| `TMPDIR`, `JOPE_*` | not read by the ports | tests only |

**Stricter than the Python, new or previously unlisted** (each is listed in the tests, with its
reason):
- an executable whose resolved name is not UTF-8, under C and `C.UTF-8` (the Python printed its raw
  bytes): cargo-artifact `R6_STRICTER` (12 cases);
- a resolved executable name longer than `PATH_MAX`: `realpath(3)` refuses (`[Errno 36] File name
  too long`) where Python's `resolve()` built and printed it (`cargo_artifact.rs:285`–`:287`; round
  five's table called this "stricter only in a race"): `select-r6-resolved-name-too-long`;
- a TOML document `tomllib` read and the `toml` crate refuses, `[a.b.c]`, `[a]`, `b.d = 1` (`duplicate
  key`): `toml-r6-implicit-then-dotted`;
- an argument that is not UTF-8 to `jope-ci` (before: a panic, exit 101; now exit 2):
  `R6_ARGV_REFUSED` (4), `R6_REPORT_ARGV` (2), `scan-r6-root-argument-not-utf8`;
- a custody whose resolved name is not UTF-8: nothing is written (the Python wrote the receipt and
  then failed printing its path, or printed its raw bytes under `C.UTF-8`): `R6_CUSTODY_NOTHING_WRITTEN`;
  a custody argument or a checksum key that is not UTF-8: `R6_REPORT_STRICTER`;
- `HOME` empty (the Python took `/`) or relative (the Python located the units from the working
  directory): `R6_REPORT_STRICTER`, and round five's `units-r5-home-empty` (`R5_UNITS_NOW_STRICTER`);
- a `parity-check` root whose resolved name is not UTF-8: `PROCESS_STRICTER`;
- unchanged and already refusing: a scan root that is not UTF-8 (`scan-r6-relative-root-from-not-utf8-cwd`
  shows it from a relative root), a family-growth root that is not UTF-8.

**Outside the P1 ports, unchanged.** `task_status.rs:88` and `:105` (the W12-T03 task-status
verifier, whose P1 change is only `int_key`) skip a work-order file name that is not UTF-8, which
Python's `glob("W*.tasks.md")` would have read: a fail-open of the older verifier, reported, not
changed here. `findings.rs:117` names a record key, not a path.

**Reproduction first.** `p1-r6-pre-20260928T082952Z`: the binaries of `3c31229ec` (the round-five
gate's build) and a new harness, `sweep6.sh` (bash and jq; the originals from `f46a9048f`'s git
bytes in scratch roots), 76 cases: `select` and `build` of executables behind links to directories named `0xff`, a
run `0xff 0xfe`, `é 0xff`, an incomplete sequence and bytes between others, under `en_US.UTF-8`,
`C.UTF-8` and C; `--path-map` host links and arguments; the two unlisted stricter cases; reports
under ten `HOME` spellings with the harness's configuration and with the case's own approved roots;
custody, configuration and tools behind names that are not UTF-8; scans from a working directory
that is not UTF-8; and the jope-records guards on a root that is a link to such a directory. 15
were refused by the original and accepted (13) or written (2) by the port: `select`, `build` and a
`--path-map` host link under `en_US.UTF-8` (7), the three `HOME=~...` spellings with a
configuration the port accepts (3), four custody cases (a link to a directory named `0xff`, and a
relative custody from such a working directory, with either configuration), and the lossy-root
`[[bin]]` in `parity-check` (the one instance of the class outside the verifier's report). Seven more
are calibration: a `goodcfg` case (the port is given the case's own approved roots) whose Python
receipt failed only at its fixed approved roots, with nothing else failing, which the harness cannot
give it (`sweep6.sh` labels them `CALIBRATION-py-fixed-roots`; `adjudicate.sh` applies the same
rule to any run). A first run, `p1-r6-pre-20260928T080903Z`, counted 18: its rule looked only at a
receipt whose path the Python printed as UTF-8, so three `C`/`C.UTF-8` custody cases whose Python
printed the path's raw bytes (and failed only at the roots) were counted; the rule now reads the
receipt the Python wrote, printed or not, and still counts a Python that failed printing it.
Applied to the verifier's own run on `3c31229ec` (`check-sweep5` `run2`), the same rule leaves
`r5-good-control-home` and `r5-good-home-link-loop` as calibration, as the verifier found.

The negative control (`p1-r6-negctl-20260928T082538Z`): with the new tests and fixtures on the
pre-fix sources (`HEAD`'s bytes of the nine product files), every round-six replay fails, and so
does round five's
units replay, which now holds the empty `HOME` to the round-six rule
(`cargo_artifact_matches_round_six`, `scans_match_round_six`, `reports_match_round_six`,
`units_match_round_five`, `process_cases_match_round_six`), each for the reason it was written:
the lossy paths printed with exit 0, the panic on an argument, the `~` receipts, the lossy-root
row. The fixed sources were restored byte for byte (SHA-256 checked) and then touched, so Cargo
rebuilt them (round five's first gate had reused a library built from sources restored with `cp -p`).

**Fixtures.** Captured in `p1-r6-capture-20260928T082121Z` by `cap6.sh` (every case as data; bash
and jq only; the originals from `f46a9048f`'s git bytes in scratch roots, run under the locale each
case names). A byte that is not UTF-8 is written `@Xhh@` in a name, an argument or `HOME`; a codec position is
written relative to the length of the root it counts (`in position @ROOT+1@`), and the replays place
it past their own root; a deep directory's printed components are placed the same way. (An earlier
capture, `p1-r6-capture-20260928T081753Z`, recorded two tracebacks with the scratch path, which the
`just lint-paths` rule forbids in `crates/*/tests`; it was rerun with the case directory written
`@CASE@`, and only the rerun is committed.)

| Fixture | Cases | Python exit 0 | Python refused (message or receipt) | Crash |
| --- | ---: | ---: | ---: | ---: |
| `jope-ci/…/cargo-artifact/negative-r6.json` | 29 | 21 | 8 (the `UnicodeEncodeError` it caught) | 0 |
| `jope-ci/…/build-report/negative-r6.json` | 25 | 3 | 22 (12 unit and 4 custody failure receipts, 4 encode errors, a root refusal, the deletion refusal) | 0 |
| `jope-records/…/process-r6.json` | 4 | 2 | 0 | 2 (`ValueError`) |
| **total** | **58** | **26** | **30** | **2** |

**Tests.** New test functions in existing targets, so `deploy/ci/coverage.json` and the test
inventory are unchanged; the capability matrix's `jope-records::*` count is now 150.
- `jope-records` `py.rs`: `names_that_are_not_utf8_render_as_python_rendered_them`; `parity.rs`:
  `entry_points_are_never_named_by_a_lossy_conversion` (`relpath_…` adapted to its `Result`);
  `layout.rs`: `a_checkout_whose_path_is_not_utf8_is_found_as_it_is`; `python_parity.rs`:
  `process_cases_match_round_six` (the process replay builds a root behind a link, `root_link`).
- `jope-ci` `build_report.rs`: `unit_files_are_located_as_path_home_located_them` extended (`~`,
  `~/x`, `~nobody/x`, `~<0xff>`, `/`, `/~x`, empty, relative); `build_report_python_parity.rs`:
  `scans_match_round_six`, the ignored `reports_match_round_six` (like the other receipt replays),
  and `units_match_round_five` holding `units-r5-home-empty` to the new rule;
  `cargo_artifact_python_parity.rs`: `cargo_artifact_matches_round_six` (the round replay now
  rebuilds names that are not UTF-8, links, a deep directory and the locale, and returns each
  outcome, so the test also holds every C and `C.UTF-8` refusal to the `en_US.UTF-8` message);
  `cargo_artifact.rs` reads `executable()`'s text.

**After.** `p1-r6-post-20260928T082733Z`, on the fix before it was committed (its diff is kept),
every harness rerun unchanged except for its scratch paths (`table.tsv`, `adjudication.txt`):

| Harness | Cases | Original refused, port accepted | Output or receipt differences | Stricter |
| --- | ---: | ---: | ---: | ---: |
| Round six (`sweep6.sh`) | 76 | 0 (1 calibration, the good-configuration control) | 10, each a refusal at an earlier step (stricter): `HOME` empty, relative or not UTF-8, and a checksum key that is not UTF-8, each with either configuration; the Python failed later, at the harness's approved roots | 28 |
| Round five (`sweep5.sh`) | 83 | 0 | 0 | 20 |
| Verifier `check-strict.sh` | 327 | 0 | 0 | 45 |
| Round four (`check-r4.sh`) | 244 | 0 | 0 | 54 |
| Verifier `check-sweep5.sh` | 192 | 0 (2 calibration: `r5-good-control-home`, `r5-good-home-link-loop`) | 3: `r5-home-relative` (stricter, the new `HOME` rule), `r5-config-link-loop2` and `r5-config-via-too-long-link` (both refuse, in other words, as in the verifier's run) | 18 |
| Round two | 137 | 0 | 0 | 9 (unchanged) |
| Round three | 132 | 0 | 1 (the known `br-int-report-config-4300`) | 17 (unchanged) |

On the verifier's harness the two blocking classes are closed: `u5-ca-*-en-us` now refuse with the
Python's message, `r5-home-tilde` and `r5-home-tilde-path` give the Python's receipt
(`Could not determine home directory.`), and the `C`/`C.UTF-8` variants are stricter. The 14 live
guards on the real tree (`guards.sh`) give byte-identical exit status, stdout, stderr and files with
the pre-fix binaries of `3c31229ec` and the post-fix ones, except Cargo's own timing line in
`artifact-build.err` (`0.13s` against `0.12s`).

**Contract after round six.** As after round five, with these corrections: a path, name, argument
or environment value that is not UTF-8 is never converted lossily; where its text would be printed,
written, recorded or compared, the port refuses (where the Python's outcome depended on the locale,
in every locale, with the message a strict UTF-8 locale gave); `HOME` is read only when set, UTF-8
and absolute, and one beginning with `~` fails as `Path.home()` raised.

**Commits.**
- `jope-records: never convert a name that is not UTF-8 lossily; refuse it where entry points are named (P1 review 6)`
- `jope-ci: refuse names and arguments that are not UTF-8 instead of printing them lossily; read HOME as Path.home() did (P1 review 6)`
- `jope-ci: let the negative report hold a replayed stdout that is not UTF-8 (P1 review 6)`
- `docs: record review round six of the P1 ports: lossy path conversion and environment values (P1 review 6)`
  (this section, the commit map and the matrix count)

**Gate.** `p1-r6-gate2-20260928T090000Z`, on the head (this file's commit): round five's gate
(`gate.sh`, with this agent's paths) — the hosted `rust-check` and `rust-test` lanes run locally as
`.github/workflows/ci.yml` runs them, fmt, whitespace, the workspace check, clippy with `-D warnings`
on `jope-ci` and `jope-records`, both crates' tests (the ignored receipt replays included), the
listed `jope-records` test count against the matrix, the family-growth tests, the fences,
`coverage validate`, `verify-test-inventory`, the record checks, the live guards,
`parity-matrix.sh --no-build`, `deploy/ci/check.sh`, the secret scan, round four's reviewer probes —
and every differential harness: round six's, round five's, the verifiers' `check-strict.sh` and
`check-sweep5.sh` (with `adjudicate.sh`), round four's, two's and three's; none may hold a case the
original refused and the port accepted.

A first run, `p1-r6-gate-20260928T084500Z`, on the head before the third commit above, passed every
step up to the `jope-ci` tests (the hosted lanes included: `rust-check` 4 of 4, `rust-test` 1620
passed and 28 skipped), where `cargo_artifact_matches_round_six` panicked in the negative-report
writer, which runs only when `JOPE_NEGATIVE_REPORT` is set, as the gate sets it and the earlier
test runs did not: it read every case's stdout as text, and round six's C and `C.UTF-8` cases hold
the Python's raw bytes as `stdout_hex`. The replay itself was not at fault. The run was stopped
(its later steps build from the working tree, which the fix changes; `STOPPED.txt` there), the
writer fixed in its own commit, and the gate run again.

## Commit map — `rr2/python-p1` to the landing

The work above was committed on `rr2/python-p1` and landed through `rr2/python-p1-landing`: its
eleven commits were replayed in order onto main at `cee1fc8bd` (after #114 and #115), followed by
landing-only commits, and the branch was then rebased onto main at `f46a9048f` (after #116). The
repository merges a pull request only by rebase or squash (merge commits are disabled, and the
head branch is deleted on merge), and both give each commit a new SHA on main, as they did for
#114, #115 and #116. So no SHA this branch has had, before or after the rebase, becomes a SHA on
main, and the table keys each landed commit by its exact subject: after a rebase merge
`git log -F --grep='<subject>' main` finds that commit, and after a squash merge it finds the
squash commit, whose message lists every subject. Where this record, the tests or the fixtures
cite a commit for P1's own provenance, they name a landed commit by its role or subject, name a
commit on main (`8a55c4fef`, `cee1fc8bd` or `f46a9048f`) or a commit in one of the custody bundles
below (an original or a pre-rebase commit), or cite the blob id of a committed file; fixture
copies of main's files keep main's own citations unchanged. An original appears only in this
table and in commit messages, which cannot change: the four `(cherry picked from commit …)`
trailers (on the landed U2, U3, U5 and archive commits) and the message of the U3 fixture commit,
which names the U3 run's tree. A pre-rebase SHA appears only in this section and in the evidence
runs made before the rebase.

| Landed commit (exact subject) | Before the rebase | Original | Patch-id |
| --- | --- | --- | --- |
| `jope-engine: grow families natively; retire the Python growth policy (P1 U8)` | `364c06fd9` | `29f4d1077` | differs |
| `jope-ci: port cargo-artifact to Rust; retire deploy/ci/cargo-artifact.py (P1 U2)` | `6b95c9208` | `5136986fb` | differs |
| `jope-records: port the four hygiene guards; retire their Python (P1 U3)` | `7382a4275` | `1f1791025` | differs |
| `jope-ci: port the build-directory report to Rust; retire jope_report.py (P1 U5)` | `6cd7672ec` | `6ce8fcfbe` | differs |
| `python: archive 703 files with no live consumer (P1 archive)` | `6e6bf5ad2` | `389898805` | differs |
| `docs: ADR-0008 Rust-only stack; say where Python still lives (P1 docs)` | `4ed0e2096` | `e9511a2fc` | same |
| `docs: renumber the Rust-only ADR to ADR-0006 (P1 follow-up)` | `ee24eae50` | `9b810cd80` | same |
| `jope-ci: replay the captured Python outputs of cargo-artifact and build-report (P1 follow-up)` | `037b8f885` | `9e8aacedc` | same |
| `jope-ci, jope-engine, jope-records: clear clippy findings in P1 port code` | `0b213a36e` | `4b4018dff` | same |
| `jope-records: replay the captured Python outputs of the four hygiene guards (P1 follow-up)` | `bc3ba2ca6` | `df9c78ae3` | same |
| `docs, gitleaks: record the stale references to archived Python as PY-P2 items (P1 follow-up)` | `575c55ea6` | `2200bc872` | same |
| `docs: name the Rust coverage validator where W01 and W09 cite coverage.py (P1 landing)` | `f7a19392c` | landing only | — |
| `docs: recount the jope-records tests in the capability matrix (P1 landing)` | `f0b1402ec` | landing only | — |
| `docs, jope-records: cite landed commits, not rr2/python-p1 originals (P1 landing)` | `309efce3b` | landing only | — |
| `docs, jope-records: cite the rebased commits after rebasing onto #116 (P1 landing)` | none, made later | landing only | — |
| `docs, jope-records: name the landed P1 commits by subject, not by branch SHA (P1 landing)` | none, made later | landing only | — |
| `jope-campaigns: floor the unit templates per directory after the P1 archive (P1 landing)` | none, made later | landing only | — |
| `jope-engine: stop observing the archived jope-parallel-console unit (P1 landing)` | none, made later | landing only | — |
| `docs: record why the nine archived unit templates stay archived (P1 landing)` | none, made later | landing only | — |
| `jope-engine: show receipts of the archived report and proposal writers as history (P1 landing)` | none, made later | landing only | — |
| `docs: correct the archived units' consumers and the host they were checked on (P1 landing)` | none, made later | landing only | — |
| `jope-records: fail closed on malformed inputs to the four hygiene guards, as their Python did (P1 review)` | none, made later | landing only | — |
| `jope-ci: compare cargo-artifact ids with Python's == and fail closed on malformed metadata and messages (P1 review)` | none, made later | landing only | — |
| `jope-ci: replay build-report configuration refusals against the Python (P1 review)` | none, made later | landing only | — |
| `docs: record the fail-closed audit of the P1 ports (P1 review)` | none, made later | landing only | — |
| `docs: record the fail-closed gate and post-fix runs (P1 review)` | none, made later | landing only | — |
| `jope-records: read Cargo's output, regex classes, NaN, integers and repr as the Python did (P1 review 2)` | none, made later | landing only | — |
| `jope-ci: CPython's messages, nextest-config order and names that are not UTF-8 (P1 review 2)` | none, made later | landing only | — |
| `docs: record round two of the fail-closed differential review (P1 review 2)` | none, made later | landing only | — |
| `jope-records, jope-ci: refuse integers of more than 4300 digits, as Python 3.12 did (P1 review 3)` | none, made later | landing only | — |
| `jope-records, jope-ci: keep POSIX's distinct '//' root where the ports compare paths (P1 review 3)` | none, made later | landing only | — |
| `docs: record review round three of the P1 ports (P1 review 3)` | none, made later | landing only | — |
| `jope-records, jope-ci: count leading zeros toward the 4300-digit limit in int() of text (P1 review 3 addendum)` | none, made later | landing only | — |
| `docs: record round three's addendum of the P1 ports (P1 review 3 addendum)` | none, made later | landing only | — |
| `jope-records, jope-ci: read every number from text through one strict grammar (P1 review 4)` | none, made later | landing only | — |
| `jope-records, jope-ci: resolve custody symlinks before '..', as Path.resolve() did (P1 review 4)` | none, made later | landing only | — |
| `docs: record review round four of the P1 ports: strict grammars (P1 review 4)` | none, made later | landing only | — |
| `jope-records, jope-ci: refuse filesystem errors pathlib raised and list what its globs listed (P1 review 5)` | none, made later | landing only | — |
| `jope-ci: read only the committed nextest config's TOML types; print select's OSError (P1 review 5)` | none, made later | landing only | — |
| `jope-engine: refuse family-growth custody that cannot be resolved (P1 review 5)` | none, made later | landing only | — |
| `docs: record review round five of the P1 ports: fail-closed filesystem and value sweep (P1 review 5)` | none, made later | landing only | — |
| `jope-records: never convert a name that is not UTF-8 lossily; refuse it where entry points are named (P1 review 6)` | none, made later | landing only | — |
| `jope-ci: refuse names and arguments that are not UTF-8 instead of printing them lossily; read HOME as Path.home() did (P1 review 6)` | none, made later | landing only | — |
| `jope-ci: let the negative report hold a replayed stdout that is not UTF-8 (P1 review 6)` | none, made later | landing only | — |
| `docs: record review round six of the P1 ports: lossy path conversion and environment values (P1 review 6)` | none, made later | landing only | — |

"Patch-id" compares `git patch-id --stable` of the commit as submitted and its original. Each
pair changes the same paths with the same status. The tree proof
(`p1-landing-post-20260927T170033Z`, `step-treeproof.log`), run before the rebase, finds that 8 of
the 829 paths the eleven commits touch differ between the original tip and `f0b1402ec` (the
capability-matrix recount before the rebase), all of them paths main also changed: 5 equal the
clean three-way merge, and the jope-records `lib.rs` and `main.rs` and the capability matrix were
resolved by hand.

The rebase onto `f46a9048f` (#116: health routing, four commits) applied all fourteen commits
without a conflict. Every commit keeps its patch-id and author, and all but one keep their
message: in the first citation commit (`309efce3b` before the rebase) the three citations of
landed commits were amended to the rebased SHAs. The rebased tip's tree is the tree
`git merge-tree --write-tree f46a9048f 309efce3b` computes (`0365499cf`). #116 and P1 change four
common paths in different lines: `deploy/ci/coverage.json`, `fast-lane.sh`, `full-lane.sh` and
`findings/test-inventory.json`. On each, the landed file is
exactly P1's line changes on top of #116's and #116's on top of P1's. #116's CI lines survive
(`check-test-routing.sh` in `full-lane.sh`, the two routing self-tests in `fast-lane.sh` and in
the hygiene `checks` list), none of them calls Python, and no retired Python tool is named
anywhere it was not named before the rebase. The capability matrix, #116's findings README and
`health-routing-20260927.md` needed no change. Runs: `p1-rebase-pre-20260927T191744Z` (identity,
both diffs, the merge prediction) and the `p1-rebase-post-*` gate (the landing gate on the
rebased head, the semantic checks above, `check-test-routing.sh` on real collections and its
self-tests).

Two landing commit messages, which cannot change either, name commits by the SHA they had on this
branch after the rebase, which the merge replaces: the first citation commit's message names the
ADR-0008 commit as `24ca97838`, the renumbering commit as `14bdfe1ef` and the U2 commit as
`947b1a1aa`, and the rebase citation commit's message names those three and the first citation
commit as `65e6f961d`. These four SHAs are given here only to decode those two messages; they are
not the landed commits.

A review of the rebased branch found that this record, `python_parity.rs` and the U3 fixture's
`source` string still cited those branch SHAs as the landed commits. Reproduction
`p1-rebase-cite-pre-20260927T195252Z`: GitHub reports merge commits disabled; the heads of #114,
#115 and #116 are not ancestors of main, whose commits have the heads' trees, one parent each and
the same authors and messages; after a simulated rebase merge and a simulated squash merge of this
branch, and in a single-branch `--no-local` clone of the simulated rebase-merged main, none of
the 14 cited branch commits resolves, while every subject resolves to exactly one commit. The last
landing commit names them by role and subject instead. `p1-rebase-cite-post-*` repeats those
checks on it (no branch SHA left but the four above; every subject in the table found once on the
branch, after either simulated merge and in the clone) and runs the gate.

The fourteen pre-rebase commits are in the custody bundle
`p1-rebase-pre-20260927T191744Z/rr2-python-p1-landing-pre-rebase.bundle` (416192 bytes, SHA-256
`bb594ade941421dc2c918689d38b3e91eda1ce21b2f70794b1afa7cbc8f8728d`, prerequisite `cee1fc8bd`,
which main contains). `p1-rebase-preserve-20260927T192159Z` restored it into a bare clone of the
rebased landing branch alone: every pre-rebase commit came back with its tree, and `git fsck
--connectivity-only` passed. To restore it into any clone that has main:

```
git fetch <bundle> refs/heads/rr2/python-p1-landing:refs/preserve/rr2-python-p1-landing-pre-rebase
```

The originals stay on `rr2/python-p1`, locally and on `origin`. Independent of any ref, all
eleven are in the custody bundle `p1-landing-preserve-20260927T172032Z/rr2-python-p1.bundle`
(406474 bytes, SHA-256 `750c4ce1abf56023380028b32717b321a4c3b674464b59c242d24a3040211cdd`,
prerequisite `8a55c4fef`, which main contains). Restored into a bare clone of the landing branch
alone, the state once that branch is deleted and gc has run, every original came back with its
tree, the four trailers resolved, `git fsck --connectivity-only` passed, and `git show` of the
U2 original's `deploy/ci/coverage.json` gave blob `5e78ee4ba`, the fixture's `mapping.json`
(`log.txt` beside the bundle). `p1-rebase-preserve-20260927T192159Z` repeated this in the clone of
the rebased branch, with the same results for the rebased commits' trailers. To restore into any
clone that has main:

```
git fetch <bundle> refs/heads/rr2/python-p1:refs/preserve/rr2-python-p1
```
