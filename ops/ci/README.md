# CI replay and security

Owner: repository maintainers listed in `.github/CODEOWNERS`.

`bash ops/ci/fast-lane.sh LANE` is the shared CI command entry point. The
implementation remains in `deploy/ci/fast-lane.sh`; `receipt.py` preserves source,
checkout, run, attempt, dirty-source inventory, tool environment and test counts.
The hosted workflows retain their declared lanes, path applicability and strict
aggregate verification. This routing change does not replace CI with local proof.

Use `bash scripts/ci-doctor.sh` to inspect prerequisites and
`bash scripts/ci-local.sh LANE` to replay one validation lane. Local receipts go to
`target/ci-local/TIMESTAMP/LANE`; rust-test defaults to an unsharded `slice:1/1`.
Set `JOPE_RUST_TEST_PARTITION=slice:1/2` and `slice:2/2` to reproduce the two CI
selections separately. A repeated output directory is refused. The changes lane
requires a GitHub event/base and is deliberately unavailable in the local helper.

The hygiene lane calls `tools/security-lane.sh`: required actionlint, full-HEAD
history gitleaks with redaction, and cargo-deny over the locked all-features graph,
including blocking advisory, source, license and ban checks. The committed
frontend lock graph also receives `npm audit --package-lock-only --audit-level=high`,
including build/test dependencies. Missing tools,
shallow history, advisory-fetch failure and scanner failures fail the lane.
Install the image-pinned versions documented in `deploy/docker/runner.Dockerfile`.
No runtime sudo, cloud credentials or Docker socket is needed by the PR jobs.

To opt into a local pre-push check, run
`git config core.hooksPath ops/git-hooks`. This runs the same hygiene lane and
does not certify full CI. Hooks remain local preferences; repository protection
and the exact-head reviewed lifecycle remain authoritative.

Run the identical full-scope Jankurai audit with
`JANKURAI_BIN=/path/to/pinned/jankurai bash ops/ci/jankurai.sh /owned/fresh/report-directory`.
The wrapper requires version1.7.5, emits timings and a repair queue, disables
badge/history writes, and always selects read-only mode. The independent audit job in full-main.yml uses the existing trusted-main
runner on main pushes and main workflow dispatches. It runs alongside the eleven
CPU families. On main it sets `JOPE_AUDIT_REPORT_ONLY=1` until the score reaches 85: a
verified below-floor audit then passes with a warning and writes `report-only.json`;
any integrity, consistency or tool failure still fails the job. Local runs enforce the floor.
The dedicated hosted workflow remains optional; its skipped runs are not evidence.

The wrapper pins its input manifest before execution. Afterward it validates the
pinned default policy and unchanged 85 floor, full scope, required reports, finding
counts and agreement between decision and exit status. It rechecks every captured
file, its read-only mode, the archive and the complete path set. The resulting
audit-verification.json records artifact integrity separately from audit_passed.
A valid failed audit retains its failure; a zero exit with invalid evidence fails.
Both passing and failing reports use the existing local/cache evidence transport.

The full auditor follows ignored `.agent/worktrees` by default. The audit wrapper
therefore exports every tracked Git blob into an immutable evidence input and
verifies the complete path set and each blob's identity before scanning. This is
an audit data artifact, not a clone, registered worktree or build lane. Untracked
files and foreign worktrees are not product input; no tracked product path is
excluded. Uncommitted tracked product changes are refused except live
`AGENT_CHAT.md` appends, which the source manifest records. The input has no `.git`
reference, so native auditor Git fields may be null; the captured full source,
checkout, tree, archive hash and per-file SHA256 manifest bind it instead.
Use the same export procedure for baseline and final comparison.

Rendered browser dependencies are SHA256- and size-pinned official Ubuntu
packages extracted without root into the task-local tool cache. Extraction runs
no maintainer scripts and changes no installed runner code, image or provenance.
Every reused runtime is checked against the contents of all verified debs.
The loader/font environment applies only to the rendered test step. The reviewed
Dockerfile dependency addition supports a later merged-source image rollout.
