# PR114 issue follow-ups — 2026-09-27

Repository: `neverhuman/JopeDime`, GitHub repository ID `1376159324`.
These observations distinguish source corrections, local fixtures and remaining live evidence.
They do not establish a passing hosted run or authorize a merge without its normal gates.

| Issue | Change or verified existing behavior | Remaining scope |
| --- | --- | --- |
| [47](https://github.com/neverhuman/JopeDime/issues/47) | Hygiene fetches complete history and `secret-scan.sh` scans HEAD and every reachable ancestor explicitly. Shallow history is refused. Unrelated custody refs no longer change the source-head result. | This is a source-head history scan, not an audit of all local refs or deleted/unreachable objects. Historical custody-ref findings remain a separate audit; no allowlist or history rewrite was introduced. |
| [57](https://github.com/neverhuman/JopeDime/issues/57) | Existing launcher rejects unexpected template/instance drop-ins and alternate service fragments. Direct launcher fixtures now prove those refusals and the admitted path before Docker or credential access. Existing immutable CPU/memory policy and cgroup checks remain unchanged. | A failed cgroup readback is still a failed job. The observed PR114 runner-cap failure must not be relabeled as passing provenance. |
| [58](https://github.com/neverhuman/JopeDime/issues/58) | Existing CI sets emitter output beneath the current workspace run/attempt/job evidence directory. A fixture executes the actual lane through its emitter dispatch with a distinct Cargo-cache path and verifies the selected destination. | Local standalone emitter runs intentionally retain evidence. No automatic deletion, retention change or historical evidence reclamation is added. |
| [62](https://github.com/neverhuman/JopeDime/issues/62) | Champion refusal now says this command has no implemented live route, without claiming host AWS credentials are absent. Existing credential-process support remains in the dataset store. | Live champion pull/push, a scoped publisher credential and its live round-trip are not completed by this diagnostic correction; leave this issue open. |
| [48](https://github.com/neverhuman/JopeDime/issues/48) | Read-only review confirms legacy C++/JavaScript source-to-program binding remains incomplete. | Keep open. Preserve frozen v1 output while adding a versioned all-target identity issued at each actual lowering boundary; require every emitted artifact to bind, including singleton C++/JS requests and wrong-program injection fixtures. |

## Verification

Custody root:
`/home/ubuntu/jope-rescue-20260921/claude-resume-20260927/pr114/issues/`.

- `runner-and-scan-tests.log`: 12 unittest tests passed. The synthetic source-head fixture
  reproduces the old scanner's failure caused only by an unrelated custody ref, then proves
  the corrected scope passes. A removed `.env` in source ancestry still fails; a shallow
  checkout is refused before scanning. Synthetic files contain no credentials.
- `head-history.log` and `explicit-secret-scan.log`: the complete 481-commit source-head
  history at `4d475b3208d1d549de500ff7f27002781ef8c917` passed with unchanged rules.
  This does not clear findings on unrelated preservation refs.
- Static coverage validation: zero unassigned/invalid entries. Workflow actionlint and
  shellcheck for the changed scripts passed.
- `hygiene.log`: the full local `deploy/ci/check.sh` passed without a scanner shim or allowlist change.
- CLI verification in `cli.log`: 2 passed, 12 unrelated tests excluded, with bounded command and exit-status JSON.
  Only the named champion and train-step refusal tests are selected; this is not a full CLI
  or live S3 test result.

Every changed PR head needs new hosted CI and independent native approval. Earlier artifact
connection resets, cancelled jobs and runner provenance failures remain failed evidence.
