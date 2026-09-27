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

## Bounded evidence publication recovery

Manual run `36299781070`, attempt 1, ran the smoke lane successfully but failed its
artifact upload (job `108565759005`). PR run `36299756410`, attempt 1, similarly
passed hygiene before a `FinalizeArtifact` connection reset. These failures remain
failed evidence; source-test success alone does not qualify either run.

The publication composite now permits three upload attempts within the unchanged
job deadline. Each uses a distinct `-pub-1` through `-pub-3` artifact name, retains
hidden files for 30 days and fails on missing files. It never overwrites an artifact.
A mandatory final gate rejects exhausted, cancelled, missing or inconsistent outcomes.
A successful upload cannot repair an earlier failed test step or job.

A lost finalize acknowledgement can leave two remote artifacts. The aggregate may
coalesce only explicitly numbered publications with identical complete file paths
and bytes, including the root receipt. Divergent, unnumbered, out-of-range or unsafe
publications fail. Its report retains all publication names; existing source, run,
attempt, status, evidence and test-count checks still apply.

Local fixture evidence in the custody directory above:

- `publication-aggregate.log`: 16 tests pass, including identical lost-ack recovery,
  divergent bytes, extra files, symlinks, invalid numbering, stale attempts and failed jobs.
- `publication-gates.log`: 14 tests pass; the publication gate checks all 125 combinations
  of five possible outcomes and admits only the three valid bounded sequences.

These fixture results do not establish hosted artifact service reliability. A fresh
complete run at the committed head must qualify composite execution and download.

`publication-hygiene3.log` records a passing full local hygiene run under 4 CPUs,
16 GiB and a 900-second ceiling. The first two launch attempts failed before completing
verification (missing Cargo PATH, then root/Git ownership mismatch); their logs are
retained as `publication-hygiene.log` and `publication-hygiene2.log`. The successful
run used the existing user service manager, without changing Git safety settings.
Receipt (6), result (5) and coverage (9) fixtures also passed; shellcheck and diff
whitespace validation passed. These are local checks, not hosted qualification.

## Network root cause and download recovery

Seven exact-head attempts at `93bfcb956` all failed only on GitHub network steps. Every source lane
passed whenever it ran. The observed failure modes were:
- the `changes` job exceeded its 3-minute cap because "Set up job" (action archive downloads) took
  2m10s;
- `ListArtifacts`, `GetSignedArtifactURL` and `FinalizeArtifact` hit `ECONNRESET` after about
  120 seconds (runs `36301428618`, `36301414927` and `36306656599`).

GitHub status reported all systems operational. Two host-side causes were measured:
- **DNS.** The runner containers used the LAN router `192.168.68.1` as their first nameserver.
  From xbabe1 it failed 18 of 90 lookups, while the host's systemd-resolved failed 0 of 90. Each
  miss costs glibc 5 seconds, which is exactly the runner's connectivity-check timeout.

  xbabe1 and xbabe2 were fixed on the host, not in the repository:
  - resolved listens on `docker0`;
  - `daemon.json` sets `dns=[172.17.0.1]` with `timeout:2 attempts:3`;
  - fresh containers then failed 0 of 90 lookups.

  Receipts are in `~/jope-evidence/runner-dns-fix-20260927/` on each host.
- **Connection blackholing.** From an xbabe1 container, 1 of 8 fresh TCP connects to the artifact
  service received no reply for 300 seconds, while the rest completed in 0.06 seconds. That is
  consistent with intermittent SYN loss on the shared LAN/NAT path. The roughly 120-second hang is
  the kernel's SYN-retry window.

The workflow now tolerates one stalled connection instead of failing on it:
- `changes` 3 → 10 minutes;
- `hygiene` 10 → 20 minutes, which gives the three bounded publication attempts room;
- `aggregate` 10 → 20 minutes, plus up to three whole-download attempts. Each retry starts from an
  empty directory. `publication-status.sh` (all 125 outcome combinations under test) requires
  exactly one complete bounded sequence.

`aggregate.py` is unchanged and still refuses any artifact without its root receipt. No test, gate,
pass criterion or evidence rule changed. Local checks: actionlint passed, `test-runner.py` 14/14,
and the full `check.sh` hygiene run exited 0.
