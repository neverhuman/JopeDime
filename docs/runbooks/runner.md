# Self-hosted CPU runner runbook

The CPU foundation uses JIT registration and one isolated container per job. Fast CI remains
required by the reviewed merge procedure. `full-main.yml` runs on every push to main and manual
dispatch of main, reports all declared CPU families, and is diagnostic until the repair phase.
Branch protection remains procedural until the API readback establishes enforcement.

## Approved allocation

| Host | Slots / labels | Per-job cap | Writable storage domain |
| --- | --- | --- | --- |
| xbabe1 | ci, ci2 / jope-ci | 16 GiB, 16 CPUs, no swap | pr-ci, pr-ci2 |
| xbabe2 | full / jope-full-cpu | 48 GiB, 16 CPUs, no swap | main-full |

The full slot also executes main-guard, serially. No other JopeDime runner role is admitted on
xbabe2. `common.sh` rejects all other host/lane combinations, including the previous ci3/ci4,
GPU and kernel roles. Their live qualification and persistent runner migration remain deferred.
The stopped runner-sizing branch's 24 GiB/28 CPU proposal is preserved but not deployed.

Each domain has separate named work, Cargo registry, target and uv volumes, for example
`jope-pr-ci-target` and `jope-main-full-target`. PR/manual fast CI is always treated as untrusted,
including a manual ci.yml dispatch of main; trusted main workflows use only the full domain.
An ephemeral registration does not clean persistent volumes. Never share these writable domains
or mount the legacy models volume into CPU jobs. Existing old volumes/evidence are retained
until the independent custody procedure permits removal.

Containers run as uid1001, with all capabilities dropped, no-new-privileges, read-only root,
4096 pids, 4 GiB temporary filesystem, and 4 GiB executable home tmpfs. They have no GPU devices,
Docker socket, host PAT, training, publishing or S3 credentials. The SDK volume is read-only.
The root PAT is used only by the host to request a single-use JIT configuration; do not print
JIT arguments, process arguments or Docker Config.Env during inspection.

The read-only root-installed `job-started.sh`/`job-policy.py` hook admits only ci.yml PR/manual
jobs to the fast domain, and main-branch push/manual full-main or push main-guard jobs to the
trusted domain. It reads a root-selected policy file and Actions context, not checkout code or
workflow overrides. Rejection kills the container's same-uid Listener/Worker processes before
an `always()` step could run. Runner v2.337.0 reads the hook path from its process environment,
invokes it before action pre-steps, and supplies GitHub context through the hook handler:
[JobExtension](https://github.com/actions/runner/blob/v2.337.0/src/Runner.Worker/JobExtension.cs),
[JobHookProvider](https://github.com/actions/runner/blob/v2.337.0/src/Runner.Worker/JobHookProvider.cs).
This isolates admitted workload domains; it does not claim that self-hosted containers withstand
every host-kernel exploit. See GitHub's [self-hosted hardening guidance](https://docs.github.com/en/actions/reference/security/secure-use#hardening-for-self-hosted-runners).

## Reviewed deployment and drain

Before changing an installed runner, obtain independent review of the exact candidate, preserve
installed files and safe fingerprints outside the checkout, and record service/runner state.
Stop the affected JopeDime timers first. Read GitHub runner `busy` state and wait for owned jobs
to finish; then stop idle owned services and confirm no owned container remains. Never stop the
unrelated `actions-runner*` fleet. Do not inspect credentials or replace unrecorded drop-ins
without preserving them. The launcher rejects any extra effective service drop-in and any surviving container owning the same lane, even after a dead docker CLI releases its lock.

From the reviewed checkout, `sudo deploy/runner/provision.sh --skip-sdk` builds the image,
creates the role-specific volumes, installs host files/units and writes
`/var/lib/jope-runner/image.json`. It enables no listener. Initial SDK provisioning follows its
pinned receipt; the CPU delivery reuses the existing SDK read-only. Re-run provisioning after
any image rebuild. Start only ci/ci2 timers on xbabe1 and only full on xbabe2; keep retired timers
disabled. Verify actual container caps/mounts, installed hashes and a real job before claiming
operational delivery. `--skip-image` requires an existing correctly built image.

The provenance action binds host runner files to reviewed merged code. On `pull_request`
events it compares the host digest (and mounted admission hooks / image entrypoint) to the
digest computed from the PR **base** ref, not the PR head — so runner-change PRs can go green
on current hosts, merge, then deploy in a short post-merge freeze. Push / workflow_dispatch /
main runs still require an exact match to the commit under test. On a shallow checkout the
action first fetches exactly the PR base commit (`provenance_bind.py ensure-base`, depth 1,
read-only job token), so no job depends on a reused workspace already holding it. A `BASE_SHA`
that is not exactly 40 lowercase hex characters is refused before git runs, the fetch is bounded
(120 s, `JOPE_PROVENANCE_FETCH_TIMEOUT`), and a base that stays unreadable fails closed. It also checks the provisioned
image identity and verifies immutable policy plus actual cgroup memory/CPU/swap values. It does
not hash every image file. Never weaken provenance to make an unreviewed deployment pass.

Idle listeners are not stopped by a service-start deadline: that deadline previously cut into
accepted jobs. Each JIT listener still executes only one job. Workflows own execution timeouts;
full-main has120minutes, with110minutes of independent family limits plus kill grace and an
upload reserve. Timers launch the next ephemeral listener after its predecessor exits.

The bounded ASM controller and the local full lane share the boot-persistent inode
`/run/lock/jope-exclusive-cpu.lock`. Provisioning creates it through tmpfiles without
replacing an existing inode. The full listener holds the lock for its whole allocation,
including idle time, and refuses populated ASM cgroups left during crash cleanup. Drain
the owned timer, wait for busy work to finish, then stop an idle owned listener before
an ASM allocation. Resume the timer afterward. The controller does not alter the runner
fleet. Both hosts require the same reviewed deployed-file inventory before exact-head
CI can pass provenance; the new lock must not be used to bypass that check.

## Evidence and operation

`deploy/ci/coverage.json` explicitly selects default workspace tests, doctests, CPU feature
variants, maintained Python files and CPU integration checks. The full main profile omits the
PR filter. Ignored Rust tests and Python xbabe2 deselections are non-execution, recorded in the
collection/results; GPU, live S3, research scripts and capability qualification remain separate.
A missing SDK is unavailable, a compilation/test failure is failed, and missing counts are
unknown. Families continue independently; no continue-on-error turns a failing job green.

Every run uploads `full-cpu-RUN-ATTEMPT`, including fresh family receipts, command logs, current
JUnit/collection reports and inventory.json. Build reuse is measured by Cargo artifact
`fresh` plus executable digest in full-build-reuse/reuse.json, and is not substituted for fresh
test execution. The target directory is job-scoped, with one persistent target per lane;
the image and host do not set CARGO_TARGET_DIR globally. Cargo builds use four compiler jobs,
CARGO_INCREMENTAL=0 and CARGO_PROFILE_DEV_DEBUG=line-tables-only.

`python3 deploy/ci/status.py RUN_ID --expected-sha FULL_SHA` requires no Actions runner and never
substitutes an older successful run. After ten minutes queued, it reports requested labels,
matching online/busy runners, service state and disk observations. A refused listener start
can leave jobs queued: the guard requires at least50 GiB free on Docker/state/tmp filesystems;
full families repeat a50 GB reserve check. The historical prune timer now only reports storage.
Nothing automatically deletes targets, volumes, Docker build cache or retained evidence.

On the canonical Ubuntu xbabe2 host, use `--host xbabe1 --host local` to inspect both CPU
hosts. The user's SSH alias `xbabe2` resolves to `agent@192.168.68.89`, a different endpoint;
a failed probe of that alias does not establish a failure of the local full runner. Account
selection, credential locations and the exact-head merge procedure are in
[AGENTS.md](../../AGENTS.md#approval-procedure-on-this-host).

For fast CI reruns use **Re-run all jobs**: receipts from a successful older attempt cannot
satisfy the current attempt. Full CPU failure repairs, strict Clippy enforcement, full CPU
qualification and learning/soak remain following work, not claims of this configuration.

## Admission negative proof

Manually dispatch `runner-admission-probe.yml` after reviewed deployment. Its two jobs target
each domain from a workflow neither domain admits. Expected outcome is rejection. Verify the
real Worker hook denial, zero execution of the `always()` bypass step, and container exit;
a generic failed job is not proof. Preserve run/attempt API data and scoped service logs.
The Worker can die before forwarding the hook's denial text. In that case, retain the exact
runner name and systemd invocation, installed hook hashes matching the reviewed source,
root-selected domain policy, deterministic `authorize()` rejection for the actual workflow
context (never call `main()` for a replay), and the hook-start-to-exit137 interval. Rule out
kernel OOM, a service stop and execution timeout in that interval. Confirm no bypass step
started, no marker was emitted as step output, and the container is absent. The Worker logs
the submitted script as JSON before admission; a marker inside that script metadata is not
execution. Label the denial message as inferred when it was not observed. Inspect only these
selected events; full runner logs and process arguments can contain registration secrets.
The probe has no token permissions, checkout or user payload, and a two-minute timeout.
Cancel the probe after both denials are retained if GitHub still waits on the terminated
Workers. Listener admission consumes a slot briefly; this remains a possible denial-of-service
vector. Main-guard can queue behind a full run. PR caches remain shared across PRs within each
fast slot; cross-PR isolation and enforced main protection are not claimed here.

## Repository transfer observation — 2026-09-25 UTC

During PR107 verification, GitHub's API reported the same repository ID
`1376159324` under `neverhuman/JopeDime`. The old `veox-ai/JopeDime` JIT
registration endpoint returned HTTP301, and the installed launchers failed before
starting a container. The policy also pinned the former namespace.

The launcher and status command now use the current canonical name. Admission
requires both `neverhuman/JopeDime` and `GITHUB_REPOSITORY_ID=1376159324`, preserving
the workflow, event, branch and storage-domain restrictions. A missing or changed
ID, foreign repository or former namespace is rejected. This is a repository
identity migration, not additional runner roles or resources. Deployment still
requires independent review, installed-file preservation, drain and fresh CI with
matching provenance; changing GitHub ownership alone does not qualify the hosts.
