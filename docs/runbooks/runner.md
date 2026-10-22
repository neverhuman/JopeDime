# Self-hosted CPU runner runbook

The CPU foundation uses JIT registration and one isolated container per job. Fast CI remains
required by the reviewed merge procedure. `full-main.yml` runs on every push to main and manual
dispatch of main, reports all declared CPU families, and is diagnostic until the repair phase.
Branch protection remains procedural until the API readback establishes enforcement.

## Approved allocation

| Host | Slots / labels | Per-job cap | Writable storage domain |
| --- | --- | --- | --- |
| xbabe1 | ci, ci2 / jope-ci | 16 GiB, 16 CPUs, no swap | pr-ci, pr-ci2 |
| xbabe2 | full / jope-full-cpu, jope-ci | 48 GiB, 16 CPUs, no swap | main-full |
| xbabe2 | ci / jope-ci | 16 GiB, 16 CPUs, no swap | pr-ci (xbabe2's own volumes) |
| xbabe3 | full / jope-full-cpu, jope-ci | 48 GiB, 16 CPUs, no swap | main-full (xbabe3's own volumes) |
| xbabe3 | ci / jope-ci | 16 GiB, 16 CPUs, no swap | pr-ci (xbabe3's own volumes) |
| xbabe0 | ci / jope-ci | 16 GiB, 16 CPUs, no swap | pr-ci (xbabe0's own volumes) |

Seven slots (owner 2026-10-22, from four): every slot runs one job at a time. The two full slots
run the full-main-cpu matrix groups in parallel and also execute main-guard. The ci slots on
xbabe2 and xbabe3 are untrusted pr-domain lanes beside the full lane: their own `jope-pr-ci-*`
volumes, never the main-full volumes, and never the exclusive CPU lock, so ASM allocations still
wait only for the full lane. xbabe0 holds one ci lane because its free disk (169 GB on
2026-10-22) fits one persistent Cargo target. `jope_runner_lanes` in `common.sh` is the per-host
lane list that `provision.sh` installs. xbabe4 is a macOS host and xbabe5 offers only a legacy SSH
host key, so neither carries a Linux container runner.
`common.sh` rejects all other host/lane combinations, including the previous ci3/ci4,
GPU and kernel roles. Their live qualification and persistent runner migration remain deferred.
The stopped runner-sizing branch's 24 GiB/28 CPU proposal is preserved but not deployed.

Each domain has separate named work, Cargo registry, target and uv volumes, for example
`jope-pr-ci-target` and `jope-main-full-target`. PR/manual fast CI is untrusted code even when it lands on a full host via `jope-ci`;
trusted main workflows continue to use the full domain. Full-lane volumes may therefore
see both trusted main jobs and untrusted ci.yml jobs (owner permanent 2026-10-14).
An ephemeral registration does not clean persistent volumes. Never share these writable domains
or mount the legacy models volume into CPU jobs. Existing old volumes/evidence are retained
until the independent custody procedure permits removal.

Containers run as uid1001, with all capabilities dropped, no-new-privileges, read-only root,
4096 pids, 4 GiB temporary filesystem, and 4 GiB executable home tmpfs. They have no GPU devices,
Docker socket, host PAT, training, publishing or S3 credentials. The SDK volume is read-only.
The root PAT is used only by the host to request a single-use JIT configuration; do not print
JIT arguments, process arguments or Docker Config.Env during inspection.

The read-only root-installed `job-started.sh`/`job-policy.py` hook admits ci.yml PR/manual
jobs to the fast (pr) domain and, on full hosts that also carry `jope-ci`, to the trusted
full lane; main-branch push/manual full-main or push main-guard jobs use the trusted domain. It reads a root-selected policy file and Actions context, not checkout code or
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
any image rebuild. Start only ci/ci2 timers on xbabe1, full and ci on xbabe2 and xbabe3, and ci on xbabe0; keep retired timers
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
(120 s; `JOPE_PROVENANCE_FETCH_TIMEOUT` may set it up to 600 s, and a larger value fails
closed), and a base that stays unreadable fails closed. A `.git/shallow.lock` left by a killed
fetch in a reused workspace is removed only when no process holds it; a held lock fails closed.
`provenance_bind.py digest REV` accepts only `WORKTREE`, a 40-hex id or a plain ref name. It also checks the provisioned
image identity and verifies immutable policy plus actual cgroup memory/CPU/swap values. It does
not hash every image file. Never weaken provenance to make an unreviewed deployment pass.

runc writes `memory.swap.max=0` but registers each `docker-<id>.scope` with
`MemorySwapMax=infinity`, so any later `daemon-reload` (enable/disable, provision.sh) flipped live
listeners to swap=max and failed their next job's swap check. The launcher now records
`MemorySwapMax=0` on the scope with `systemctl set-property --runtime` and kills a listener
whose cap systemd does not hold. Listeners started before this deployment still need the
drain-and-restart procedure after a reload. This changes deployed files: provision it only after
the PR is merged, in a post-merge freeze coordinated with ENG:watcher, never from an unmerged head.

Idle listeners are not stopped by a service-start deadline: that deadline previously cut into
accepted jobs. Each JIT listener still executes only one job. Workflows own execution timeouts;
full-main has 145 minutes, with 137.75 minutes of independent family limits plus kill grace and an
upload reserve. Timers launch the next ephemeral listener after its predecessor exits.

The bounded ASM controller and the local full lane share the boot-persistent inode
`/run/lock/jope-exclusive-cpu.lock`. Provisioning creates it through tmpfiles without
replacing an existing inode. The full listener holds the lock for its whole allocation,
including idle time, and refuses populated ASM cgroups left during crash cleanup. Drain
the owned timer, wait for busy work to finish, then stop an idle owned listener before
an ASM allocation. Resume the timer afterward. The controller does not alter the runner
fleet. Each full host has its own lock inode and ASM check. Every runner host requires the
same reviewed deployed-file inventory before exact-head CI can pass provenance; the new lock
must not be used to bypass that check.

## Second full lane on xbabe3

xbabe3/full is a second trusted-main slot with the xbabe2/full cap (48 GiB, 16 CPUs, no swap)
and its own `jope-main-full-*` volumes and SDK volume on xbabe3. It changes `common.sh`, which
is part of the provenance digest. After that PR merges, every runner host must be re-provisioned
from merged main in the same post-merge freeze, or every job fails provenance:

1. Pause the merge train and stop the JopeDime timers on xbabe1 and xbabe2. Wait until the
   GitHub runners are idle (`busy` false), then confirm no JopeDime container is running.
2. On xbabe1 and xbabe2, as root from a clean checkout of merged main, run
   `deploy/runner/provision.sh --skip-sdk --skip-image`. This reinstalls host files only;
   the image is unchanged.
3. On xbabe3 (`hostname -s` must print `xbabe3`), as root from the same checkout, run
   `deploy/runner/provision.sh`. This builds the image, provisions the SDK volume from its
   receipt, creates the volumes and units, and creates an empty
   `/etc/jope-runner/github-pat` (root:root 0600).
4. The owner places the runner PAT in that file. Agents never read, print or copy it.
5. Run `systemctl enable --now jope-runner@full.timer` on xbabe3. Re-enable the stopped
   timers on xbabe1 and xbabe2.
6. Verify the GitHub runner list shows `xbabe3-full` online with the `jope-full-cpu` label.
   Then dispatch `full-main.yml` on main and confirm, for both legs: provenance, the
   container caps, the `cpu-1` and `cpu-2` legs on different hosts, and a passing aggregate.

xbabe3 also runs GPU and research CPU work outside this fleet. The full lane takes only its
16-CPU/48 GiB cap and the host-local ASM lock; it does not stop or wait for unrelated work.

## Build-cache trim

Each lane's persistent Cargo target keeps every build it ever made. On 2026-10-22 xbabe1's two pr
targets had reached 355 GB each, and xbabe3's main-full target held up to 76 builds of one crate.
The 50 GB disk guard then refused every listener on both hosts. `run.sh` therefore calls
`jope_runner_trim_lane` (common.sh) after taking the lane lock and checking that no container of
the lane exists, and before the disk guard. That is the one moment no build of the lane can be
running: an idle listener holds the lock while it waits, so a separate timer would almost never
find the lane free.

At most once per day per lane (`JOPE_RUNNER_TRIM_INTERVAL_S`), `trim-deps.sh` runs in a
network-less, read-only container, as uid 1001, mounting only that lane's target volume. It
removes files directly in `<volume>/*/debug/deps` that are named `<stem>-<16 hex>[.<ext>]` and
are older than the newest `JOPE_RUNNER_TRIM_KEEP` (6) builds of the same directory, crate stem
and kind, so each feature and profile variant keeps its recent builds. Nothing else is touched:
other names, `build/`, `incremental/`, `tmp/`, deeper paths, and any top-level directory whose
name contains `evidence`.

Every removed path, size and mtime is listed in
`/var/lib/jope-runner/trim/<UTC>-<lane>.tsv` before deletion, and its sha256 is logged. A listing
error deletes nothing. A failed trim never blocks the listener, and it keeps the previous stamp,
so the next start retries. The nightly prune timer still only reports storage.

## Seven-slot deployment (post-merge freeze)

The seven-slot change edits `common.sh`, `run.sh`, `provision.sh` and adds `trim-deps.sh`, all of
which are in `JOPE_RUNNER_DEPLOYED`. After the PR merges, every runner host must be
re-provisioned from merged main in the same freeze, or every push/manual job fails provenance:

1. Pause merges to main. Stop the JopeDime timers on every host (`jope-runner@ci.timer` and
   `jope-runner@ci2.timer` on xbabe1, `jope-runner@full.timer` on xbabe2 and xbabe3). Wait until
   GitHub shows each JopeDime runner `busy` false, stop idle listeners, and confirm no JopeDime
   container runs.
2. On xbabe1, xbabe2 and xbabe3 (check `hostname -s` each time), as root from a clean checkout of
   merged main, run `deploy/runner/provision.sh --skip-sdk --skip-image`. This reinstalls the host
   files and creates any missing lane volumes. Volumes are created before the image step, so
   xbabe2 and xbabe3 gain their `jope-pr-ci-{work,cargo,target,uv}` volumes even with
   `--skip-image`. The existing image is used to set volume ownership.
3. On xbabe0, as root from a clean checkout of merged main (a registered checkout, never under
   `/tmp`), run `deploy/runner/provision.sh --skip-sdk`. This builds the image, creates the ci
   volumes and an empty `jope-sdk` volume (ci lanes do not need the SDK), installs the units and
   creates an empty `/etc/jope-runner/github-pat` (root:root 0600).
4. The owner places the runner PAT in xbabe0's `/etc/jope-runner/github-pat`. Agents never read,
   print or copy it.
5. Re-enable the existing timers. Then run `systemctl enable --now jope-runner@ci.timer` on xbabe2,
   xbabe3 and xbabe0.
6. Verify that the GitHub runner list shows `xbabe1-ci`, `xbabe1-ci2`, `xbabe2-full`, `xbabe2-ci`,
   `xbabe3-full`, `xbabe3-ci` and `xbabe0-ci` online. Dispatch `ci.yml` on a branch and
   `full-main.yml` on main, and confirm for jobs on the new lanes: provenance, container caps,
   the pr-domain policy file, and the `jope-pr-ci-*` volume mounts. Check the first
   `/var/lib/jope-runner/trim/*.tsv` manifest on each host.

## Evidence and operation

`deploy/ci/coverage.json` explicitly selects default workspace tests, doctests, CPU feature
variants, maintained Python files and CPU integration checks. The full main profile omits the
PR filter. Ignored Rust tests and Python xbabe2 deselections are non-execution, recorded in the
collection/results; GPU, live S3, research scripts and capability qualification remain separate.
A missing SDK is unavailable, a compilation/test failure is failed, and missing counts are
unknown. Families continue independently; no continue-on-error turns a failing job green.

Every run retains each group's evidence on its runner's persistent target volume under
/var/cache/jope/target/ci-evidence-retained/REPOSITORY/NAME/. Names include run and attempt;
a failed local copy or SHA-256 readback fails the job. Logs and step summaries record host,
lane, destination and the SHA-256 manifest. Retain these logs as the independent hash reference:
later untrusted jobs can write their lane volume, so a manifest beside the files alone does
not establish tamper-evident custody. Age produces review candidates only, in both log and
summary. Publication never deletes evidence. Reclamation requires the separate preservation,
independent recovery and fresh consumer checks in AGENTS.md.

Run/attempt-scoped Actions caches hand receipts to the aggregate without GitHub artifact
storage. Save warnings are checked through lookup before publication succeeds. Save and restore
use the same workspace-relative paths across ci, ci2 and full. The full-main download uses
logical group names, preserving its verifier's layout. The aggregate re-verifies the declared
families; unavailable, stale or changed evidence fails closed. Build reuse is measured by Cargo artifact
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
