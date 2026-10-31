# jope-diskd

`jope-diskd.py` is a user-level disk observer and local-only janitor. The policy file is JSON syntax, which is valid YAML 1.2, so Linux hosts need only Python's standard library. The default is `--dry-run`; apply requires `JOPE_DISKD_LOCAL_ONLY=1`, one of the exact policy hosts (`xbabe1`, `xbabe2`, or `xbabe3`), and an explicit `--apply`.

## Policy and safety boundary

- At 85% filesystem use, expired entries in the allowlisted pip cache may be pruned toward its 5 GiB size cap and the 80% low watermark. At 90%, the TTL shortens to 72 hours. At 95%, an idempotent daily alert is written to `~/nudges/DISK_ALERT.md` (a dry run prints the alert instead). The much larger uv and sccache caches remain report-only. The installer sets `SCCACHE_CACHE_SIZE=60G` in the systemd user manager and future login shells on each supported host; this caps future sccache processes and does not stop or clear a live server.
- The only automatic cache root is the exact owned, non-symlink path `~/.cache/pip`; policy cannot retarget its allowlist ID. `uv-cache` is observe-only. Files must be regular, single-link, owned by the user, past TTL, and not found open by the process-handle scan. Cache marker/lock files and paths inside cached `.git` directories are retained. Nested symlinks, other-owner entries and special files are skipped without traversal; unknown process/metadata state fails closed. Each deletion records an intent and completion event in `~/.local/state/jope-diskd/audit.jsonl`, flushed to disk.
- Remote per-job test scratch remains observe-only. The xbabe3 shared Cargo target has the separate exact-path owner exception documented below.
- A registered worktree's `target/` is eligible only if GitHub confirms that the exact local branch/head was merged, the checkout is clean, the target is older than 24 hours, all target entries are ordinary files/directories owned by the user, no protected evidence/report/receipt family is present, `/proc/*/fd` is complete and empty for the tree, and no local process cwd or `CARGO_TARGET_DIR` refers to its target/worktree. Any uncertainty retains the entire target. Worktree targets are considered only above the 85% watermark and at most enough directories are removed to approach the configured low watermark (one full target can overshoot it). Docker storage and `/tmp` remain observe-only.
- The shipped policy rotates only `~/.local/state/jope-diskd/jope-diskd.log` after it exceeds 256 MiB. It creates the archive exclusively and verifies it before removing the active file. Existing archive paths, including symlinks and hardlinks, remain untouched and leave the active log available for a later rotation; open or uncertain handles also prevent rotation. Compressed archives remain local indefinitely. Report offload and S3 access are disabled by policy; the daemon service receives no AWS profile.
- The user service omits `ProtectKernelModules`: unprivileged systemd user managers on xbabe1 and xbabe3 cannot apply that sandbox property and fail with status 218 before starting. Other supported sandbox settings remain enabled.
- `jope-data`, `jope_plan`, rescue/custody, veox, jeryu, admiralty, checkpoints, datasets, receipts/evidence, sealed/holdout and flywheel paths are hard-denied. Live flywheel output changes remain proposal-only until an approved restart.
- A single process lock prevents overlapping apply runs. The policy is the allowlist; unknown IDs or roots are never added to it automatically.

The daemon does not reclaim Docker, arbitrary `/tmp`, or flywheel data. Shared-target cleanup is limited to the exact `~/remote_tests/target` on xbabe1 and xbabe3 via the root-scoped cleaner. It checks all process references and removes only known Cargo output below host-specific free-space floors; top-level `target/tmp`, unknown contents, and active or uncertain targets are retained. On each supported host, a separate user timer also removes only old targets from clean, registered worktrees whose exact branch/head is merged, with no open handles or local process references.

## Local proof

```sh
nice -n 10 python3 -m unittest discover -s tools/ops -p 'test_jope_diskd.py' -v
python3 -m json.tool tools/ops/disk-policy.yaml >/dev/null
python3 tools/ops/jope_diskd.py --config tools/ops/disk-policy.yaml --dry-run
```

The policy is local-only and supports only Linux `xbabe1`, `xbabe2`, and `xbabe3`; it never targets remote scratch on another host. Run the dry-run on each host after merge and inspect proposed paths and handle decisions before enabling the user timer. The shared target cleaner continues to use its dedicated root-scoped service.

## Install after merge

`install_jope_diskd.sh` defaults to a print-only preview. After merge and a host-local dry-run, `--enable` installs the script and policy, sets the 60G sccache environment cap, and configures a ten-minute systemd user timer on the selected host. The installer refuses unsupported hosts, Darwin, symlinked paths, an unmerged source tree, or missing Linux user linger; it never changes account settings. `--enable` is intentionally not run from a feature branch.

The source policy does not alter CI's Cargo target selection or incremental settings. Shared build-target coordination and remote PID handling require separate changes outside this task's allowed repository paths.

## xbabe3 low-space owner exception

The 2026-10-31 xbabe3 owner call permits cleanup of exactly `~/remote_tests/target` while free space is below 60 GiB, after a complete all-process FD/cwd/`CARGO_TARGET_DIR` scan is clear. `tools/ops/remote_test.sh` is the tracked source for the host-local `~/bin/remote_test.sh`; its xbabe3 probe returns unavailable below 60 GiB, and it delegates x3 cleanup to the root-scoped `jope-remote-target-cleaner`. After merge, update the host-local launcher with `install_remote_test_guard.sh`. The cleaner removes only known Cargo output subtrees under `debug/` and `release/`, after verifying ownership, file type, evidence-name exclusions and internal hardlink counts. It retains `target/tmp`, test artifacts, marker files, unknown directories and all `src.<PID>` scratch. Each removed subtree gets flushed delete intent/completion records in `/var/log/jope-diskd-xbabe3/remote-target-cleaner.jsonl`. Below 60 GiB free, it also considers only Docker's dangling volumes matching `jope-*-target`; it rechecks that no running container mounts each volume and requires the local driver and exact Docker mountpoint. Only an empty, canonical, non-symlink mountpoint with complete clear process metadata is eligible; non-empty or uncertain volumes are retained because a Cargo target name does not certify their contents. It rechecks emptiness and process references after the audit intent, then invokes non-force `docker volume rm` so a container-use race is refused by Docker. CI target volumes used by live containers and all non-target volumes remain untouched. The cleaner has no cache, log rotation, or S3 capability. Install it only on xbabe3 from merged source with `install_jope_xbabe3_target_cleaner.sh`.

This exception does not permit deleting or offloading `jarambit_runs`, checkpoints, datasets, receipts, `jope-evidence`, sealed/holdout material, per-job scratch, or any other target. `jarambit_runs` inventory is read-only; a retention/S3 proposal is recorded separately and requires that project's owner approval before action.
# Disk cleanup deployment

The local `jope-diskd` timer is installed on xbabe2. The separate shared-target
cleaner is pinned to xbabe1 and xbabe3, checks every process reference before
removing only verified Cargo output under `~/remote_tests/target`, and retains
`target/tmp` and unknown contents. Its xbabe1 floor is 120 GiB free; xbabe3
remains 60 GiB. Install only from a clean checkout contained in merged
`origin/main` using the matching `install_jope_xbabe1_target_cleaner.sh` or
`install_jope_xbabe3_target_cleaner.sh` script.

The shared policy enables rotation only for the daemon's own oversized log.
Report offload and all S3 operations remain disabled. Apply the policy separately
on xbabe1, xbabe2, and xbabe3 using the merged-source installer and verify each
host's dry-run and timer state. The shared-target cleaners remain separately
installed root services on xbabe1 and xbabe3.
