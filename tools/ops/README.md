# jope-diskd

`jope-diskd.py` is a user-level disk observer and local-only janitor. The policy file is JSON syntax, which is valid YAML 1.2, so Linux and macOS runners need only Python's standard library. The default is `--dry-run`; apply requires `JOPE_DISKD_LOCAL_ONLY=1`, the policy-pinned host (`xbabe2`), and an explicit `--apply`.

## Policy and safety boundary

- At 85% filesystem use, expired entries in the allowlisted pip cache may be pruned toward its configured size cap and the 80% low watermark. At 90%, the TTL shortens to 72 hours. At 95%, an idempotent daily alert is written to `~/nudges/DISK_ALERT.md` (a dry run prints the alert instead). The much larger uv and sccache caches remain report-only. The installer sets `SCCACHE_CACHE_SIZE=60G` in the systemd user manager and future login shells; the daemon never stops or clears a live sccache server.
- The only automatic cache root is the exact owned, non-symlink path `~/.cache/pip`; policy cannot retarget its allowlist ID. `uv-cache` is observe-only. Files must be regular, single-link, owned by the user, past TTL, and not found open by the process-handle scan. Cache marker/lock files and paths inside cached `.git` directories are retained. Nested symlinks, other-owner entries and special files are skipped without traversal; unknown process/metadata state fails closed. Each deletion records an intent and completion event in `~/.local/state/jope-diskd/audit.jsonl`, flushed to disk.
- All remote test scratch and the shared `remote_tests/target` are observe-only. The launcher has a host/PID mismatch, so no remote scratch is eligible for automated deletion.
- A registered worktree's `target/` is eligible only if GitHub confirms that the exact local branch/head was merged, the checkout is clean, the target is older than 24 hours, all target entries are ordinary files/directories owned by the user, no protected evidence/report/receipt family is present, `/proc/*/fd` is complete and empty for the tree, and no local process cwd or `CARGO_TARGET_DIR` refers to its target/worktree. Any uncertainty retains the entire target. Worktree targets are considered only above the 85% watermark and at most enough directories are removed to approach the configured low watermark (one full target can overshoot it). Docker storage and `/tmp` remain observe-only.
- The daemon log rotates only after exceeding 256 MiB, gzip-verifies the archive, and retains local archives for 14 days. Older daemon logs and jankurai reports under the canonical repo/worktree target roots are uploaded to `s3://veox-jopedime-learning-use2/offload/<host>/<YYYYMMDD>/...`; a content SHA-256 is embedded in the key, downloaded and verified before local removal. Upload failures, unreadable handles, protected paths, and source changes retain local files. Checkpoints, datasets, receipts, evidence, and flywheel paths are excluded.
- `jope-data`, `jope_plan`, rescue/custody, veox, jeryu, admiralty, checkpoints, datasets, receipts/evidence, sealed/holdout and flywheel paths are hard-denied. Live flywheel output changes remain proposal-only until an approved restart.
- A single process lock prevents overlapping apply runs. The policy is the allowlist; unknown IDs or roots are never added to it automatically.

The daemon does not reclaim Docker, arbitrary `/tmp`, or flywheel data. Remote-scratch cleanup remains disabled until the launcher PID mismatch is fixed in a separate change.

## Local proof

```sh
nice -n 10 python3 -m unittest discover -s tools/ops -p 'test_jope_diskd.py' -v
python3 -m json.tool tools/ops/disk-policy.yaml >/dev/null
python3 tools/ops/jope_diskd.py --config tools/ops/disk-policy.yaml --dry-run
```

The policy is deliberately host-bound to local Linux `xbabe2`. Do not use it for remote hosts. Run the dry-run on xbabe2 after merge, inspect every proposed path and handle decision, then let the merger/owner enable the user timer. No remote target cleanup is enabled.

## Install after merge

`install_jope_diskd.sh` defaults to a print-only preview. After merge and the local xbabe2 dry-run, `--enable` installs the script and policy, then configures a ten-minute systemd user timer on xbabe2. The installer refuses other hosts, Darwin, or missing Linux user linger; it never changes account settings. `--enable` is intentionally not run from a feature branch.

The source policy does not alter CI's Cargo target selection or incremental settings. Shared build-target coordination and remote PID handling require separate changes outside this task's allowed repository paths.
