# jope-diskd

`jope-diskd.py` is a small user-level disk observer and conservative janitor. The policy file is JSON syntax, which is valid YAML 1.2, so Linux and macOS runners need only Python's standard library. The default is `--dry-run`; deletion requires an explicit `--apply`.

## Policy and safety boundary

- At 85% filesystem use, expired entries in the allowlisted pip cache may be pruned toward its configured size cap and the 80% low watermark. At 90%, the TTL shortens to 72 hours. At 95%, an idempotent daily alert is written to `~/nudges/DISK_ALERT.md` (a dry run prints the alert instead). The much larger uv cache remains report-only until its cache structure can be pruned with per-entry audit evidence.
- The only automatic cache root is the exact owned, non-symlink path `~/.cache/pip`; policy cannot retarget its allowlist ID. `uv-cache` is observe-only. Files must be regular, single-link, owned by the user, past TTL, and not found open by the process-handle scan. Cache marker/lock files and paths inside cached `.git` directories are retained. Nested symlinks, other-owner entries and special files are skipped without traversal; unknown process/metadata state fails closed. Each deletion records an intent and completion event in `~/.local/state/jope-diskd/audit.jsonl`, flushed to disk.
- Under the exact `~/remote_tests` root, only direct `src.<PID>` directories are candidates, and only when the matching PID is absent, the directory is older than 24 hours, every entry is a regular user-owned file/directory, the recursive handle scan is complete and empty, and the aggregate root exceeds 60 GiB. Policy cannot retarget this root. The shared `remote_tests/target` is always report-only because remote test jobs share it and the current launcher has its own cleanup guard.
- Per-worktree Cargo targets, Docker storage, and `/tmp` are observed manually, not pruned by the timer. JopeDime targets can contain evidence anywhere in the tree; Docker ownership is shared; `/tmp` currently contains artifacts with unknown owners and active consumers. These limits follow `deploy/maintenance/README.md` and `MASTER_CLEAN_UP.md`.
- `jope-data`, `jope_plan`, rescue/custody, veox, jeryu, admiralty, checkpoints, datasets, receipts/evidence, sealed/holdout and flywheel paths are hard-denied. Live flywheel output changes remain proposal-only until an approved restart.
- A single process lock prevents overlapping apply runs. The policy is the allowlist; unknown IDs or roots are never added to it automatically.

The daemon does not claim to reclaim Docker, worktree target, arbitrary `/tmp`, logs, or flywheel data. Those need an owner-bound preservation/consumer procedure before automatic cleanup is safe.

## Local proof

```sh
nice -n 10 python3 -m unittest discover -s tools/ops -p 'test_jope_diskd.py' -v
python3 -m json.tool tools/ops/disk-policy.yaml >/dev/null
python3 tools/ops/jope_diskd.py --config tools/ops/disk-policy.yaml --dry-run
```

For an archived candidate on another host, set `JOPE_DISKD_POLICY_B64` to the base64 encoding of `disk-policy.yaml` and pipe `jope_diskd.py` to `python3 - --dry-run`. This leaves no staged candidate files on the host. Do not run `--apply` until the reviewed PR is merged and all three host dry runs have been reviewed.

## Install after merge

`install_jope_diskd.sh` defaults to a print-only preview. After merge and host dry-runs, `--enable` installs the script and policy, then configures a ten-minute systemd user timer on Linux or an idempotent ten-minute user crontab entry on macOS. Linux installation refuses to enable the timer unless user linger is already enabled; it never changes account settings. `--enable` is intentionally not run from a feature branch.

The source policy does not alter CI's Cargo target selection or incremental settings. Shared build-target coordination requires a separate change to the remote test launcher, which is outside this task's allowed repository paths.
