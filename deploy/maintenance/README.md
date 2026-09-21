# Report-only maintenance

`cargo-build-dirs.sh ROOT [--stale HOURS]` reports Cargo candidates as JSON. It
requires `.rustc_info.json` or `debug/release/.fingerprint`; neither `target`
nor `CACHEDIR.TAG` identifies Cargo output. It reports outermost candidates,
uses descendant modification times, never traverses directory symlinks, and
fails visibly on missing roots, traversal errors or failed disk measurements.
`--delete` is rejected before scanning. Allocated bytes come from `/usr/bin/du -s -B1`.
The candidate sum can count shared hardlinks more than once; it is not reclaimable
space. These measurements may cross mount points and do not equal a single `df` total.

Cargo markers and age do not certify disposable contents. Evidence can occur
at any depth, including `debug`, `release`, and `tmp`. Reclamation requires a
separate logged batch: preserve complete source/evidence, independently restore
and verify it, check current consumers, then remove the explicit approved paths.

The installed entrypoints remain `~/lane-tools/cargo-build-dirs.sh` and the
user units `jope-build-sweep.service` / `.timer`. The service calls
`~/lane-tools/jope-build-report.sh`; install `jope_report.py` beside both helpers.
Python 3 and GNU coreutils are required. Pass the existing custody location through
the service's `--custody` argument, independently of `--config`, so malformed
configuration still gets a failure receipt. Custody inside `/tmp` or a Git working
tree is rejected. If custody itself cannot be created/accessed, diagnostics remain
in the systemd journal. The wrapper saves candidate output,
stderr, script/config/unit checksums, systemd invocation ID, exit status, timestamps, per-root allocated
bytes and filesystem usage beneath an existing custody directory outside
working trees. Failed and interrupted runs retain their diagnostics. A killed
process may leave a `running` receipt; inspect the systemd journal and treat it
as incomplete. The service has a 30-minute timeout.

The machine-local `~/lane-tools/jope-report-roots.json` has this structure:

```json
{
  "cargo_roots": ["/home/ubuntu/JopeDime", "/home/ubuntu/jain-split/jope-dime"],
  "stale_hours": 48,
  "measurement_roots": [
    {"group": "jope", "path": "/home/ubuntu/JopeDime"},
    {"group": "jope", "path": "/home/ubuntu/jain-split/jope-dime"},
    {"group": "custody", "path": "/home/ubuntu/jope-rescue-20260921"}
  ]
}
```

Only the two specified roots are scanned for Cargo candidates. Measurement roots
are explicit resolved directories, with no duplicates or ancestor overlaps;
record deliberately resolved symlink targets in the custody ledger. Preserve
separate unrelated-root measurements for comparison, never scan Bullet or the
broad Claude `/tmp` root for Cargo output. Per-root `du` measurements can include
shared hardlinks, so do not claim their sum equals filesystem use.
For retained stores containing root-owned evidence, an individual measurement row
may set `"privileged": true` to invoke only `/usr/bin/sudo -n /usr/bin/du -s -B1 -- ROOT`.
This read-only measurement requires existing noninteractive sudo permission;
failure is recorded, never bypassed. Cargo scanning remains unprivileged.

Before installation, claim the files and preserve their installed bytes and
systemd readback under custody. Stop the existing timer and service. Run shell
syntax checks, ShellCheck, the fixture test below, and `systemd-analyze --user
verify` on the units. Install the three helper files, local root configuration
and two units, then `systemctl --user daemon-reload`. Manually start the service,
inspect its receipt and `systemctl --user cat/show` readback, then enable and
start the timer. Its calendar is explicitly `04:00:00 UTC`, with
`Persistent=true`. On failure leave automatic deletion stopped.

Run `python3 deploy/maintenance/test_report.py --scratch CUSTODY_FIXTURE_DIR`.
Run `python3 deploy/maintenance/test_wrapper.py --scratch CUSTODY_WRAPPER_FIXTURE_DIR`
for configuration, success, overlap, measurement, interruption and failure receipts.
The tests reject Python optimization mode because their assertions are required.
Fixtures and their before/after hashes stay under that explicit non-`/tmp`
directory. The fixture directory is proof evidence, not a working lane. Test the
installed helper too with `--helper /home/ubuntu/lane-tools/cargo-build-dirs.sh`.
The next-day measurement is a separate acceptance gate: scheduling it does not
mean it happened. Keep PR #49 open until its actual successful report is recorded.
