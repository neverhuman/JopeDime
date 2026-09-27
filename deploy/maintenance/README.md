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
`~/lane-tools/jope-build-report.sh`. Both wrappers run the report core
`jope-ci build-report` (crate `crates/jope-ci`), which must sit beside them; it replaced
`jope_report.py` (Python removal P1, archived at `8a55c4fef`) with the same candidate rule,
JSON documents, receipts and failure behaviour. The two approved Cargo roots are host layout
and are passed by `jope-build-report.sh` (`--approved-root`, twice); the configuration must name
exactly those. GNU coreutils are required; Python is not.
Pass the custody location through
the service's `--custody` argument, independently of `--config`, so malformed
configuration still gets a failure receipt. Custody inside `/tmp` or a Git working
tree is rejected. If custody itself cannot be created/accessed, diagnostics remain
in the systemd journal. The wrapper saves candidate output,
stderr, script/config/unit checksums, systemd invocation ID, exit status, timestamps, per-root allocated
bytes and filesystem usage beneath a custody directory outside
working trees. Failed and interrupted runs retain their diagnostics. A killed
process may leave a `running` receipt or truncate a receipt during a write. Preserve
either result and inspect the systemd journal; neither establishes a complete run.
The next-day comparator refuses malformed JSON rather than accepting partial evidence.
The service has a 30-minute timeout.

The wrapper creates custody if needed and records both supplied and resolved paths.
Missing unit files are recorded explicitly. Completed Cargo scans are retained
even when a separate measurement root is missing; the overall run fails.

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
`group` is a label for per-root comparison, not an aggregated total. Label custody
separately to distinguish report/archive growth from product and research growth.
For retained stores containing root-owned evidence, an individual measurement row
may set `"privileged": true` to invoke only `/usr/bin/sudo -n /usr/bin/du -s -B1 -- ROOT`.
This read-only measurement requires existing noninteractive sudo permission;
failure is recorded, never bypassed. Cargo scanning remains unprivileged.

Before installation, claim the files and preserve their installed bytes and
systemd readback under custody. Stop the existing timer and service. Run shell
syntax checks, ShellCheck, the fixture tests below, and `systemd-analyze --user
verify` on the units. Build the core with `cargo build --release --locked -p jope-ci`
from a clean checkout and install the three helper files (`install -m 0755
target/release/jope-ci deploy/maintenance/cargo-build-dirs.sh
deploy/maintenance/jope-build-report.sh ~/lane-tools/`), the local root configuration
and two units, then `systemctl --user daemon-reload`. Remove the superseded
`~/lane-tools/jope_report.py` only after preserving its bytes under custody. Manually start the service,
inspect its receipt and `systemctl --user cat/show` readback, then enable and
start the timer. Verify `loginctl show-user ubuntu -p Linger` reports `Linger=yes`
so the user manager can run without a login session. Its calendar is explicitly `04:00:00 UTC`, with
`Persistent=true`. On failure leave automatic deletion stopped.

Run the fixture tests with
`JOPE_MAINTENANCE_SCRATCH=CUSTODY_FIXTURE_DIR cargo test --locked -p jope-ci --test build_report -- --ignored`:
`scan_preserves_fixtures_and_fails_visibly` covers candidate selection, refusals, traversal
and measurement failures and exact fixture preservation; `report_receipts_record_every_outcome`
covers configuration, success, overlap, measurement, interruption (a real SIGTERM while the
approved roots are scanned) and failure receipts. Fixtures, inventories and `results.json`
stay under that explicit directory, which must be outside `/tmp` and outside Git working
trees. The fixture directory is proof evidence, not a working lane. Test the installed
helper too with `JOPE_MAINTENANCE_HELPER=/home/ubuntu/lane-tools/cargo-build-dirs.sh`.
The next-day measurement is a separate acceptance gate: scheduling it does not
mean it happened. Keep PR #49 open until its actual successful report is recorded.

Until the installed copy is replaced, `~/lane-tools` still runs the Python `jope_report.py`;
this repository no longer carries it (restore with `git checkout 8a55c4fef --
deploy/maintenance/jope_report.py` if that installed copy must be re-verified).
