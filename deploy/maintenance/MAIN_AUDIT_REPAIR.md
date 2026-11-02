# Main-audit source-deletion reconciliation

The hourly DSC08 helper failed before execution at `8aa627bad4cb7855916bd476e2f60d81576fc772`:
it reported 42 shared Python module removals as unexpected and tried to read
`crates/smartcluster/Cargo.toml`, which is not a workspace package manifest.

`main-audit-deletions.json` records the exact 42 paths, their shared import
locations, source hashes and package search-path bootstraps at that commit and
the reviewed fastlane source. The removals are from #1929 and related
#1921/#1960/#2021/#2049 deduplication landings. Static resolution located each
module through the declared package paths and parsed its source. No package
initializer, model, native runtime or module body was executed; this is source
import-closure evidence, not packaged-runtime or learning qualification.

`reconcile-main-audit` applies a narrow patch to the two existing host-local
operator helpers. It refuses a changed predecessor, preserves the old scripts,
parses the replacement source and installs under the existing hourly mutex.
The exact 42-path policy is installed beside the helpers and SHA-pinned by the
consumer. The original eleven leak/custody removals remain required, including
the immutable external-custody recovery-map pin. Other removals and new model
or bulk copies remain refused. A shared module restored in a later tree is not
required to be deleted again.

The package selector reads the pinned root workspace's member paths and each
changed member's actual manifest. Nested smartcluster packages are selected by
their package names; no aggregate `crates/smartcluster/Cargo.toml` is assumed.

From this source tree, prepare a fresh preservation directory outside Git:

```sh
nice -n 19 python3 -B deploy/maintenance/reconcile-main-audit \
  --preserve "$AUDIT_REPAIR_RECEIPTS/check"
nice -n 19 python3 -B deploy/maintenance/reconcile-main-audit --install \
  --preserve "$AUDIT_REPAIR_RECEIPTS/install"
```

Use different fresh check/install directories. The command does not start an
audit, build, test, source update or timer; the existing owner retains those
operations. Historical audit receipts and their FAIL/NOT_RUN verdicts stay
unchanged. New audit evidence must come from a later actual owner run.
