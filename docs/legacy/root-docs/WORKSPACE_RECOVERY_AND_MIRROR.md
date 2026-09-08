JOPE-Dime recovery and workspace mirror
=====================================

`/home/ubuntu/jope-dime` on xbabe2 is the merge point. The root application,
`parallel/`, and their packaged source dependencies belong to one source
generation. `vendor/feat-math` remains authoritative; regenerate its development
copy with `python3 parallel/tools/kernel_release.py sync-mirror`.

The GPU workspace remains an active v44 runtime. Workspace synchronization writes
only beneath `/home/ubuntu/jope-dime-mirror` and the separate recovery directory.
It does not replace the live checkout, restart services, change executable routes,
or qualify an upgraded runtime.

Recovery evidence
-----------------

The custody root is `/home/ubuntu/jope-dime-recovery/20260908` on each host.
`root.bundle` was restored into an isolated bare repository and checked with Git
fsck. Twenty-one commits outside ordinary refs were anchored before bundling under
`refs/recovery/20260908/`. Ten existing worktrees have tar snapshots whose payload
hashes, executable modes and link targets were independently verified. The
unavailable temporary worktrees were recorded in `worktrees.txt`; no directory
was removed. Remote source was captured and independently extracted and hashed
on xbabe2. Custody is narrower than complete durable inventory: ignored datasets,
run state, worktree evidence and build-directory evidence require backfill.

The router implementation was recovered byte-for-byte from Jain's August 11
orphan-work archive. Its SHA-256 is
`0fa85f4c5ec3b1078e494f45b4e166325b8e46114888880ffa5ca06d33c4716b`.
Only the test crate import changed. Dime registers the module and retains the
Hyperion compatibility exports. The four archived tests passed within Dime.

The machine-readable ledger is in
[`audits/workspace-recovery-20260908`](audits/workspace-recovery-20260908):

- `provenance.json` binds the router archive, bundle and conversion sources.
- `remote-reconciliation.json` accounts for 50 differing and 147 remote-only
  paths from the initial source comparison. Missing conversion dependencies and
  source references were recovered. Each remaining difference has an explicit
  disposition and a preserved diff in custody.
- `git-recovery-ledger.json` records the 21 anchored commits and individual
  patch-unique branch candidates, including multiclass frozen-input/resume work.
  Their semantic integration remains unresolved; no historical branch was
  merged automatically.
- `worktrees.json` links each source snapshot to its independently verified hash.

`sotarena-refresh` is still the original gitlink to
`acf77fe401a99b5c420b7687d42a825f8d3151d5`. Its directory is empty, there is no
`.gitmodules`, and the object was unavailable in both the root and sotarena
object stores. The gitlink has not been replaced with invented source.

Commands and publication
------------------------

All controller commands run on xbabe2:

```sh
python3 scripts/workspace_sync.py inventory
python3 scripts/workspace_sync.py plan
python3 scripts/workspace_sync.py sync --publish <reviewed-local-snapshot-sha256>
python3 scripts/workspace_sync.py sync
python3 scripts/workspace_sync.py verify
python3 scripts/workspace_sync.py verify --durable
python3 scripts/workspace_sync.py status
python3 scripts/install_workspace_sync.py
```

`plan` captures unfinished source on both nodes and records each differing path.
Publication requires the exact current local snapshot identity; modification
times never select a winning source version. Later timer runs capture new edits
as incoming snapshots while continuing to mirror the explicitly published
canonical generation. They never silently publish a new canonical generation.

`manifests/workspace-sync.v1.json` defines source/dependency roots, mutable roots,
external custody coverage and the SSH route. Inventory walks build directories
and worktrees; custom source and receipts are retained even under `target`.
Known caches, secrets, locks and sockets are excluded. Symlinks are recorded as
pending, never followed. Databases are recognized by their header as well as
their filename and require application-supported snapshots. Open queues are
never declared resumable state.

Each regular durable file is captured as a versioned byte snapshot. It receives
a verified SHA-256 object on both nodes, with its path, size, executable bit,
owner and observation signature in `durable-index.json`. Earlier versions remain
addressable in its history. The index itself is mirrored under its content hash
in `receipts/`; each sync receipt identifies that hash. A successful copy of a
checkpoint, corpus or run does not establish application consistency. That gate
stays pending until the application's completion/checkpoint receipt is verified.

Objects transfer sequentially at an aggregate maximum of 8192 KiB/s (8 MiB/s).
Source and small evidence precede historical backfill. Rsync partials remain in
an unpublished namespace and are resumable; publication independently rehashes
the complete file. Generations materialize into a temporary directory, then
publish atomically only after validating their objects. Conflicting bytes are
preserved under different identities. Existing published corruption is an error
and is never repaired by overwriting another worker's bytes.

`systemd/jope-workspace-sync.timer` runs every five minutes only on xbabe2. Systemd
and the controller lock prevent overlap. The service uses idle I/O priority,
nice 15, a half-core CPU quota and private devices; it performs no GPU work.
The installer freezes the controller, its helpers and configuration under
`/home/ubuntu/jope-dime-mirror/controller/<identity>` and verifies the bytes before
selecting that installation. Source edits do not change the running controller;
reinstall explicitly after reviewing an update. Metadata inventory is resumable
across bounded timer passes, with unvisited coverage reported as unknown.
Fresh protected-route, GP, campaign-cycle and hardware observations are required
before staging and throughout transfers. Missing/stale evidence, invocation
changes, new failed views, or exhausted disk/memory reserves pause synchronization.
A paused receipt never triggers automatic restoration of another worker's edits.

Status reports canonical source parity, incoming conflicts, artifact bytes still
pending, pending live files, observation age and active executable identities
separately. It does not equate command success with complete synchronization.
External dependencies and unavailable historical material remain visible gates.

Restoration
-----------

Source generations are independently verifiable at
`/home/ubuntu/jope-dime-mirror/generations/<generation>`. Copy a verified generation
to a new, empty directory for development or a build. Do not copy it over an
active checkout. Objects are regular files under `objects/<sha256>`; the immutable
manifest in `manifests/<generation>.json` supplies paths and executable modes.
Use a durable index receipt to restore historical bytes to a fresh directory,
then verify the referenced hashes. Restore SQLite/queue state only through its
application's supported recovery procedure and consistency check.

Git history restoration uses `git clone --mirror root.bundle <new-directory>`
followed by `git -C <new-directory> fsck --full`. Worktree tar archives can be
restored into new, empty directories; their adjacent JSON inventories contain
the expected payload hashes and modes. Archive links must be inspected before
extraction into any location used by a running application.

Build and runtime gates
-----------------------

Rust is pinned to 1.97.1 in both application workspaces; Cargo.lock files are
retained. Build from a verified generation with `--locked` and a separate
`CARGO_TARGET_DIR`. The default CPU build does not establish neural build closure.
The two ignored `seeds/generation-zero*.txt` compile-time dependencies were found
by the isolated build and added explicitly to packaged dependency coverage.
CUDA, LibTorch, Python conversion environments, models and their source/license
references still need their own qualification. Remote Fable/Chimera and
target-maker notices were preserved; missing upstream license/lock evidence is
explicitly unresolved.

The first integration checks passed: 211 Dime library tests, 13 Hyperion gate
tests, four recovered router tests, and the default-feature parallel test suite
(99 passed, one pre-existing ignored test). Verification logs live in custody.
Mirror fault tests cover interruption/corruption, disk exhaustion, concurrent
source changes, distinct conflicting versions, symlink escapes, repeated
publication, database detection and preservation of build-directory evidence.
The broader kernel run passed all 83 golden fixtures and the frozen
parser/RNG/fold/prediction/emission checks. It exposed two token-estimator
failures: the estimator's formula fragments omitted the existing clip-failure
guards. Those fragments were repaired to match the emitters; all four estimator
tests then passed, including exact counts on 200 random genomes and the speed
gate. Expected fixture hashes and emitted source were not changed by this repair.

After the current campaign finishes, verify its terminal receipt and saved
checkpoints, stop old producers and drain the finite queue, preserve rollback
runtime/dependencies, and move all consumers of conflicting mutable paths before
publishing a new live checkout. Activation remains subject to the original paired
seed quality/performance gates, 8 → 32 → 104 workers with 100 successful jobs per
stage, eight-GPU rank-failure recovery, 24-hour integrated qualification, fresh
state namespaces, checkpoint migration lineage and demonstrated rollback.

Selective Jain ports remain separate work:

| Jain source | Pending Dime package | Required compatibility evidence |
|---|---|---|
| `registry.rs`: operator IDs, parameter domains, shape/task/license descriptors | Package 2: common fitting/search admission registry | Preserve Dime parser, RNG, folds, supported heads and canonical identities |
| `plan.rs`: typed resolved operators and registry-bound plans | Packages 2–3: typed admission and bounded execution | Compare accepted/rejected plans and budget exhaustion against Dime fixtures |
| Fitted-state bindings associated with typed plans | Packages 5–6: output/state ABI and backend pipeline | Roundtrip old codecs and predictions; reject wrong-state bindings |
| Cooperative cancellation and bounded work mechanisms (candidate audit pending) | Packages 3 and 8: execution and recovery | Process/rank interruption tests and bounded cancellation latency |
| `diagnostics/`: aggregation, uncertainty and cost | Package 7: diagnostics and guarded search | Training-only inputs, complete diagnostic/fallback costs and abstention behavior |

Registry, plan and diagnostics reference hashes are recorded in provenance and
their source copies retained in custody. No Jain kernel replacement or unqualified
port was made. Simplification research stays isolated until rejected-candidate
false negatives, mechanism holdouts and full diagnostic/fallback costs are measured.
