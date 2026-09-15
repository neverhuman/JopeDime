# Native SmartCluster worker contract

`jope-smartcluster-worker` is a fixed registered executable. Submit bootstrap tasks
through `jain_smartcluster::jope::job_spec` and `Client::submit`; its only execution
arguments are `--task-manifest ABSOLUTE --manifest-sha256 SHA256`.
The daemon owns the inherited control socket and exclusive resident allocation.
Never invoke this executable as an unmanaged numerical launcher.

Root must install `/etc/jope-smartcluster/worker.json` with root-owned ancestors
and no group/other write permission. The deny-unknown-fields policy requires:

- `root`: physical canonical JOPE root; `state_root`: durable ext4 worker state.
- `daemon_socket`, `cgroup_root`: native socket and delegated attempt parent.
- `daemon_uid`, `daemon_sha256`, `worker_sha256`: qualified peer/binary identity.
- `allowed_runtime_sha256`: qualified loaded runtime descriptor hashes.
- `cpu_evaluator`, `fit_authority`: optional fixed existing numerical authorities.
- `minimum_available_memory_bytes`: 24 GiB deployment floor.
- `storage_reserves`: absolute mount/directory paths to free-byte floors.
- `storage_devices`: exact mountpoint to expected block-device path (external
  `/dev/disk/by-uuid/CC2EB7CA2EB7AC38`). Missing/mismatched storage fails closed.
- `scratch_root`: daemon's configured trusted scratch root.

`--inspect-runtime` prints a runtime observation without authorizing numerical
work. Qualify linked CUDA libraries separately; startup observation cannot prove
libraries that are loaded lazily. Allow only independently qualified descriptors.
The worker checks the actual inherited peer PID/UID/executable, the socket peer,
process birth identity, live JobRecord attempt and the exact cgroup path. Worker
UID needs read access to its own native JobRecord. Every resident subtask repeats
live-attempt, RAM, disk/device, dependency and runtime checks.

For `task.json`, put the companion input at `task.input.json`. Hash its exact bytes
into the typed TaskManifest input_manifest_sha256. The companion schema is
`jope.smartcluster_input.v1`; fields are `root`, `operation`, `input_artifacts`
(relative file path to SHA256), and `output_receipt` (relative path).
For small inputs, task dependencies must equal the input_artifacts map.
For large closures, omit input_artifacts and set `closure` to
`{"path":"relative/closure.jsonl","sha256":"EXACT_FILE_SHA256"}`. Task
 dependencies then contain only that closure path and hash. Each JSONL record is
`{"path":"relative/artifact","sha256":"ARTIFACT_SHA256"}`. The worker streams
and verifies the closure hash, rejects duplicate paths, limits records to 8192
bytes, files to 256 MiB and entries to one million, then verifies every file and
required transitive edge. This preserves the native task envelope limit. GPU corpus identity
is index.json SHA256; parent identity is checkpoint state.json SHA256.
Search binds complete prepared observations; sealed-test search is refused.
GPU operations also verify the existing resident-protocol dependency closure.

Operations are serde tagged with `kind`: Resident, Search, Pack, Verify, Train,
Proposal. Their exact typed fields are in `smartcluster_worker::Operation`.
Search with seeds maps only to jope-continuation; without seeds only jope-gp-search.
Train/Proposal use jope-resident-gpu, same run identity and bounded allocation.
Proposal requires an explicit hash-bound champion_snapshot file.
Train freezes initial_optimizer_step, requested_step_increment and
 target_optimizer_step; 60000 + 3000 must remain 63000 across interruption.
The native wrappers reuse the existing evaluator, optimizer carry, numerical
training and proposal implementations. Legacy original-grant checks remain held.

Use `dispatch_subtask` to fsync immutable dispatch intent before sending the
`jope-subtask-v1` application command. Identical completed retries verify/reuse the
same immutable receipt; conflicting identities fail. Unfinished nontraining work
requires explicit reconciliation. Training retries preserve unpublished remnants
using checkpoint_recovery and resume the original window. Immutable records use
fsynced temporary files and atomic no-replace publication; interrupted temporary
files remain available for preservation. Native Progress events
reference the durable receipt. ControlAccepted proves delivery only.

Control EOF/checkpoint requests set the numerical stop flag; training checkpoints
at an optimizer boundary then exits. The worker enforces the explicit allowance
with a process exit watchdog if numerical work hangs. Set daemon checkpoint cap
from measured latency. New attempts require the daemon's complete old-cgroup drain.
Recovery-state receipts alone never prove numerical completion.

Scratch admission is a reservation, not a filesystem hard quota. The worker gives
numerical code a per-task TMPDIR beneath inherited SCQ_SCRATCH_ROOT and monitors
its bytes every 100 ms against SCQ_SCRATCH_RESERVATION_BYTES, exiting on overflow or
symlink escape. Permanent corpus/checkpoint outputs live under dataset authority;
monitor root free-space separately. A single write can overshoot the cooperative
scratch budget, so reserve headroom and qualify actual peak writes.

Publication remains a dataset-authority operation: this worker refuses a generic
publication handler. Pack reuses the existing qualification corpus builder;
production teaching-store publication must use the dataset owner's integration.
No source build or CPU test establishes numerical/runtime qualification. Hardware
recovery, real GP/continuation, CUDA optimizer and end-to-end receipt restoration
must pass before activation.
