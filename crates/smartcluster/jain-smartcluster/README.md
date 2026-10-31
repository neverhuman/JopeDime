# `jain-smartcluster`

Async Unix-domain socket client for the Jain Smartcluster daemon, plus JOPE task-manifest helpers. The package exposes the shared wire and domain types from `jain-smartcluster-core`; it does not run a scheduler, persist jobs, or execute workloads.

## Client and protocol

`Client` sends typed requests to the daemon socket. The default path is `/run/jain-smartcluster/scqd.sock`; use `Client::new` or `ClientConfig` to select another path and configure connection, request, wait, frame-size, and stream-buffer limits. Operations return `DispatchError` for configuration, validation, transport, protocol, timeout, and daemon failures.

`FrameCodec` encodes Postcard messages behind a four-byte big-endian length prefix. It rejects empty, oversized, truncated, malformed, and trailing-byte frames. The default frame limit is 8 MiB; the absolute limit is 16 MiB.

Job events are delivered with offsets. Subscribe again from the first unseen offset to resume; `JobMonitor` reconnects after transient transport errors with bounded backoff. A compacted event history is reported as `DispatchError::StreamGap`, so callers can decide whether the daemon's available offset is an acceptable resume point. Persist the last delivered event or the next offset in the caller's own durable state when restart continuity matters.

## JOPE task manifests

The `jope` module accepts a bounded JSON task manifest (`jope.smartcluster_task.v1`) and turns it into a validated native `JobSpec`. `job_spec` reads at most 1 MiB from an absolute normalized regular-file path without following a symlink, checks the caller-provided SHA-256 against the exact bytes, parses and validates the manifest, and binds the digest and task identities into the submitted workload. The service rechecks the digest and dependencies before doing work.

Manifest submissions are declarations and scheduling requests. A successful submit or control acknowledgement is not a result receipt, does not prove that a service executed the task, and does not certify its output. The registered service is responsible for dependency checks, execution, and its durable receipt. The `resident_gpu` task kind is a contract for an explicitly reserved exclusive GPU and checkpoint allowance; this client does not qualify or authorize a GPU run.

## Validation

From the workspace root:

```sh
cargo test --locked -p jain-smartcluster
cargo clippy --locked --all-targets -p jain-smartcluster -- -D warnings
```

The tests cover client behavior against local Unix sockets, event reconnect/offset handling, manifest validation, frame bounds, and checked-in wire fixtures. They do not qualify a production daemon, multi-host scheduler, service deployment, GPU execution, or end-to-end receipt pipeline.
