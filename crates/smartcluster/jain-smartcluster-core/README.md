# `jain-smartcluster-core`

This crate defines the serializable contracts shared by Jain Smartcluster
clients, workers, and the daemon. It contains workload and job specifications,
typed identities, resource requests, fenced attempt and state-transition
records, events, and the versioned client/server protocol. It does not implement
scheduling, persistence, transport, or worker execution.

## Contract areas

- `Workload`, `JainWorkload`, and `ServiceName` describe validated workload
  requests.
- `JobSpec`, `JobRecord`, retry/preemption policy, and `IdempotencyScope`
  define submission and job identity contracts.
- `AttemptNumber`, `FenceToken`, `FencedAttempt`, and `FencedTransition`
  represent attempt fencing and legal state transitions.
- `Request`, `Response`, frame types, event types, and protocol version
  constants define the wire-level messages. The protocol is positionally
  encoded; compatible peers must use the same major and minor version until
  feature negotiation is defined.
- `Validate` and `ValidationError` provide semantic checks for decoded values.

The public types are re-exported from the crate root. Validation errors are
returned to callers; this crate does not choose how a daemon reports or retries
them.

## Checks

From the workspace root:

```sh
cargo test --locked -p jain-smartcluster-core
cargo clippy --locked --all-targets -p jain-smartcluster-core -- -D warnings
```

The package denies unsafe Rust. Its contract tests check validation, protocol
compatibility, state transitions, identity scoping, and serialization; they do
not qualify a daemon, scheduler, transport, or worker deployment.
