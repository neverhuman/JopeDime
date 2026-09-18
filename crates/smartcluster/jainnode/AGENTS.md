@/home/ubuntu/.codex/RTK.md

# jainnode OCI executor

This is a trusted host-side OCI composition root. Workloads receive exact argv and never receive
the Docker socket, host namespaces, privileges, inbound ports, caller host paths, or unselected
devices. There is no native or shell fallback.

Keep inputs read-only through `ShardPort`, outputs private until sealed, smoke mandatory before the
real attempt, events bounded and fenced, and terminal success conditional on durable Shard
receipts. Missing isolation, network enforcement, metrics, cancellation, or storage capability is
a typed failure, not an emulation.

Run `cargo test --locked -p jainnode` and strict Clippy after changes. Hardware tests must not claim
qualification when Docker, cgroup, namespace, network-filter, NVIDIA, or Shard quorum is absent.
