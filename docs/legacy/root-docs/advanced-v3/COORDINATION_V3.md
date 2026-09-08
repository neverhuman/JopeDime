# Advanced V3 coordination protocol

This protocol supersedes the live-lease role of `ROADMAP.md` for Epoch 6 and
Search V3 work. It does not rewrite V1 or V2 history.

## Authorities

1. `manifests/advanced-v3/task-graph.v3.json` is task-state authority.
2. `manifests/advanced-v3/work-ledger.v3.jsonl` is verified-event authority.
3. `REALTIME_AGENT_CHAT.md` is the append-only real-time coordination surface.
4. Scientific claims live in content-addressed receipts and artifacts. A chat
   acknowledgement is coordination evidence, not scientific evidence.
5. V1/V2 graphs, ledgers, receipts, and `ROADMAP.md` are immutable history.

## Canonical event line

Append one seven-field Markdown line with
`scripts/append_realtime_agent_chat.py`:

```text
timestamp | actor | EVENT lease=<id> task=<id> ... | evidence | next action | blocker | handoff
```

The helper locks the canonical chat file, collapses whitespace in every field,
rejects newlines and field separators, appends exactly one line, fsyncs the
file and its parent directory, and never rewrites existing bytes. Read the
latest 20 entries before claiming a file family or live resource.

## Required events

- `STATE_FREEZE` records HEAD, branch, staged/unstaged/untracked inventory,
  submodule state, active runtime, protected job IDs, and content hashes for
  every pre-existing dirty path.
- `CLAIM_REQUEST` names exact files/resources, baseline hashes, interfaces,
  intended mutations, tests, reviewer, and explicit exclusions.
- `CLAIM_ACK` or `CLAIM_CONFLICT` cites the lease and exact overlap.
- `HEARTBEAT` is appended every 15 minutes during source work and every five
  minutes during submission, cleanup, synchronization, service changes, or
  cutover. Append one before and after long builds or tests.
- `CHECKPOINT` lists changed paths and hashes, completed checks, failures, and
  remaining work.
- `ACTION_PROPOSAL` plus bilateral `ACTION_ACK` is required before job
  cancellation, priority changes, garbage collection, SCQ submission,
  immutable publication, service changes, bundle staging/activation,
  quarantine movement, or pointer mutation.
- `HANDOFF`, independent `REVIEW_ACK`, and `LEASE_RELEASE` end a lease. A lease
  is not released merely because its author stopped writing.

## Source and index isolation

Use clean, separate worktrees and Git indexes for concurrent actors. Before a
commit, stage explicit paths only, inspect `git diff --cached --name-only`, and
record the staged list in chat. Preserve `.claude/`, dirty submodules,
bytecode, caches, and unrelated user changes.

Shared public boundaries (`Cargo.toml`, `src/lib.rs`, task/Epoch 6 APIs,
graphs, ledgers, artifact schemas, service/spec registries, and production
queue interfaces) are serialized by the root coordinator. Search V3 owns only
new `src/search_v3/**`, `src/bin/search-v3.rs`, and Search-specific tests and
artifacts unless a narrower shared-boundary claim is acknowledged.

## Evidence and transitions

Ledger entries are canonical compact JSON objects, one per line, with
contiguous sequence numbers and a SHA-256 chain. Existing lines are never
edited. A correction is a new event citing the earlier sequence. Task state
normally advances `queued -> ready -> in_progress -> review -> verified` only
after its dependencies and acceptance evidence validate.

Failure to read an authority, a stale operational receipt, an identity drift,
or a missing reciprocal acknowledgement is a fail-closed condition. Preserve
the last verified state and do not perform the affected live action.
