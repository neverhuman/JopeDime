# Advanced V3 coordination protocol

This protocol governs Search V3 work after the V1-to-V2 metadata migration. It does not
reinterpret or replace `task-graph.v1.json` or `work-ledger.v1.jsonl`.

## Authorities

Use the authorities in this order:

1. `ROADMAP.md` is the live lease, heartbeat, blocker, and handoff board.
2. `task-graph.v2.json` defines task dependencies, intended paths, interfaces, deliverables,
   and acceptance checks. Its imported V1 tasks preserve V1 owner, dependencies, and state. The
   graph remains a reconciled draft until `P-00` is verified with a deterministic snapshot.
3. `work-ledger.v2.jsonl` is the append-only verified-event chain. Its genesis entry seals V1
   through sequence 51 and the initial V2 graph. V1 remains immutable evidence.
4. Immutable state-transition receipts live under
   `manifests/advanced-v3/task-receipts/<task-id>/<sequence>.v1.json`; test and benchmark
   evidence remains under `feedback/implementation/advanced-v3/`.
5. Content-addressed archives under `/home/ubuntu/jope-dime-snapshots/` are byte-recovery
   authority because the Jope checkout is not a Git repository.

This program writes only `/home/ubuntu/jope-dime`. Any future maintained-Jain change requires a
separate user authorization and its repository's own governed review process.

## Lease lifecycle

`/root` is the sole physical writer of `ROADMAP.md`. Workers send events to `/root`; the
coordinator records them promptly. This prevents concurrent patches from losing lease data.

1. **claim_request**: name the task, exact paths, baseline hashes, dependencies, interfaces,
   intended checks, and reviewer.
2. **lease_granted**: the coordinator confirms that no active lease overlaps the paths.
3. **heartbeat**: report progress at least every fifteen minutes and before and after long tests.
4. **checkpoint**: report changed paths, current hashes, completed checks, and remaining work.
5. **blocked**: identify the blocker and whether current bytes are safe to snapshot.
6. **handoff**: stop writing and report exact paths, final hashes, interfaces, checks, evidence,
   risks, and requested reviewer.
7. **review**: a different actor inspects the handoff and reruns proportionate checks.
8. **verified**: `/root` records evidence in the V2 ledger and seals shared-path work when
   recovery value warrants it.
9. **lease_released**: the coordinator removes the active lease only after handoff or recovery.

A lease never expires automatically. Before recovery or transfer, the coordinator confirms
that the owner is inactive, records the last known hashes, and preserves WIP in a labeled
snapshot. A recovered lease receives a new event; history is never rewritten.

## Path and evidence rules

- A worker edits only its leased exact paths. A directory entry in the graph is planning scope,
  not permission to overlap a live lease.
- Shared public boundaries belong to `/root` unless a narrower lease explicitly grants them.
- Caches and prose status are not scientific evidence. Missing or corrupt caches cause
  recomputation.
- V1 and V2 commands, fixtures, receipts, frozen hashes, and selection behavior remain frozen.
- Every ledger event is one canonical JSON object per line. Sequence numbers are contiguous;
  `prev_entry_sha256` names the preceding entry; and `entry_sha256` hashes the canonical object
  with that field removed.
- Use UTF-8 JSON serialized with sorted keys and compact separators for the entry hash.
- Never edit an existing ledger line. If an event is wrong, append a correcting event that
  cites the incorrect sequence and explains the correction.
- Run `rtk python3 scripts/validate_advanced_v3_v2.py` before governance handoff and before a
  release gate consumes V2 metadata.

## Task transitions

The normal transition is `queued -> ready -> in_progress -> review -> verified`. `blocked` and
`rejected` are terminal records unless a later recovery task explicitly supersedes them. A task
may become ready only after all dependencies are verified. Acceptance checks and evidence must
be recorded before verification; status prose alone cannot verify a task.

The additive Search V3 milestone tasks (`REC-00`, `M0-CANON`, and `M1-EVIDENCE` through
`M8-PACKAGE`) are preserved planning history. The reconciled portable/search-v2 gate spine is
`P-00`, `P-1A`, `P-1B`, `P-1C`, `P-1D`, and `P-2` through `P-6`. `P-6` alone is optional and
starts only after the core release gate. Legacy task IDs remain the implementation authorities
referenced by the P-gates; completing a legacy task does not by itself satisfy a wave gate.
