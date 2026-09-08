# Unified Recovery, Encoder V2, and Five-Pillar Search V3

This document is the human-readable companion to the authoritative
`manifests/advanced-v3/task-graph.v3.json` and append-only
`manifests/advanced-v3/work-ledger.v3.jsonl`.

The V1 and V2 graphs, ledgers, receipts, and `ROADMAP.md` are frozen inputs.
They remain valid history but cannot authorize new work. Every former task ID
maps exactly once to a canonical V3 milestone in the graph migration.

Release A is CPU-first regression/binary Search V3 with std-only Rust and
dependency-free Python export. All four tasks remain exhaustive in type and
artifact contracts. Multiclass, timeseries, GPU, Hyperion, C++17, and ES2020
are independent additive lanes and cannot change Release A identities.

## Current authority state

- `U0-AUTHORITY` is in progress. The external fixture index contains 422
  records and its 254-file workspace-test profile materializes from pinned Git
  objects into a clean, read-only tree. Full verification remains fail-closed:
  46 Binary42 train/test files have pinned SHA-256 values but the pinned
  metadata records zero bytes and no retrieval URL, so their exact sizes and
  bytes cannot yet be verified.
- Epoch 6 task semantics and foundational V3 Rust contracts are an offline
  implementation handoff; their milestone cannot advance while U0 is open.
- Production pointers, queues, catalogs, training, promotion, expansion, and
  accepted runtime `9521ffce…` remain unchanged.
- SCQ80 is permanently test-adaptive and benchmark-invalid. It is retained
  only for an integrity audit followed by archival.
- The dirty `sotarena` submodule and `.claude/` directory remain outside this
  change and are recorded in the migration handoff.

## Canonical sequence

1. Authority and external fixtures.
2. Epoch 6 operational foundation.
3. Executable recovery and production Encoder V2.
4. Recovery cutover and 86,400-second continuity.
5. Search scientific evidence and no-forgetting selection.
6. Typed compilation and one fitted Rust/Python model bundle.
7. Deterministic CPU execution and benchmark authority.
8. Regression/binary shadow release and one bounded CPU canary.
9. Independent GPU and Hyperion qualification.
10. Independent multiclass and timeseries activation.

No failed gate is waivable. An incomplete phase leaves the last verified CPU
and recovery state runnable and does not advance a production pointer.
