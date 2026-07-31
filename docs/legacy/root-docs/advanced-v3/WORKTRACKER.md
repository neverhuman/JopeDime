# Advanced-v3 work tracker

This tracker is the human-readable companion to `manifests/advanced-v3/work-ledger.v1.jsonl`.
The JSONL ledger is append-only. Source implementation is locked until G-01 records a passing
legacy-v2 freeze. Existing manifests and run roots are immutable and remain legacy-v2.

## Baseline

- Takeover date: 2026-07-22 UTC
- Source control: none; no repository was initialized
- Canonical pre-change snapshot: `/home/ubuntu/jope-dime-snapshots/2285c594ec8d12ae5d66a4e751db92c806e97a6a84d7c940c0dca75a25f3a096.tar.zst`
- Snapshot SHA-256: `2285c594ec8d12ae5d66a4e751db92c806e97a6a84d7c940c0dca75a25f3a096`
- Snapshot exclusions: `target/`, `runs/`, `datasets/`, `data/`, and cache directories
- Baseline implementation hash: `vendor/feat-math/src/advanced_v3.rs` = `057298f8685f57630593c0cdda3aad7c41d7d5313bd5da8ca0788bee0d2bfc22`

## Ownership

| Actor | Exclusive paths |
|---|---|
| `/root` | public module/re-export boundaries, root `Cargo.toml`, `src/main.rs`, shared public interfaces, ledgers, integration and gates |
| `/root/kernel_workstream` | advanced-v3 kernel leaf modules except grammar/profile/evolution/screening |
| `/root/search_campaign_workstream` | grammar/profile/evolution/screening, battle-gpu logistic additions, and `src/advanced_v3/` application modules |
| `/root/verification_workstream` | new tests, fixtures, benchmark/admission harnesses and evidence validators |

## Current phase

| Task | State | Notes |
|---|---|---|
| G-00 | verified | Deterministic archive created, hashed, and listed successfully |
| G-01 | verified | 83-test broad golden passed; fast 3-test immutable freeze added and independently rerun |
| G-02 | verified | Ledger, ownership validator, task graph, and snapshot procedure established |
| G-03 | verified | Five-run emission, Ridge-GCV, search, CPU, RSS, and environment receipt recorded |
| G-04 | verified | Machine-readable acceptance policy skeleton created and validated |
| K-00 | verified | Monolith mechanically split; all 18 v3 tests and legacy freeze pass |
| K-01 | verified | Eight opaque shared internal contracts compile; existing public paths remain unchanged |
| K-02 | verified | Public parity/property targets added; 18 unchanged private subsystem assertions retained |
| K-10/K-11 | verified | Policy-bound schema hash and strict runtime CSV parser pass focused/public/legacy gates |
| K-12 | verified | Canonical lineage, masks, sparse indexes, and independent raw/dense/sparse width accounting |
| K-13/K-14 | verified | Input modes plus honest IID and ordered historical encodings pass leakage and legacy gates |
| K-15 | in progress | Group/order-aware lag and rolling transforms with retained fitted tails |
| S-10/S-11 | verified | Typed registry, strict compiler, canonical defaults, width checks, and semantic hashes |
| S-20 | verified | Exact 24-statistic train-only profile, bounded sketches, and fold-local landmarks |
| S-21 | in progress | Four protected 30-slot islands and typed seed catalogs |

## Gate procedure

1. The owner reports changed paths, interfaces, risks, targeted checks, and evidence.
2. Root checks the changed-path inventory against ownership and reruns targeted checks.
3. The legacy-v2 freeze must pass after every implementation task.
4. A different actor reviews the handoff when practical.
5. Root appends evidence hashes to the JSONL ledger.
6. Phase-closing tasks receive a deterministic archive under `/home/ubuntu/jope-dime-snapshots/`.

Task states are `queued -> ready -> in_progress -> review -> verified`, with terminal
`blocked` and `rejected` branches. Admission and held-out failures are never waived.
