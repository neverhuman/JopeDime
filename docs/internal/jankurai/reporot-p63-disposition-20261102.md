# P63 repo-rot finding dispositions

This is a narrow, source-backed false-positive disposition for the three HLT-040 findings in P63. It does not change scanner rules, suppress findings, or classify unrelated paths. All three filenames carry API/schema versions that identify active implementations; none is an abandoned or shadow copy.

## Pinned source and finding identities

| Finding fingerprint | Referencing source file | Current source SHA-256 |
|---|---|---|
| `sha256:9a5b08e3f6c942114b886f110c71ae54bc3962205b85fce34e9fbefcf616ca75` | [`crates/jope-campaigns/src/evaluation_v2.rs`](../../../crates/jope-campaigns/src/evaluation_v2.rs) | `865ad21acae020fd6b9966a98315420f16b814af6b0b9860ada76988f5c65965` |
| `sha256:f148c7a481f1ca174dbe9f0fe7554807e75908e677901bd4bdf7e5edb0b1a90c` | [`crates/jope-campaigns/src/golden42_v3.rs`](../../../crates/jope-campaigns/src/golden42_v3.rs) | `4d6d422780b0c1e20709ccd3e6abbc0177176fb6df8825fbdd7bd4e161973f1e` |
| `sha256:28ce6edadf342d825021d47da54124970306cb09d84b2057e61fe22e81623367` | [`crates/jope-campaigns/src/router_contracts_v2.rs`](../../../crates/jope-campaigns/src/router_contracts_v2.rs) | `0fa85f4c5ec3b1078e494f45b4e166325b8e46114888880ffa5ca06d33c4716b` |

The source hashes above pin the reviewed bytes at the P63 disposition commit. Recompute them before relying on this disposition after any source edit.

## Evidence of active versioned contracts

- `evaluation_v2.rs` is the live module declared by `crates/jope-campaigns/src/main.rs:24` and imported by `crates/jope-campaigns/src/commands/exact_cv_v2.rs`. It implements explicit V2 run, fold, prediction, and receipt schemas. `compute_implementation_sha256` includes this file in the evaluator identity, so moving or renaming it changes a compatibility identity with recorded fixtures and receipts.
- `golden42_v3.rs` is declared by `crates/jope-campaigns/src/main.rs:27`; the CLI dispatches admission, run, validation, worker, and confirmation commands to it at `main.rs:1303-1329`. Its V3 checkpoint, manifest, candidate, and runtime contracts are exercised by its in-file tests.
- `router_contracts_v2.rs` is exported as `pub mod router_contracts_v2` by `crates/jope-campaigns/src/lib.rs` and its public V2 contract types are re-exported by `crates/jope-campaigns/src/hyperion_router.rs`. It is the canonical task-neutral API consumed by the router; the version suffix distinguishes serialized/public contract shape.

The detector labels these paths “fake-versioned-source” based on their names. The imports, CLI dispatch, public exports, schema names, and evaluator identity show that the version suffixes are intentional compatibility labels on current implementations. Deleting or collapsing them would remove live behavior or silently change a persisted identity. No replacement copy or migration target exists.

## Disposition ownership and review

- Owner: `luna_jk_reporot_1_x3`, P63 repo-rot lane.
- Proof lane: current source review plus full `jankurai audit`; source references above are the independent checks for live use.
- Expiry: review when these schema contracts receive an intentional successor/migration, or when the detector gains a semantic check for active versioned modules. Until then this remains a finding-specific false-positive record, not an allowlist or scanner exception.
- Scope: only the three pinned finding/source pairs in this table. Any source-byte change invalidates that row's source pin and requires a fresh review.
