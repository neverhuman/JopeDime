# P35 immutable corpus mapping identity transition (2026-11-02)

The rustunsafe repair stores `Records` in a read-only `memmap2::Mmap` only after verifying that
the backing file has `F_SEAL_WRITE`, `F_SEAL_GROW`, `F_SEAL_SHRINK`, and `F_SEAL_SEAL`. The
unverified `Records::open` path now snapshots the count-bounded input into sealed memfd storage
before mapping it. `Records::get` uses safe slice bounds and preserves the parsed record values.

`crates/jope-engine/src/corpus.rs` and `Cargo.lock` are ADR-0004 identity inputs. This repair
therefore creates a new current evaluator identity. The identity-only `compat_freeze` rebaseline
was generated against source commit `1af2e6e337a3aa1d27c85bcdf3a1d68840222d53`; its retained
receipt is `/home/ubuntu/jope_plan/master/audit/receipts/P35/identity-1af2e6e/identity-regenerate.log`.
The fixture SHA-256 values are `98138e61ecab06d77285ac755503506f650c1998baa4f401db965a69db96703c`
(`evaluator_identity.v1.json`) and `8c01e09b7ceb3c8d27c38ad44ce3fc00d09106662f394a8787d58cc4ede7773c`
(`lineage_sources.v1.json`).

| Field | Before | After |
| --- | --- | --- |
| `implementation_sha256` | `e4dd3486d173c65209da3df7fdc6cc8aaccdbc5d7e0102eab512bd6f7f491716` | `143fefc220a06431b4217c95dfa98dd8a23b5297ed69cee3acdc35cc3075128d` |
| `cargo_lock_sha256` | `e0f1d2f90337bf2df4930b051cf6b3115ee0b27bc5533c56de4b8a927678a401` | `06f623742889266c3493eee1be2a1bc2453d61e0afab042e08a3de0db55487a9` |
| `identity_digest` | `6ff84deb38b32b725960d9b18a1a7775ea7df6593fc4eb575f54a6611e22fd1d` | `545bf844bebb344e6f17dba49c6050bf98bcabe330326a0beb75e506aafe6f39` |
| `dsl_abi_sha256` | `c8b4babfa7dc5b65bf614ffc4c9cdb3a8ed08f42ce1b7087481bbdcce92a8389` | unchanged |

The identity-only rebaseline ran with `JOPE_COMPAT_FREEZE_REGENERATE=identity` and passed all
13 `compat_freeze` tests. It changed only the evaluator identity and lineage source fixtures;
emitted bytes, model serialization, numerical values, seeded streams, DSL ABI, and historical
identity rows were retained. The known-identity table remains unchanged because this new identity
has no shipped receipt to add as a historical row.
