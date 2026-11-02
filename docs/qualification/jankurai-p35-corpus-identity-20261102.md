# P35 immutable corpus mapping identity transition (2026-11-02)

The rustunsafe repair stores `Records` in a read-only `memmap2::Mmap` only after verifying that
the backing file has `F_SEAL_WRITE`, `F_SEAL_GROW`, `F_SEAL_SHRINK`, and `F_SEAL_SEAL`. The
unverified `Records::open` path now snapshots the count-bounded input into sealed memfd storage
before mapping it. `Records::get` uses safe slice bounds and preserves the parsed record values.

`crates/jope-engine/src/corpus.rs` and `Cargo.lock` are ADR-0004 identity inputs. This repair
therefore creates a new current evaluator identity. The identity-only `compat_freeze` rebaseline
was first generated against source commit `66bf3b548de2f39aef8b26128cbf1331ee2bfb2a`; its retained
receipt is `/home/ubuntu/jope_plan/master/audit/receipts/P35/identity-66bf3b5/identity-regenerate.log`.
After rebasing onto `228c0f32f5fd7fa14afd1e9ad51b51c73151f091`, the same identity-only regeneration
passed against source commit `0c3232528f45c05c1547c9763e6dd695bf7cdabc`; see
`/home/ubuntu/jope_plan/master/audit/receipts/P35/identity-0c32325/identity-regenerate.log`.
The current fixture SHA-256 values are `72a5da670dd4e92572952eed1e88b2edb4aac070f725306b9cb34a308dc345ab`
(`evaluator_identity.v1.json`) and `bb1c0e80206215604c941fa8188e273d5b84194e7cdffbf814fad99aeb52e2e2`
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
has no shipped receipt to add as a historical row. The append-only edge in
`crates/feat-math/tests/fixtures/compat_freeze/transitions/p35_sealed_corpus_identity_rebaseline.v1.json`
links both fixture successors; `historical_v1` passes 3/3.
