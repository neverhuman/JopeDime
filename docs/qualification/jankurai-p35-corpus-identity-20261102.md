# P35 immutable corpus mapping identity transition (2026-11-02)

The rustunsafe repair stores `Records` in a read-only `memmap2::Mmap` only after verifying that
the backing file has `F_SEAL_WRITE`, `F_SEAL_GROW`, `F_SEAL_SHRINK`, and `F_SEAL_SEAL`. The
unverified `Records::open` path now snapshots the count-bounded input into sealed memfd storage
before mapping it. `Records::get` uses safe slice bounds and preserves the parsed record values.

`crates/jope-engine/src/corpus.rs` and `Cargo.lock` are ADR-0004 identity inputs. This repair
therefore creates a new current evaluator identity. The identity-only `compat_freeze` rebaseline
was first generated against source commit `66bf3b548de2f39aef8b26128cbf1331ee2bfb2a`; its retained
receipt is `/home/ubuntu/jope_plan/master/audit/receipts/P35/identity-66bf3b5/identity-regenerate.log`.
After rebasing onto `03c9d035ef31aa832a63b87241a5ded4f5cbcdb1`, the same identity-only regeneration
passed against source commit `44840c49c0ceba7f161d42fa7726089343bf7d49`; see
`/home/ubuntu/jope_plan/master/audit/receipts/P35/identity-0c32325/identity-regenerate.log`.
The current fixture SHA-256 values are `316c26afb04bb94b53147b7b2bf309f2148a09b465dc55d501ef0008e49dd7a2`
(`evaluator_identity.v1.json`) and `9de2dce45cc7852aa86d5e1eb0b1e313a6aed340942e17b503b22780add11e5c`
(`lineage_sources.v1.json`).

| Field | Before | After |
| --- | --- | --- |
| `implementation_sha256` | `e5b6edc767a192b3ce7b14b132b68cd29fa72407282ca3231562671ea236e3d5` | `43c6ae985f652b408c94f1e5f60c93a4ecf99e801f5a11a189cda1a55d747c4f` |
| `cargo_lock_sha256` | `e0f1d2f90337bf2df4930b051cf6b3115ee0b27bc5533c56de4b8a927678a401` | `7d90c1d3f86960fcfbbf66e848d22e665c446debec38e386f512ef09c8a28042` |
| `identity_digest` | `2febbecb129d61930a625d5aa15f7f1a49ecf035c352e4999fc1719b88b20e86` | `9dab8f04a9bef7f36a5b432eb2123a8ff2fac9acba434ccc4331ba4cf25bf5c8` |
| `dsl_abi_sha256` | `c8b4babfa7dc5b65bf614ffc4c9cdb3a8ed08f42ce1b7087481bbdcce92a8389` | unchanged |

The identity-only rebaseline ran with `JOPE_COMPAT_FREEZE_REGENERATE=identity` and passed all
13 `compat_freeze` tests. It changed only the evaluator identity and lineage source fixtures;
emitted bytes, model serialization, numerical values, seeded streams, DSL ABI, and historical
identity rows were retained. The known-identity table remains unchanged because this new identity
has no shipped receipt to add as a historical row. The append-only edge in
`crates/feat-math/tests/fixtures/compat_freeze/transitions/p35_sealed_corpus_identity_rebaseline.v1.json`
links both fixture successors; `historical_v1` passes 3/3.
