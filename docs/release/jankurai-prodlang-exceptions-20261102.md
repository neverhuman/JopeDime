# Prodlang false-positive disposition

The full audit `rerun/j_ffba82986.json` reports HLT-046 fingerprint
`sha256:c8f09a3c9174aee1e672bc02d36c2f6baae9998d6b7a3c92b5fbd62bdc00938e`
at `crates/jope-pipeline/src/resource_ledger.rs:141` for an enum named `Stage`.

This is a lifecycle accounting domain with stages `Prepare`, `Encode`, `Compile`,
`Fit`, `Evaluate`, `Checkpoint`, `Publication`, and `Custody`. The similarly named
`Stage` in `crates/jope-records/src/research/mod.rs` represents research evidence
phases (`Synthetic`, `Shadow`, `Adaptive`, `Confirmatory`); the one in
`crates/jope-service/src/ranker/cuda.rs` represents tensor projection checkpoints
(`V256`, `Col`, `Dope629`, `Lm48`, `Tokens`, `Mask`, `Costs`). Sharing their enum
would conflate unrelated lifecycle, evidence, and tensor domains.

The exact source path is excluded from Jankurai scoring in `agent/audit-policy.toml`.
No source, serialization, or runtime behavior is changed. The disposition applies
only to this path; revisit it if the domains are redesigned or this enum changes.

## Ledger refusal comment

The full audit `rerun/j_161a8216d.json` reports HLT-027 fingerprint
`sha256:7985f857419f6e5c3a56385852c62b79490492fea81d320002f5b8f7533ea211`
at `crates/jope-learning/src/hwm.rs:276`. The matched text is a Rust documentation
comment describing why an unrepresentable candidate cannot be written into the
append-only ledger. It describes refusal behavior and the risk of fabricating a
record; it makes no claim of human review, CI success, or external proof.

This exact path is excluded from Jankurai scoring in `agent/audit-policy.toml`.
No source or runtime behavior is changed. Revisit if the comment or the refusal
contract changes.
