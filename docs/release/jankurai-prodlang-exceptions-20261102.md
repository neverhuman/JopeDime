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
