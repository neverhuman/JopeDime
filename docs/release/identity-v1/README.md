# v1.0.0 evaluator identity transition

Local release preparation on qualified receiving main
`aa18ebfbf38c23d6f708b8fbd803ddb93b627db3`. The authorized product version bump changes
only version fields in hashed `crates/jope-campaigns/Cargo.toml` and workspace `Cargo.lock`.
The sealed-runtime libc dependency and all receiving identity successor ledgers are retained.
Evaluator source, DSL ABI, behavioral fixtures and original history/source snapshots do not change.

| Identity field | Receiving main aa18ebfbf | Version-bumped source 8b3111e7 |
| --- | --- | --- |
| implementation_sha256 | c18d66a0504d892ca1bb41e5222f987efce18185eb1ed1ac8017d095f12b110a | b2f5222e2c97d29f86baa224cda5f80cce898e90a3996edb1692af2a438bf715 |
| cargo_lock_sha256 | 211457420a679d8b3be5a96f84879cb47fa400a3ce885780c15ac077f319348d | 02a993e813e0206cce1c35bc5c1cf49513576fc27d238f500642b86406c691f2 |
| identity_digest | e767781cec10739eafa8242bd0c74885ecdf376b363a918f251e567fe4926bd0 | 214451a6ddf1262e1a30c95339c91c918be555aa72eaa73eb683bd5ff95f616a |

| Changed identity input | Old SHA-256 | New SHA-256 |
| --- | --- | --- |
| `crates/jope-campaigns/Cargo.toml` | 06955340d10a2c70ce71717b52c5f984224c49a37c394191bd85896b2b09c9c0 | 355a7a154e5ec0b763f015eaa0233823e5e35202b2dadbe979390186fc41e32e |
| `Cargo.lock` | 211457420a679d8b3be5a96f84879cb47fa400a3ce885780c15ac077f319348d | 02a993e813e0206cce1c35bc5c1cf49513576fc27d238f500642b86406c691f2 |

The maintained identity-only generator ran at committed source
`8b3111e7fb47d9c842c23358c0f93a53590e2dae` on xbabe3 under MemoryMax 24 GB:
13 compatibility tests passed, zero failures or skips. Its recorded fixture is produced by
that maintained generator; the lineage fixture's header-only write was restored, so only the
announced evaluator fixture changes. The named release transition follows the receiving
sealed-runtime successor and binds both fixture hashes. Existing historical assertions and
all value/model/emission/source pins remain strict. Final compare-mode results are recorded
separately from generation.

[ADR-0004](../../adr/ADR-0004-evaluator-identity-known-table.md) requires an actual receipt
before appending a known identity. [prior-receipt.json](prior-receipt.json) remains the original
twice-produced, byte-identical native receipt at clean
`adf24ba6f7d68973eaa6fdbbe00e5654863b4c03`, whose complete identity inputs match historical
main `4fc1492e4755b0777134a78705feedb0a2f73d30`.
[prior-source-proof.json](prior-source-proof.json) retains its commands, source, binary and
receipt hashes. Its twelve-row training CSV and manifest are synthetic fixture inputs.
The known table retains both receiving rows and appends only this earlier receipt-backed pair;
that row is not relabeled as the latest receiving identity.

Before release publication, retain an actual receipt-backed aa18 receiving identity if its
receipts must be accepted as historical identities after the version transition. No row may
be guessed from this fixture. Final RC no-regression controls, non-author verification,
exact-head hosted evidence and the release lifecycle remain open. This announcement does not
authorize a push, tag, signing, asset publication or a live training claim.
