# Contract completion after this skeleton

Jepson-approved t526u (2026-10-29 22:18 MDT) requires API-01 first, then API-06
workspace/CI registration. No nested workspace is introduced here.

Existing LUNA cards retain their ownership:

- API-02: `src/error.rs`, closed Appendix D ErrorCode and RFC 9457 Problem.
- API-03: `src/ids.rs`, validated ID newtypes (including JsonSchema derives so
  the containing types continue deriving schemas).
- API-04: `src/decimal.rs`, canonical decimal-string DecimalU64 with JsonSchema.
- API-05: generated OpenAPI and drift check, after the operation registry.
- API-06: root workspace members and CI product partition, after this PR.

The inline error/ids/decimal aliases are bootstrap stubs, not substitutes for
these tasks. Their files are intentionally absent. API-03's newtypes will also
need the existing workspace ulid and proptest dependencies: this first crate
manifest intentionally has only the three dependencies in API-01's card.

A LUNA integration follow-up has been filed as
`~/jope_plan/master/tasks/API-01-F01-integrate-contract-modules.md`. It specifies
external module wiring, dependency additions, schema regeneration and rerunning
the same JSON tests after API-02/03/04 source is available. It must be scheduled
with those owners, not treated as authorization to modify their files here.

Before contract freeze, review the candidate shapes for Algorithm, exports,
prediction, system and dataset profiles, which the spec names without complete
JSON examples. Engine behavior, validation and conditional capability refusals
remain the service/worker/export tasks' responsibility. Multiclass is retained
as a vocabulary item but runtime qualification is deferred under t525u.

Regenerate all 13 top-level schema snapshots from this owning proof lane with:

```sh
cargo run --locked -p jope-api --example schema_snapshots
```

Capture successful stdout as `tests/snapshots/schemas.json`. Until API-06 has
registered the crate, the same example can run via the source-referencing proof
manifest described in the PR's local evidence. Normal tests never rewrite the
snapshot. The `run_snapshot.json` fixture expands the illustrative shortened
artifact ID to 24 hex characters; the other three §4.3/§8.4 examples are copied
without value changes. Extra fixtures are illustrative wire values, not dataset
or model-performance evidence.
