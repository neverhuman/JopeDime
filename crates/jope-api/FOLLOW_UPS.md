# Contract maintenance and qualification

The original API-01 skeleton is now integrated into the root workspace. The
crate uses external `error`, `ids` and `decimal` modules, including validated ID
deserialization, canonical decimal-string integers and the closed error catalog.
These are implemented types, rather than the original inline string aliases.

Current source and proof locations:

- `src/error.rs`: Appendix D `ErrorCode` and RFC 9457 `Problem`.
- `src/ids.rs`: parsing, ordering, serialization and validated ID newtypes.
- `src/decimal.rs`: canonical decimal-string `DecimalU64` and its JSON schema.
- `src/registry.rs`: operation/surface metadata and named payload schemas.
- `src/bin/jope-api-openapi.rs`: registry-driven OpenAPI generator.
- `tests/contract.rs` and `tests/registry.rs`: wire examples, schema snapshots
  and registry invariants; module tests also check exchange proofs.
- `../../deploy/ci/check-api-schema.sh`: generated OpenAPI drift check against
  `../../schemas/jope.api.v1.openapi.json`.

From the repository root, check the existing contract without rewriting its
committed fixtures:

```sh
cargo test --locked -p jope-api
cargo check --locked -p jope-api
cargo clippy --locked -p jope-api --all-targets -- -D warnings
bash deploy/ci/check-api-schema.sh
```

Normal tests compare snapshots and never rewrite them. The ignored
`regenerate_exchange_proofs` test explicitly owns exchange-snapshot generation;
it is excluded from normal tests because it writes the committed fixture. For an intentional,
reviewed wire change, the `schema_snapshots` example emits the top-level schemas
to stdout, and `jope-api-openapi` accepts an explicit output path. Review changes
before replacing a committed snapshot or OpenAPI document. Documentation edits
do not require regeneration.

`DESIGN.md` records the historical pre-registration API-01 design. Its pending
module/registration statements describe that initial landing, not current source.

Passing wire/schema checks does not establish handler availability, runtime
state-transition enforcement, numerical parity, learning-loop qualification or
serving promotion. Runtime owners enforce semantic limits and capability
refusals. Contract freeze still requires its own review/approval evidence; the
presence of registered types is not that evidence. Wire fixtures are illustrative
values, not dataset or model-performance evidence.
