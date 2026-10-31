# jope-api

Shared JopeDime v1 wire types, JSON schemas and operation metadata for the
service, batch client, terminal and MCP adapters. This crate defines the data
contract; it does not run an engine or implement transports.

## Modules

| Module | Contract |
| --- | --- |
| `runs` | Run specifications, plans, snapshots, states and budgets |
| `events` | Durable public event envelopes/bodies and ephemeral telemetry |
| `artifacts` | Algorithms, exports and prediction requests |
| `system`, `metrics`, `warnings` | System/dataset descriptions, measurements and warnings |
| `ids` | Validated run, dataset, artifact and algorithm identifier newtypes |
| `decimal` | `DecimalU64`: a `u64` encoded as a canonical JSON decimal string |
| `error` | Closed error catalog, retry policies and RFC 9457 problem documents |
| `registry` | Operation/surface metadata and concrete request/response schemas |

Run, event, artifact, system, metric and warning types are re-exported from the
crate root. Use qualified imports for identifier, decimal, error and registry
types, such as `jope_api::decimal::DecimalU64`.

Identifier parsing/deserialization validates each ID format. `DecimalU64`
rejects JSON numbers, leading zeros, signs and values above `u64::MAX`.
`ErrorCode` rejects unknown codes; retry policies retain conditional and unknown
outcomes. Callers still enforce state transitions, capabilities and other
producer-side semantics. A schema or wire fixture is not runtime qualification.

## Verify from the workspace root

```sh
cargo fmt -p jope-api -- --check
cargo check --locked -p jope-api
cargo test --locked -p jope-api
cargo clippy --locked -p jope-api --all-targets -- -D warnings
bash deploy/ci/check-api-schema.sh
```

Package tests cover ID/decimal/error validation, spec JSON round-trips, event
kind/body matching, committed schemas, exchange definitions, operation/surface
consistency and OpenAPI generation. The schema drift script generates a
candidate and compares it with `schemas/jope.api.v1.openapi.json`; it does not
update that file.

## Maintain generated contracts

`tests/support/mod.rs` defines the top-level snapshot inventory. After an
approved contract change, generate candidates for review:

```sh
mkdir -p target
cargo run --quiet --locked -p jope-api --example schema_snapshots > target/jope-api-schemas.json
cargo run --quiet --locked -p jope-api --bin jope-api-openapi -- target/jope-api-openapi.json
```

Review the candidates against `tests/snapshots/schemas.json` and
`schemas/jope.api.v1.openapi.json`. Update generated snapshots through their
owning proof lane and retain the required frozen-contract review. Ordinary
package tests never rewrite them. Changes to registry exchanges also require
review of `tests/snapshots/exchanges.json`.

[FOLLOW_UPS.md](FOLLOW_UPS.md) describes current contract maintenance,
source/proof locations and remaining qualification requirements.
[DESIGN.md](DESIGN.md) preserves the original bootstrap design. The external
`error`, `ids` and `decimal` modules, workspace registration and OpenAPI generator
are integrated; this guide does not assert runtime or release qualification.
