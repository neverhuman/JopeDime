# Operator registry v2

`crates/feat-math/registry/operators.toml` governs the additive
`operator-registry-v2` profile, revision 2. Its 90 entries distinguish 61 legacy
recipe spellings from 29 advanced-v3 spellings. An entry is identified by dialect
and spelling. Shared names do not imply shared semantics, tasks or emitters.

The generated table lives in `kernel::registry_v2`, under the existing kernel
module. `kernel::profile::registry_v2_profile` supplies its opt-in validator and
head sampler; `kernel::capabilities::registry_v2_manifest` supplies its report.
These consumers read the same generated entries. The shipped legacy and kernel2
search profile registry, IDs, revisions, serialization and selection stay intact.
There is no new CLI selection, strategy, arithmetic or emitter in this change.

## Inheritance and the consumer census

The source names its compatibility projections instead of modifying frozen
readers. `registry_v1` inherits IDs 0–52, semantic keys/versions, parameter
types/defaults/domains, shape, fit/label facts, license and provenance through
`registry::operator_registry`/`operator`/`generatable_operators`. Aliases retain
their distinct historical wire IDs. Compatibility and quarantined entries cannot
enter new draws through a metadata-only lifecycle change.

`recipe_v1` inherits the existing full-Rust schema for the five expansion
transforms and `ridge_gcv`/compact `portfolio`. Their fit/label/width facts and
new-profile policy are in the data file; the report also materializes the
existing canonical defaults. `hurdle2_v1` records the four parameters consumed by
`heads::fit_head`, with finite bounded domains for this new profile. These
supplemental entries use namespaced identities, **not new wire IDs**. A future
opcode needs a coordinated allocation; reservations 53+ are not appropriated.

`advanced_v3` inherits its private `OperationDescriptor` through a narrow adapter
in `advanced_v3/grammar.rs`. It reuses `materialize_stage` in production mode,
including task domains, defaults, aliases, inactive parameters and validation.
The descriptor does not declare a general fitted-state Boolean, so that report
field remains unknown (`null`); it is not inferred from a spelling. Task-dependent
aliases such as `logistic` still resolve through the existing task-aware lookup.
The report's alias projection uses the first historical descriptor; task-specific
validation remains authoritative. Parameter validation alone does not confer
schema, backend, receipt or search admission.

| Existing consumer | Compatibility authority | New-profile route |
| --- | --- | --- |
| `registry.rs`: registry/aliases/hash | Existing public APIs and frozen bytes | Inherited facts, no registry edits |
| `spec.rs`: head recognition and planning | Existing 14-name list and planner | Generated recognition, followed by actual target support check |
| `evolve.rs`: recognition and RNG | Existing 14-name recognition, 9 head slots, 18 transforms | Separate opt-in weighted head draw |
| `heads.rs`: recognition and fitting | Existing 15 names and `fit_head` | Generated seeds execute existing fitting |
| `full_rust.rs`: schema/validation/generation | Existing schemas, 31 transforms, 11 heads, 5 expansion ops | Strict generated profile, inherited schemas |
| `advanced_v3/grammar.rs`: compiler | Existing descriptors and compiler modes | Narrow descriptor/materializer adapter |
| `kernel/capabilities.rs`: historical union | Existing `Manifest::current` and render bytes | Separate dialect-specific manifest |
| `kernel/profile.rs`: eight ID/revision pairs | Existing profiles and canonical hashes | Separate `registry_v2_profile` lookup |
| Both engine grammar assets | Existing tokens, IDs, bytes, authority hashes | No automatic union/generation admission |

The agreement test logs the full generated census, serialized legacy registry,
historical manifest (including grammar-only entries), and all eight existing
profile hashes/canonical serializations. Existing manifest/vocabulary tests check
both immutable grammar assets against their historical authorities. Historical
grammar-only entries absent from v2 remain outside the new profile; a union does
not grant support.

## Validation, draws and target honesty

`RegistryProfile::parse(source, target)` is an explicit legacy-recipe seed route.
It requires a terminal head and valid stage positions, rejects unknown, no-op,
inactive, mistyped and out-of-domain parameters, and keeps the historical parser
permissive. `poly_interact topk` is refused; no numeric implementation is added.
`target q` is active only for `winsor`; `impute value` only for `strategy=const`;
`histgbm2 delta` only for `loss=huber`. Unspecified inactive defaults are removed.
Supplemental numeric parameters use the bounded hull of the inherited schema's
birth values, retaining integer types. They do not redefine old fit behavior.

Legacy parameter defaults/domains come from the frozen descriptor; recipe
defaults remain in the existing implementation. Target checks also call the
actual planner/emitter support functions. An individual stage declaration is
not a guarantee for every possible composition or parameter configuration.

`draw_head` consumes the ordered generated entries and their explicit weights,
filters admission/license/provenance through the inherited public registry API,
and filters by requested target. It is separate from all old RNG consumers.
The data records the old head slots verbatim: `histgbm` and `histgbm2` each occur
twice. The three frozen generated target kinds remain separate from four valid
kinds; `zscore` does not enlarge the historical draw modulus.

| Head/form | New runtime seed and head draw | Python/Rust recipe | C++17/JavaScript recipe |
| --- | --- | --- | --- |
| Compact `portfolio` | Accepted, actual fitting tested | Actual emission tested | Named unsupported target refusal |
| `hurdle2` | Accepted, actual fitting tested | Named refusal; existing planner also refuses | Named refusal |
| `ridge_gcv` | Accepted, actual fitting tested | Actual emission tested | Named refusal |
| Explicit nested portfolio | Separate historical runtime form | Existing refusal retained | Existing refusal retained |

Fitted portable/C++ APIs are separate existing routes, not targets admitted by
this profile. This work does not assert or implement their completeness.

Legacy `lag` shifts feature positions and `roll` rotates feature positions
within each row. Advanced-v3's same spellings operate over ordered sample rows.
The latter remain disabled for production in the existing descriptor. The
agreement fixture executes within-row lag; the separately collected existing
advanced-v3 transform fixture executes group/time-ordered lag and rolling mean,
including restored caller order. A new registry declaration does not promote
the compatibility ordered-row implementation.

## Generation and checks

One std-only parser/generator is shared by the build script, example and tests.
Its documented TOML subset accepts full-line comments, single-line quoted
strings/string arrays and unsigned integers. Escapes, multiline values, inline
comments, other TOML syntax and unknown keys are rejected rather than silently
ignored. Duplicate names/identities, incomplete effects, invalid targets,
unallocated identities and unsupported transitions into generation fail closed.

```sh
cargo run --locked -p feat-math --example registry_codegen -- --write
cargo run --locked -p feat-math --example registry_codegen -- --check
cargo test --locked -p feat-math --test registry_agreement -- --test-threads=1
```

Only `--write` writes the generated source. The build validates metadata and
warns about stale output, preserving the ability to run the repair example;
`--check` and the agreement test fail on stale bytes. Builds never implicitly
write source. The existing C++ runtime command, ISA selection and floating-point
flags are preserved. `source_sha256()` identifies the exact data bytes; it is
not an evaluator or HWM comparison identity.

No evaluator-ledger input, frozen file, Cargo manifest/lock, grammar asset or
existing golden fixture changes. Therefore this implementation needs no
ADR-0004 rebaseline. Executable bytes still change and must be bound to the
integration receipt. The integrator owns I02 rebinding, coverage/example
registration, inventory regeneration, the full required gate, exact-head CI and
independent review. Local fixture evidence does not establish live learning,
serving, numerical benefit or promotion. Rollback disables the additive profile
while preserving old readers and all candidate/failure receipts.
