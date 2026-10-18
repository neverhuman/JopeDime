# Typed graph and edit ABI v1

`feat_math::kernel::ir` adds an opt-in bounded program representation over the
existing numeric interpreter. Its payload discriminator is
`jope.selfplay.graph-effects/v1`, version 1. No old `Program::LegacyV1` variant,
codec tag, identity function, grammar, search profile or receipt is reinterpreted.

## Supported executable subset

| Form | Typed execution | Otherwise |
| --- | --- | --- |
| Linear native pipeline | I07 legacy-recipe operators with known fitting effects and exact inherited Preserve/Parameter/Fixed widths; final ridge, ridge_gcv or elasticnet | Opaque legacy bridge or explicit graph admission refusal |
| Input column selection | Ordered references directly after input; native fitting width at most 512 | Refuse before fit |
| Symbolic expression | Constants, schema-bound columns, existing unary/binary/function AST variants; optional existing affine scaling | Legacy row-reduction/element variables remain opaque |
| Native target pseudo-stage, expansion layouts, advanced-v3 raw-table layouts | Unavailable in this adapter | Preserve original legacy bytes/ID where applicable |
| Stack, residual, compact/explicit portfolio, ensemble heads | Unavailable: nested cross-fitting not proven | Never emitted as legal actions; explicit refusal before fit |

This subset introduces no fitting algorithm or arithmetic primitive. Native
stages use the existing `model::fit`; symbolic graphs construct the existing
executable `MathExpr` AST directly, without reparsing a lossy printer. Historical
whole-pipeline refitting still applies to target-aware transforms followed by a
supported terminal head. There is no shipping sampler, trainer, CLI or policy
integration in this change.

The registry remains I07's generated `operator-registry-v2`, revision 2. Each
operator reference binds dialect, spelling, stable identity, semantic version
and exact registry-source digest. Effects, parameter validation, lifecycle,
license/provenance checks and public inherited shape rules come through I07 and
its frozen source API. The three supported heads and shape-rule cases describe
this adapter's coverage; they do not constitute another operator registry.

## Graph, schemas, ports and limits

An immutable `ProgramGraph` owns an ordered node vector, one schema input and a
terminal prediction output. Dependencies must refer to earlier nodes. Every node
must reach the output; cycles, forward/missing references, dead nodes, matrix/
prediction port mismatches and non-linear stage chains fail admission. Operand,
column and node ordering is retained; no reassociation, commutation, pruning or
hash-consing is performed.

`ColumnRef` binds the input-schema identity, checked zero-based index and exact
column key. The schema identity includes ordered keys and row semantics. An index
above seven is valid: the fixtures select column 260 of a 300-column schema and
execute a direct wide-column expression. Out-of-schema indices, changed keys or
different schema identities fail before fitting. Selection must precede native
stages; references are never silently rebound after a fitted width change.

The ABI hard limits are 64 graph/expression nodes, 128 graph edges, depth 32,
4096 input columns, 512 native generated/fitting columns, 4096 dataset rows, and
1 MiB transport bytes. Defaults further limit graph bytes to 256 KiB. Callers can
tighten these constraints and cannot enlarge the hard limits. Known widths and
append widths are checked with overflow protection before fitting. Training
identity expansion is also bounded before allocating its exact-bit projection.
Generic JSON decoding retains serde's recursion limit; semantic graph limits
are checked before lowering or execution.

Row-order requirements remain separate from feature-position operations. Only
the legacy-recipe adapter is executable here; an advanced-v3 ordered-row spelling
cannot acquire admission by sharing a name. Row semantics are included in schema
identity, not guessed from a column's spelling.

## Effects and fold closure

`Effects` propagates fitting on features (`fit_x`), fitting on targets (`fit_y`),
target derivation, ordered-row requirements and seeded state over dependencies.
Unknown effects refuse admission. A supported head reads training targets;
unscaled symbolic expressions read no target values, while affine scaling does.
Per-operator seeds remain exact stage parameters. The plan seed is a separate
replay context; a stable seed does not establish equal outcomes or comparability.

`FitPlan::bind` fixes graph/operator versions, schema, declared source/input/
target/split identities, sorted disjoint fitting and prediction row sets, exact
training-feature/target digests, effects and a full-width hexadecimal seed.
Initial identities and row permission come from the caller's admitted split
contract; this module does not authenticate dataset manifests or issue access
authority. Actual accessed training values are rebound to their exact bits.
Changing training content or any declared identity fails before invoking fit.

`FitPlan` has private fields and no deserializer. Its `narrow` operation records
the parent plan identity and permits only subsets of the parent's fitting and
prediction rows, unchanged source/input/target/split/schema identity and seed,
and no newly granted target access. Plan persistence/readback and general nested
fold execution are later integration work; serialized metadata is not accepted
as an authority-bearing plan.

Execution selects fitting rows before it hands data to the existing interpreter.
Held prediction labels are never passed to it. Prediction reads only requested,
authorized feature rows and preserves their requested order (sorted unique row
IDs in this ABI). Observational `ExecutionStats` counts the actual interpreter
boundary and accessed row IDs; it is not an authorization object. Shape, version,
closure and content checks precede its `fit_calls` increment.

The poison fixture compares fitted state and pre-score predictions under changed
held features/labels, then runs a deliberately leaky whole-data fit whose
predictions change. Explicit target-derived stack/residual graphs and substituted
composition heads are refused before any fit. This proves the declared primitive
fold boundary and a detectable negative control. It does **not** prove safe nested
stacking/residual learning. Those actions remain unavailable until a later adapter
refits every target-derived transform/statistic and produces nested cross-fitted
training predictions inside the appropriate child closures.

## Identity and compatibility bridge

New graph bytes follow I02's `jope.canonical-json/v1` projection: UTF-8 scalar-key
ordering, exact strings without Unicode normalization, ordered arrays, safe
integers and tagged finite `f64_bits`. Seeds use 16-character lowercase hex.
Signed zero and subnormals remain distinct. Raw floats, nonfinite graph constants,
self-referential identity fields, unknown fields/versions, duplicate keys and
noncanonical byte spellings are refused. Decode/re-encode byte comparison detects
duplicate map keys even where a map deserializer would otherwise overwrite them.

`ProgramIdV2` applies the adopted framing:

```text
SHA256(schema_utf8 || 0x00 || u64_be(payload_length) || canonical_payload_bytes)
```

The small local projection has no new Cargo edge; agreement tests compare all ten
adopted I02 canonical-byte and digest vectors. Constraints do not rewrite program
identity. Model-state snapshots use exact-bit tags for pre-score fixture evidence;
they are not measured-outcome authority or a new evaluator identity.

`LegacyBridge` stores original binary codec bytes, the unchanged binary program
identity and original `legacy_identity_text`. Typed status adds a separate graph
and relation; opaque status preserves an explicit reason. Reading a typed bridge
reimports the original program to verify the relation, so a different valid graph
cannot be attached to an old identity. Original bytes, including NaN payloads,
unsafe old integers and lossy historical text, remain resolvable without forcing
them into the new numeric contract. Bridge transport itself is bounded to 1 MiB;
larger original programs require external custody references, not truncation.

The committed historical corpus contributes 73 binary programs. Tests report
typed/opaque totals and require zero roundtrip errors. Available qualified winner
sets are **UNKNOWN / not admitted or accessed**; no corpus coverage is inferred
from these committed fixtures. Existing historical/freeze/codec tests retain their
own source and artifact inventories unchanged.

## Legal edits and ancestry

`legal_edits(graph, schema, profile, constraints)` enumerates an explicit no-op,
supported head replacements, inherited width-preserving stage insertions,
nonterminal stage deletions, inherited parameter variants and schema-column
replacements. Enumeration is deterministic, bounded to 4096 candidates, and
filters every candidate through the same `apply` validator. This is a legal-action
ABI, not a proposal distribution, mutation prior or strategy.

`apply` revalidates the parent, schema/profile/version, exact parent inventory,
edited site, parameters, output width and whole child graph. It never fits,
draws randomness or mutates the parent. Forged actions receive the same checks.
`crossover_edit` validates and records both parent identities, transplants only a
supported donor head and retains both links in the serialized edit record. A
no-op or identical child is explicitly `unchanged`; a changed representation is
not a claim of behavioral novelty, improvement or a learned outcome. Refusal is
an error with no child artifact.

Exhaustive tiny fixtures apply, validate, fit and predict for every enumerated
action. Illegal graphs/actions demonstrate rejection before the interpreter
counter changes. Full-main gates, target discovery/inventory registration,
independent review and CI remain with the integrator. No top-level feat-math
source, old codec, evaluator-ledger input, Cargo file, frozen asset or old golden
fixture changes; I08 is an identity-neutral source-ledger checkpoint. Linked
executable bytes still change and must be bound by the integration receipt.
