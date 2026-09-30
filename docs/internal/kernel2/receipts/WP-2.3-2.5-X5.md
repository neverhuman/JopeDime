# WP-2.3–2.5 and X5: authority, export report and closure gate

PR 10 is based on merged PR 09 main `ad0488a073c50320fcd1e6aed842dd7f650656b3`.
Its branch is `kernel2/claude-m2-closure-gate`. The first closure phase checks
the defined `kernel2-baseline` and `kernel2-experimental` r1 profiles. No
shipping search selects either yet, so the shipped kernel2 set has zero profiles.
The legacy wrappers are frozen exceptions: they have no kernel2 tier, and
their expansion rows are intentionally NotGenerated for kernel2.

## Authority and drift

`authority(grammar_sha256)` selects Registry for only the frozen v3 grammar
(`8d55fee4366acd2c175cf55bfd769cebc904770f5227a6fd50a613723dc85787`)
and jain-v1 grammar
(`570b2e5e8d859a73b888f5d0bee27f59da0e2376536babce24c241604f818e81`).
Every later grammar, including v4, selects Manifest. Under Registry authority,
generated operations are registry-generatable, one of the eight accepted
LegacyDrift rows, or GrammarOnly. The frozen v3 grammar has 53 operations:
35 generatable, 8 LegacyDrift and 10 GrammarOnly. Jain-v1 has 43: 35, 8 and 0.
The eight drift rows are `map`, `derive`, `norm_row`, `norm_features`,
`winsor_moments`, `logistic`, `gbm` and `ensemble_ridge_gbm`.
Only `portfolio` and `ridge_gcv` are disabled in the frozen maps; both are
GrammarOnly. Under Manifest authority, each operation has a manifest row and
its generate flag equals its kernel2 verdict. A Quarantined operation or a
nonsearchable expression token fails. The rule uses the file hash and parsed
JSON only. In-memory tests prove `map` passes on the Manifest branch, while
`count_sketch` and `norm_row` with `generate: true` fail; a frozen-hash clone
with generated `lag` fails on the Registry branch. No declared drift remains
unclassified.

| Consumer | Authority | Source |
| --- | --- | --- |
| External jain product generators | Registry | `registry::generatable_operators`, `crates/feat-math/src/registry.rs` |
| Frozen v3 and jain-v1 token samplers | Each file's generate flag, checked under Registry | `authority` and `grammar_violations`, `crates/feat-math/src/kernel/capabilities.rs` |
| Kernel2 profiles and later grammar v4 | Manifest | `Manifest::current`, `crates/feat-math/src/kernel/capabilities.rs`; owner decision in `MASTER_PLAN.md` |

The five expansion operations `symlog`, `select_corr`, `impute`, `pair_ratio`
and `spline` have manifest rows and no new registry ids. The owner decision
keeps `registry.rs` frozen. Its hash remains
`8173b07f37fde39bdc8bbf2dfe2e496aa5ccaeb4e0274843c235ec1fc6972c62`;
the registry dump still equals the jain executable snapshot, and the frozen
pins test reads the same vocabulary hashes. PR 09 checks 4 and 9 also run in
the feat-math PR tier.

## Export report

X5 lives in new `kernel/export.rs`, outside the ADR-0004 input `transpile.rs`.
The report returns separate Rust recipe, Python recipe, C++17 recipe,
declared C++17 fitted, and JavaScript routes. Each refusal is an ordered list
of element kinds. Planning errors first reduce to `portfolio`, a manifest op
name, `target`, `gpr kernel`, `symbolic expression`, or `unknown op` before any
untrusted parameter or program text can escape. An emitter label is cut at
its first `(` or `{`, trimmed and deduplicated in order. The report gives no
CLI strings in this PR.

The four in-memory planning refusals are `portfolio blend=caruana`,
`portfolio add1=nope`, conflicting `add1`/`drop1`, and an unknown operation.
The first three reduce to `portfolio`; the fourth reduces to `unknown op`.
Their Rust, Python, C++17 recipe and JavaScript routes carry those kinds;
the first three retain a declared fitted Ok route because the report reads
element rows, not the invalid parameters. All kinds pass `lint_shipped_text`
and contain no `(`, `{` or quotation mark.

Against all 55 recorded fixture programs, Rust and Python are Ok 55/55,
C++17 recipe and JavaScript each are Ok 7/55 and have the recorded or a
strictly narrowed refusal on the other 48. Declared fitted is Ok 55/55;
54 recorded programs fit and pass the real fitted support check, while the
fixture's compact portfolio skip remains skipped. A separate cheap nested
portfolio fits and passes. The report also agrees with the live recipe
support functions for every fixture and probed operation. The fixture's 25
recorded supported operation names are a subset of each 27-name manifest
set; the extra two are the supported aliases `norm_features` and
`winsor_moments`.

The C++17 fitted basis is explicitly `Declared`. PR 16 adds the executed
boundary parity gate. PR 17 chooses whether its CLI exclusion strings use
these sanitized kinds or retain byte-identical legacy strings; today those
strings join raw reasons and can include plan debug payloads. PR 18 may
re-export `export_report` from the identity-bearing `transpile.rs` when it
already mints the planned identity.

## Closure gate

The new `kernel2_closure` test runs in every feat-math PR tier from this PR
on. It checks every shipped kernel2 profile, then selects the highest defined
revision for each kernel2 profile id. Today's shipped set is empty; the gate
does not assume it remains empty. The r1 count pins require a deliberate update
when a newer revision is defined. It passes the `&Manifest` into every phase.
Clause (c) compares interpreter
and compiled row/element values bit for bit and fault sets exactly, plus
compiled map cells against interpreter element evaluations. Boundary inputs
are the 66 recorded binary64 values; rows of width 0–9 include finite values
and nonfinite last cells. The code probes every manifest expression row,
including rows outside r1, so later promotions have an established fault
proof. Clause (d) checks each profile's stage and head domain points, each
scope-valid expression carrier, probe text and every birth constant through
lossless canonical text. Clause (e) checks each manifest row's three declared
routes against the report and cross-checks fitted support on a real small
model. All failures name an element and clause.

| Set | (a) elements | (c) comparison triples | (d) fixed points | declared (e) row-route pairs |
| --- | ---: | ---: | ---: | ---: |
| `kernel2-baseline` r1 | 51 | 956,412 | 233 | 153 |
| `kernel2-experimental` r1 | 51 | 956,412 | 233 | 153 |
| Manifest-wide expression rows | — | 971,397 | — | — |
| Shipped kernel2 profiles | 0 profiles | 0 profile items | 0 profile items | 0 profile items |

The r1 counts are literals in the test, so an omitted clause cannot pass
quietly. The phase currently runs (a), (c), (d) and declared (e). Clause (b),
which requires create/change/remove coverage and zero invalid children in
100,000, is pending PR 12. Executed (e) is pending PR 16. The C++17 recipe
gaps for RG-13 are `map` and `derive` (X2a), `ridge_gcv` (X2e), `ksmooth`
(X2f), `histgbm` (X2g), `histgbm2` (X2h), and `gpr` (X2i). JavaScript remains
a subset and is informational for this gate.

| Negative | Pair or direct predicate | Evidence |
| --- | --- | --- |
| Runtime-only `exp` in a profile | `(exp, (a))` | Not admitted at the r1 tier |
| Quarantined `count_sketch` stage | `(count_sketch, (a))` | No admitted manifest row |
| Columns through 31 | `(columns, (a))` | `Col(i >= 8)` is runtime only |
| Removed `sqrt` expression row | `(sqrt, (a))` | No manifest row |
| `abs` declares LOG_DOMAIN | `(abs, (c))` | Declared class never raised |
| `/` declares no class | `(/, (c))` | DIV_NEAR_ZERO is raised undeclared |
| `step` domain includes `1e-13` | `(step, (d))` | Canonical printer rounds it; direct `2.0000000000000004` and `map expr="0.5"` are also lossy |
| `hurdle2` added as a head | `(hurdle2, (e))` | Recipe routes refuse it |
| Map Python route changed in manifest | `(map, (e))` | Manifest disagrees with live report |
| Negative literal `Const(-0.5)` | direct (d) | Reparse changes it to `Neg(Const(0.5))` |
| Ridge fitted route changed in manifest | `(ridge, (e))` | Report says Unsupported, real fitted check says Ok |
| Birth bin `0.3` added to r1 | `(constants, (a))` | Not bit-equal to any admitted bin |
| Negation policy row made runtime only | `(negation, (a))` | Passed manifest policy row is read |
| Manifest constant bin `0.5` removed | `(constants, (a))` | Passed manifest bins are read |

## Decisions and unchanged surfaces

| Decision | Resolution and reason |
| --- | --- |
| D-M2-5 | Codex is informed when the PR opens; additive public kernel items are checked against all six frozen Lambda workspaces. |
| D-M2-6 | X5 lives in `kernel/export.rs` because `transpile.rs` is an ADR-0004 input; PR 18 can re-export when it already changes that identity. |
| D-M2-7 | No CLI switch here. PR 17 rewrites those call sites once and records its sanitized-kind versus byte-identical string choice. |
| D-M2-8 | The gate is required green for both defined kernel2 profiles even while the shipped set is empty, preventing a vacuous gate. |
| D-M2-9 | `docs/emitters.md` links to the generated, machine-checked matrix and describes its own hand table as a summary. The hand table itself is not claimed checked. |

The 38 ADR-0004 evaluator input paths and the identity fixture are unchanged.
The jope-engine lineage source list, registry, both frozen grammars, every
compatibility fixture, Cargo manifests and locks, and the ablation protocol
are unchanged. Jope-engine lineage digests also hash the running executable;
their value may change with feat-math code even though their listed source
inputs do not. No emitted byte or CLI code is changed.

## Local verification

The focused authority test, export target (5 tests) and closure target
(15 tests) passed. `cargo nextest run --locked -P pr -p feat-math` passed
616/616 selected with 11 skipped. `just test-main` passed 3,924/3,924
selected with 66 skipped. The six frozen Lambda workspaces built and tested:
66/66 passed. Its local cargo-deny subcheck was skipped by its version gate:
the host has 0.19.8 while the CI hygiene lane requires 0.20.2. Strict
feat-math Clippy, formatting, the repository check, inventory validation,
the frozen registry dump comparison, the selected compatibility pin test,
and the IP scrub passed. The Rust inventory has 3,990 tests and the Python
inventory retains 172; coverage validation found 295 targets and zero invalid.

`just docs` built rustdoc, then its relative-link scan failed on 14 links in
unchanged `docs/internal/core-ip/` files. The new link from `docs/emitters.md`
resolves to the generated matrix. None of the 14 source files differs from the
base of this PR. This existing repository-wide doc failure is retained as a
limitation; it is not reported as passing verification.

The exact-head local gate and hosted runs are recorded at the final PR head
before merge. Local logs are under
`/home/ubuntu/jope-evidence/kernel2/pr10-local-20260930/`.
