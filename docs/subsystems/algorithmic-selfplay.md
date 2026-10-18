# Algorithmic self-play subsystem map

Task: **I62 — Plan maintainability changes without disrupting held consumers**. Intended repository path: `docs/subsystems/algorithmic-selfplay.md`.

Source pin C=`1a8a66c502ca2cede77c197b4f5b5bbc24ff398e`; tree `1c6d5882718434ae6f109854bff4597896f1210f`. Exact direct dependencies: I01, I18, I45, I53. Their implementation/acceptance is not inferred from prepared packets. Completion criterion (verbatim tasks.json): Reviewable ownership/archive plan; no broad source or documentation churn in the first closure path.

This is the complete staged document contract/plan for the assigned scope. Source facts are tied to the verified C citations below; proposed behavior, owner bindings and future proof are explicitly distinguished. No Git worktree, source, Git state, live service or canonical ledger is changed by this document; no Cargo command or product experiment is executed.

## Authority and ownership

This is a map of source boundaries and proposed adapters, not a claim that the 66-task programme is implemented or an alternative task/status ledger. The exact backlog remains tasks.json; accepted source/schema/profile/receipt IDs and agy's current stopped-head claims determine what may run. Unknown owner or consumer means HOLD. No source move is performed by this map.

| Boundary | Current pinned source / role | Owner responsibility / interface rule |
| --- | --- | --- |
| Wire / records / evidence | jope-records validators and baseline reader; immutable planning inputs; existing learning progress receipts. | agy owns canonical schemas and root registration. New views consume authority receipts; they do not create a second authoritative store. |
| Roles / fit / measurements | Engine task/fit/label/eval-registry boundaries and versioned source identities. | Authentic principal, purpose, row role, fold and exposure ancestry before reading or fitting; missing labels/outcomes remain typed unavailable, not favorable zero. |
| Language / numeric evaluation | feat-math models, kernel codecs/profile/search admission, registry and emitters. | Preserve accepted grammar/IDs/frozen behavior. Profile is vocabulary/capability; strategy is search algorithm. A new representation or dispatch hook requires identity review. |
| Search consumers | Bench methods reaches evolve_lite_race; pipeline kernel2 has its own run_cut path and one-thread/one-island guard. | Keep controls and actual budgets separate. Shared vocabulary is not proof of equivalent algorithms, discovery cost or execution. |
| Native learner / continuation | Engine package jope-parallel, native_loop and neural-gated training/continuation/checkpoint paths. | Reuse existing authority/trainer. Actual active updates, optimizer state and committed consumption/child/next use require accepted closure; no second trainer/scheduler. |
| Policy / export / actor | jope-policy bound loaders, grammar/decode and cfg(test) challenger module; testing integration targets are separately gated. | Preserve accepted identity; a legitimate descendant is a separately admitted challenger, not a relaxation of accepted release checks. |
| Publication / storage | Existing datasets observed-version conditional operations, learning mirror/sync and progress publication closure. | One authority adapter, immutable exact-byte closure, expected-version CAS and retry verification. Derived lake/cache views are rebuildable, not authority. |
| Qualification / decisions | Learning qualify gate/roster/promotion and round qualification. | Independent adopted roster/issuer/look budget and prospective primary before outcomes. Research-parent adoption and serving are distinct events; diagnostic output is not permission. |
| Custody / retention | Retention dry-run model and lifecycle hazard inputs; report-only maintenance. | No deletion from classification. Strong refs/recovery promises/current consumers block retirement until independent recovery and separate exact action admission. |
| CLI / operator views | Existing jope Doctor and Linux Train routes; records and learning binaries retain compatibility responsibilities. | Thin library adapters only. Stale evidence remains dated; aliases do not mint grants or duplicate business rules. |
| Research memory | I44 capsules and I45 derived compact state are accepted dependency deliverables when implemented. | Preserve origin records, negative/null/inconclusive results and evidence roots. No correlated automatic pooling or competing narrative authority. |

## Current caller facts versus future shape

At C campaigns main.rs declares evaluation_v2; feat-math lib.rs exports model, kernel and evolve; bench methods imports and calls evolve_lite_race. CLI Doctor delegates to jope_bench::panel::demo::doctor, while Train dispatch is Linux-gated. The existing train source still includes a Python compatibility argument/worker; ADR-0006's direction is not proof its consumers have disappeared.

The future shared-kernel, lineage-aware update, verified assessment and compact-state interfaces are governed by their individual packets and accepted dependencies. Keep old entry points available until named callers migrate under parity/identity/custody proof. Do not perform campaign/evolve/Python refactors on the first closure path.

## Required inventory comparison for a later change

Before proposing any source/ownership move, attach current and proposed path/module/caller graphs; package/default/feature/required_features/platform cfg inventories; library/bin/integration/doctest selections; source versus runtime ignore categories; coverage registration/routing; historical replay and deployed/future consumer closure. The current documentation proposal has an empty source delta.

The committed C listing is 5,013 Rust names with 72 ignored, not fresh execution. The source scan has 86 ignore attributes across different feature/build populations. Preserve neural/LibTorch, testing, ASM/Linux, partial cfg, private inputs and path-skipped categories separately. New target/example registration is an explicit agy hunk in coverage.json, test-inventory.json and README; zero collection cannot prove a migrated subsystem.

## Identity, archival and failure boundary

Source-byte changes to evaluator inputs such as evaluation_v2.rs, evolve.rs or feat-math lib.rs require an agy-owned task-named transition even if a refactor appears behavior-preserving. Model/task-contract sources may also move lineage/HWM comparison; runtime executable and witness signed-policy identities are separate gates. Preserve frozen bytes and the committed 12-file Clover audit.

ARCHIVE_CANDIDATES.md contains the full seven-surface checklist and candidate register. Unknown consumers block archive. Module/feature inventories must compare before/after the proposed transition. Hash-moving changes require explicit migration and independent old replay. I53 additionally holds protected/symlink trees and filesystem obligations. This document changes none of those permissions.

## Authoritative acceptance coverage

| Verbatim tasks.json acceptance case | Required evidence / refusal | Current evidence ceiling |
| --- | --- | --- |
| Unknown consumers block archive | Mark every candidate with unknown consumers as HOLD, including configured future and foreign consumers; a presumed lack of current processes is insufficient. | SPECIFIED; implementation/runtime NOT_EXECUTED |
| Module/feature test inventories compared before/after proposed transition | Attach an explicit before/after proposal inventory of modules, features, targets and required_features, preserving missing/ignored/neural categories without claiming a proposed state was executed. | SPECIFIED; implementation/runtime NOT_EXECUTED |
| Source hashing changes get explicit identity migration | Any proposed source-byte hash change includes a named agy transition, historical replay and comparator/lineage impact plan; keep actual source untouched. | SPECIFIED; implementation/runtime NOT_EXECUTED |

The case table is a review/implementation contract, not a manufactured runtime result. Future proof retains exact source/input/argv/redacted environment/resource/UTC/exit and actual collected/executed/pass/fail/ignored/filtered counts at `/home/ubuntu/jope-evidence/spec-integration-20261017/work-packets/evidence/I62/<full-head>/`. Missing capabilities and unknown counts do not pass.

## Verified source citations

| Full-pin source citation | Verified boundary |
| --- | --- |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-campaigns/src/main.rs:24` | Campaign evaluation module |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/feat-math/src/lib.rs:27` | Kernel export |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/feat-math/src/lib.rs:33` | Model export |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/feat-math/src/lib.rs:54` | Evolution export |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-bench/src/methods.rs:6` | Bench evolution imports |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-bench/src/methods.rs:190` | Actual lite-race call |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-pipeline/src/kernel2/search.rs:75` | One-thread/one-island check |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-pipeline/src/kernel2/search.rs:237` | Kernel2 run_cut entry point |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-cli/src/main.rs:224` | Doctor delegates to existing library |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-cli/src/main.rs:231` | Train platform-gated dispatch |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-cli/src/train.rs:77` | Python compatibility argument |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:docs/adr/ADR-0006-rust-only-stack.md:30` | Rust-only direction and phased preservation |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-learning/src/qualify/gate.rs:32` | Existing fail-closed gate |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-learning/src/retention/plan.rs:1` | Dry-run outputs |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-learning/src/sync.rs:109` | Existing publication entry point |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-campaigns/src/evaluation_v2.rs:1` | Authoritative read-first file |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:AGENTS.md:1` | Authoritative read-first file |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:CLAUDE.md:1` | Authoritative read-first file |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:docs/plans/algorithmic-selfplay-20261017/tasks.json:2997` | Exact task dependencies, completion and acceptance cases |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-engine/Cargo.toml:2` | Engine package is jope-parallel |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-campaigns/Cargo.toml:2` | Campaigns package is jope-dime-regressor |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-encoders/Cargo.toml:2` | Encoders package is starforge |

These exact lines, line hashes and whole-blob SHA-256 values are recorded in `wave3-staged-docs-receipt.json`. Parent paths, feature/test declarations and new symbols proposed by the packet must be rebound to the actual accepted implementation head before use; C is an inspection pin, not an assertion that future code exists.

## Integration boundary

agy (`2830191.jopedime_agy`) owns the one integration PR, accepted dependencies, source/Cargo/schema/profile/root hooks, all seven identity/guard surfaces, coverage/inventory/README and claims. This stage adds only the requested document/configuration candidates. It moves no evaluator/lineage/frozen/witness/Clover/retention/discovery surface and grants no source scope, deployment, cleanup, campaign, serving or scientific authority. Preserve accepted readers/parent and every failed/rejected/partial/inconclusive receipt; disable an additive candidate rather than reset shared history or destroy evidence.

Execution grant (verbatim tasks.json): NONE; planning is not campaign admission
