# Archive candidates and ownership plan

Task: **I62 — Plan maintainability changes without disrupting held consumers**. Intended repository path: `docs/plans/algorithmic-selfplay-20261017/ARCHIVE_CANDIDATES.md`.

Source pin C=`1a8a66c502ca2cede77c197b4f5b5bbc24ff398e`; tree `1c6d5882718434ae6f109854bff4597896f1210f`. Exact direct dependencies: I01, I18, I45, I53. Their implementation/acceptance is not inferred from prepared packets. Completion criterion (verbatim tasks.json): Reviewable ownership/archive plan; no broad source or documentation churn in the first closure path.

This is the complete staged document contract/plan for the assigned scope. Source facts are tied to the verified C citations below; proposed behavior, owner bindings and future proof are explicitly distinguished. No Git worktree, source, Git state, live service or canonical ledger is changed by this document; no Cargo command or product experiment is executed.

## Current decision: retain; no archive execution

This plan proposes review boundaries only. It authorizes no move, rename, deletion, module split, Python removal, log rotation or lock/history relocation. Unknown active, scheduled, historical-replay, foreign or future consumers block archive. The older paper tree and retired exp-next-02 retain I53's HOLD/UNPROBED dispositions; no new observation of either is made.

The controlling ADR is `docs/adr/ADR-0006-rust-only-stack.md`, not a guessed alternate filename. Its Rust-only direction and phased differential-port requirements do not prove that all Python consumers have disappeared at C. In particular, CI orchestration still uses Python and the CLI's train compatibility source still exposes a Python argument/worker. Do not remove a live path on the basis of its intended replacement.

## Candidate register

| Candidate / proposed boundary | Verified source consumers or constraint at C | Owner and current decision | Evidence required before a later transition |
| --- | --- | --- | --- |
| crates/jope-campaigns/src/evaluation_v2.rs split | main.rs declares evaluation_v2; evaluator identity hashes this file, related exact-CV/data files, Cargo inputs and feat-math source bytes. | agy + evaluator subsystem owner; HOLD. | Full caller/feature graph, exact independent old replay, proposed module/target inventory and a task-named evaluator identity transition. No first-closure-path refactor. |
| crates/feat-math/src/evolve.rs extraction | lib.rs exports evolve; jope-bench methods imports and calls evolve_lite_race. It is evaluator-hashed and one of the Clover 12 audited files. | agy + search owner; HOLD. | Preserve existing entry points, deterministic controls, profile semantics and all audit literals; compare exact before/after selection, predictions, emitted bytes and costs under admitted proof. |
| CLI/pipeline/search consolidation | Existing bench lite-race and pipeline kernel2 routes have distinct callers/configuration; shared vocabulary does not prove shared search behavior. | agy + search/CLI owners; DESIGN ONLY. | Read actual caller graph, bind strategy versus profile IDs, retain compatibility aliases and compare feature/test/cost inventories before proposing a source move. |
| Existing Python orchestration/tooling | ADR-0006 names staged ports; pinned CI parser/runner code remains live and was exercised in the separate I63/I64 staging receipt. No new rerun is claimed here. | agy + each installed/tool/CI owner; HOLD while consumer unknown or active. | Differential captures on the same inputs, installed service/caller mapping, inventory-floor/coverage parity, exact replacement route and stopped-head release. No active Python training exception is created. |
| GP-emitted Python / parity harness | ADR-0006 explicitly preserves the emitted-code/harness category and pinned grammar assets. | Emitter/compatibility owner; RETAIN. | Preserve exact artifact/parity obligations; removal is outside this plan. |
| Old plans, logs, locks, claim history and generated evidence | Coordination, reproduction, future consumers and custody may depend on path/bytes; age does not discharge them. | Current owner + custody reviewer; HOLD. | Owner-specific stopped-head release, a specific successor, complete filesystem/consumer closure and independent restoration through I53. |
| Bulk payloads, checkpoints, replay and model state | Git does not protect all local bytes; exact-resume, rollback, claim-reproduction and future-use promises differ. | Storage/custody owner; HOLD. | Exact physical-version dependency and recovery obligations under I50; this archive plan cannot become a retirement executor. |

No row is a present remove candidate. A missing consumer is an unresolved fact, not a proposed silent default.

## Before / after proposal inventory

The current proposal deliberately changes no source path or module; its expected source delta is empty. A later nonempty proposal must supply the following paired inventories before review, rather than labeling an unbuilt proposed state “verified.”

| Inventory axis | Baseline at C | Proposed state in this documentation-only stage | Later transition comparison |
| --- | --- | --- | --- |
| Paths and Rust modules | Exact git tree and parent mod/pub mod declarations; campaigns main.rs → evaluation_v2, feat-math lib.rs → evolve/model/kernel. | Same paths and declarations; no moves. | Added/removed/renamed modules, visibility and every caller/entry point; unexplained disappearance blocks archive. |
| Cargo package names | Engine directory = jope-parallel; campaigns = jope-dime-regressor; encoders = starforge; manifests are the authority. | Unchanged; no Cargo edit. | Explicit dependency graph, package/features/defaults/required-features/platform cfg before and after; Cargo.lock changes trigger identity review. |
| Registered test targets | coverage.json discovered targets and required_features, committed inventory, library/bin/integration/doctest distinctions. | Unchanged; no new target. | Exact set difference and routing, including new/removed/gated/ignored names; agy updates coverage.json, test-inventory.json and README from actual discovery. |
| Committed default listing | 5,013 Rust names, 72 inventoried ignored names at C; listing, not fresh execution. | Same source baseline; not recollected. | Compare the same feature/profile/toolchain/platform first, then separate supplementary variants. Unknown execution stays unknown. |
| Static ignore census | 86 source attributes; 72 matched + 3 ASM + 5 neural + 5 testing + 1 not(debug_assertions). | Preserved as unlike populations. | Keep category attribution; never force totals to agree by dropping tests or changing an inventory number by hand. |
| Neural / testing | Neural uses optional tch/LibTorch; policy testing targets and partial cfg differ from default collection. | Unchanged / not run. | Nonzero identical-feature collection and execution, supported SDK/ABI and explicit unavailable routes. A default zero is not parity. |
| Historical replay | Accepted old identity, frozen inputs, source blobs, exact reference behavior and independent route. | Retained; not rerun. | Demonstrate the original remains restorable/executable with the source tree moved or absent; do not rewrite its historical hashes. |
| Deployed/configured/future consumers | UNKNOWN until owner-admitted current inventory. | UNKNOWN / HOLD. | Installed units, CLI paths, imports, report links, mounts, timers, logs/locks and foreign consumers; unknown blocks archive. |

A later proposed inventory is a review artifact, not execution proof. Attach exact commands/source hashes and collected/executed/pass/fail/ignored/filtered counts only when actually run under admission.

## Identity migration and guard checklist

1. Intersect every proposed changed source path AND parent export/Cargo hook with evaluator_identity.v1.json and evaluation_v2.rs's actual input list. Byte-equivalent behavior does not imply equal implementation hash after a source move.
2. Intersect teaching/corpus/eval-query lineage sources. model.rs or task_contract.rs moves/edits can change lineage and HWM comparability; linked executable changes also mint runtime identities even with an empty listed-source intersection.
3. Preserve all FROZEN_FILES at compat_freeze.rs:2705-2726: vocabulary/grammar/accepted semantic manifest/registry/champion/runtime/emitted pins. No migration mode grants permission to change frozen behavior or registry line positions.
4. Any witness/authority source or executable change needs its own original signed policy/runtime gate; routine rotation is not implicit authority for new semantics.
5. Check all 12 Clover audited sources, including evolve.rs, Cargo.lock and authority.rs. Audited bytes must equal committed HEAD and required literals remain; no bypass or dirty-tree rebaseline.
6. Retention's exact-directory/include_str and forbidden-substring guard remains intact; an archive plan cannot remove a guarded module to hide an effectful path.
7. agy serializes coverage/discovery, inventory/README, parent exports, Cargo, known identities and transition fixtures. Record old/new implementation and identity digests, HWM implications, task-named transition, historical_v1 coverage, legacy comparisons and receipt-backed known-table rows.

The source-defined identity-only mode is for an explicitly admitted, committed source transition. Never use blanket regeneration to conceal semantic drift, fabricate an old receipt, silently rewrite accepted historical IDs, or call a simple path rename identity-neutral.

## Future release / rollback rule

Retain the current route until exact replacement, consumer, identity/parity and custody proofs are accepted. Stop on an unaccounted caller, changed source binding, missing SDK/test target, stale ownership, failed/empty required comparison or unknown recovery obligation. Rollback disables the additive route and restores the accepted entry point through the same reviewed integration branch, retaining candidate/failure evidence; no reset, forced history rewrite or cleanup.

The companion subsystem map describes present boundaries and future adapters. Neither document is a new coordination board or second backlog; tasks.json, existing claims, task receipts and agy's one integration PR remain authoritative.

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
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:docs/adr/ADR-0006-rust-only-stack.md:30` | Rust-only decision, preserved emitted Python and phased ports |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-campaigns/src/main.rs:24` | Active evaluation_v2 module declaration |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-campaigns/src/evaluation_v2.rs:130` | Actual evaluator implementation source hash inputs |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/feat-math/src/lib.rs:54` | Public evolve module |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-bench/src/methods.rs:190` | Active evolve_lite_race consumer |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-cli/src/train.rs:77` | Existing Python compatibility argument remains visible |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:docs/parity/test-inventory.json:9` | Committed default collection count |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/feat-math/tests/fixtures/compat_freeze/evaluator_identity.v1.json:9` | Evaluator source census |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/feat-math/tests/fixtures/compat_freeze/lineage_sources.v1.json:4` | Lineage source and executable identity obligations |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/feat-math/tests/compat_freeze.rs:126` | Identity-only transition mode |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/feat-math/tests/compat_freeze.rs:2705` | Frozen bytes |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-learning/src/clover/expansion/config.rs:37` | 12 audited sources |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-learning/tests/retention.rs:918` | Exact retention module census |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:crates/jope-records/src/coverage.rs:440` | Unassigned target additions fail |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:docs/adr/ADR-0004-evaluator-identity-known-table.md:17` | Receipt-backed append-only known identity rule |
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
