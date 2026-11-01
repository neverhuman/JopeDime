# Accepted-route task scope and writer/map review — t2345

Reviewed source `6419154c116175945502f6a570afafe356e8cf62` on 1 November 2026. This pass reads local Git source and existing paper artifacts only. No runtime/model/dataset/sealed/holdout payloads, fresh logs or scientific runs are used.

## Scope gap and corrections

The paper already called broader task types prospective, but did not explicitly limit the accepted native checkpoint route and fixed comparators to tabular regression. The E-2 finding is supported by these source boundaries:

| Boundary | Current source evidence | Paper correction |
| --- | --- | --- |
| Accepted context path | `crates/jope-policy/src/context_producer.rs`: `CpuProducer::encode_prepared` uses `TargetTask::Regression` for target windows/encoding | Section 6's opener and encoder description explicitly say regression |
| Accepted proposal | `crates/jope-policy/src/seeds.rs`: `propose` enforces the accepted checkpoint identity and delegates to `sample_rank_assemble`, which sets `task = "regression"` | Section 4.5 and the Section 6 opener limit the accepted route |
| Subsequent search | `crates/jope-pipeline/src/neural_search.rs`: `RegressionVariation::generic` requests regression grammar; `crates/jope-bench/src/panel/inference_gp.rs` constructs that variation | No inference from a generic search interface to task coverage |
| Fitting | `crates/jope-engine/src/gp.rs`: `fit` uses regression preprocessing and records a regression predictor | Proposed game-day input and narrative now specify regression |
| Fixed comparators | `crates/jope-bench/regressors/README.md:20–21` says entrants are regression-only; `crates/jope-bench/src/regressors.rs` refuses any other adapter task | Section 4 identifies fixed comparators as regressors; Section 5 scopes the discussed writer costs to regression |
| Separate classification code | `crates/jope-policy/src/context_producer/classification.rs` implements a separate `produce_classification` entry point | No claim that classification code is absent; that component does not widen the accepted end-to-end proposal/search/fit route |
| Retained adapter data | Existing admission and program-window evidence is regression-only, distinct from native writer generation | Section 6.8 now calls the later record a regression admission record, without inferring optimizer updates or transfer |

Binary/multiclass classification, ranking, forecasting and relational prediction remain prospective boards needing their own qualified encoding, language, metric and protocol. The shared formal interface and generic writer/map mathematics do not establish broader accepted capability.

## Remaining writer/value boundaries

Re-read every Section 4–6 scope, writer, value/map, training-target and search statement against the existing H56 source audit and 27 current source/note pins. The preceding 21 pins are unchanged. The native writer and map use separate inputs/embeddings; ranking neither screens nor conditions the writer, but ranks weight search-parent selection. The writer objective remains schematic with native smoothing and normalization stated separately. Historical reference training specifications, current implementation, later adapter ranking and proposed one-draw deployment remain distinguished. Exact runtime/checkpoint binding, update evidence and independent first-proposal results remain required; no further mismatch found beyond task scope.

All eight substitutions preserve physical source line counts. No equations, scalar values, float source order/options/widths, exhibit identities, figures/data/generators, bibliography or reserved source are changed. All 28 TBD bodies/source positions and 20 Needs anchors remain intact. Results, Meaning and abstract are carried byte-identically.

## Verification and authoring boundary

Clean nice19 serial make, source-bound diff/TBD/number/citation/import/link checks and page proofs are retained under `writer-pass103-*`. The requested branch is authored with a private Git index from the pinned main tree. Only a paper build export is materialized; no additional worktree/checkout or canonical source/index/HEAD mutation occurs at the over-cap host. The complete Git tree preserves every other path. No native test execution, installed-checkpoint qualification or scientific coverage is claimed by these documentation checks.

Final build/visual receipt: 27 pages and zero TeX/Bib/undefined/box warnings. Twelve before/after page proofs cover pages 8–13; context pages were inspected. Only Section 7.3 and Figure 13 change page assignments (to 15 and 16). Numbered exhibit identities and float placement settings/order remain intact. All literal numbers/equations, 124 cited bibliography keys, 194 citation occurrences, 110 external targets and 52 DOI targets are preserved.
