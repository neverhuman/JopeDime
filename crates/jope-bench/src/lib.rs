//! `jope-bench`: one cut for everyone.
//!
//! For every dataset in a directory tree, the benchmark builds train/held-out cuts with the one
//! fold authority (`feat_math::fold_of` keyed on `"<dataset_hash>:<seed>"`, the same key the
//! product pipeline uses), fits every tier-1 baseline as a single-head program such as
//! `(pipe X (ridge))`, fits any extra program the caller names, optionally runs the search on the
//! cut's training rows only, and scores **all of them on the same held-out rows with the same
//! evaluator**. Then it follows the honest screen-then-deepen protocol: every method is scored
//! on one screen cut, the top N are re-scored on every cut, and the number that counts is the
//! pooled out-of-fold one, penalised by fold variance
//! ([`eval::robust_fitness`]).
//!
//! What it deliberately does not do: touch the catalog's sealed `test.csv` (cuts are made from
//! training rows only), reuse the pipeline's private evaluator (it carries its own ~40 lines so
//! a change there cannot silently change a benchmark), or depend on any crate outside this
//! workspace. Tier-2 external adapters are a later step.
//!
//! Outputs are `jope.bench-report/v1` files ([`report`]) and, from those, `scores.csv` rows plus
//! registries for a local `jope-arena` track ([`emit`]), so the same Elo ladder code that ranks the
//! published benchmark ranks this one.

pub mod cuts;
pub mod emit;
pub mod eval;
pub mod load;
pub mod methods;
pub mod protocol;
pub mod report;

/// Identity of the evaluator every score in a report was produced by.
pub const EVALUATOR: &str = "jope-bench.feat-math-f64.v1";
/// Default fold seed: the product pipeline's `DEFAULT_SEED`, so cuts agree with `jope pipeline run`.
pub const DEFAULT_SEED: u64 = 20_260_907;
/// Default number of folds.
pub const DEFAULT_FOLDS: usize = 5;
