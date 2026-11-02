//! feat-math — the vendored new-math engine: a pipeline-program DSL with a Rust
//! reference interpreter, dual standalone-code emitters, and an in-process
//! search over pipelines.
//!
//! Programs are s-expressions like
//! `(pipe X (winsor_clip q=0.1) (cheb degree=6) (robust_scale) (histgbm depth=5 lr=0.15 rounds=150))`.
//! Three implementations agree bit-for-bit: (1) this crate's f64 interpreter
//! (`model::fit/predict`, used to EXECUTE programs during synthesis — the Rust
//! version), (2) the emitted standalone Rust artifact, and (3) the emitted
//! standalone dependency-free Python artifact (`transpile::emit`).
//!
//! Fairness note: `cv_r2` refits the ENTIRE pipeline (fitted transforms,
//! target-aware encoders, head) inside every fold, so target-aware operators
//! are out-of-fold by construction in honest CV.

// DSL + interpreter (the execution engine).
pub mod advanced_v3;
pub mod artifact_manifest;
pub mod boost2;
pub mod combinators;
pub mod cxx;
pub mod dsl;
pub mod feasibility;
pub mod fm_oof;
pub mod folds;
pub mod full_rust;
pub mod genome;
pub mod gpr;
pub mod heads;
pub mod kernel;
pub mod krr;
pub mod ksmooth;
pub mod math_expr;
pub mod members;
pub mod metrics;
pub mod model;
pub mod numeric_v2;
pub mod numeric_v2_emit;
pub mod ops;
pub mod portable_math;
pub mod portfolio;
pub mod public_names;
pub mod registry;
pub mod rng;
pub mod router_policy;
pub mod stack;
pub mod stage_cache;
pub mod target_transform;

// Standalone-code emitters (genome -> full Python + full Rust).
mod emit_py;
pub mod emit_rs;
mod mathexpr;
pub mod source_stream;
pub mod spec;
pub mod tokens;
pub mod transpile;
pub mod typescript;

// Invention: evolutionary search over pipeline genomes + token-credit accounting.
pub mod credit;
pub mod evolve;
pub mod explain;

pub use advanced_v3::{
    balanced_accuracy_at, bound_from_retained_legacy_model, check_direct_bound_emission_support,
    crossover_candidates, deterministic_seed_enumeration, emit_bound, emit_bound_rust_only,
    emit_fitted_cpp17, fit_legacy_numeric_bound, fit_table, initial_population, macro_f1,
    multiclass_accuracy, mutate_candidate, next_generation, predict_labels_table,
    predict_multiclass_table, predict_probabilities_table, predict_table, supports_fitted_cpp17,
    try_bound_pipeline_from_retained_legacy_model, BoundEmitted, BoundModel, ColumnKind,
    ColumnSpec, DataSchema, FeatureBlock, FeatureLineage, FeatureOrigin, FeatureRole,
    FeatureWidths, FittedCppArtifact, MulticlassPrediction, Objective, RawTable, RowSemantics,
    TaskKind, TaskSpec, V3Archive, V3Candidate, V3EvolutionConfig, V3ScoredCandidate,
    MAX_GENERATED_DENSE_WIDTH, MULTICLASS_MAX_TRANSFORMED_FEATURES,
};
/// The fitted-model codec and Rust host runtime for C++17 model artifacts (feature `fitted-cpp`, default-on).
#[cfg(feature = "fitted-cpp")]
pub use advanced_v3::{
    encode_fitted_cpp17, validate_fitted_cpp_codec, EncodedTable, FittedCppError, FittedCppModel,
    FittedCppRuntime, FittedModelCodec,
};
pub use folds::fold_of;
pub use genome::GenomeIr;
pub use metrics::{
    accuracy_at, auc, average_precision, best_f1_threshold, brier, f1_at, lift_at, logloss, mae,
    pearson, r2, rmse, youden_threshold, LOGLOSS_EPS,
};
pub use model::{cv_r2, fit, fit_and_predict_train, fit_predict, CvReport, Model};
pub use rng::Pcg32;

/// The reigning vendored champion genome (honest 5-fold CV r2 = 0.924 on its
/// source dataset); used to seed the invention population.
pub const CHAMPION_GENOME: &str = include_str!("../assets/champions/8944cbdf.dsl");

/// Hard cap on pipeline width after any stage (matches the nm_ append cap).
pub const MAX_WIDTH: usize = 512;

/// Rows are plain row-major `Vec<Vec<f64>>` — deliberately the same shape as the
/// emitted artifacts (Python list-of-lists / Rust Vec<Vec<f64>>) so arithmetic
/// order is identical by construction.
pub type Rows = Vec<Vec<f64>>;

/// Replace non-finite values with 0.0, applied after every stage (mirrors the
/// engine's `finite_or_zero`).
pub fn sanitize(x: &mut Rows) {
    for row in x.iter_mut() {
        for v in row.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
    }
}

/// Complete native model export for C17 callers.
pub mod c17;

pub mod cuda;
/// Go output backed by the complete Rust WebAssembly implementation.
pub mod go;

static PARALLEL_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Enable or disable internal Rayon parallelism inside `feat_math` transforms and heads.
/// When set to `false` before any Rayon call, `feat_math` executes strictly on the calling
/// OS thread without initializing or waking a background Rayon thread pool.
pub fn set_parallel_enabled(enabled: bool) {
    PARALLEL_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
pub fn is_parallel_enabled() -> bool {
    PARALLEL_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}
