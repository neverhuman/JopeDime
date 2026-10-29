//! Append-only advanced-v3 table and task boundary.
//!
//! This module intentionally does not change the legacy numeric-matrix API.
//! A v3 model owns its fitted raw-table encoder, task contract, and schema
//! hashes; legacy `model::fit` and `transpile::emit` continue down their
//! historical paths byte-for-byte.

mod calibration;
mod csv;
pub(crate) mod emit;
mod evolution;
mod feature_block;
mod fitted_cpp;
#[cfg(feature = "fitted-cpp")]
mod fitted_cpp_runtime;
mod fitted_portable;
#[cfg(feature = "fitted-cpp")]
mod fitted_runtime_image;
mod grammar;
mod heads;
mod input;
pub(crate) mod interfaces;
mod pipeline;
mod profile;
mod schema;
mod transforms;
mod types;

pub use csv::RawTable;
pub use emit::{emit_bound, emit_bound_rust_only, BoundEmitted};
pub use evolution::{
    crossover_candidates, deterministic_seed_enumeration, initial_population, mutate_candidate,
    next_generation, V3Archive, V3Candidate, V3EvolutionConfig, V3ScoredCandidate,
};
pub use feature_block::{
    FeatureBlock, FeatureLineage, FeatureOrigin, FeatureRole, FeatureWidths,
    MAX_GENERATED_DENSE_WIDTH,
};
// B12: distinct fitted v2 row-vector route.
pub use fitted_cpp::emit_row_vector_cpp17;
pub use fitted_cpp::{emit_fitted_cpp17, supports_fitted_cpp17, FittedCppArtifact};
#[cfg(feature = "fitted-cpp")]
pub use fitted_cpp_runtime::{
    encode_fitted_cpp17, validate_fitted_cpp_codec, EncodedTable, FittedCppError, FittedCppModel,
    FittedCppRuntime, FittedModelCodec,
};
pub use fitted_portable::{
    emit_fitted_portable, emit_fitted_public_cpp17, fitted_javascript_unsupported,
    fitted_portable_templates, fitted_portable_unsupported, fitted_public_cpp_unsupported,
    lint_fitted_public_text, FittedPortableSources, PortableUnsupported, FITTED_PORTABLE_EMITTER,
    FITTED_PUBLIC_CPP_EMITTER,
};
pub use heads::MULTICLASS_MAX_TRANSFORMED_FEATURES;
pub use pipeline::{
    bound_from_retained_legacy_model, check_direct_bound_emission_support,
    fit_legacy_numeric_bound, fit_table, predict_labels_table, predict_multiclass_table,
    predict_probabilities_table, predict_table, try_bound_pipeline_from_retained_legacy_model,
    validate_v3_genome, BoundModel, MulticlassPrediction,
};
pub use schema::{ColumnKind, ColumnSpec, DataSchema, RowSemantics};
pub use types::{
    balanced_accuracy_at, macro_f1, multiclass_accuracy, stratified_fold_assignments, Objective,
    TaskKind, TaskSpec, ARTIFACT_SCHEMA,
};

#[cfg(test)]
use input::{FittedInput, InputConfig};
#[cfg(test)]
mod tests;
