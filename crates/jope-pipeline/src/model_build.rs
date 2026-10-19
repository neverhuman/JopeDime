//! Shared model-building options for `--automl` and `--ensemble` (SC-DEC-184 JA-02).
//!
//! One resolver turns a configuration file and command-line switches into the four
//! `(automl, ensemble)` routes. Source-level only: no route executes here. A route whose
//! checkpoint-bound neural input is absent is refused with the reason, and no generic CPU GP
//! or ridge fallback is substituted. A route whose stages are not yet implemented is reported
//! as `Pending`, never as completed.
pub mod config;

pub use config::{
    disposition, resolve, route, Disposition, ModelBuildArgs, ModelBuildFile, ModelBuildFlags,
    NeuralCheckpoint, ResolvedModelBuild, Route, Source, OPTIONS_SCHEMA,
};
