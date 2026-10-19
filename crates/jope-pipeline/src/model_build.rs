//! Shared model-building options for `--automl` and `--ensemble` (SC-DEC-184 JA-02).
//!
//! One resolver turns a configuration file and command-line switches into the four
//! `(automl, ensemble)` routes. A route whose checkpoint input does not verify is refused with
//! the reason, and no generic CPU GP or ridge fallback is substituted. A route whose stages are
//! not yet implemented is reported as `Pending`, never as completed.
pub mod config;

pub use config::{
    disposition, resolve, route, verify_checkpoint, CheckpointClaim, CheckpointRefusal,
    Disposition, ModelBuildArgs, ModelBuildFile, ModelBuildFlags, NeuralCheckpoint, ReleaseClosure,
    ResolvedModelBuild, Route, Source, VerifiedNeuralCheckpoint, OPTIONS_SCHEMA,
};
