//! Shared model-building options for `--automl` and `--ensemble` (SC-DEC-184 JA-02).
//!
//! One resolver turns a configuration file and command-line switches into the four
//! `(automl, ensemble)` routes. A route whose checkpoint input does not verify is refused with
//! the reason, and no generic CPU GP or ridge fallback is substituted. A route whose stages are
//! not yet implemented is reported as `Pending`, never as completed.
pub mod artifact;
pub mod candidates;
pub mod config;
pub mod ensemble;
pub mod oof;
pub mod selection;

pub use ensemble::{build_winner_ensemble, EnsembleDisposition, WinnerEnsemble, ENSEMBLE_SCHEMA};

pub use candidates::{
    bakeoff_roster, Admission, CandidateSpec, CatalogBinding, RuntimeCapability, CATALOG_BINDING,
};
pub use oof::{ExpectedPlan, FoldRecord, FoldStatus, OofPredictionBundle, OOF_SCHEMA};
pub use selection::{
    run_bakeoff, BakeoffPlan, BakeoffReceipt, BudgetExhausted, CandidateFitter, CandidateReport,
    CandidateStatus, EvaluationBudget, FitEvidence, FoldFit, NotStarted, APPROVED_FOLD_COUNT,
};

pub use config::{
    disposition, resolve, route, verify_checkpoint, CheckpointClaim, CheckpointRefusal,
    Disposition, ModelBuildArgs, ModelBuildFile, ModelBuildFlags, NeuralCheckpoint, ReleaseClosure,
    ResolvedModelBuild, Route, Source, VerifiedNeuralCheckpoint, OPTIONS_SCHEMA,
};
