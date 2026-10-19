//! Shared model-building options for `--automl` and `--ensemble` (SC-DEC-184 JA-02).
//!
//! One resolver turns a configuration file and command-line switches into the four
//! `(automl, ensemble)` routes. A route whose checkpoint input does not verify is refused with
//! the reason, and no generic CPU GP or ridge fallback is substituted. A route whose stages are
//! not yet implemented is reported as `Pending`, never as completed.
pub mod artifact;
pub mod automl;
pub mod candidates;
pub mod catalog;
pub mod config;
pub mod driver;
pub mod ensemble;
pub mod executor;
pub mod identity;
pub mod native;
pub mod oof;
pub mod refit;
pub mod selection;
pub mod test_look;

pub use automl::{run_automl, AutomlOutput};
pub use ensemble::{
    admit_winner_ensemble, EnsembleDisposition, ReportPredictions, WinnerEnsemble, ENSEMBLE_SCHEMA,
};
pub use identity::ExecutionIdentity;
pub use refit::{refit_members, MemberRefit, OuterRefitRecord, RefitOutcome, REFIT_RECORD_SCHEMA};

pub use candidates::{
    bakeoff_roster, Admission, CandidateSpec, CatalogBinding, RuntimeCapability, CATALOG_BINDING,
};
pub use oof::{ExpectedPlan, FoldRecord, FoldStatus, OofPredictionBundle, OOF_SCHEMA};
pub use selection::{
    run_bakeoff, BakeoffPlan, BakeoffReceipt, BudgetCounters, BudgetExhausted, BudgetKind,
    CandidateFitter, CandidateReport, CandidateStatus, EvaluationBudget, FitEvidence, FoldFit,
    NotStarted, VerifiedBundle, APPROVED_FOLD_COUNT,
};

pub use config::{
    disposition, resolve, route, route_of, verify_checkpoint, CheckpointClaim, CheckpointRefusal,
    Disposition, ModelBuildArgs, ModelBuildFile, ModelBuildFlags, NeuralCheckpoint, ReleaseClosure,
    ResolvedModelBuild, Route, Source, VerifiedNeuralCheckpoint, OPTIONS_SCHEMA,
};
