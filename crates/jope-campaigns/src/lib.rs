//! JOPE benchmark campaigns and the shared library of the former root crate.
//!
//! Python is intentionally restricted to the PyTorch training/proposal
//! boundary. Queueing, provenance, fitness bookkeeping, freshness, datasets,
//! and deployment policy live in this crate (the `selfplay_*` modules are the
//! library half that `jope-selfplay`, the flywheel control plane, builds on).
//!
//! The crate's binary carries the benchmark-campaign programs
//! (`commands::PROGRAMS`, dispatched by `dispatch::Table`); the former root
//! binary `jope-dime-regressor` is `commands::jope_dime_regressor`.

pub mod aggressive;
pub mod args;
pub mod commands;
pub mod core;
pub mod data;

pub mod dime_activation;
/// One binary, many programs: the argv[0] dispatcher shared with `jope-selfplay`.
pub mod dispatch;
pub mod dime_runtime_contracts;
pub mod encoder_v2_authority;
pub mod encoder_v2_capacity;
pub mod encoder_v2_execution;
pub mod encoder_v2_identity;
pub mod encoder_v2_store;
pub mod encoder_v2_worker;
pub mod epoch6;
pub mod epoch6_wave;
pub mod evaluation_v2;
pub mod followup;
pub mod four_task;
pub mod genome_boost;
pub mod golden42_v3;
pub mod hyperion_gate;
pub mod hyperion_router;
pub mod jope_dlgp_best5;
pub mod jope_generalized_gp;
/// ADR-0003 locations: every former home-directory path, by root.
pub mod paths;
pub mod ratchet;
pub mod recovery_contracts;
pub mod router_contracts_v2;
pub mod runtime;
pub mod score_corpus;
pub mod search_v3;
pub mod selfplay_core;
pub mod selfplay_data;
pub mod selfplay_expansion;
pub mod selfplay_ingest;
pub mod selfplay_registry;
pub mod selfplay_runtime;
pub mod selfplay_shadow;
/// WS-2: the single source of truth for scq job specs, and the drift test that
/// fails when two renderers disagree.
pub mod spec_drift;
pub mod spec_policy;
/// R2: the library's task-vocabulary authority (TaskKind + corpus-dir and
/// primary-metric maps); `selfplay_shadow::Task` remains the wire type.
pub mod task;
pub mod task_preflight;
pub mod task_rating;
pub mod test_adaptive;
pub mod timeseries;
pub mod training_history;

/// Items the campaign modules reach as `crate::…` because they sat at the root
/// of the former binary crate (`src/main.rs`, now `commands::jope_dime_regressor`).
pub(crate) use commands::jope_dime_regressor::{
    canonical_json, dataset_seed, elo_standings, hash_file, load_baselines, pairwise_results,
    score_dataset, seed_bytes, sha256_bytes, validate_static_inputs, ScoredDataset, DATASETS,
    PANEL_ANALYSIS_SHA256, PANEL_SHA256, PREDICTION_FILE, REPORT_FILE,
};

/// The `jope-campaigns` binary's program table.
pub static TABLE: dispatch::Table = dispatch::Table {
    binary: "jope-campaigns",
    about: "JOPE benchmark campaigns: Golden42/46 panels, exact-cv, test-adaptive, DLGP best-5, generalized GP, search v3, corpus_eval, Hyperion gate/router, canary/capacity/causal tools",
    programs: commands::PROGRAMS,
};
