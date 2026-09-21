//! Rust-first JOPE self-play control plane.
//!
//! Python is intentionally restricted to the PyTorch training/proposal
//! boundary. Queueing, provenance, fitness bookkeeping, freshness, datasets,
//! and deployment policy live in this crate.

pub mod dime_activation;
pub mod dime_runtime_contracts;
pub mod encoder_v2_authority;
pub mod encoder_v2_capacity;
pub mod encoder_v2_execution;
pub mod encoder_v2_identity;
pub mod encoder_v2_store;
pub mod encoder_v2_worker;
pub mod epoch6;
pub mod epoch6_wave;
pub mod four_task;
pub mod hyperion_gate;
pub mod hyperion_router;
pub mod jope_dlgp_best5;
pub mod jope_generalized_gp;
pub mod recovery_contracts;
pub mod router_contracts_v2;
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
pub mod timeseries;
pub mod training_history;
