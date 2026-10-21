/// ADR-0003 runtime roots (`JOPE_HOME`, `JOPE_STATE`, `JOPE_REGISTRY`, …).
pub use jope_paths as paths;

pub mod audit;
pub mod build_receipt;
pub mod champions;
pub mod checkpoint_recovery;
pub mod controller;
pub mod dashboard;
pub mod encoding;
pub mod families;
pub mod family_growth;
pub mod fit_boundary;
pub mod gp;
pub mod host_os;
pub mod learning_history;
pub mod manifest;
pub mod native_graph;
#[cfg(feature = "neural")]
pub mod neural;
pub mod operations;
#[cfg(feature = "neural")]
pub mod policy_export;
pub mod prepare;
pub mod proposal_population;
pub mod qualification_slots;
pub mod queue;
pub mod runtime_custody;
pub mod service;
pub mod snapshot;
pub mod telemetry;
pub mod tournament;
pub mod training_exposure;
pub mod transport;

pub mod hyperion;

#[cfg(feature = "neural")]
pub mod components;

#[cfg(feature = "neural")]
pub mod contexts;

#[cfg(feature = "neural")]
pub mod auxiliary_training;
pub mod corpus;
pub mod corpus_projection;
pub mod corpus_publication;
pub mod eval_registry;
#[cfg(feature = "neural")]
pub mod gpu_worker;
// The original grammar.rs remains byte-frozen for historical conformance.
#[path = "grammar/automaton.rs"]
pub mod grammar;
pub mod jain_vocabulary;
pub mod label_admission;
#[cfg(feature = "neural")]
pub mod pipeline;
#[cfg(feature = "neural")]
pub mod proposal_ranking;
#[cfg(feature = "neural")]
pub mod proposals;
#[cfg(feature = "neural")]
pub mod ranked_proposals;
pub mod shard_index;
pub mod task_contract;
pub mod teaching_store;
#[cfg(feature = "neural")]
pub mod training;

pub mod campaign;

#[cfg(feature = "neural")]
pub mod monitoring;

pub mod sealed_storage;

pub mod resident_protocol;
pub mod smartcluster_worker;

pub mod native_genome;
pub mod native_loop;

pub mod continuation_corpus;

pub mod native_observer;

mod native_persistence;

pub(crate) mod native_review;

pub mod native_provisioner;

pub mod preparation_overlay;

pub(crate) mod native_input_cache;
