//! Synchronous training on saved real contexts and adaptive GP teaching labels.
use crate::{
    corpus::{Index, SEQUENCE},
    manifest::*,
    neural::{AdamW, Bf16, Config, Nccl, Policy},
    telemetry::now,
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufWriter, Write},
    path::Path,
    time::{Duration, Instant},
};
use tch::{nn, Device, Kind, Tensor};

pub mod ancestry;
pub mod checkpoint_commit;
pub mod consumption;
pub mod continuation;
pub mod inputs;
pub mod intent;
pub mod objective;
pub mod partition;
pub mod replay_bridge;
pub mod topology;

mod checkpoint;
mod rank_api;
mod rank_impl;
mod sampling;

#[cfg(test)]
#[path = "training/native_launch_tests.rs"]
mod native_launch_tests;
#[cfg(test)]
#[path = "training/sampler_id_tests.rs"]
mod sampler_id_tests;
#[cfg(test)]
#[path = "training/checkpoint_tests.rs"]
mod tests;

pub(crate) use checkpoint::parameter_hash;
use checkpoint::{checkpoint, hash_parameter_values};
use rank_api::native_attempt_launch;
pub(crate) use rank_api::smartcluster_rank;
pub use rank_api::{
    bounded_rank, communicator_source, qualify_batch_rank, rank, rank_replay, rank_with_policy,
    verify_admitted_bootstrap, CommunicatorSource,
};
use rank_impl::rank_impl;
pub use sampling::{
    active_sampler_id, mix, sample, sample_replay, sample_replay_fresh, validation_panel,
    validation_panel_partitioned, REPLAY_SAMPLER_ID, SAMPLER_ID, SEARCH_BEATING_SAMPLER_ID,
};
use sampling::{validation_families, ValidationPanel};
