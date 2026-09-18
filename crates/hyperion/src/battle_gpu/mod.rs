//! jain-battle-gpu's cross-process GPU lease queue (`queue.rs`, `queue/storage.rs`,
//! `control.rs`, `queue_tests.rs`), copied verbatim: std + anyhow only, no cudarc. Kept under the
//! `battle_gpu` name so `hyperion_v7::backend` and `hyperion::runtime` keep their import paths.

pub mod control;
pub mod queue;
