//! Offline fixtures, CPU Hyperion encode, feat-math fit/emit/search, local router.

mod admission;
mod champions;
mod encode;
mod generator;
mod kernel;
mod router;
mod schema;
mod storage;

pub use admission::{verify_inputs, Verification};
pub use champions::{
    local_champions, local_train_step, refuse_s3, write_json, ChampionBank, TrainStepReport,
    UnsupportedDataset,
};
pub use encode::{encode_dataset, EncodeReport};
pub use generator::generate_fixtures;
pub use kernel::{
    emit_genome, fit_dataset, search_dataset, EmitReport, FitReport, SearchReport, DEFAULT_GENOME,
};
pub use router::{load_router, route_dataset, RouteReport};
pub use storage::resolve_root;

#[cfg(test)]
mod tests;
