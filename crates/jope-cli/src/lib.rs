//! Offline synthetic fixtures, CPU Hyperion encoding, GP fit/emit/search, and a local router.
//!
//! The library exposes the bounded operations behind the `jope` command-line tool. Fixture
//! admission verifies deterministic generator identity before consumers use the admitted data.

mod admission;
mod champions;
mod encode;
mod generator;
mod kernel;
pub mod research_view;
mod router;
mod schema;
mod storage;

pub use admission::{verify_inputs, Verification};
pub use champions::{
    local_champions, local_train_step, refuse_s3, write_json, ChampionBank, TrainStepReport,
    UnsupportedDataset,
};
pub use encode::{encode_dataset, encode_dataset_with_landmarks, EncodeReport};
pub use generator::generate_fixtures;
pub use kernel::{
    emit_genome, emit_genome_with_options, emit_genome_with_train, emit_numeric_v2, fit_dataset,
    search_dataset, EmitReport, FitReport, SearchReport, DEFAULT_GENOME,
};
pub use router::{load_router, route_dataset, RouteReport};
pub use storage::resolve_root;

#[cfg(test)]
mod tests;
