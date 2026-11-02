//! Reporting-only nested procedure audit. No fitting or search entry point is linked here.
//! Hashes authenticate retained bytes, not the truth of an executor's provenance claim.
mod planning;
mod report;
mod types;

pub use planning::plan;
use planning::{asset, build, digest, hash, read, PROTOCOL};
pub use report::report;
use report::{average, gains, rmse, sign};
pub use types::{
    Arm, Asset, Dataset, FailedPrediction, Freeze, Limits, Plan, Prediction, Release, Row, Search,
    SelectionReceipt, Specification,
};

#[cfg(test)]
#[path = "transfer_tests.rs"]
mod tests;
