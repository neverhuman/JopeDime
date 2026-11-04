//! Reporting-only nested procedure audit. No fitting or search entry point is linked here.
//! Hashes authenticate retained bytes, not the truth of an executor's provenance claim.
mod planning;
mod report;
mod types;

pub use planning::plan;
pub use report::report;
pub use types::{
    Arm, Asset, Dataset, FailedPrediction, Freeze, Limits, Plan, Prediction, Release, Row, Search,
    SelectionReceipt, Specification,
};

#[cfg(test)]
#[path = "transfer_tests.rs"]
mod tests;
