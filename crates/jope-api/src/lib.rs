//! JopeDime v1 data contract (spec §4, §8.4 and Appendix B).
//!
//! This first landing supplies the schema skeleton before API-06 registers the
//! crate. The inline compatibility modules below intentionally reserve the
//! API-02/03/04 names without writing their owned files. Their string aliases
//! preserve the example JSON; they do **not** validate IDs, catalog codes or
//! canonical decimal integers. Replace these modules with external declarations
//! when those tasks land, before claiming a frozen contract. See FOLLOW_UPS.md.
//! No handler, runtime, store, tensor or transport dependency belongs here.

pub mod artifacts;
pub mod events;
pub mod metrics;
pub mod runs;
pub mod system;
pub mod warnings;

/// API-02 supplies the closed catalog and RFC 9457 Problem in `error.rs`.
pub mod error;

/// API-03 supplies parsing, ordering and validated newtypes in `ids.rs`.
pub mod ids {
    pub type RunId = String;
    pub type DatasetId = String;
    pub type AlgorithmId = String;
    pub type ArtifactId = String;
}

/// API-04 supplies canonical string serialization and validation in `decimal.rs`.
pub mod decimal {
    pub type DecimalU64 = String;
}

pub use artifacts::*;
pub use events::*;
pub use metrics::*;
pub use runs::*;
pub use system::*;
pub use warnings::*;
