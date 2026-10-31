//! JopeDime v1 data contract (spec §4, §8.4 and Appendix B).
//!
//! Shared wire types for clients and service adapters. [`ids`] supplies validated
//! identifier newtypes, [`decimal`] encodes `u64` values as canonical decimal
//! strings, and [`error`] defines the closed error catalog and problem document.
//! Run, artifact, metric, system, warning and event records derive JSON schemas;
//! [`registry`] binds named payload types to the operation and terminal surfaces.
//!
//! The `jope-api-openapi` binary generates the OpenAPI document from that registry.
//! Contract tests check JSON round trips and committed schema/exchange snapshots;
//! the repository's API-schema drift check compares the generated OpenAPI bytes.
//! See `FOLLOW_UPS.md` for maintenance commands and qualification boundaries.
//!
//! These types and checks describe the wire contract. They do not implement
//! handlers, storage, transport, runtime state transitions or numerical behavior.
//! Runtime semantic validation and release qualification remain separate gates.

pub mod artifacts;
/// Canonical decimal-string serialization and validation for wire integers.
pub mod decimal;
pub mod events;
pub mod metrics;
pub mod registry;
pub mod runs;
pub mod system;
pub mod warnings;

/// Closed error catalog and RFC 9457 problem documents.
pub mod error;

/// Parsing, ordering and validated identifier newtypes.
pub mod ids;

pub use artifacts::*;
pub use events::*;
pub use metrics::*;
pub use runs::*;
pub use system::*;
pub use warnings::*;
