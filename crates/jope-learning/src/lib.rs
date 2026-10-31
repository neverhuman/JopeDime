//! JOPE-Dime learning store (WP11 L1/L2): the unified high-water-mark ledger `jope.hwm/v1`,
//! the score ledger `jope.score/v1`, append-only hash-chained JSONL batches, and conditional
//! `current.json` pointers, all under `$JOPE_STATE` with an optional `object_store` mirror.
//!
//! Layout under `$JOPE_STATE`:
//!
//! ```text
//! blobs/sha256/<64hex>                               content-addressed record blobs
//! hwm/ledger/<yyyy>/<mm>/<ulid>.jsonl               every HWM event, sealed, hash-chained
//! hwm/ledger/HEAD.json                               tip of the HWM chain
//! hwm/<task>/<dataset_hash>/<cut_id>/<identity>/current.json    exact-comparison current record
//! scores/ledger/<yyyy>/<mm>/<ulid>.jsonl            every score row, sealed, hash-chained
//! scores/ledger/HEAD.json
//! revocations/ledger/<yyyy>/<mm>/<ulid>.jsonl       revoked evidence imported from the claims graph
//! progress/outbox/...                               the W8 progress outbox
//! ```

#![cfg_attr(doc, deny(missing_docs))]

pub mod arena_exact;
pub mod bank;
pub mod canon;
pub mod champion;
pub mod civil;
pub mod clover;
pub mod custody;
pub mod eligibility;
pub mod encoder_receipt;
pub mod exe_cache;
pub mod hwm;
pub mod lake;
pub mod ledger;
pub mod metric;
pub mod mirror;
pub mod pools;
pub mod progress;
#[path = "selfplay/promote.rs"]
pub mod promote;
pub mod proposals;
pub mod qualify;
pub mod record;
pub mod replay;
pub mod research;
pub mod retention;
pub mod retrieval;
pub mod revocation;
pub mod scores;
pub mod selfplay;
pub mod sources;
pub mod state;
pub mod support;
pub mod sync;
pub mod universe;

/// Version stamped into ledger seals.
pub const TOOL_VERSION: &str = concat!("jope-learning/", env!("CARGO_PKG_VERSION"));

/// The committed `jope.hwm/v1` schema (tests prove it agrees with [`record::HwmRecord`]).
pub const HWM_SCHEMA_JSON: &str = include_str!("../../../schemas/jope.hwm.v1.schema.json");
/// The committed `jope.score/v1` schema (tests prove it agrees with [`record::ScoreRecord`]).
pub const SCORE_SCHEMA_JSON: &str = include_str!("../../../schemas/jope.score.v1.schema.json");
