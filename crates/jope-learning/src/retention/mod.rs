//! S3 retention brain (gap S3-RET; `S3_STORAGE_STRATEGY_20261013.md` §7 and the binding §8).
//!
//! A read-only, fail-closed classifier. It maps every S3 object group to exactly one action and
//! writes a dry-run plan; it never calls S3 and has no code path that tags, transitions, copies
//! or deletes anything. Applying a plan is a separate, owner-approved step keyed by the plan's
//! sha256.
//!
//! Inputs are local snapshots, each pinned by sha256 in `brain-inputs.lock.json`:
//! - the uncapped S3 object ledger (`jopedime.s3-object-ledger/v1`), flattened into prefix groups
//!   ([`groups`]); S3 Inventory replaces it once the daily reports exist;
//! - the GP results index (`jopedime.gp-results-index/v1`), aggregated per group ([`index`]);
//! - rosters, seed packets and reference scans of the main tree and the evidence store
//!   ([`refs`]);
//! - each bucket's lifecycle configuration ([`lifecycle`]);
//! - the retention policy ([`policy`]): prices, thresholds and the prefix-to-kind map.
//!
//! The rules (§8): "poor", "dominated", "old" or "NON-QUAL" is never a reason to delete; training
//! value TV(g) is computed first and fails closed to true. A group may only EXPIRE when TV(g) is
//! false, an E1–E5 rule is proven with object-level evidence, and an independent reference scan
//! agrees. The ledger only gives prefix-level counts, where no E-rule can be proven, so this
//! version never emits EXPIRE; [`classify::expire_allowed`] states the gate.
//!
//! Write side (strategy §4 and §8.3), library only: [`tags`] builds validated write-time tags,
//! [`pack`] turns small result objects into >= 64 MiB zstd packs with a verifiable index, and
//! [`distilled`] writes and reconciles the `gp_distilled/v1` table through the minimal
//! [`parquet`] writer. None of them talks to S3; writers own their uploads.

pub mod classify;
pub mod distilled;
pub mod groups;
pub mod index;
pub mod inputs;
pub mod inventory;
pub mod lifecycle;
pub mod md5;
pub mod obligations;
pub mod pack;
pub mod parquet;
pub mod plan;
pub mod policy;
pub mod refs;
pub mod snappy;
pub mod tags;
pub mod value;

use serde::Serialize;

/// Schema identifier for plan row values.
pub const PLAN_ROW_SCHEMA: &str = "jopedime.retention-plan-row/v1";
/// Schema identifier for plan summary values.
pub const PLAN_SUMMARY_SCHEMA: &str = "jopedime.retention-plan-summary/v1";
/// Schema identifier for inputs lock values.
pub const INPUTS_LOCK_SCHEMA: &str = "jopedime.retention-brain-inputs/v1";

/// The one action a group gets. Serialized names follow strategy §8.4 and §8.7.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Action {
    /// Never delete; no transition proposed by this plan (T0).
    #[serde(rename = "KEEP_T0")]
    KeepT0,
    /// Keep readable in its current class (T1): live writer, active roster input, encodings.
    #[serde(rename = "KEEP_HOT")]
    KeepHot,
    /// Pack small objects into >= 64 MiB zstd shards plus an index; stays readable (T2).
    #[serde(rename = "COMPACT")]
    Compact,
    /// Distill rows into `gp_distilled/v1`, then pack the raw bytes and archive them (T3).
    #[serde(rename = "DISTILL+COLD")]
    DistillCold,
    /// Opaque archive (tarball, git bundle): cold storage only, nothing to distill (T3).
    #[serde(rename = "COLD")]
    Cold,
    /// No change; flagged for review (default rule, owner decision pending, or out of scope).
    #[serde(rename = "HOLD")]
    Hold,
    /// Provably redundant bytes (E1–E5 with object-level evidence). Never emitted by this version.
    #[serde(rename = "EXPIRE")]
    Expire,
}

impl Action {
    /// Constant identifying the all.
    pub const ALL: [Action; 7] = [
        Action::KeepT0,
        Action::KeepHot,
        Action::Compact,
        Action::DistillCold,
        Action::Cold,
        Action::Hold,
        Action::Expire,
    ];

    /// Returns the stable string representation of this value.
    pub fn as_str(self) -> &'static str {
        match self {
            Action::KeepT0 => "KEEP_T0",
            Action::KeepHot => "KEEP_HOT",
            Action::Compact => "COMPACT",
            Action::DistillCold => "DISTILL+COLD",
            Action::Cold => "COLD",
            Action::Hold => "HOLD",
            Action::Expire => "EXPIRE",
        }
    }

    /// Performs the tier operation and returns an error if it cannot be completed.
    pub fn tier(self) -> &'static str {
        match self {
            Action::KeepT0 => "T0",
            Action::KeepHot => "T1",
            Action::Compact => "T2",
            Action::DistillCold | Action::Cold => "T3",
            Action::Hold => "HOLD",
            Action::Expire => "T4",
        }
    }
}

/// Rounds a dollar or GB figure so plan bytes do not depend on float noise.
pub fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}
