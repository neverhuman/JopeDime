//! Retention policy: thresholds, list prices and the prefix-to-kind map.
//!
//! The policy is data (`policy.v1.json`, embedded as the default) and its sha256 is recorded in
//! every plan. It can classify prefixes and add soft T0 prefixes; it cannot remove a hard T0 rule,
//! which lives in code (`classify::hard_t0`).

use super::refs::{Role, SCAN_EXTENSIONS};
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

/// Schema identifier for policy values.
pub const POLICY_SCHEMA: &str = "jopedime.retention-policy/v1";
/// Constant identifying the default policy JSON.
pub const DEFAULT_POLICY_JSON: &str = include_str!("policy.v1.json");

/// What kind of bytes a prefix holds. The action for each kind is decided in code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    /// The `Encoding` case of `Kind`.
    Encoding,
    /// The `DatasetPayload` case of `Kind`.
    DatasetPayload,
    /// The `GpResult` case of `Kind`.
    GpResult,
    /// The `GpResultUnconsumed` case of `Kind`.
    GpResultUnconsumed,
    /// The `Telemetry` case of `Kind`.
    Telemetry,
    /// The `Receipts` case of `Kind`.
    Receipts,
    /// The `Archive` case of `Kind`.
    Archive,
    /// The `Checkpoint` case of `Kind`.
    Checkpoint,
    /// The `Report` case of `Kind`.
    Report,
    /// The `Replica` case of `Kind`.
    Replica,
    /// The `Custody` case of `Kind`.
    Custody,
    /// The `Review` case of `Kind`.
    Review,
}

impl Kind {
    /// Returns the stable string representation of this value.
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Encoding => "encoding",
            Kind::DatasetPayload => "dataset-payload",
            Kind::GpResult => "gp-result",
            Kind::GpResultUnconsumed => "gp-result-unconsumed",
            Kind::Telemetry => "telemetry",
            Kind::Receipts => "receipts",
            Kind::Archive => "archive",
            Kind::Checkpoint => "checkpoint",
            Kind::Report => "report",
            Kind::Replica => "replica",
            Kind::Custody => "custody",
            Kind::Review => "review",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Prices` is a record for the prices contract in this module: Retention policy: thresholds, list prices and the prefix-to-kind map.
pub struct Prices {
    /// Label recorded for this prices.
    pub label: String,
    /// Mapping of storage gb month values carried by this prices.
    pub storage_gb_month: BTreeMap<String, f64>,
    /// Put per 1k recorded for this prices.
    pub put_per_1k: f64,
    /// Get per 1k recorded for this prices.
    pub get_per_1k: f64,
    /// Transition deep archive per 1k recorded for this prices.
    pub transition_deep_archive_per_1k: f64,
    /// Egress per gb recorded for this prices.
    pub egress_per_gb: f64,
    /// Byte count for the associated archive overhead standard.
    pub archive_overhead_standard_bytes: u64,
    /// Byte count for the associated archive overhead archive.
    pub archive_overhead_archive_bytes: u64,
}

impl Prices {
    /// $/month for `bytes` in `class` (decimal GB, list price).
    pub fn storage(&self, bytes: u64, class: &str) -> Result<f64> {
        let p = self
            .storage_gb_month
            .get(class)
            .with_context(|| format!("no storage price for class {class}"))?;
        Ok(bytes as f64 / 1e9 * p)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PrefixRule` is a record for the prefix rule contract in this module: Retention policy: thresholds, list prices and the prefix-to-kind map.
pub struct PrefixRule {
    /// Bucket recorded for this prefix rule.
    pub bucket: String,
    /// Prefix recorded for this prefix rule.
    pub prefix: String,
    /// Record kind or category represented by this value.
    pub kind: Kind,
    /// Optional family value; `None` means it is not available.
    pub family: Option<String>,
    /// Why recorded for this prefix rule.
    pub why: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PrefixNote` is a record for the prefix note contract in this module: Retention policy: thresholds, list prices and the prefix-to-kind map.
pub struct PrefixNote {
    /// Bucket recorded for this prefix note.
    pub bucket: String,
    /// Prefix recorded for this prefix note.
    pub prefix: String,
    /// Why recorded for this prefix note.
    pub why: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
/// `ClassOverride` is a record for the class override contract in this module: Retention policy: thresholds, list prices and the prefix-to-kind map.
pub struct ClassOverride {
    /// Bucket recorded for this class override.
    pub bucket: String,
    /// Prefix recorded for this class override.
    pub prefix: String,
    /// Class recorded for this class override.
    pub class: String,
    /// Why recorded for this class override.
    pub why: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
/// `E1Candidate` is a record for the e1 candidate contract in this module: Retention policy: thresholds, list prices and the prefix-to-kind map.
pub struct E1Candidate {
    /// Bucket recorded for this e1 candidate.
    pub bucket: String,
    /// Prefix recorded for this e1 candidate.
    pub prefix: String,
    /// Counterpart recorded for this e1 candidate.
    pub counterpart: String,
    /// Why recorded for this e1 candidate.
    pub why: String,
}

/// Which scanned refs pin (owner decision D1). Refs found by a scan of one of `prose_roles` in a
/// file whose extension is in `prose_extensions` are training-value mentions, not T0 pins.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceFormats {
    /// Collection of prose roles values carried by this reference formats.
    pub prose_roles: Vec<String>,
    /// Collection of prose extensions values carried by this reference formats.
    pub prose_extensions: Vec<String>,
    /// Why recorded for this reference formats.
    pub why: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Policy` is a record for the policy contract in this module: Retention policy: thresholds, list prices and the prefix-to-kind map.
pub struct Policy {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Note recorded for this policy.
    pub note: String,
    /// Frozen v1 l6 recorded for this policy.
    pub frozen_v1_l6: f64,
    /// L6 direction recorded for this policy.
    pub l6_direction: String,
    /// Open run window hours recorded for this policy.
    pub open_run_window_hours: u64,
    /// Reservoir rate recorded for this policy.
    pub reservoir_rate: f64,
    /// Number of diversity min scored rows represented here.
    pub diversity_min_scored_rows: u64,
    /// K ctx recorded for this policy.
    pub k_ctx: u64,
    /// K lineage recorded for this policy.
    pub k_lineage: u64,
    /// Margin rel recorded for this policy.
    pub margin_rel: f64,
    /// Grace days recorded for this policy.
    pub grace_days: u64,
    /// Noncurrent days recorded for this policy.
    pub noncurrent_days: u64,
    /// Byte count for the associated pack.
    pub pack_bytes: u64,
    /// Byte count for the associated compact max mean object.
    pub compact_max_mean_object_bytes: u64,
    /// Byte count for the associated cold min mean object.
    pub cold_min_mean_object_bytes: u64,
    /// Byte count for the associated distilled row.
    pub distilled_row_bytes: u64,
    /// Prices recorded for this policy.
    pub prices: Prices,
    /// Mapping of known score identities values carried by this policy.
    pub known_score_identities: BTreeMap<String, String>,
    /// Collection of in scope bucket prefixes values carried by this policy.
    pub in_scope_bucket_prefixes: Vec<String>,
    /// Reference formats recorded for this policy.
    pub reference_formats: ReferenceFormats,
    /// Collection of soft t0 values carried by this policy.
    pub soft_t0: Vec<PrefixNote>,
    /// Collection of class overrides values carried by this policy.
    pub class_overrides: Vec<ClassOverride>,
    /// Collection of e1 candidates values carried by this policy.
    pub e1_candidates: Vec<E1Candidate>,
    /// Collection of rules values carried by this policy.
    pub rules: Vec<PrefixRule>,
}

/// `pattern` is an exact bucket name, or a name prefix ending in `*`.
pub fn bucket_matches(pattern: &str, bucket: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(stem) => bucket.starts_with(stem),
        None => pattern == bucket,
    }
}

impl Policy {
    /// Parses and validates a policy. Any inconsistency is an error: the brain fails closed.
    pub fn parse(bytes: &[u8]) -> Result<Policy> {
        let p: Policy = serde_json::from_slice(bytes).context("retention policy does not parse")?;
        ensure!(
            p.schema == POLICY_SCHEMA,
            "policy schema {} is not {POLICY_SCHEMA}",
            p.schema
        );
        ensure!(
            p.l6_direction == "lower_is_better",
            "policy l6_direction must be lower_is_better (L6 is a validation loss)"
        );
        ensure!(
            p.frozen_v1_l6 > 0.0 && p.frozen_v1_l6.is_finite(),
            "frozen_v1_l6 must be positive"
        );
        ensure!(
            p.reservoir_rate > 0.0 && p.reservoir_rate <= 1.0,
            "reservoir_rate out of (0, 1]"
        );
        ensure!(p.margin_rel >= 0.0, "margin_rel must be >= 0");
        ensure!(
            p.noncurrent_days >= 30,
            "noncurrent_days below the 30-day floor (§8.6)"
        );
        ensure!(
            p.pack_bytes >= 64 * 1024 * 1024,
            "pack_bytes below 64 MiB (§4)"
        );
        for class in ["STANDARD", "DEEP_ARCHIVE", "GLACIER_IR"] {
            ensure!(
                p.prices.storage_gb_month.contains_key(class),
                "policy lacks a {class} price"
            );
        }
        ensure!(
            p.known_score_identities.contains_key("l6"),
            "policy lacks the l6 score identity"
        );
        let rf = &p.reference_formats;
        for r in &rf.prose_roles {
            let role = Role::parse(r)?;
            ensure!(
                role.is_scan_pin(),
                "reference_formats.prose_roles: {r} refs can never be demoted to prose mentions"
            );
        }
        for e in &rf.prose_extensions {
            ensure!(
                SCAN_EXTENSIONS.contains(&e.as_str()) && *e == e.to_ascii_lowercase(),
                "reference_formats.prose_extensions: {e:?} is not a scanned text extension"
            );
            ensure!(
                !matches!(e.as_str(), "json" | "jsonl"),
                "reference_formats.prose_extensions: structured {e} refs are T0 pins (D1)"
            );
        }
        let mut seen = BTreeSet::new();
        for r in &p.rules {
            if !seen.insert((r.bucket.clone(), r.prefix.clone())) {
                bail!("duplicate policy rule s3://{}/{}", r.bucket, r.prefix);
            }
            ensure!(
                r.prefix.is_empty() || r.prefix.ends_with('/'),
                "rule prefix {} must end with /",
                r.prefix
            );
        }
        for o in &p.class_overrides {
            ensure!(
                p.prices.storage_gb_month.contains_key(&o.class),
                "class override {} has no price",
                o.class
            );
        }
        Ok(p)
    }

    /// The prose extensions a scan of `role` demotes to mentions (empty: every ref pins).
    pub fn prose_extensions_for(&self, role: Role) -> &[String] {
        let rf = &self.reference_formats;
        if rf.prose_roles.iter().any(|r| r == role.as_str()) {
            &rf.prose_extensions
        } else {
            &[]
        }
    }

    /// Performs the in scope operation and returns an error if it cannot be completed.
    pub fn in_scope(&self, bucket: &str) -> bool {
        self.in_scope_bucket_prefixes
            .iter()
            .any(|p| bucket.starts_with(p.as_str()))
    }

    /// The most specific rule whose bucket matches and whose prefix is a prefix of `prefix`.
    pub fn rule_for(&self, bucket: &str, prefix: &str) -> Option<&PrefixRule> {
        self.rules
            .iter()
            .filter(|r| bucket_matches(&r.bucket, bucket) && prefix.starts_with(r.prefix.as_str()))
            .max_by_key(|r| r.prefix.len())
    }

    /// Performs the soft t0 for operation and returns an error if it cannot be completed.
    pub fn soft_t0_for(&self, bucket: &str, prefix: &str) -> Option<&PrefixNote> {
        self.soft_t0
            .iter()
            .filter(|n| bucket_matches(&n.bucket, bucket) && prefix.starts_with(n.prefix.as_str()))
            .max_by_key(|n| n.prefix.len())
    }

    /// Performs the class override for operation and returns an error if it cannot be completed.
    pub fn class_override_for(&self, bucket: &str, prefix: &str) -> Option<&ClassOverride> {
        self.class_overrides
            .iter()
            .filter(|o| bucket_matches(&o.bucket, bucket) && prefix.starts_with(o.prefix.as_str()))
            .max_by_key(|o| o.prefix.len())
    }

    /// Performs the e1 candidate for operation and returns an error if it cannot be completed.
    pub fn e1_candidate_for(&self, bucket: &str, prefix: &str) -> Option<&E1Candidate> {
        self.e1_candidates
            .iter()
            .filter(|c| bucket_matches(&c.bucket, bucket) && prefix.starts_with(c.prefix.as_str()))
            .max_by_key(|c| c.prefix.len())
    }
}
