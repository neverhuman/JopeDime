//! Classification of one object group: T0 checks, the training-value test TV(g) (§8.1), E-rule
//! status (§8.2), the action, list-price costs and the reason string (§8.7).
//!
//! Order (first match wins): hard or referenced T0 → out of scope → referenced by key from an
//! active roster/seed → live writer → the kind's default. A group whose kind is unknown gets
//! HOLD (§7.3 rule 6). EXPIRE is gated by [`expire_allowed`], which nothing here can satisfy at
//! prefix granularity.

use super::groups::Group;
use super::index::GroupIndex;
use super::inventory::{E1Overlap, GroupInv};
use super::lifecycle::Coverage;
use super::policy::{Kind, Policy};
use super::refs::Role;
use super::{round4, Action};
use anyhow::Result;
use serde::Serialize;
use std::collections::BTreeMap;

/// Hard T0 rules. They are code, not policy, so no policy file can remove them.
pub fn hard_t0(bucket: &str, prefix: &str) -> Option<String> {
    if bucket.contains("sealed") {
        return Some("hard T0: sealed bucket (NEVER delete)".into());
    }
    if bucket == "veox-jopedime-models-use2" {
        return Some("hard T0: model bundles (frozen v1 / promoted checkpoints)".into());
    }
    for seg in prefix.split('/').filter(|s| !s.is_empty()) {
        let s = seg.to_ascii_lowercase();
        let why = if s.contains("sealed") {
            "sealed"
        } else if s.contains("evidence") {
            "evidence"
        } else if s.starts_with("split") {
            "split / split-registry"
        } else if s.contains("prereg") {
            "prereg"
        } else if s.contains("roster") {
            "roster"
        } else if matches!(
            s.as_str(),
            "schema-registry" | "catalog" | "catalogs" | "full-source-registry"
        ) {
            "registry / catalog"
        } else if matches!(
            s.as_str(),
            "tombstones" | "gp_distilled" | "retention-plans" | "expiry-manifests"
        ) {
            "retention records (plans, tombstones, distilled tables)"
        } else {
            continue;
        };
        return Some(format!("hard T0: {why} prefix segment {seg:?}"));
    }
    None
}

/// EXPIRE needs all three (§8.4): no training value, a proven E-rule, and an agreeing
/// independent reference check.
pub fn expire_allowed(tv: &Tv, e_rule_proven: bool, independent_check_agrees: bool) -> bool {
    !tv.value && e_rule_proven && independent_check_agrees
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
/// `Tv` is a record for the tv contract in this module: Classification of one object group: T0 checks, the training-value test TV(g) (§8.1), E-rule status (§8.2), the action, list-price costs and the reason string (§8.7).
pub struct Tv {
    /// Measured or derived value carried by this result.
    pub value: bool,
    /// Collection of uses values carried by this tv.
    pub uses: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
/// `Cost` is a record for the cost contract in this module: Classification of one object group: T0 checks, the training-value test TV(g) (§8.1), E-rule status (§8.2), the action, list-price costs and the reason string (§8.7).
pub struct Cost {
    /// Class now recorded for this cost.
    pub class_now: String,
    /// Class after recorded for this cost.
    pub class_after: String,
    /// Now usd mo recorded for this cost.
    pub now_usd_mo: f64,
    /// After usd mo recorded for this cost.
    pub after_usd_mo: f64,
    /// Delta usd mo recorded for this cost.
    pub delta_usd_mo: f64,
    /// One time in region usd recorded for this cost.
    pub one_time_in_region_usd: f64,
    /// The same work run from the xbabe hosts: adds internet egress for every byte read.
    pub one_time_off_aws_usd: f64,
    /// The `after` figure needs the loose originals expired under an approved E1 once the
    /// pack copy is verified; until then storage is `now` plus the pack.
    pub after_needs_e1_of_originals: bool,
    /// An applied lifecycle rule already moves this group to `class_after`.
    pub already_applied_by_lifecycle: bool,
}

/// Everything known about one group before deciding.
pub struct Facts<'a> {
    /// Group recorded for this facts.
    pub group: &'a Group,
    /// Storage class recorded for this facts.
    pub storage_class: String,
    /// Mapping of refs by role values carried by this facts.
    pub refs_by_role: BTreeMap<Role, u64>,
    /// Mapping of ctx roles values carried by this facts.
    pub ctx_roles: BTreeMap<Role, u64>,
    /// Mapping of prog roles values carried by this facts.
    pub prog_roles: BTreeMap<Role, u64>,
    /// Optional index value; `None` means it is not available.
    pub index: Option<&'a GroupIndex>,
    /// Number of family scored rows represented here.
    pub family_scored_rows: Option<u64>,
    /// Coverage recorded for this facts.
    pub coverage: Coverage,
    /// As of unix recorded for this facts.
    pub as_of_unix: u64,
    /// S3 Inventory facts for this group, when its bucket has a report.
    pub inv: Option<&'a GroupInv>,
    /// The inventory's current objects and bytes equal the ledger's, so its exact dates and
    /// classes describe this group (the report is older than the ledger).
    pub inv_fresh: bool,
    /// The inventory and the ledger disagree although the ledger shows no write after the report
    /// (`GroupInv::conflicts_with_ledger`): one input is wrong for this group.
    pub inv_conflict: bool,
    /// Key/size/ETag overlap of this group's E1 candidate pair, when both sides are inventoried.
    pub e1: Option<&'a E1Overlap>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
/// `Decision` is a record for the decision contract in this module: Classification of one object group: T0 checks, the training-value test TV(g) (§8.1), E-rule status (§8.2), the action, list-price costs and the reason string (§8.7).
pub struct Decision {
    /// Action recorded for this decision.
    pub action: Action,
    /// Record kind or category represented by this value.
    pub kind: Option<&'static str>,
    /// Optional family value; `None` means it is not available.
    pub family: Option<String>,
    /// Rule recorded for this decision.
    pub rule: String,
    /// Optional t0 value; `None` means it is not available.
    pub t0: Option<String>,
    /// Live recorded for this decision.
    pub live: bool,
    /// Tv recorded for this decision.
    pub tv: Tv,
    /// Collection of e candidates values carried by this decision.
    pub e_candidates: Vec<String>,
    /// Cost recorded for this decision.
    pub cost: Cost,
    /// Explanation associated with the recorded disposition.
    pub reason: String,
}

mod costing;
mod decision;

pub(super) use costing::{
    cost, e_candidates, exact_max, human_bytes, is_live, roles_str, storage_now, tv_for,
};
pub use decision::decide;
