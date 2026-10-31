//! G23 / J17: is a sealed-family roster proposal ready for adoption?
//!
//! The input is the `jope.rlfw-sealed-roster-proposal/v1` document that
//! `deploy/dl-flywheel/pipelines/rlfw/judge/roster.py` writes. [`check`] reads it and sorts what
//! it finds into three lists:
//!
//! - **defects**: the proposal is unsound and must be regenerated. A family in two roles, a
//!   sealed family whose members carry another role, an identifier shared by two sealed
//!   families, a failed or miscounted disjointness proof, sealed views or contexts inside the
//!   training closure, or a proposal that claims adoption, a seal or qualification itself.
//! - **owner decisions**: parameters only the owner sets. Today that is the alpha budget (U13).
//! - **limitations**: facts that narrow what an adopted roster could support, such as an
//!   unmeasured encoder pretraining closure (§8.2) or an empty development role.
//!
//! A proposal is adoptable when it has no defects and no undecided owner parameter. Adoptable
//! is not adopted: [`super::request_seal`] still refuses.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::canon::sha256_hex;

/// The proposal schema this check reads.
pub const PROPOSAL_SCHEMA: &str = "jope.rlfw-sealed-roster-proposal/v1";
/// The report schema [`check`] writes.
pub const REPORT_SCHEMA: &str = "jope.qualify-roster-report/v1";
/// The four §8.1 boundaries, in the proposal's order.
pub const ROLES: [&str; 4] = [
    "meta_training",
    "development",
    "confirmation",
    "sealed_final",
];
/// The role whose families the seal would lock.
pub const SEALED: &str = "sealed_final";

/// How many identifiers a finding names before it only counts the rest.
const NAMED: usize = 5;

#[derive(Deserialize)]
struct Proposal {
    schema: String,
    roles: Vec<String>,
    partition: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    sealed_family_records: BTreeMap<String, FamilyRecord>,
    proof: Proof,
    alpha_budget: Value,
    encoder_pretraining_closure: Value,
    adopted: bool,
    seal_created: bool,
    seal_spent: bool,
    qualifying: bool,
    #[serde(default)]
    sources: Sources,
}

#[derive(Deserialize)]
struct FamilyRecord {
    #[serde(default)]
    context_ids: Vec<String>,
    #[serde(default)]
    dataset_hashes: Vec<String>,
    #[serde(default)]
    member_roles: Vec<String>,
    #[serde(default)]
    view_ids: Vec<String>,
}

#[derive(Deserialize)]
struct Proof {
    passed: bool,
    failures: Vec<Value>,
    families: u64,
}

#[derive(Default, Deserialize)]
struct Sources {
    closure: Option<Closure>,
}

#[derive(Deserialize)]
struct Closure {
    sealed_context_ids_in_closure: Option<u64>,
    sealed_views_in_closure: Option<u64>,
}

/// What [`check`] found in one proposal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RosterReport {
    /// Schema identifier and version for the serialized value.
    pub schema: &'static str,
    /// SHA-256 of the exact proposal bytes.
    pub proposal_sha256: String,
    /// Distinct families per declared role.
    pub families: BTreeMap<String, usize>,
    /// Collection of defects values carried by this roster report.
    pub defects: Vec<String>,
    /// Collection of owner decisions values carried by this roster report.
    pub owner_decisions: Vec<String>,
    /// Collection of limitations values carried by this roster report.
    pub limitations: Vec<String>,
    /// Always false: a check adopts nothing.
    pub adopted: bool,
    /// Always false: a check creates no seal.
    pub seal_created: bool,
}

impl RosterReport {
    /// No defects and no undecided owner parameter.
    #[must_use]
    pub fn adoptable(&self) -> bool {
        self.defects.is_empty() && self.owner_decisions.is_empty()
    }

    /// Defects, then owner decisions, each prefixed with its kind.
    #[must_use]
    pub fn reasons(&self) -> Vec<String> {
        let defects = self.defects.iter().map(|item| format!("defect: {item}"));
        let decisions = self
            .owner_decisions
            .iter()
            .map(|item| format!("owner decision: {item}"));
        defects.chain(decisions).collect()
    }
}

fn named(ids: &BTreeSet<&str>) -> String {
    let mut text = ids
        .iter()
        .take(NAMED)
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    if ids.len() > NAMED {
        text.push_str(&format!(" and {} more", ids.len() - NAMED));
    }
    text
}

/// Check one proposal. Read-only.
///
/// # Errors
///
/// The bytes are not JSON of the proposal's shape, or the schema is not [`PROPOSAL_SCHEMA`].
pub fn check(proposal: &[u8]) -> Result<RosterReport, String> {
    let parsed: Proposal = serde_json::from_slice(proposal).map_err(|e| e.to_string())?;
    if parsed.schema != PROPOSAL_SCHEMA {
        return Err(format!(
            "schema {:?} is not {PROPOSAL_SCHEMA}",
            parsed.schema
        ));
    }
    let mut defects = Vec::new();
    let mut owner_decisions = Vec::new();
    let mut limitations = Vec::new();

    if parsed.roles != ROLES {
        defects.push(format!("roles {:?} are not {ROLES:?}", parsed.roles));
    }
    let declared: BTreeSet<&str> = parsed.partition.keys().map(String::as_str).collect();
    let expected: BTreeSet<&str> = ROLES.into_iter().collect();
    for role in declared.difference(&expected) {
        defects.push(format!("partition has unknown role {role}"));
    }
    for role in expected.difference(&declared) {
        defects.push(format!("partition has no {role} list"));
    }

    // Each family belongs to exactly one role, once.
    let mut role_of: BTreeMap<&str, &str> = BTreeMap::new();
    let mut families = BTreeMap::new();
    let mut repeated = BTreeSet::new();
    let mut split: BTreeMap<(&str, &str), BTreeSet<&str>> = BTreeMap::new();
    for (role, ids) in &parsed.partition {
        let mut distinct = BTreeSet::new();
        for id in ids {
            if !distinct.insert(id.as_str()) {
                repeated.insert(id.as_str());
                continue;
            }
            match role_of.get(id.as_str()) {
                Some(first) => {
                    split.entry((first, role)).or_default().insert(id);
                }
                None => {
                    role_of.insert(id, role);
                }
            }
        }
        families.insert(role.clone(), distinct.len());
    }
    if !repeated.is_empty() {
        defects.push(format!(
            "families listed twice within a role: {}",
            named(&repeated)
        ));
    }
    for ((first, second), ids) in &split {
        defects.push(format!(
            "families in both {first} and {second}: {}",
            named(ids)
        ));
    }

    let sealed: BTreeSet<&str> = parsed
        .partition
        .get(SEALED)
        .map(|ids| ids.iter().map(String::as_str).collect())
        .unwrap_or_default();
    if sealed.is_empty() {
        defects.push(format!("{SEALED} is empty"));
    }
    let recorded: BTreeSet<&str> = parsed
        .sealed_family_records
        .keys()
        .map(String::as_str)
        .collect();
    let unrecorded: BTreeSet<&str> = sealed.difference(&recorded).copied().collect();
    if !unrecorded.is_empty() {
        defects.push(format!(
            "sealed families without a record: {}",
            named(&unrecorded)
        ));
    }
    let stray: BTreeSet<&str> = recorded.difference(&sealed).copied().collect();
    if !stray.is_empty() {
        defects.push(format!(
            "records for families outside {SEALED}: {}",
            named(&stray)
        ));
    }

    // A sealed family is one independent unit: all of its members are sealed, it has an
    // identifier to prove disjointness with, and it shares no identifier with another family.
    let mut role_split = BTreeSet::new();
    let mut unidentified = BTreeSet::new();
    let mut owner_of: BTreeMap<(&str, &str), &str> = BTreeMap::new();
    let mut shared: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (family, record) in &parsed.sealed_family_records {
        if record.member_roles != [SEALED] {
            role_split.insert(family.as_str());
        }
        let kinds = [
            ("context_ids", &record.context_ids),
            ("dataset_hashes", &record.dataset_hashes),
            ("view_ids", &record.view_ids),
        ];
        if kinds.iter().all(|(_, ids)| ids.is_empty()) {
            unidentified.insert(family.as_str());
        }
        for (kind, ids) in kinds {
            for id in ids {
                match owner_of.get(&(kind, id.as_str())) {
                    Some(other) if *other != family.as_str() => {
                        shared.entry(kind).or_default().insert(id.as_str());
                    }
                    Some(_) => {}
                    None => {
                        owner_of.insert((kind, id), family);
                    }
                }
            }
        }
    }
    if !role_split.is_empty() {
        defects.push(format!(
            "sealed families with members outside {SEALED}: {}",
            named(&role_split)
        ));
    }
    if !unidentified.is_empty() {
        defects.push(format!(
            "sealed families with no context, dataset or view id: {}",
            named(&unidentified)
        ));
    }
    for (kind, ids) in &shared {
        defects.push(format!(
            "{kind} shared by two sealed families: {}",
            named(ids)
        ));
    }

    let proof = &parsed.proof;
    if !proof.passed || !proof.failures.is_empty() {
        defects.push(format!(
            "disjointness proof did not pass ({} failures)",
            proof.failures.len()
        ));
    }
    if usize::try_from(proof.families).ok() != Some(sealed.len()) {
        defects.push(format!(
            "proof covers {} families, {SEALED} lists {}",
            proof.families,
            sealed.len()
        ));
    }
    match &parsed.sources.closure {
        None => defects.push("sources.closure is absent".to_owned()),
        Some(closure) => {
            for (name, count) in [
                (
                    "sealed_context_ids_in_closure",
                    closure.sealed_context_ids_in_closure,
                ),
                ("sealed_views_in_closure", closure.sealed_views_in_closure),
            ] {
                match count {
                    Some(0) => {}
                    Some(n) => defects.push(format!("{name} is {n}, not 0")),
                    None => defects.push(format!("{name} is absent")),
                }
            }
        }
    }
    for (name, claimed) in [
        ("adopted", parsed.adopted),
        ("seal_created", parsed.seal_created),
        ("seal_spent", parsed.seal_spent),
        ("qualifying", parsed.qualifying),
    ] {
        if claimed {
            defects.push(format!("a proposal cannot claim {name}"));
        }
    }

    match parsed.alpha_budget.as_f64() {
        Some(alpha) if alpha > 0.0 && alpha < 1.0 => {}
        _ => owner_decisions.push(format!(
            "alpha budget (U13) is {}, not a decided level in (0, 1)",
            parsed.alpha_budget
        )),
    }
    if parsed.encoder_pretraining_closure.as_str() == Some("unmeasured") {
        limitations.push(
            "encoder pretraining closure is unmeasured: no claim that a sealed family was unseen \
             by the encoder (§8.2)"
                .to_owned(),
        );
    }
    if families.get("development").copied().unwrap_or(0) == 0 {
        limitations.push("no development families: design choices tune on other roles".to_owned());
    }

    Ok(RosterReport {
        schema: REPORT_SCHEMA,
        proposal_sha256: sha256_hex(proposal),
        families,
        defects,
        owner_decisions,
        limitations,
        adopted: false,
        seal_created: false,
    })
}
