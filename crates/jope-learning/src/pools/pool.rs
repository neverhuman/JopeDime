//! Frozen candidate pools (CLOVER §4.1).
//!
//! A pool is built from measurement rows by a rule that never consults a selector: group by
//! context, keep one row per distinct program, refuse conflicting duplicates and unknown
//! outcomes, then (optionally) cap the pool by a salted sha256 order that no selector can
//! influence. The frozen pool and the manifest over all pools are canonical JSON with sha256
//! identities, written before any selector is evaluated.

use crate::canon::{canonical_sha256, is_hex64};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Schema identifier for pool values.
pub const POOL_SCHEMA: &str = "jope.j9-pool/v1";
/// Schema identifier for manifest values.
pub const MANIFEST_SCHEMA: &str = "jope.j9-pool-manifest/v1";

/// Roles whose outcomes may be read by this shadow evaluation. Validation, development, sealed
/// and test labels never enter a pool (they are refused, not skipped).
pub const ALLOWED_ROLES: &[&str] = &["train", "train_fit"];

/// Execution outcome of one measured attempt. `Unknown` rows are refused at freeze time: CLOVER
/// §5.1 forbids fabricating `q` for them, so their context is reported as excluded instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// The `Complete` case of `Status`.
    Complete,
    /// The `ProgramFailed` case of `Status`.
    ProgramFailed,
    /// The `Timeout` case of `Status`.
    Timeout,
    /// The `Unknown` case of `Status`.
    Unknown,
}

/// One input measurement, as written by a source adapter.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementRow {
    /// Identifier for the associated context.
    pub context_id: String,
    /// Parent / lineage cluster: cuts sharing a parent are one bootstrap unit.
    pub parent_id: String,
    /// Data or authority role assigned to this value.
    pub role: String,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Outcome or lifecycle status recorded for this value.
    pub status: Status,
    /// Measured reward, higher is better; required exactly when `status` is complete.
    pub q: Option<f64>,
    /// The frozen comparable incumbent `h_c` on the same protocol and scale, when one exists.
    pub incumbent_q: Option<f64>,
    /// Evaluator/protocol identity; one pool never mixes two.
    pub protocol_sha256: String,
    /// Program token ids: a pre-measurement descriptor selectors may read.
    pub tokens: Option<Vec<u32>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Member` is a record for the member contract in this module: Frozen candidate pools (CLOVER §4.1).
pub struct Member {
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Outcome or lifecycle status recorded for this value.
    pub status: Status,
    /// Optional q value; `None` means it is not available.
    pub q: Option<f64>,
    /// Optional tokens value; `None` means it is not available.
    pub tokens: Option<Vec<u32>>,
}

impl Member {
    /// The scored outcome: `Some(q)` when complete, `None` for a failure that used its slot.
    pub fn outcome(&self) -> Option<f64> {
        match self.status {
            Status::Complete => self.q,
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Pool` is a record for the pool contract in this module: Frozen candidate pools (CLOVER §4.1).
pub struct Pool {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated context.
    pub context_id: String,
    /// Identifier for the associated parent.
    pub parent_id: String,
    /// Data or authority role assigned to this value.
    pub role: String,
    /// SHA-256 digest identifying the associated protocol.
    pub protocol_sha256: String,
    /// Optional incumbent q value; `None` means it is not available.
    pub incumbent_q: Option<f64>,
    /// Sorted by program sha256; distinct.
    pub members: Vec<Member>,
}

/// The value `h_c` that gains are measured from (CLOVER §5.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Reference {
    /// The pool's frozen comparable incumbent; a pool without one is an error.
    Incumbent,
    /// No-incumbent protocol: the lowest measured `q` in the pool. Use it when the recorded
    /// incumbent is not independent of the pool (for example, the HWM selected from these same
    /// measurements). Paired selector differences then equal differences in best `q` at `B`.
    PoolFloor,
}

impl Pool {
    /// Computes the canonical SHA-256 digest of this value.
    pub fn sha256(&self) -> Result<String> {
        canonical_sha256(self)
    }

    /// What a selector may see: the context and the candidate identities, never outcomes.
    pub fn view(&self) -> PoolView<'_> {
        PoolView {
            context_id: &self.context_id,
            parent_id: &self.parent_id,
            incumbent_q: self.incumbent_q,
            programs: self
                .members
                .iter()
                .map(|m| m.program_sha256.as_str())
                .collect(),
            tokens: self.members.iter().map(|m| m.tokens.as_deref()).collect(),
        }
    }

    /// Performs the outcomes operation and returns an error if it cannot be completed.
    pub fn outcomes(&self) -> Vec<Option<f64>> {
        self.members.iter().map(Member::outcome).collect()
    }

    /// Performs the reference operation and returns an error if it cannot be completed.
    pub fn reference(&self, mode: Reference) -> Result<f64> {
        match mode {
            Reference::Incumbent => self
                .incumbent_q
                .with_context(|| format!("pool {} has no incumbent", self.context_id)),
            Reference::PoolFloor => self
                .outcomes()
                .into_iter()
                .flatten()
                .min_by(f64::total_cmp)
                .with_context(|| format!("pool {} has no measured member", self.context_id)),
        }
    }
}

/// The outcome-free projection of a pool handed to selectors.
#[derive(Clone, Debug)]
pub struct PoolView<'a> {
    /// Identifier for the associated context.
    pub context_id: &'a str,
    /// Identifier for the associated parent.
    pub parent_id: &'a str,
    /// Optional incumbent q value; `None` means it is not available.
    pub incumbent_q: Option<f64>,
    /// Collection of programs values carried by this pool view.
    pub programs: Vec<&'a str>,
    /// Collection of tokens values carried by this pool view.
    pub tokens: Vec<Option<&'a [u32]>>,
}

/// Schema identifier for view values.
pub const VIEW_SCHEMA: &str = "jope.j9-pool-view/v1";

/// The serialized outcome-free view handed to external scorers (the torch q-head adapter): it
/// carries identities and tokens only, so an external selector cannot read a measured `q`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportedView {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated context.
    pub context_id: String,
    /// Identifier for the associated parent.
    pub parent_id: String,
    /// SHA-256 digest identifying the associated pool.
    pub pool_sha256: String,
    /// Collection of programs values carried by this exported view.
    pub programs: Vec<ExportedProgram>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `ExportedProgram` is a record for the exported program contract in this module: Frozen candidate pools (CLOVER §4.1).
pub struct ExportedProgram {
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Optional tokens value; `None` means it is not available.
    pub tokens: Option<Vec<u32>>,
}

/// Outcome-free views of verified pools, in manifest order.
pub fn export_views(pools: &[Pool], manifest: &Manifest) -> Result<Vec<ExportedView>> {
    verify(pools, manifest)?;
    Ok(pools
        .iter()
        .zip(&manifest.pools)
        .map(|(p, e)| ExportedView {
            schema: VIEW_SCHEMA.into(),
            context_id: p.context_id.clone(),
            parent_id: p.parent_id.clone(),
            pool_sha256: e.pool_sha256.clone(),
            programs: p
                .members
                .iter()
                .map(|m| ExportedProgram {
                    program_sha256: m.program_sha256.clone(),
                    tokens: m.tokens.clone(),
                })
                .collect(),
        })
        .collect())
}

/// Construction rule, recorded in the manifest. It names no selector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FreezeRule {
    /// Contexts with fewer distinct members are excluded (`min_pool`).
    pub min_pool: usize,
    /// When set, keep the first `max_pool` members by `sha256(salt ‖ context ‖ program)`.
    pub max_pool: Option<usize>,
    /// Salt recorded for this freeze rule.
    pub salt: String,
    /// Contexts need at least this many complete members with distinct `q`.
    pub min_distinct_measured: usize,
    /// What the measurements are, e.g. `native_nonqualifying_R22`; never upgraded by this crate.
    pub source_grade: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Exclusion` is a record for the exclusion contract in this module: Frozen candidate pools (CLOVER §4.1).
pub struct Exclusion {
    /// Identifier for the associated context.
    pub context_id: String,
    /// Explanation associated with the recorded disposition.
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `ManifestEntry` is a record for the manifest entry contract in this module: Frozen candidate pools (CLOVER §4.1).
pub struct ManifestEntry {
    /// Identifier for the associated context.
    pub context_id: String,
    /// Identifier for the associated parent.
    pub parent_id: String,
    /// SHA-256 digest identifying the associated pool.
    pub pool_sha256: String,
    /// Members recorded for this manifest entry.
    pub members: usize,
    /// Measured recorded for this manifest entry.
    pub measured: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Manifest` is a record for the manifest contract in this module: Frozen candidate pools (CLOVER §4.1).
pub struct Manifest {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Shadow only recorded for this manifest.
    pub shadow_only: bool,
    /// Qualified claim recorded for this manifest.
    pub qualified_claim: bool,
    /// Rule recorded for this manifest.
    pub rule: FreezeRule,
    /// sha256 of the exact input bytes the pools were built from.
    pub input_sha256: String,
    /// Number of input rows represented here.
    pub input_rows: usize,
    /// Collection of pools values carried by this manifest.
    pub pools: Vec<ManifestEntry>,
    /// Collection of excluded values carried by this manifest.
    pub excluded: Vec<Exclusion>,
}

impl Manifest {
    /// Computes the canonical SHA-256 digest of this value.
    pub fn sha256(&self) -> Result<String> {
        canonical_sha256(self)
    }
}

fn hex64(label: &str, value: &str) -> Result<()> {
    ensure!(
        is_hex64(value),
        "{label} must be a lowercase sha256, got {value:?}"
    );
    Ok(())
}

/// Context-level invariants shared by input rows and frozen pools.
fn check_header(
    context_id: &str,
    parent_id: &str,
    role: &str,
    protocol_sha256: &str,
    incumbent_q: Option<f64>,
) -> Result<()> {
    hex64("context_id", context_id)?;
    hex64("protocol_sha256", protocol_sha256)?;
    ensure!(!parent_id.trim().is_empty(), "parent_id is required");
    ensure!(
        ALLOWED_ROLES.contains(&role),
        "role {role:?} may not enter a shadow pool (allowed: {ALLOWED_ROLES:?})"
    );
    ensure!(
        incumbent_q.is_none_or(f64::is_finite),
        "incumbent_q must be finite"
    );
    Ok(())
}

/// Member-level invariants shared by input rows and frozen pools.
fn check_member(
    program_sha256: &str,
    status: Status,
    q: Option<f64>,
    tokens: Option<&[u32]>,
) -> Result<()> {
    hex64("program_sha256", program_sha256)?;
    if let Some(tokens) = tokens {
        ensure!(!tokens.is_empty(), "empty token row");
    }
    match (status, q) {
        (Status::Complete, Some(q)) => ensure!(q.is_finite(), "measured q must be finite"),
        (Status::Complete, None) => bail!("complete row without q"),
        (_, Some(_)) => bail!("q present on a {status:?} row"),
        (_, None) => {}
    }
    Ok(())
}

fn check_row(row: &MeasurementRow) -> Result<()> {
    check_header(
        &row.context_id,
        &row.parent_id,
        &row.role,
        &row.protocol_sha256,
        row.incumbent_q,
    )?;
    check_member(
        &row.program_sha256,
        row.status,
        row.q,
        row.tokens.as_deref(),
    )
}

fn distinct_measured(members: &[Member]) -> usize {
    members
        .iter()
        .filter_map(Member::outcome)
        .map(f64::to_bits)
        .collect::<BTreeSet<u64>>()
        .len()
}

fn check_rule(rule: &FreezeRule) -> Result<()> {
    ensure!(
        rule.min_pool >= 2,
        "a pool needs at least two members to rank"
    );
    if let Some(max) = rule.max_pool {
        ensure!(max >= rule.min_pool, "max_pool below min_pool");
    }
    Ok(())
}

/// Every invariant `freeze` establishes, re-checked on a loaded pool: hashes alone do not make a
/// caller-supplied artifact train-only, outcome-consistent or within the recorded rule.
fn check_pool(pool: &Pool, rule: &FreezeRule) -> Result<()> {
    ensure!(pool.schema == POOL_SCHEMA, "unknown pool schema");
    check_header(
        &pool.context_id,
        &pool.parent_id,
        &pool.role,
        &pool.protocol_sha256,
        pool.incumbent_q,
    )?;
    for m in &pool.members {
        check_member(&m.program_sha256, m.status, m.q, m.tokens.as_deref())?;
        ensure!(
            m.status != Status::Unknown,
            "unknown outcome inside a frozen pool"
        );
    }
    ensure!(
        pool.members
            .windows(2)
            .all(|w| w[0].program_sha256 < w[1].program_sha256),
        "members are not distinct and sorted"
    );
    ensure!(
        pool.members.len() >= rule.min_pool,
        "{} members < min_pool {}",
        pool.members.len(),
        rule.min_pool
    );
    if let Some(max) = rule.max_pool {
        ensure!(pool.members.len() <= max, "members above max_pool {max}");
    }
    ensure!(
        distinct_measured(&pool.members) >= rule.min_distinct_measured,
        "fewer distinct measured q than the rule requires"
    );
    Ok(())
}

fn cap_key(salt: &str, context: &str, program: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    for part in [salt, context, program] {
        h.update((part.len() as u64).to_le_bytes());
        h.update(part.as_bytes());
    }
    h.finalize().into()
}

/// Build the frozen pools. `input_sha256` is the digest of the bytes `rows` were parsed from.
pub fn freeze(
    rows: &[MeasurementRow],
    rule: &FreezeRule,
    input_sha256: &str,
) -> Result<(Vec<Pool>, Manifest)> {
    hex64("input_sha256", input_sha256)?;
    check_rule(rule)?;
    let mut by_context: BTreeMap<&str, Vec<&MeasurementRow>> = BTreeMap::new();
    for (i, row) in rows.iter().enumerate() {
        check_row(row).with_context(|| format!("input row {i}"))?;
        by_context.entry(&row.context_id).or_default().push(row);
    }
    let (mut pools, mut entries, mut excluded) = (Vec::new(), Vec::new(), Vec::new());
    for (context, group) in by_context {
        let first = group[0];
        let mut exclude = |reason: String| {
            excluded.push(Exclusion {
                context_id: context.to_owned(),
                reason,
            });
        };
        let header_conflict = group.iter().find(|r| {
            r.parent_id != first.parent_id
                || r.role != first.role
                || r.protocol_sha256 != first.protocol_sha256
                || r.incumbent_q.map(f64::to_bits) != first.incumbent_q.map(f64::to_bits)
        });
        if header_conflict.is_some() {
            exclude("rows disagree on parent, role, protocol or incumbent".into());
            continue;
        }
        if group.iter().any(|r| r.status == Status::Unknown) {
            exclude("unknown outcome present; no imputation".into());
            continue;
        }
        let mut members: BTreeMap<&str, Member> = BTreeMap::new();
        let mut conflict = false;
        for r in &group {
            let m = Member {
                program_sha256: r.program_sha256.clone(),
                status: r.status,
                q: r.q,
                tokens: r.tokens.clone(),
            };
            match members.get(r.program_sha256.as_str()) {
                Some(prev) if prev != &m => conflict = true,
                Some(_) => {}
                None => {
                    members.insert(&r.program_sha256, m);
                }
            }
        }
        if conflict {
            exclude("one program has conflicting measurements".into());
            continue;
        }
        let mut members: Vec<Member> = members.into_values().collect();
        if let Some(max) = rule.max_pool {
            members.sort_by_key(|m| cap_key(&rule.salt, context, &m.program_sha256));
            members.truncate(max);
            members.sort_by(|a, b| a.program_sha256.cmp(&b.program_sha256));
        }
        if members.len() < rule.min_pool {
            exclude(format!(
                "{} members < min_pool {}",
                members.len(),
                rule.min_pool
            ));
            continue;
        }
        let distinct = distinct_measured(&members);
        if distinct < rule.min_distinct_measured {
            exclude(format!(
                "{} distinct measured q < {}",
                distinct, rule.min_distinct_measured
            ));
            continue;
        }
        let pool = Pool {
            schema: POOL_SCHEMA.into(),
            context_id: context.to_owned(),
            parent_id: first.parent_id.clone(),
            role: first.role.clone(),
            protocol_sha256: first.protocol_sha256.clone(),
            incumbent_q: first.incumbent_q,
            members,
        };
        entries.push(ManifestEntry {
            context_id: pool.context_id.clone(),
            parent_id: pool.parent_id.clone(),
            pool_sha256: pool.sha256()?,
            members: pool.members.len(),
            measured: pool
                .members
                .iter()
                .filter(|m| m.outcome().is_some())
                .count(),
        });
        pools.push(pool);
    }
    let manifest = Manifest {
        schema: MANIFEST_SCHEMA.into(),
        shadow_only: true,
        qualified_claim: false,
        rule: rule.clone(),
        input_sha256: input_sha256.to_owned(),
        input_rows: rows.len(),
        pools: entries,
        excluded,
    };
    Ok((pools, manifest))
}

/// Check that loaded pools and their manifest carry every invariant `freeze` establishes:
/// schema and shadow flags, rule bounds, train-only roles, identities, outcome consistency (no
/// unknowns), manifest metadata and digests. Correct hashes alone are never sufficient.
pub fn verify(pools: &[Pool], manifest: &Manifest) -> Result<()> {
    ensure!(
        manifest.schema == MANIFEST_SCHEMA,
        "unknown manifest schema"
    );
    ensure!(
        manifest.shadow_only && !manifest.qualified_claim,
        "a J9 manifest is shadow-only"
    );
    hex64("manifest input_sha256", &manifest.input_sha256)?;
    check_rule(&manifest.rule)?;
    ensure!(
        pools.len() == manifest.pools.len(),
        "pool count differs from the manifest"
    );
    ensure!(
        manifest
            .pools
            .windows(2)
            .all(|w| w[0].context_id < w[1].context_id),
        "manifest contexts are not distinct and sorted"
    );
    let pooled: BTreeSet<&str> = manifest
        .pools
        .iter()
        .map(|e| e.context_id.as_str())
        .collect();
    ensure!(
        manifest
            .excluded
            .iter()
            .all(|e| !pooled.contains(e.context_id.as_str())),
        "a context is both pooled and excluded"
    );
    for (pool, entry) in pools.iter().zip(&manifest.pools) {
        check_pool(pool, &manifest.rule).with_context(|| format!("pool {}", pool.context_id))?;
        ensure!(
            pool.context_id == entry.context_id,
            "pool order differs from the manifest"
        );
        ensure!(
            pool.parent_id == entry.parent_id
                && pool.members.len() == entry.members
                && pool
                    .members
                    .iter()
                    .filter(|m| m.outcome().is_some())
                    .count()
                    == entry.measured,
            "pool {} metadata differs from the manifest",
            pool.context_id
        );
        ensure!(
            pool.sha256()? == entry.pool_sha256,
            "pool {} digest mismatch",
            pool.context_id
        );
    }
    Ok(())
}
