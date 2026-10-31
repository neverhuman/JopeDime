//! Exact measured outcomes for an explicitly requested development support.
//!
//! This module never predicts missing values, broadens a support set, or upgrades an input to
//! an admitted view. Callers must supply a verifier backed by the accepted I27/I28 interfaces.
//! The verifier's contract covers source-view authenticity, provenance closure and revocation.

use crate::canon::{canonical_sha256, is_hex64};
use anyhow::{ensure, Context, Result};
use jope_records::wire_contract::VerifiedMeasuredOutcome;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Schema identifier for universe values.
pub const UNIVERSE_SCHEMA: &str = "jope.measured-universe/v1";
/// Schema identifier for source view values.
pub const SOURCE_VIEW_SCHEMA: &str = "jope.immutable-measurement-view/v1";
/// Constant identifying the metric name.
pub const METRIC_NAME: &str = "development_score";

/// A complete evaluator identity. Every field participates in equality and table joins.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementScope {
    /// SHA-256 digest identifying the associated context.
    pub context_sha256: String,
    /// SHA-256 digest of the checkpoint bound to this evidence.
    pub checkpoint_sha256: String,
    /// SHA-256 digest identifying the associated policy.
    pub policy_sha256: String,
    /// SHA-256 digest identifying the evaluator used for this evidence.
    pub evaluator_sha256: String,
    /// SHA-256 digest identifying the associated target.
    pub target_sha256: String,
    /// SHA-256 digest identifying the associated profile.
    pub profile_sha256: String,
    /// Fidelity recorded for this measurement scope.
    pub fidelity: String,
    /// Data or authority role assigned to this value.
    pub role: String,
    /// Metric name recorded for this measurement scope.
    pub metric_name: String,
}

impl MeasurementScope {
    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("context", &self.context_sha256),
            ("checkpoint", &self.checkpoint_sha256),
            ("policy", &self.policy_sha256),
            ("evaluator", &self.evaluator_sha256),
            ("target", &self.target_sha256),
            ("profile", &self.profile_sha256),
        ] {
            ensure!(
                is_hex64(value),
                "{name} identity must be a lowercase sha256"
            );
        }
        ensure!(!self.fidelity.trim().is_empty(), "fidelity is required");
        ensure!(!self.role.trim().is_empty(), "measurement role is required");
        ensure!(
            self.metric_name == METRIC_NAME,
            "unsupported metric identity"
        );
        Ok(())
    }

    fn same_subject(&self, other: &Self) -> bool {
        self.context_sha256 == other.context_sha256
            && self.checkpoint_sha256 == other.checkpoint_sha256
            && self.policy_sha256 == other.policy_sha256
    }

    fn mismatches(&self, other: &Self) -> Vec<String> {
        let mut out = Vec::new();
        macro_rules! mismatch {
            ($field:ident) => {
                if self.$field != other.$field {
                    out.push(stringify!($field).to_owned());
                }
            };
        }
        mismatch!(context_sha256);
        mismatch!(checkpoint_sha256);
        mismatch!(policy_sha256);
        mismatch!(evaluator_sha256);
        mismatch!(target_sha256);
        mismatch!(profile_sha256);
        mismatch!(fidelity);
        mismatch!(role);
        mismatch!(metric_name);
        out
    }
}

/// One exact requested cell. Replicates are independent only when the source contract gives
/// them distinct stable IDs; `None` is a single unreplicated key, never an invitation to average.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeKey {
    /// Scope label describing the population covered by this result.
    pub scope: MeasurementScope,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Identifier for the associated replicate.
    pub replicate_id: Option<String>,
}

impl OutcomeKey {
    fn validate(&self) -> Result<()> {
        self.scope.validate()?;
        ensure!(
            is_hex64(&self.program_sha256),
            "program identity must be a lowercase sha256"
        );
        if let Some(id) = &self.replicate_id {
            ensure!(!id.trim().is_empty(), "replicate id cannot be empty");
        }
        Ok(())
    }

    fn same_subject(&self, other: &Self) -> bool {
        self.scope.same_subject(&other.scope) && self.program_sha256 == other.program_sha256
    }
}

/// Statuses remain distinct so failure, censoring and unknown evidence retain their denominator.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeStatus {
    /// The `Complete` case of `OutcomeStatus`.
    Complete,
    /// The `Failed` case of `OutcomeStatus`.
    Failed,
    /// The `Timeout` case of `OutcomeStatus`.
    Timeout,
    /// The `Rejected` case of `OutcomeStatus`.
    Rejected,
    /// The `Unknown` case of `OutcomeStatus`.
    Unknown,
}

/// An immutable source locator and its declared outcome. Complete values use exact I02 f64 bits.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedOutcome {
    /// Key recorded for this observed outcome.
    pub key: OutcomeKey,
    /// Outcome or lifecycle status recorded for this value.
    pub status: OutcomeStatus,
    /// Optional metric f64 bits value; `None` means it is not available.
    pub metric_f64_bits: Option<String>,
    /// SHA-256 digest identifying the associated evidence receipt.
    pub evidence_receipt_sha256: String,
    /// SHA-256 digest identifying the associated source locator.
    pub source_locator_sha256: String,
}

impl ObservedOutcome {
    fn validate(&self) -> Result<()> {
        self.key.validate()?;
        ensure!(
            is_hex64(&self.evidence_receipt_sha256)
                && self.evidence_receipt_sha256.bytes().any(|b| b != b'0'),
            "nonzero evidence receipt sha256 required"
        );
        ensure!(
            is_hex64(&self.source_locator_sha256),
            "source locator must be a lowercase sha256"
        );
        match (self.status, self.metric_f64_bits.as_deref()) {
            (OutcomeStatus::Complete, Some(bits)) => {
                let exact =
                    jope_records::wire_contract::ExactF64::from_hex("/metric_f64_bits", bits)?;
                ensure!(exact.to_hex() == bits, "metric bits must be canonical");
            }
            (OutcomeStatus::Complete, None) => anyhow::bail!("complete outcome has no metric"),
            (_, None) => (),
            (_, Some(_)) => anyhow::bail!("unavailable outcome cannot carry a metric"),
        }
        Ok(())
    }
}

/// Source view pinned by I27/I28. Its sorted locator inventory binds every observation admitted
/// by this build; the verifier must authenticate the view and prove that none of its inputs were
/// revoked. This local type is a draft adapter until the upstream accepted view DTO is integrated.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImmutableSourceView {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Source commit recorded for this immutable source view.
    pub source_commit: String,
    /// SHA-256 digest identifying the associated source receipt.
    pub source_receipt_sha256: String,
    /// SHA-256 digest identifying the associated revocation snapshot.
    pub revocation_snapshot_sha256: String,
    /// SHA-256 digest identifying the associated observation locators.
    pub observation_locators_sha256: Vec<String>,
    /// SHA-256 digest identifying the associated view.
    pub view_sha256: String,
}

impl ImmutableSourceView {
    /// Performs the seal operation and returns an error if it cannot be completed.
    pub fn seal(&mut self) -> Result<()> {
        self.view_sha256.clear();
        self.view_sha256 = canonical_sha256(self)?;
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SOURCE_VIEW_SCHEMA,
            "unsupported source view schema"
        );
        ensure!(
            (self.source_commit.len() == 40 || self.source_commit.len() == 64)
                && self
                    .source_commit
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "source commit must be a lowercase Git object id"
        );
        for (name, value) in [
            ("source receipt", &self.source_receipt_sha256),
            ("revocation snapshot", &self.revocation_snapshot_sha256),
            ("view", &self.view_sha256),
        ] {
            ensure!(is_hex64(value), "{name} must be a lowercase sha256");
        }
        ensure!(
            self.observation_locators_sha256
                .windows(2)
                .all(|pair| pair[0] < pair[1])
                && self
                    .observation_locators_sha256
                    .iter()
                    .all(|id| is_hex64(id)),
            "source locators must be unique, sorted sha256 identities"
        );
        let mut unsealed = self.clone();
        let expected = unsealed.view_sha256.clone();
        unsealed.view_sha256.clear();
        ensure!(
            canonical_sha256(&unsealed)? == expected,
            "source view digest mismatch"
        );
        Ok(())
    }
}

/// Trusted adapter boundary. An accepted I27/I28 implementation must verify the actual immutable
/// receipts, exact source closure and revocation snapshot. Complete outcomes must return I02's
/// sealed `VerifiedMeasuredOutcome`; unavailable outcomes must be authenticated and return None.
pub trait UniverseEvidenceVerifier {
    /// Performs the verify view operation and returns an error if it cannot be completed.
    fn verify_view(&self, view: &ImmutableSourceView) -> Result<()>;

    /// Performs the verify observation operation and returns an error if it cannot be completed.
    fn verify_observation(
        &self,
        view: &ImmutableSourceView,
        observation: &ObservedOutcome,
    ) -> Result<Option<VerifiedMeasuredOutcome>>;
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
/// `OutcomeCell` is a record for the outcome cell contract in this module: Exact measured outcomes for an explicitly requested development support.
pub struct OutcomeCell {
    /// Key recorded for this outcome cell.
    pub key: OutcomeKey,
    /// Outcome or lifecycle status recorded for this value.
    pub status: CellStatus,
    /// Set only for a single, verified complete value.
    pub metric_f64_bits: Option<String>,
    /// Every exact source occurrence is retained, including duplicate equal rows and conflicts.
    pub observations: Vec<ObservedOutcome>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
/// `CellStatus` is an enum for the cell status contract in this module: Exact measured outcomes for an explicitly requested development support.
pub enum CellStatus {
    /// The `Complete` case of `CellStatus`.
    Complete,
    /// The `Failed` case of `CellStatus`.
    Failed,
    /// The `Censored` case of `CellStatus`.
    Censored,
    /// The `Rejected` case of `CellStatus`.
    Rejected,
    /// The `Unknown` case of `CellStatus`.
    Unknown,
    /// The `Conflicting` case of `CellStatus`.
    Conflicting,
    /// The `Missing` case of `CellStatus`.
    Missing,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
/// `JoinRefusal` is a record for the join refusal contract in this module: Exact measured outcomes for an explicitly requested development support.
pub struct JoinRefusal {
    /// SHA-256 digest identifying the associated requested key.
    pub requested_key_sha256: Option<String>,
    /// SHA-256 digest identifying the associated observed locator.
    pub observed_locator_sha256: String,
    /// Collection of mismatched fields values carried by this join refusal.
    pub mismatched_fields: Vec<String>,
}

/// Per-context/checkpoint/policy coverage. Probability mass is always explicitly unknown until an
/// adopted typed frozen-law DTO is available; observed counts are never converted to mass.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UniverseCoverage {
    /// SHA-256 digest identifying the associated context.
    pub context_sha256: String,
    /// SHA-256 digest of the checkpoint bound to this evidence.
    pub checkpoint_sha256: String,
    /// SHA-256 digest identifying the associated policy.
    pub policy_sha256: String,
    /// Support cardinality recorded for this universe coverage.
    pub support_cardinality: usize,
    /// Observed recorded for this universe coverage.
    pub observed: usize,
    /// Complete measured recorded for this universe coverage.
    pub complete_measured: usize,
    /// Failed recorded for this universe coverage.
    pub failed: usize,
    /// Number of attempts whose outcomes were censored before completion.
    pub censored: usize,
    /// Rejected recorded for this universe coverage.
    pub rejected: usize,
    /// Unknown recorded for this universe coverage.
    pub unknown: usize,
    /// Conflicting recorded for this universe coverage.
    pub conflicting: usize,
    /// Missing recorded for this universe coverage.
    pub missing: usize,
    /// Full support recorded for this universe coverage.
    pub full_support: bool,
    /// Optional probability mass value; `None` means it is not available.
    pub probability_mass: Option<String>,
    /// Probability mass reason recorded for this universe coverage.
    pub probability_mass_reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
/// `Universe` is a record for the universe contract in this module: Exact measured outcomes for an explicitly requested development support.
pub struct Universe {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// SHA-256 digest identifying the associated source view.
    pub source_view_sha256: String,
    /// The rows in this table are limited to this exact requested finite support.
    pub requested_support_sha256: String,
    /// Collection of outcomes values carried by this universe.
    pub outcomes: Vec<OutcomeCell>,
    /// Collection of join refusals values carried by this universe.
    pub join_refusals: Vec<JoinRefusal>,
    /// Collection of coverage values carried by this universe.
    pub coverage: Vec<UniverseCoverage>,
}

impl Universe {
    /// Build an exact development table. No row outside `requested` becomes a table member.
    pub fn build(
        view: &ImmutableSourceView,
        requested: Vec<OutcomeKey>,
        observations: Vec<ObservedOutcome>,
        verifier: &dyn UniverseEvidenceVerifier,
    ) -> Result<Self> {
        view.validate()?;
        verifier.verify_view(view)?;
        ensure!(!requested.is_empty(), "requested support must be nonempty");
        let mut requested_set = BTreeSet::new();
        for key in &requested {
            key.validate()?;
            ensure!(
                requested_set.insert(key.clone()),
                "duplicate requested outcome key"
            );
        }
        let requested_support_sha256 = canonical_sha256(&requested_set)?;
        let locator_set: BTreeSet<_> = view.observation_locators_sha256.iter().cloned().collect();
        let mut exact: BTreeMap<OutcomeKey, Vec<ObservedOutcome>> = BTreeMap::new();
        let mut join_refusals = Vec::new();
        for observation in observations {
            observation.validate()?;
            ensure!(
                locator_set.contains(&observation.source_locator_sha256),
                "observation locator is outside the authenticated view"
            );
            let verified = verifier.verify_observation(view, &observation)?;
            match observation.status {
                OutcomeStatus::Complete => {
                    let verified =
                        verified.context("complete outcome lacks I02 verified authority")?;
                    ensure!(
                        verified.metric_name() == observation.key.scope.metric_name,
                        "verified metric name differs from outcome key"
                    );
                    ensure!(
                        Some(verified.metric().to_hex()) == observation.metric_f64_bits,
                        "verified exact metric differs from observed value"
                    );
                    ensure!(
                        verified.evidence_receipt_sha256() == observation.evidence_receipt_sha256,
                        "verified evidence receipt differs from observation"
                    );
                }
                _ => ensure!(
                    verified.is_none(),
                    "unavailable outcome cannot gain measured authority"
                ),
            }
            if requested_set.contains(&observation.key) {
                exact
                    .entry(observation.key.clone())
                    .or_default()
                    .push(observation);
            } else {
                let candidates: Vec<_> = requested_set
                    .iter()
                    .filter(|request| request.same_subject(&observation.key))
                    .collect();
                if let Some(request) = candidates.first() {
                    let mut mismatched_fields = request.scope.mismatches(&observation.key.scope);
                    if request.replicate_id != observation.key.replicate_id {
                        mismatched_fields.push("replicate_id".into());
                    }
                    join_refusals.push(JoinRefusal {
                        requested_key_sha256: Some(canonical_sha256(request)?),
                        observed_locator_sha256: observation.source_locator_sha256,
                        mismatched_fields,
                    });
                } else {
                    join_refusals.push(JoinRefusal {
                        requested_key_sha256: None,
                        observed_locator_sha256: observation.source_locator_sha256,
                        mismatched_fields: vec!["no_requested_support_key".into()],
                    });
                }
            }
        }
        let mut outcomes = Vec::with_capacity(requested_set.len());
        for key in requested_set {
            let mut rows = exact.remove(&key).unwrap_or_default();
            rows.sort_by(|a, b| {
                (
                    &a.status,
                    &a.metric_f64_bits,
                    &a.evidence_receipt_sha256,
                    &a.source_locator_sha256,
                )
                    .cmp(&(
                        &b.status,
                        &b.metric_f64_bits,
                        &b.evidence_receipt_sha256,
                        &b.source_locator_sha256,
                    ))
            });
            let different_values: BTreeSet<_> = rows
                .iter()
                .map(|row| (row.status, row.metric_f64_bits.clone()))
                .collect();
            let (status, metric_f64_bits) = if rows.is_empty() {
                (CellStatus::Missing, None)
            } else if different_values.len() > 1 {
                (CellStatus::Conflicting, None)
            } else {
                let first = &rows[0];
                let status = match first.status {
                    OutcomeStatus::Complete => CellStatus::Complete,
                    OutcomeStatus::Failed => CellStatus::Failed,
                    OutcomeStatus::Timeout => CellStatus::Censored,
                    OutcomeStatus::Rejected => CellStatus::Rejected,
                    OutcomeStatus::Unknown => CellStatus::Unknown,
                };
                (status, first.metric_f64_bits.clone())
            };
            outcomes.push(OutcomeCell {
                key,
                status,
                metric_f64_bits,
                observations: rows,
            });
        }
        join_refusals.sort_by(|a, b| {
            (&a.requested_key_sha256, &a.observed_locator_sha256)
                .cmp(&(&b.requested_key_sha256, &b.observed_locator_sha256))
        });
        let coverage = build_coverage(&outcomes);
        Ok(Self {
            schema: UNIVERSE_SCHEMA.into(),
            source_view_sha256: view.view_sha256.clone(),
            requested_support_sha256,
            outcomes,
            join_refusals,
            coverage,
        })
    }

    /// Only verified complete cells are exposed as measured values.
    pub fn measured(&self) -> impl Iterator<Item = (&OutcomeKey, &str)> {
        self.outcomes.iter().filter_map(|cell| {
            (cell.status == CellStatus::Complete)
                .then(|| {
                    cell.metric_f64_bits
                        .as_deref()
                        .map(|bits| (&cell.key, bits))
                })
                .flatten()
        })
    }

    /// Computes the canonical SHA-256 digest of this value.
    pub fn digest(&self) -> Result<String> {
        canonical_sha256(self)
    }
}

fn build_coverage(outcomes: &[OutcomeCell]) -> Vec<UniverseCoverage> {
    let mut groups: BTreeMap<(String, String, String), Vec<&OutcomeCell>> = BTreeMap::new();
    for cell in outcomes {
        groups
            .entry((
                cell.key.scope.context_sha256.clone(),
                cell.key.scope.checkpoint_sha256.clone(),
                cell.key.scope.policy_sha256.clone(),
            ))
            .or_default()
            .push(cell);
    }
    groups
        .into_iter()
        .map(
            |((context_sha256, checkpoint_sha256, policy_sha256), cells)| {
                let mut report = UniverseCoverage {
                    context_sha256,
                    checkpoint_sha256,
                    policy_sha256,
                    support_cardinality: cells.len(),
                    observed: 0,
                    complete_measured: 0,
                    failed: 0,
                    censored: 0,
                    rejected: 0,
                    unknown: 0,
                    conflicting: 0,
                    missing: 0,
                    full_support: false,
                    probability_mass: None,
                    probability_mass_reason: "unknown: no adopted typed frozen-law DTO supplied"
                        .into(),
                };
                for cell in cells {
                    if !cell.observations.is_empty() {
                        report.observed += 1;
                    }
                    match cell.status {
                        CellStatus::Complete => report.complete_measured += 1,
                        CellStatus::Failed => report.failed += 1,
                        CellStatus::Censored => report.censored += 1,
                        CellStatus::Rejected => report.rejected += 1,
                        CellStatus::Unknown => report.unknown += 1,
                        CellStatus::Conflicting => report.conflicting += 1,
                        CellStatus::Missing => report.missing += 1,
                    }
                }
                report.full_support = report.complete_measured == report.support_cardinality
                    && report.conflicting == 0
                    && report.missing == 0;
                report
            },
        )
        .collect()
}
