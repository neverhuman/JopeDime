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

mod identity;
pub use identity::{
    ImmutableSourceView, MeasurementScope, ObservedOutcome, OutcomeKey, OutcomeStatus,
    UniverseEvidenceVerifier,
};
