//! Frozen bridge from measured, admitted fresh observations to the existing
//! native objective. This module selects complete comparisons; it does not
//! admit evidence or launch training.

use super::objective::{self, RowLabels, COMPARISON_ROWS};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SCHEMA: &str = "jope.native_fresh_replay_bundle.v1";
const ACTIVE_TERMS: [&str; 5] = [
    "sequence_nll",
    "ranking_bt",
    "ranking_listwise",
    "runtime_huber",
    "contextfree_bt",
];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayIdentity {
    pub schema: String,
    pub bundle_id: String,
    pub intent_sha256: String,
    pub lineage_sha256: String,
    pub corpus_sha256: String,
    pub parent_state_sha256: String,
    pub objective_sha256: String,
    pub recipe_sha256: String,
    pub partition_sha256: String,
    pub window_id: String,
    pub rng_identity: String,
    pub sampler_cursor: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    Chosen,
    Rejected,
}

/// Eligibility is recorded per measured occurrence. Since the existing
/// objective consumes a complete eight-row group across all active terms, a
/// group with any missing active-head authority is omitted as a whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadEligibility {
    pub sequence_nll: bool,
    pub ranking_bt: bool,
    pub ranking_listwise: bool,
    pub runtime_huber: bool,
    pub contextfree_bt: bool,
}

impl HeadEligibility {
    fn missing_active_term(self) -> Option<&'static str> {
        ACTIVE_TERMS.into_iter().find(|term| match *term {
            "sequence_nll" => !self.sequence_nll,
            "ranking_bt" => !self.ranking_bt,
            "ranking_listwise" => !self.ranking_listwise,
            "runtime_huber" => !self.runtime_huber,
            "contextfree_bt" => !self.contextfree_bt,
            _ => unreachable!(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FreshOccurrence {
    /// Stable draw/measurement occurrence, distinct even when a record is
    /// measured or selected more than once.
    pub occurrence_id: String,
    pub evidence_sha256: String,
    pub comparison_id: String,
    pub record_id: usize,
    pub context_id: usize,
    pub disposition: Disposition,
    pub value: f32,
    pub percentile: f32,
    pub runtime: f32,
    pub token_length: usize,
    pub eligibility: HeadEligibility,
}

impl FreshOccurrence {
    pub fn row_label(&self) -> RowLabels {
        RowLabels {
            record: self.record_id,
            context: self.context_id,
            value: self.value,
            percentile: self.percentile,
            runtime: self.runtime,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayBundle {
    pub schema: String,
    pub identity: ReplayIdentity,
    pub occurrences: Vec<FreshOccurrence>,
    pub bundle_sha256: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Denominators {
    pub comparisons: usize,
    pub occurrences: usize,
    pub chosen_occurrences: usize,
    pub rejected_occurrences: usize,
    pub sequence_supervised_positions: usize,
    pub sequence_row_weight_sum: f64,
    pub ranking_pairs: usize,
    pub runtime_rows: usize,
    pub occurrence_multiplicity_by_record: BTreeMap<usize, usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExcludedComparison {
    pub comparison_id: String,
    pub reason: String,
    pub occurrences: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Selection {
    pub bundle_sha256: String,
    pub occurrences: Vec<FreshOccurrence>,
    pub labels: Vec<RowLabels>,
    pub token_lengths: Vec<usize>,
    pub excluded: Vec<ExcludedComparison>,
    pub denominators: Denominators,
}

impl ReplayBundle {
    /// Freeze a bundle after validating its complete comparison groups.
    pub fn seal(identity: ReplayIdentity, occurrences: Vec<FreshOccurrence>) -> Result<Self> {
        let mut bundle = Self {
            schema: SCHEMA.into(),
            identity,
            occurrences,
            bundle_sha256: String::new(),
        };
        bundle.validate_shape()?;
        bundle.bundle_sha256 = bundle.digest()?;
        Ok(bundle)
    }

    /// Check both the immutable bundle digest and the caller's frozen I12/I27
    /// identity. A label edit or changed parent/window cannot reuse this seal.
    pub fn verify(&self, expected: &ReplayIdentity) -> Result<()> {
        ensure!(self.schema == SCHEMA, "unknown fresh replay bundle schema");
        ensure!(
            &self.identity == expected,
            "fresh replay bundle identity changed"
        );
        self.validate_shape()?;
        ensure!(
            self.digest()? == self.bundle_sha256,
            "fresh replay bundle digest changed"
        );
        Ok(())
    }

    /// Produce the only rows that may enter the existing native composite.
    /// Any group with an ineligible active term is excluded in full; the
    /// bridge never fills a masked label from another head or from a default.
    pub fn select(&self, expected: &ReplayIdentity) -> Result<Selection> {
        self.verify(expected)?;
        let mut grouped = BTreeMap::<&str, Vec<&FreshOccurrence>>::new();
        for occurrence in &self.occurrences {
            grouped
                .entry(&occurrence.comparison_id)
                .or_default()
                .push(occurrence);
        }

        let mut selected = Vec::new();
        let mut excluded = Vec::new();
        for (comparison_id, group) in grouped {
            let labels = group
                .iter()
                .map(|item| item.row_label())
                .collect::<Vec<_>>();
            ensure!(
                labels.len() == COMPARISON_ROWS,
                "comparison {comparison_id} is incomplete"
            );
            objective::validate_rows(&labels)?;
            let chosen = group
                .iter()
                .filter(|item| item.disposition == Disposition::Chosen)
                .count();
            ensure!(
                chosen == 1,
                "comparison {comparison_id} must retain one chosen occurrence"
            );
            let first_context = group[0].context_id;
            ensure!(
                group.iter().all(|item| item.context_id == first_context),
                "comparison {comparison_id} crosses contexts"
            );
            ensure!(
                group
                    .iter()
                    .all(|item| (2..=crate::corpus::SEQUENCE).contains(&item.token_length)),
                "comparison {comparison_id} has an invalid token length"
            );
            if let Some(term) = group
                .iter()
                .find_map(|item| item.eligibility.missing_active_term())
            {
                excluded.push(ExcludedComparison {
                    comparison_id: comparison_id.to_owned(),
                    reason: format!("ineligible active term: {term}"),
                    occurrences: group.len(),
                });
            } else {
                selected.extend(group);
            }
        }

        let mut denominators = Denominators::default();
        let mut occurrences = Vec::with_capacity(selected.len());
        let mut labels = Vec::with_capacity(selected.len());
        let mut token_lengths = Vec::with_capacity(selected.len());
        for item in selected {
            denominators.occurrences += 1;
            match item.disposition {
                Disposition::Chosen => denominators.chosen_occurrences += 1,
                Disposition::Rejected => denominators.rejected_occurrences += 1,
            }
            denominators.sequence_supervised_positions += item.token_length - 1;
            denominators.sequence_row_weight_sum +=
                f64::from(objective::row_weight(item.percentile));
            *denominators
                .occurrence_multiplicity_by_record
                .entry(item.record_id)
                .or_default() += 1;
            occurrences.push(item.clone());
            labels.push(item.row_label());
            token_lengths.push(item.token_length);
        }
        denominators.comparisons = denominators.occurrences / COMPARISON_ROWS;
        denominators.runtime_rows = denominators.occurrences;
        for comparison in labels.chunks(COMPARISON_ROWS) {
            for (i, left) in comparison.iter().enumerate() {
                denominators.ranking_pairs += comparison[i + 1..]
                    .iter()
                    .filter(|right| right.value != left.value)
                    .count();
            }
        }
        Ok(Selection {
            bundle_sha256: self.bundle_sha256.clone(),
            occurrences,
            labels,
            token_lengths,
            excluded,
            denominators,
        })
    }

    fn validate_shape(&self) -> Result<()> {
        validate_sha(&self.identity.intent_sha256, "intent")?;
        validate_sha(&self.identity.lineage_sha256, "lineage")?;
        validate_sha(&self.identity.corpus_sha256, "corpus")?;
        validate_sha(&self.identity.parent_state_sha256, "parent state")?;
        validate_sha(&self.identity.objective_sha256, "objective")?;
        validate_sha(&self.identity.recipe_sha256, "recipe")?;
        validate_sha(&self.identity.partition_sha256, "partition")?;
        ensure!(
            self.identity.schema == "jope.training-intent.v1"
                && !self.identity.bundle_id.is_empty()
                && !self.identity.window_id.is_empty()
                && !self.identity.rng_identity.is_empty(),
            "fresh replay identity is incomplete"
        );
        ensure!(
            !self.occurrences.is_empty(),
            "fresh replay bundle has no observations"
        );
        let mut ids = BTreeSet::new();
        for occurrence in &self.occurrences {
            ensure!(
                !occurrence.occurrence_id.is_empty()
                    && ids.insert(occurrence.occurrence_id.as_str()),
                "fresh replay occurrence ID is empty or duplicated"
            );
            ensure!(
                !occurrence.comparison_id.is_empty(),
                "fresh comparison identity is empty"
            );
            validate_sha(&occurrence.evidence_sha256, "measurement evidence")?;
        }
        let mut grouped = BTreeMap::<&str, Vec<&FreshOccurrence>>::new();
        for occurrence in &self.occurrences {
            grouped
                .entry(&occurrence.comparison_id)
                .or_default()
                .push(occurrence);
        }
        for (comparison_id, group) in grouped {
            let labels = group
                .iter()
                .map(|item| item.row_label())
                .collect::<Vec<_>>();
            ensure!(
                labels.len() == COMPARISON_ROWS,
                "comparison {comparison_id} is incomplete"
            );
            objective::validate_rows(&labels)?;
            ensure!(
                group
                    .iter()
                    .filter(|item| item.disposition == Disposition::Chosen)
                    .count()
                    == 1,
                "comparison {comparison_id} must retain one chosen occurrence"
            );
            ensure!(
                group
                    .iter()
                    .all(|item| (2..=crate::corpus::SEQUENCE).contains(&item.token_length)),
                "comparison {comparison_id} has an invalid token length"
            );
        }
        Ok(())
    }

    fn digest(&self) -> Result<String> {
        let body = serde_json::json!({
            "schema": &self.schema,
            "identity": &self.identity,
            "occurrences": &self.occurrences,
        });
        Ok(crate::manifest::hash(&serde_json::to_vec(&body)?))
    }
}

fn validate_sha(value: &str, field: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "{field} digest is malformed"
    );
    Ok(())
}

/// Build the journal's per-occurrence supervision summary from the selected
/// native rows and corpus token lengths. Callers still invoke the existing
/// `objective::composite` and optimizer; this records the exact denominator
/// submitted to that training step.
pub fn supervision(selection: &Selection) -> Result<objective::Supervision> {
    ensure!(
        !selection.labels.is_empty() && selection.labels.len() == selection.token_lengths.len(),
        "selected replay has no complete supervision rows"
    );
    objective::supervision(&selection.labels, &selection.token_lengths)
        .context("derive native composite supervision from selected fresh occurrences")
}
