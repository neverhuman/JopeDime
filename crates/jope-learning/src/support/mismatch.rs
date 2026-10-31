use crate::canon::is_hex64;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, Serialize)]
/// `MismatchRow` is a public mismatch row record.
pub struct MismatchRow {
    /// Identifier for the associated context.
    pub context_id: String,
    /// Parent recorded for this mismatch row.
    pub parent: String,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Predicted tokens including eos recorded for this mismatch row.
    pub predicted_tokens_including_eos: usize,
    /// Trie sequence logprob float64 recorded for this mismatch row.
    pub trie_sequence_logprob_float64: f64,
    /// Grammar sequence logprob float64 recorded for this mismatch row.
    pub grammar_sequence_logprob_float64: f64,
    /// Sampler recompute abs error recorded for this mismatch row.
    pub sampler_recompute_abs_error: f64,
}

#[derive(Debug, Serialize)]
/// `MismatchReport` is a public mismatch report record.
pub struct MismatchReport {
    /// Schema identifier and version for the serialized value.
    pub schema: &'static str,
    /// Qualified recorded for this mismatch report.
    pub qualified: bool,
    /// Contexts recorded for this mismatch report.
    pub contexts: usize,
    /// Parents recorded for this mismatch report.
    pub parents: usize,
    /// Mean grammar extra nll recorded for this mismatch report.
    pub mean_grammar_extra_nll: f64,
    /// Median grammar extra nll recorded for this mismatch report.
    pub median_grammar_extra_nll: f64,
    /// Lower bound on grammar extra nll.
    pub min_grammar_extra_nll: f64,
    /// Upper bound on grammar extra nll.
    pub max_grammar_extra_nll: f64,
    /// Upper bound on sampler recompute abs error.
    pub max_sampler_recompute_abs_error: f64,
    /// Aggregation recorded for this mismatch report.
    pub aggregation: &'static str,
    /// Training objective changed recorded for this mismatch report.
    pub training_objective_changed: bool,
}

impl MismatchReport {
    /// Performs the from rows operation and returns an error if it cannot be completed.
    pub fn from_rows(rows: &[MismatchRow]) -> Result<Self> {
        ensure!(!rows.is_empty(), "nonempty measured comparison required");
        let mut contexts = BTreeSet::new();
        let mut parents = BTreeSet::new();
        let mut differences = Vec::new();
        let mut max_error: f64 = 0.;
        for r in rows {
            ensure!(
                is_hex64(&r.context_id) && is_hex64(&r.parent) && is_hex64(&r.program_sha256),
                "bound comparison identities required"
            );
            ensure!(
                contexts.insert(&r.context_id),
                "duplicate comparison context"
            );
            ensure!(
                r.predicted_tokens_including_eos > 0 && r.predicted_tokens_including_eos < 640,
                "EOS-inclusive count required"
            );
            ensure!(
                r.trie_sequence_logprob_float64.is_finite()
                    && r.trie_sequence_logprob_float64 <= 0.
                    && r.grammar_sequence_logprob_float64.is_finite()
                    && r.grammar_sequence_logprob_float64 <= 0.
                    && r.sampler_recompute_abs_error.is_finite()
                    && r.sampler_recompute_abs_error >= 0.,
                "finite measured probabilities required"
            );
            let difference = r.trie_sequence_logprob_float64 - r.grammar_sequence_logprob_float64;
            ensure!(difference.is_finite(), "comparison overflow");
            differences.push(difference);
            parents.insert(&r.parent);
            max_error = max_error.max(r.sampler_recompute_abs_error);
        }
        differences.sort_by(f64::total_cmp);
        let n = differences.len();
        let median = if n % 2 == 0 {
            differences[n / 2 - 1] / 2. + differences[n / 2] / 2.
        } else {
            differences[n / 2]
        };
        Ok(Self {
            schema: "jope.support-probability-mismatch/v1",
            qualified: false,
            contexts: n,
            parents: parents.len(),
            mean_grammar_extra_nll: differences.iter().map(|d| d / n as f64).sum(),
            median_grammar_extra_nll: median,
            min_grammar_extra_nll: differences[0],
            max_grammar_extra_nll: differences[n - 1],
            max_sampler_recompute_abs_error: max_error,
            aggregation: "sum_including_eos",
            training_objective_changed: false,
        })
    }
}

/// Performs the require same support reference operation and returns an error if it cannot be completed.
pub fn require_same_support_reference(
    actor: &str,
    reference: &str,
    parameter: &str,
    expected_parameter: &str,
) -> Result<()> {
    ensure!(
        [actor, reference, parameter, expected_parameter]
            .iter()
            .all(|x| is_hex64(x)),
        "bound reference identity required"
    );
    ensure!(actor == reference, "cross-support forward KL refused");
    ensure!(
        parameter == expected_parameter,
        "frozen reference parameter mismatch"
    );
    Ok(())
}
