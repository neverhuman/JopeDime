//! Selectors over a frozen pool (CLOVER §6 Experiment A).
//!
//! A selector scores the members of a `PoolView`, which carries no outcomes. Fitted selectors
//! learn only from the training pools of their cross-fitting fold. Every selector records its
//! exact rule; deterministic selectors are never given a propensity. The uniform baseline is not
//! a `Selector`: its expected metrics are computed exactly in `metrics::uniform_context`.

use super::pool::{Pool, PoolView};
use crate::canon::canonical_sha256;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod contextual_surrogate;
mod external;
mod funnel;
mod retrieval;
mod ridge;
#[cfg(test)]
mod tests;

pub use contextual_surrogate::{
    ContextualSurrogateQHat, MAX_BILINEAR_CTX_DIM, MAX_BILINEAR_PROG_DIM, MAX_SURROGATE_DESIGN_DIM,
};
pub use external::{
    External, ExternalDocument, ExternalProgramPrior, ExternalScore, EXTERNAL_SCHEMA,
};
pub use funnel::{pre_screen_funnel, FunnelAllocation};
pub use retrieval::RetrievalKnn;
pub(super) use ridge::cholesky_solve;
pub use ridge::RidgeProgram;

/// Whether a selector is a simple baseline or a learned candidate under test. A candidate passes
/// only against the strongest baseline, never by beating uniform alone (CLOVER §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectorClass {
    /// The `Baseline` case of `SelectorClass`.
    Baseline,
    /// The `Candidate` case of `SelectorClass`.
    Candidate,
}
/// Compatibility name retained for callers of the original selector API.
pub type Class = SelectorClass;

/// Interface for the selector contract used by Selectors over a frozen pool (CLOVER §6 Experiment A).
pub trait Selector {
    /// Performs the name operation.
    fn name(&self) -> &str;
    /// Performs the class operation.
    fn class(&self) -> SelectorClass;
    /// Exact rule and version, recorded in the receipt.
    fn rule(&self) -> String;
    /// One score per `view.programs`, higher first; ties are broken by program sha256.
    fn scores(&self, view: &PoolView<'_>) -> Result<Vec<f64>>;
}

/// Within-context normalized training target: the mid-rank percentile of each member's outcome
/// among its pool, failures tied lowest. It uses only that pool's own (training) outcomes, so it
/// is fixed before any held-out pool is read.
pub fn rank_targets(pool: &Pool) -> Vec<f64> {
    let outcomes = pool.outcomes();
    let n = outcomes.len();
    if n < 2 {
        return vec![0.5; n];
    }
    let key = |o: &Option<f64>| o.unwrap_or(f64::NEG_INFINITY);
    outcomes
        .iter()
        .map(|o| {
            let k = key(o);
            let below = outcomes.iter().filter(|p| key(p) < k).count() as f64;
            let ties = outcomes.iter().filter(|p| key(p) == k).count() as f64;
            (below + (ties - 1.0) / 2.0) / (n - 1) as f64
        })
        .collect()
}

/// Mean training target of each program over the training pools that contain it.
fn program_means(train: &[&Pool]) -> BTreeMap<String, (f64, usize)> {
    let mut acc: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for pool in train {
        for (m, t) in pool.members.iter().zip(rank_targets(pool)) {
            let e = acc.entry(m.program_sha256.clone()).or_insert((0.0, 0));
            e.0 += t;
            e.1 += 1;
        }
    }
    acc.into_iter()
        .map(|(k, (s, c))| (k, (s / c as f64, c)))
        .collect()
}

/// Popularity prior: a program's mean within-context percentile on the training pools, shrunk
/// toward 0.5 by `prior_weight` pseudo-observations. Context-blind.
pub struct ProgramPrior {
    means: BTreeMap<String, (f64, usize)>,
    prior_weight: f64,
}

impl ProgramPrior {
    /// Performs the fit operation.
    pub fn fit(train: &[&Pool], prior_weight: f64) -> Self {
        Self {
            means: program_means(train),
            prior_weight,
        }
    }

    fn score(&self, program: &str) -> f64 {
        match self.means.get(program) {
            Some(&(mean, count)) => {
                let c = count as f64;
                (mean * c + 0.5 * self.prior_weight) / (c + self.prior_weight)
            }
            None => 0.5,
        }
    }
}

impl Selector for ProgramPrior {
    fn name(&self) -> &str {
        "program_prior"
    }
    fn class(&self) -> SelectorClass {
        SelectorClass::Baseline
    }
    fn rule(&self) -> String {
        format!(
            "program_prior/v1: mean training within-context mid-rank percentile, shrunk to 0.5 by {} pseudo-observations; unseen = 0.5",
            self.prior_weight
        )
    }
    fn scores(&self, view: &PoolView<'_>) -> Result<Vec<f64>> {
        Ok(view.programs.iter().map(|p| self.score(p)).collect())
    }
}

/// Dense per-key feature vectors (context encodings or program descriptors).
#[derive(Clone, Debug, Default)]
pub struct Features {
    /// Dim recorded for this features.
    pub dim: usize,
    /// Number of rows represented by this value.
    pub rows: BTreeMap<String, Vec<f64>>,
}

/// A target-free descriptor that can be independently reconstructed from the public context ID.
/// Arbitrary `id/vector` files do not carry fitting provenance and are refused by nested/v3.
/// The first 32 coordinates are normalized ID bytes; further coordinates are zero padding.
/// No target-derived external provider is admitted by this version.
pub const CONTEXT_DESCRIPTOR: &str = "context-id-bytes/v1";

impl Features {
    /// Performs the context ID vector operation and returns an error if it cannot be completed.
    pub fn context_id_vector(id: &str, dim: usize) -> Result<Vec<f64>> {
        ensure!(
            dim > 0 && id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit()),
            "context descriptor requires a 64-hex context ID and positive dimension"
        );
        (0..dim)
            .map(|j| {
                if j >= 32 {
                    return Ok(0.0);
                }
                let offset = 2 * j;
                Ok((u8::from_str_radix(&id[offset..offset + 2], 16)? as f64 - 127.5) / 127.5)
            })
            .collect()
    }

    /// Performs the context ID descriptors operation and returns an error if it cannot be completed.
    pub fn context_id_descriptors(pools: &[Pool], dim: usize) -> Result<Self> {
        Ok(Self {
            dim,
            rows: pools
                .iter()
                .map(|p| {
                    Ok((
                        p.context_id.clone(),
                        Self::context_id_vector(&p.context_id, dim)?,
                    ))
                })
                .collect::<Result<_>>()?,
        })
    }

    /// Performs the verify context descriptor operation and returns an error if it cannot be completed.
    pub fn verify_context_descriptor(&self) -> Result<()> {
        ensure!(!self.rows.is_empty(), "context descriptor has no rows");
        for (id, vector) in &self.rows {
            let expected = Self::context_id_vector(id, self.dim)?;
            ensure!(vector.len() == expected.len()
                && vector.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                "unbound context provenance: {id} is not {CONTEXT_DESCRIPTOR}; target-derived features require a fold-fitted provider");
        }
        Ok(())
    }
}

/// Constant identifying the nested procedure.
pub const NESTED_PROCEDURE: &str = "contextual-surrogate-parent-nested/v3";
const INNER_FOLDS: usize = 5;
const INNER_SALT: &str = "jope.j9-inner-ridge-folds/v1";

/// Denominator unit: pool-member rows. A refused whole procedure scores none of its rows.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct CvCounts {
    /// Attempted units recorded for this cross-validation counts.
    pub attempted_units: usize,
    /// Scored units recorded for this cross-validation counts.
    pub scored_units: usize,
    /// Excluded units recorded for this cross-validation counts.
    pub excluded_units: usize,
    /// Failed units recorded for this cross-validation counts.
    pub failed_units: usize,
    /// Parents recorded for this cross-validation counts.
    pub parents: usize,
    /// Folds recorded for this cross-validation counts.
    pub folds: usize,
    /// Failed candidates recorded for this cross-validation counts.
    pub failed_candidates: usize,
    /// Collection of failure details values carried by this cross-validation counts.
    pub failure_details: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
/// `CvUnavailableReason` is an enum for the cross-validation unavailable reason contract in this module: Selectors over a frozen pool (CLOVER §6 Experiment A).
pub enum CvUnavailableReason {
    /// The `InsufficientParents` case of `CvUnavailableReason`.
    InsufficientParents,
    /// The `EmptyUsableFolds` case of `CvUnavailableReason`.
    EmptyUsableFolds,
    /// The `ShapeMismatch` case of `CvUnavailableReason`.
    ShapeMismatch,
    /// The `NonfiniteSystem` case of `CvUnavailableReason`.
    NonfiniteSystem,
    /// The `SingularSystem` case of `CvUnavailableReason`.
    SingularSystem,
    /// The `ComponentFit` case of `CvUnavailableReason`.
    ComponentFit,
    /// The `UnboundContextProvenance` case of `CvUnavailableReason`.
    UnboundContextProvenance,
    /// The `ExternalPriorProvenance` case of `CvUnavailableReason`.
    ExternalPriorProvenance,
    /// The `NoFiniteCandidate` case of `CvUnavailableReason`.
    NoFiniteCandidate,
}

/// Local estimate adapter: finite binary64 values use I02's exact bit spelling. Unavailable
/// carries no metric at all. This is a CV report, not an outcome authority envelope.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CvEstimate {
    /// The `Supported` case of `CvEstimate`.
    Supported {
        /// Procedure recorded for this cross-validation estimate.
        procedure: String,
        /// Metric used to evaluate the represented result.
        metric: ExactCvLoss,
        /// Counts recorded for this cross-validation estimate.
        counts: CvCounts,
    },
    /// Required evidence is unavailable, so no exact result can be reported.
    Unavailable {
        /// Procedure recorded for this cross-validation estimate.
        procedure: String,
        /// Explanation associated with the recorded disposition.
        reason: CvUnavailableReason,
        /// Detail recorded for this cross-validation estimate.
        detail: String,
        /// Counts recorded for this cross-validation estimate.
        counts: CvCounts,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
/// `ExactCvLoss` is a record for the exact cross-validation loss contract in this module: Selectors over a frozen pool (CLOVER §6 Experiment A).
pub struct ExactCvLoss {
    f64_bits: String,
}

impl CvEstimate {
    /// Performs the loss operation.
    pub fn loss(&self) -> Option<f64> {
        match self {
            Self::Supported { metric, .. } => Some(f64::from_bits(
                u64::from_str_radix(&metric.f64_bits, 16).expect("constructed exact finite loss"),
            )),
            Self::Unavailable { .. } => None,
        }
    }

    /// Performs the counts operation.
    pub fn counts(&self) -> &CvCounts {
        match self {
            Self::Supported { counts, .. } | Self::Unavailable { counts, .. } => counts,
        }
    }
}

#[derive(Clone, Debug)]
/// `CvUnavailable` is a record for the cross-validation unavailable contract in this module: Selectors over a frozen pool (CLOVER §6 Experiment A).
pub struct CvUnavailable(pub CvEstimate);

impl std::fmt::Display for CvUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
impl std::error::Error for CvUnavailable {}

#[derive(Clone, Debug, PartialEq, Serialize)]
/// `InnerFoldFit` is a record for the inner fold fit contract in this module: Selectors over a frozen pool (CLOVER §6 Experiment A).
pub struct InnerFoldFit {
    /// Fold recorded for this inner fold fit.
    pub fold: usize,
    /// Collection of training parents values carried by this inner fold fit.
    pub training_parents: Vec<String>,
    /// Collection of held parents values carried by this inner fold fit.
    pub held_parents: Vec<String>,
    /// SHA-256 digest identifying the associated training input.
    pub training_input_sha256: String,
    /// SHA-256 digest identifying the associated training features.
    pub training_features_sha256: String,
    /// SHA-256 digest identifying the associated split.
    pub split_sha256: String,
    /// SHA-256 digest identifying the associated state.
    pub state_sha256: String,
    /// SHA-256 digest identifying the associated held design.
    pub held_design_sha256: String,
    /// One digest per locally finite candidate in grid order, independent of other folds failing.
    pub prediction_sha256: Vec<String>,
    /// Collection of selected prediction bits values carried by this inner fold fit.
    pub selected_prediction_bits: Vec<Vec<String>>,
}

fn evidence_parent_ids<'p>(train: &[&'p Pool]) -> Vec<&'p str> {
    train
        .iter()
        .map(|p| p.parent_id.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn component_error(error: anyhow::Error, counts: CvCounts) -> anyhow::Error {
    let detail = format!("{error:#}");
    let reason = if detail.contains("nonfinite") || detail.contains("non-finite") {
        CvUnavailableReason::NonfiniteSystem
    } else if detail.contains("not positive definite") {
        CvUnavailableReason::SingularSystem
    } else {
        CvUnavailableReason::ComponentFit
    };
    unavailable(reason, detail, counts)
}

fn unavailable(
    reason: CvUnavailableReason,
    detail: impl ToString,
    mut counts: CvCounts,
) -> anyhow::Error {
    counts.failed_units = counts.attempted_units - counts.excluded_units;
    counts.scored_units = 0;
    CvUnavailable(CvEstimate::Unavailable {
        procedure: NESTED_PROCEDURE.into(),
        reason,
        detail: detail.to_string(),
        counts,
    })
    .into()
}

/// One line of a feature file.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureLine {
    /// Stable identifier for this value.
    pub id: String,
    /// Collection of vector values carried by this feature line.
    pub vector: Vec<f64>,
}

impl Features {
    /// JSON Lines of `{"id": ..., "vector": [...]}`; one dimension, finite values, unique ids.
    pub fn from_jsonl(text: &str) -> Result<Self> {
        let mut out = Features::default();
        for (i, line) in text
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty())
        {
            let row: FeatureLine =
                serde_json::from_str(line).with_context(|| format!("feature line {}", i + 1))?;
            ensure!(
                !row.vector.is_empty(),
                "empty feature vector for {}",
                row.id
            );
            ensure!(
                row.vector.iter().all(|v| v.is_finite()),
                "non-finite feature for {}",
                row.id
            );
            if out.rows.is_empty() {
                out.dim = row.vector.len();
            }
            ensure!(
                row.vector.len() == out.dim,
                "feature dimension changes at {}",
                row.id
            );
            ensure!(
                out.rows.insert(row.id.clone(), row.vector).is_none(),
                "duplicate feature id {}",
                row.id
            );
        }
        Ok(out)
    }
}

/// `token_bag/v1`: per-token frequency (count / length) over ids `0..=max_id` plus `ln(length)`,
/// keyed by program sha256. Tokens are pre-measurement descriptors, so the vocabulary may be read
/// from every pool. `None` unless every member of every pool carries tokens.
pub fn token_bag_features(pools: &[Pool]) -> Option<Features> {
    let members = || pools.iter().flat_map(|p| &p.members);
    let max_id = members()
        .map(|m| m.tokens.as_ref().and_then(|t| t.iter().max().copied()))
        .collect::<Option<Vec<u32>>>()?
        .into_iter()
        .max()?;
    let dim = max_id as usize + 2;
    let mut rows = BTreeMap::new();
    for m in members() {
        let tokens = m.tokens.as_ref()?;
        let mut v = vec![0.0; dim];
        let len = tokens.len() as f64;
        for &t in tokens {
            v[t as usize] += 1.0 / len;
        }
        v[dim - 1] = len.ln();
        rows.insert(m.program_sha256.clone(), v);
    }
    Some(Features { dim, rows })
}
