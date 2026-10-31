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

/// Whether a selector is a simple baseline or a learned candidate under test. A candidate passes
/// only against the strongest baseline, never by beating uniform alone (CLOVER §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    /// The `Baseline` case of `Class`.
    Baseline,
    /// The `Candidate` case of `Class`.
    Candidate,
}

/// Interface for the selector contract used by Selectors over a frozen pool (CLOVER §6 Experiment A).
pub trait Selector {
    /// Performs the name operation.
    fn name(&self) -> &str;
    /// Performs the class operation.
    fn class(&self) -> Class;
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
    fn class(&self) -> Class {
        Class::Baseline
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

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Similar-task reuse: score each program by its training targets on the `k` training contexts
/// whose context features are most cosine-similar, weighting by similarity; programs absent from
/// those neighbours fall back to the program prior.
pub struct RetrievalKnn<'a> {
    k: usize,
    context_features: &'a Features,
    train: Vec<(&'a Pool, Vec<f64>)>,
    prior: ProgramPrior,
}

impl<'a> RetrievalKnn<'a> {
    /// Performs the fit operation and returns an error if it cannot be completed.
    pub fn fit(
        train: &[&'a Pool],
        context_features: &'a Features,
        k: usize,
        prior_weight: f64,
    ) -> Result<Self> {
        ensure!(k >= 1, "k must be positive");
        for pool in train {
            ensure!(
                context_features.rows.contains_key(&pool.context_id),
                "no context features for training context {}",
                pool.context_id
            );
        }
        Ok(Self {
            k,
            context_features,
            train: train.iter().map(|p| (*p, rank_targets(p))).collect(),
            prior: ProgramPrior::fit(train, prior_weight),
        })
    }
}

impl Selector for RetrievalKnn<'_> {
    fn name(&self) -> &str {
        "retrieval_knn"
    }
    fn class(&self) -> Class {
        Class::Baseline
    }
    fn rule(&self) -> String {
        format!(
            "retrieval_knn/v1: k={} cosine over context features, similarity-weighted training percentile, excluding the evaluated parent; fallback program_prior",
            self.k
        )
    }
    fn scores(&self, view: &PoolView<'_>) -> Result<Vec<f64>> {
        let query = self
            .context_features
            .rows
            .get(view.context_id)
            .with_context(|| format!("no context features for {}", view.context_id))?;
        let mut neighbours: Vec<(f64, &str, &Pool, &Vec<f64>)> = self
            .train
            .iter()
            .filter(|(p, _)| p.parent_id != view.parent_id)
            .map(|(p, t)| {
                (
                    cosine(query, &self.context_features.rows[&p.context_id]),
                    p.context_id.as_str(),
                    *p,
                    t,
                )
            })
            .collect();
        neighbours.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        neighbours.truncate(self.k);
        Ok(view
            .programs
            .iter()
            .map(|program| {
                let (mut num, mut den) = (0.0, 0.0);
                for (sim, _, pool, targets) in &neighbours {
                    let w = sim.max(0.0);
                    if let Ok(i) = pool
                        .members
                        .binary_search_by(|m| m.program_sha256.as_str().cmp(program))
                    {
                        num += w * targets[i];
                        den += w;
                    }
                }
                if den > 0.0 {
                    num / den
                } else {
                    self.prior.score(program)
                }
            })
            .collect())
    }
}

/// Simple supervised predictor: ridge regression of the training percentile on program
/// features (with intercept), solved in closed form. Context-blind.
pub struct RidgeProgram<'a> {
    lambda: f64,
    program_features: &'a Features,
    weights: Vec<f64>,
}

/// Solve `A x = b` for symmetric positive definite `A` (row-major, `n x n`) by Cholesky.
fn cholesky_solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Result<Vec<f64>> {
    ensure!(
        a.len() == n * n && b.len() == n && a.iter().chain(&b).all(|v| v.is_finite()),
        "nonfinite or malformed ridge system"
    );
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        ensure!(d > 0.0, "ridge system is not positive definite");
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
    }
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= a[k * n + i] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    ensure!(b.iter().all(|v| v.is_finite()), "nonfinite ridge solution");
    Ok(b)
}

impl<'a> RidgeProgram<'a> {
    /// Performs the fit operation and returns an error if it cannot be completed.
    pub fn fit(train: &[&Pool], program_features: &'a Features, lambda: f64) -> Result<Self> {
        ensure!(
            lambda > 0.0 && lambda.is_finite(),
            "lambda must be positive"
        );
        let d = program_features.dim + 1;
        let (mut xtx, mut xty) = (vec![0.0; d * d], vec![0.0; d]);
        let mut rows = 0usize;
        for pool in train {
            for (m, t) in pool.members.iter().zip(rank_targets(pool)) {
                let phi = program_features
                    .rows
                    .get(&m.program_sha256)
                    .with_context(|| format!("no program features for {}", m.program_sha256))?;
                let x: Vec<f64> = std::iter::once(1.0).chain(phi.iter().copied()).collect();
                for i in 0..d {
                    xty[i] += x[i] * t;
                    for j in 0..d {
                        xtx[i * d + j] += x[i] * x[j];
                    }
                }
                rows += 1;
            }
        }
        ensure!(rows > 0, "ridge needs training rows");
        // The intercept is not penalized.
        for i in 1..d {
            xtx[i * d + i] += lambda;
        }
        xtx[0] += 1e-12;
        Ok(Self {
            lambda,
            program_features,
            weights: cholesky_solve(xtx, xty, d)?,
        })
    }
}

impl Selector for RidgeProgram<'_> {
    fn name(&self) -> &str {
        "ridge_program"
    }
    fn class(&self) -> Class {
        Class::Baseline
    }
    fn rule(&self) -> String {
        format!(
            "ridge_program/v1: closed-form ridge (lambda={}, unpenalized intercept) of training percentile on {} program features",
            self.lambda, self.program_features.dim
        )
    }
    fn scores(&self, view: &PoolView<'_>) -> Result<Vec<f64>> {
        view.programs
            .iter()
            .map(|p| {
                let phi = self
                    .program_features
                    .rows
                    .get(*p)
                    .with_context(|| format!("no program features for {p}"))?;
                Ok(self.weights[0]
                    + phi
                        .iter()
                        .zip(&self.weights[1..])
                        .map(|(x, w)| x * w)
                        .sum::<f64>())
            })
            .collect()
    }
}

/// Schema identifier for external values.
pub const EXTERNAL_SCHEMA: &str = "jope.j9-external-scores/v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `ExternalScore` is a record for the external score contract in this module: Selectors over a frozen pool (CLOVER §6 Experiment A).
pub struct ExternalScore {
    /// Identifier for the associated context.
    pub context_id: String,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Measured score for this outcome; absence denotes a failed draw.
    pub score: f64,
}

/// An externally produced score document: the PolicyLM q-head adapter, a native GP ranking
/// export, or any learned selector. `provenance` (checkpoint, source and input digests) is
/// carried into the receipt verbatim.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalDocument {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Name recorded for this external document.
    pub name: String,
    /// Class recorded for this external document.
    pub class: Class,
    /// Rule recorded for this external document.
    pub rule: String,
    /// Provenance recorded for this external document.
    pub provenance: serde_json::Value,
    /// Collection of scores values carried by this external document.
    pub scores: Vec<ExternalScore>,
}

/// One score per (context, program). Missing scores are an error, never a default.
pub struct External {
    name: String,
    class: Class,
    rule: String,
    scores: BTreeMap<(String, String), f64>,
}

impl External {
    /// Performs the from document operation and returns an error if it cannot be completed.
    pub fn from_document(doc: &ExternalDocument) -> Result<Self> {
        ensure!(
            doc.schema == EXTERNAL_SCHEMA,
            "unknown external score schema {:?}",
            doc.schema
        );
        ensure!(
            !doc.name.trim().is_empty() && !doc.rule.trim().is_empty(),
            "external selector needs a name and rule"
        );
        let mut scores = BTreeMap::new();
        for (i, row) in doc.scores.iter().enumerate() {
            ensure!(
                row.score.is_finite(),
                "{}: non-finite score at {i}",
                doc.name
            );
            ensure!(
                scores
                    .insert(
                        (row.context_id.clone(), row.program_sha256.clone()),
                        row.score
                    )
                    .is_none(),
                "{}: duplicate score at {i}",
                doc.name
            );
        }
        Ok(Self {
            name: doc.name.clone(),
            class: doc.class,
            rule: doc.rule.clone(),
            scores,
        })
    }
}

impl Selector for External {
    fn name(&self) -> &str {
        &self.name
    }
    fn class(&self) -> Class {
        self.class
    }
    fn rule(&self) -> String {
        self.rule.clone()
    }
    fn scores(&self, view: &PoolView<'_>) -> Result<Vec<f64>> {
        view.programs
            .iter()
            .map(|p| {
                self.scores
                    .get(&(view.context_id.to_owned(), (*p).to_owned()))
                    .copied()
                    .with_context(|| {
                        format!("{}: no score for {} / {p}", self.name, view.context_id)
                    })
            })
            .collect()
    }
}

/// Cross-family S3 `gp_distilled/v1` program prior statistics restricted to an outer training split.
#[derive(Clone, Debug, Default)]
pub struct ExternalProgramPrior {
    /// Mapping of tot stats values carried by this external program prior.
    pub tot_stats: BTreeMap<String, (f64, usize)>,
    /// Mapping of parent stats values carried by this external program prior.
    pub parent_stats: BTreeMap<(String, String), (f64, usize)>,
}

impl ExternalProgramPrior {
    fn restrict_to(&self, train: &[&Pool]) -> Result<Self> {
        let mut total: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        for ((_, program), &(sum, count)) in &self.parent_stats {
            ensure!(
                sum.is_finite() && sum >= 0.0 && sum <= count as f64 && count > 0,
                "invalid external prior statistics"
            );
            let e = total.entry(program.clone()).or_default();
            e.0 += sum;
            e.1 += count;
        }
        ensure!(
            total.len() == self.tot_stats.len()
                && total
                    .iter()
                    .all(|(p, (s, c))| self.tot_stats.get(p).is_some_and(
                        |(t, n)| c == n && (s - t).abs() <= 1e-10 * (*c as f64).max(1.0)
                    )),
            "external prior has unattributed aggregates"
        );
        let parents: BTreeSet<_> = train.iter().map(|p| &p.parent_id).collect();
        let mut out = Self::default();
        for ((parent, program), &(sum, count)) in &self.parent_stats {
            if parents.contains(parent) {
                out.parent_stats
                    .insert((parent.clone(), program.clone()), (sum, count));
                let e = out.tot_stats.entry(program.clone()).or_default();
                e.0 += sum;
                e.1 += count;
            }
        }
        Ok(out)
    }
}

/// Contextual + program-interaction ridge surrogate (`GB-09` / `J9`) trained on dense GP trajectories.
///
/// Combines unpenalized intercept, inner-CV cross-fitted `RidgeProgram` prediction,
/// leave-one-parent-out `ProgramPrior`, full-dimensional cosine `RetrievalKnn`, hierarchical
/// Empirical-Bayes program/kNN shrinkage, program novelty gating, and standardized compact
/// bilinear context-program interactions `\tilde{z}_i(c) * \tilde{\phi}_j(p)` into a closed-form
/// ridge predictor capable of ranking unseen `N+1` candidate programs on high-dimensional (2,051-d)
/// S3 contexts without exceeding `MAX_SURROGATE_DESIGN_DIM`.
pub struct ContextualSurrogateQHat<'a> {
    lambda: f64,
    base_mult: f64,
    proj_mult: f64,
    inner_cv: CvEstimate,
    inner_fold_fits: Vec<InnerFoldFit>,
    prior_weight: f64,
    k_neighbours: usize,
    context_features: &'a Features,
    program_features: &'a Features,
    ridge: RidgeProgram<'a>,
    train_targets: Vec<(&'a Pool, Vec<f64>, Vec<f64>)>,
    tot_stats: BTreeMap<String, (f64, usize)>,
    parent_stats: BTreeMap<(String, String), (f64, usize)>,
    sos_tot_stats: BTreeMap<String, (f64, usize)>,
    sos_parent_stats: BTreeMap<(String, String), (f64, usize)>,
    external_prior: Option<ExternalProgramPrior>,
    ctx_proj: Vec<(usize, f64, f64)>,
    prog_proj: Vec<(usize, f64, f64)>,
    weights: Vec<f64>,
}

/// Configured upper or lower bound for max surrogate design dim.
pub const MAX_SURROGATE_DESIGN_DIM: usize = 512;
/// Configured upper or lower bound for max bilinear ctx dim.
pub const MAX_BILINEAR_CTX_DIM: usize = 8;
/// Configured upper or lower bound for max bilinear prog dim.
pub const MAX_BILINEAR_PROG_DIM: usize = 8;
const SURROGATE_BASE_FEATURES: usize = 13;

impl<'a> ContextualSurrogateQHat<'a> {
    /// Returns the `(base_mult, proj_mult)` ridge regularization multipliers selected by 5-fold
    /// parent-grouped inner cross-validation on `train`.
    pub fn selected_multipliers(&self) -> (f64, f64) {
        (self.base_mult, self.proj_mult)
    }

    /// Returns the mean inner cross-validation loss achieved by `selected_multipliers()` on `train`.
    pub fn inner_cv_loss(&self) -> Option<f64> {
        self.inner_cv.loss()
    }

    /// Performs the inner cross-validation estimate operation.
    pub fn inner_cv_estimate(&self) -> &CvEstimate {
        &self.inner_cv
    }

    /// Performs the inner fold fits operation.
    pub fn inner_fold_fits(&self) -> &[InnerFoldFit] {
        &self.inner_fold_fits
    }

    /// Raw unprojected bilinear design dimension (`2 + d_ctx + d_prog + d_ctx * d_prog`).
    pub fn design_dim(d_ctx: usize, d_prog: usize) -> Option<usize> {
        let cross = d_ctx.checked_mul(d_prog)?;
        2usize
            .checked_add(d_ctx)
            .and_then(|v| v.checked_add(d_prog))
            .and_then(|v| v.checked_add(cross))
    }

    /// Effective design dimension after training-fold variance projection to at most
    /// `MAX_BILINEAR_CTX_DIM x MAX_BILINEAR_PROG_DIM` bilinear coordinates.
    pub fn effective_design_dim(d_ctx: usize, d_prog: usize) -> Option<usize> {
        if d_ctx == 0 || d_prog == 0 {
            return None;
        }
        let dz = d_ctx.min(MAX_BILINEAR_CTX_DIM);
        let dp = d_prog.min(MAX_BILINEAR_PROG_DIM);
        let cross = dz.checked_mul(dp)?;
        SURROGATE_BASE_FEATURES
            .checked_add(dz)
            .and_then(|v| v.checked_add(dp))
            .and_then(|v| v.checked_add(cross))
    }

    /// Leave-one-parent-cluster-out Strength-of-Schedule (`SoS`) context-difficulty calibration.
    ///
    /// Estimates each context pool's difficulty offset `ctx_offset` from out-of-parent-cluster
    /// program rank means `mu_p` shrunk toward `0.5` with `prior_weight`, shifting all members in
    /// context `c` additively by `bt + ctx_offset` (without top/bottom clamping) so all within-pool
    /// pairwise target differences `bt_i - bt_j` are identically preserved. Single-member pools
    /// (`bt = 0.5`) evaluate to `0.5 + ctx_offset = mu_p` (`0.5` when unseen outside the parent
    /// cluster).
    fn compute_sos_targets(
        raw_targets: &[(&'a Pool, Vec<f64>)],
        prior_weight: f64,
    ) -> Vec<Vec<f64>> {
        let mut tot: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
        let mut par: BTreeMap<(&str, &str), (f64, usize)> = BTreeMap::new();
        for (pool, targets) in raw_targets {
            for (m, &t) in pool.members.iter().zip(targets) {
                let te = tot.entry(m.program_sha256.as_str()).or_insert((0.0, 0));
                te.0 += t;
                te.1 += 1;
                let pe = par
                    .entry((pool.parent_id.as_str(), m.program_sha256.as_str()))
                    .or_insert((0.0, 0));
                pe.0 += t;
                pe.1 += 1;
            }
        }
        raw_targets
            .iter()
            .map(|(pool, base_t)| {
                let nm = pool.members.len();
                if nm == 0 {
                    return Vec::new();
                }
                let diff_sum: f64 = pool
                    .members
                    .iter()
                    .map(|m| {
                        let (ts, tc) = tot
                            .get(m.program_sha256.as_str())
                            .copied()
                            .unwrap_or((0.0, 0));
                        let (ps, pc) = par
                            .get(&(pool.parent_id.as_str(), m.program_sha256.as_str()))
                            .copied()
                            .unwrap_or((0.0, 0));
                        let s_p = ts - ps;
                        let c_p = tc.saturating_sub(pc);
                        let mu_p = (s_p + 0.5 * prior_weight) / (c_p as f64 + prior_weight);
                        mu_p - 0.5
                    })
                    .sum();
                let ctx_offset = diff_sum / (nm as f64);
                base_t.iter().map(|&bt| bt + ctx_offset).collect()
            })
            .collect()
    }

    fn select_top_variance_coords(
        vectors: &[&[f64]],
        dim: usize,
        max_coords: usize,
    ) -> Vec<(usize, f64, f64)> {
        let n = vectors.len().max(1) as f64;
        let mut stats: Vec<(usize, f64, f64)> = (0..dim)
            .map(|j| {
                let mean = vectors.iter().map(|v| v[j]).sum::<f64>() / n;
                let var = vectors
                    .iter()
                    .map(|v| {
                        let d = v[j] - mean;
                        d * d
                    })
                    .sum::<f64>()
                    / n;
                (j, mean, var.sqrt())
            })
            .collect();
        stats.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        stats.truncate(dim.min(max_coords));
        stats
            .into_iter()
            .map(|(j, mean, std)| (j, mean, if std > 1e-8 { std } else { 1.0 }))
            .collect()
    }

    fn project(v: &[f64], proj: &[(usize, f64, f64)]) -> Vec<f64> {
        proj.iter()
            .map(|&(j, mean, scale)| (v[j] - mean) / scale)
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn design_row(
        zt: &[f64],
        pt: &[f64],
        r_s: f64,
        prior_s: f64,
        sos_prior_s: f64,
        s3_prior: f64,
        knn_s: f64,
        sos_knn_s: f64,
        eb_prior: f64,
        eb_sos_prior: f64,
        eb_knn: f64,
        eb_sos_knn: f64,
        nov: f64,
    ) -> Vec<f64> {
        let mut x =
            Vec::with_capacity(SURROGATE_BASE_FEATURES + zt.len() + pt.len() + zt.len() * pt.len());
        x.push(1.0);
        x.push(r_s);
        x.push(prior_s);
        x.push(sos_prior_s);
        x.push(s3_prior);
        x.push(knn_s);
        x.push(sos_knn_s);
        x.push(eb_prior);
        x.push(eb_sos_prior);
        x.push(eb_knn);
        x.push(eb_sos_knn);
        x.push(nov);
        x.push(nov * (r_s - 0.5));
        x.extend_from_slice(zt);
        x.extend_from_slice(pt);
        for &zi in zt {
            for &pj in pt {
                x.push(zi * pj);
            }
        }
        x
    }

    fn view_design_rows_with_ridge(
        &self,
        view: &PoolView<'_>,
        ridge_scores: &[f64],
    ) -> Result<Vec<Vec<f64>>> {
        ensure!(
            ridge_scores.len() == view.programs.len(),
            "ridge prediction shape mismatch"
        );
        let z = self
            .context_features
            .rows
            .get(view.context_id)
            .with_context(|| format!("no context features for {}", view.context_id))?;
        let zt = Self::project(z, &self.ctx_proj);

        type NeighbourRow<'b> = (f64, &'b str, &'b Pool, &'b [f64], &'b [f64]);
        let mut neighbours: Vec<NeighbourRow<'_>> = self
            .train_targets
            .iter()
            .filter(|(p, _, _)| p.parent_id != view.parent_id)
            .map(|(p, t, st)| {
                (
                    cosine(z, &self.context_features.rows[&p.context_id]),
                    p.context_id.as_str(),
                    *p,
                    t.as_slice(),
                    st.as_slice(),
                )
            })
            .collect();
        neighbours.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        neighbours.truncate(self.k_neighbours);

        let mut rows = Vec::with_capacity(view.programs.len());
        for (&prog_sha, &r_s) in view.programs.iter().zip(ridge_scores) {
            let phi = self
                .program_features
                .rows
                .get(prog_sha)
                .with_context(|| format!("no program features for {prog_sha}"))?;
            let pt = Self::project(phi, &self.prog_proj);
            let (tot_s, tot_c) = self.tot_stats.get(prog_sha).copied().unwrap_or((0.0, 0));
            let (par_s, par_c) = self
                .parent_stats
                .get(&(view.parent_id.to_owned(), prog_sha.to_owned()))
                .copied()
                .unwrap_or((0.0, 0));
            let s_p = tot_s - par_s;
            let c_p = tot_c.saturating_sub(par_c);

            let (sos_tot_s, _) = self
                .sos_tot_stats
                .get(prog_sha)
                .copied()
                .unwrap_or((0.0, 0));
            let (sos_par_s, _) = self
                .sos_parent_stats
                .get(&(view.parent_id.to_owned(), prog_sha.to_owned()))
                .copied()
                .unwrap_or((0.0, 0));
            let sos_s_p = sos_tot_s - sos_par_s;

            let s3_prior = if let Some(ext) = &self.external_prior {
                let (es, ec) = ext.tot_stats.get(prog_sha).copied().unwrap_or((0.0, 0));
                let (eps, epc) = ext
                    .parent_stats
                    .get(&(view.parent_id.to_owned(), prog_sha.to_owned()))
                    .copied()
                    .unwrap_or((0.0, 0));
                let sc = ec.saturating_sub(epc);
                let ss = (es - eps).clamp(0.0, sc as f64);
                (ss + 0.5 * self.prior_weight) / (sc as f64 + self.prior_weight)
            } else {
                0.5
            };
            let prior_s = (s_p + 0.5 * self.prior_weight) / (c_p as f64 + self.prior_weight);
            let sos_prior_s =
                (sos_s_p + 0.5 * self.prior_weight) / (c_p as f64 + self.prior_weight);
            let eb_prior = (s_p + self.prior_weight * r_s) / (c_p as f64 + self.prior_weight);
            let eb_sos_prior =
                (sos_s_p + self.prior_weight * r_s) / (c_p as f64 + self.prior_weight);

            let (mut num, mut sos_num, mut den) = (0.0, 0.0, 0.0);
            for (sim, _, pool, targets, sos_targets) in &neighbours {
                let w = sim.max(0.0);
                if let Ok(i) = pool
                    .members
                    .binary_search_by(|m| m.program_sha256.as_str().cmp(prog_sha))
                {
                    num += w * targets[i];
                    sos_num += w * sos_targets[i];
                    den += w;
                }
            }
            let knn_s = if den > 0.0 { num / den } else { prior_s };
            let sos_knn_s = if den > 0.0 {
                sos_num / den
            } else {
                sos_prior_s
            };
            let eb_knn = (num + self.prior_weight * eb_prior) / (den + self.prior_weight);
            let eb_sos_knn =
                (sos_num + self.prior_weight * eb_sos_prior) / (den + self.prior_weight);
            let nov = 1.0 / (1.0 + c_p as f64);
            rows.push(Self::design_row(
                &zt,
                &pt,
                r_s,
                prior_s,
                sos_prior_s,
                s3_prior,
                knn_s,
                sos_knn_s,
                eb_prior,
                eb_sos_prior,
                eb_knn,
                eb_sos_knn,
                nov,
            ));
        }
        Ok(rows)
    }

    fn view_design_rows(&self, view: &PoolView<'_>) -> Result<Vec<Vec<f64>>> {
        let ridge_scores = self.ridge.scores(view)?;
        self.view_design_rows_with_ridge(view, &ridge_scores)
    }

    /// Performs the fit operation and returns an error if it cannot be completed.
    pub fn fit(
        train: &[&'a Pool],
        context_features: &'a Features,
        program_features: &'a Features,
        k_neighbours: usize,
        prior_weight: f64,
        lambda: f64,
    ) -> Result<Self> {
        Self::fit_with_prior(
            train,
            context_features,
            program_features,
            k_neighbours,
            prior_weight,
            lambda,
            None,
        )
    }

    fn fit_state(
        train: &[&'a Pool],
        context_features: &'a Features,
        program_features: &'a Features,
        k_neighbours: usize,
        prior_weight: f64,
        lambda: f64,
        external_prior: Option<&ExternalProgramPrior>,
    ) -> Result<Self> {
        ensure!(k_neighbours >= 1, "k_neighbours must be positive");
        ensure!(
            prior_weight > 0.0 && prior_weight.is_finite(),
            "prior_weight must be positive and finite"
        );
        ensure!(
            lambda > 0.0 && lambda.is_finite(),
            "lambda must be positive and finite"
        );
        ensure!(
            context_features.dim > 0 && program_features.dim > 0,
            "non-empty context and program feature dimensions required"
        );
        let d = Self::effective_design_dim(context_features.dim, program_features.dim)
            .context("design dimension overflow")?;
        ensure!(
            d <= MAX_SURROGATE_DESIGN_DIM,
            "effective design dimension {d} exceeds MAX_SURROGATE_DESIGN_DIM ({MAX_SURROGATE_DESIGN_DIM})"
        );

        let ridge = RidgeProgram::fit(train, program_features, lambda)?;
        let mut raw_train_targets: Vec<(&'a Pool, Vec<f64>)> = Vec::with_capacity(train.len());
        let mut z_vecs: Vec<&[f64]> = Vec::with_capacity(train.len());
        let mut p_vecs: Vec<&[f64]> = Vec::new();
        for &pool in train {
            let z = context_features
                .rows
                .get(&pool.context_id)
                .with_context(|| format!("no context features for {}", pool.context_id))?;
            ensure!(
                z.len() == context_features.dim && z.iter().all(|v| v.is_finite()),
                "context feature shape/nonfinite"
            );
            z_vecs.push(z.as_slice());
            let targets = rank_targets(pool);
            for m in &pool.members {
                let phi = program_features
                    .rows
                    .get(&m.program_sha256)
                    .with_context(|| format!("no program features for {}", m.program_sha256))?;
                ensure!(
                    phi.len() == program_features.dim && phi.iter().all(|v| v.is_finite()),
                    "program feature shape/nonfinite"
                );
                p_vecs.push(phi.as_slice());
            }
            raw_train_targets.push((pool, targets));
        }
        ensure!(
            !p_vecs.is_empty(),
            "contextual surrogate q_hat needs training rows"
        );

        let sos_all = Self::compute_sos_targets(&raw_train_targets, prior_weight);
        let mut train_targets = Vec::with_capacity(train.len());
        let mut tot_stats: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        let mut parent_stats: BTreeMap<(String, String), (f64, usize)> = BTreeMap::new();
        let mut sos_tot_stats: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        let mut sos_parent_stats: BTreeMap<(String, String), (f64, usize)> = BTreeMap::new();

        for ((pool, targets), sos_targets) in raw_train_targets.into_iter().zip(sos_all) {
            for ((m, &t), &tsos) in pool.members.iter().zip(&targets).zip(&sos_targets) {
                let tot = tot_stats
                    .entry(m.program_sha256.clone())
                    .or_insert((0.0, 0));
                tot.0 += t;
                tot.1 += 1;
                let par = parent_stats
                    .entry((pool.parent_id.clone(), m.program_sha256.clone()))
                    .or_insert((0.0, 0));
                par.0 += t;
                par.1 += 1;

                let stot = sos_tot_stats
                    .entry(m.program_sha256.clone())
                    .or_insert((0.0, 0));
                stot.0 += tsos;
                stot.1 += 1;
                let spar = sos_parent_stats
                    .entry((pool.parent_id.clone(), m.program_sha256.clone()))
                    .or_insert((0.0, 0));
                spar.0 += tsos;
                spar.1 += 1;
            }
            train_targets.push((pool, targets, sos_targets));
        }

        let ctx_proj =
            Self::select_top_variance_coords(&z_vecs, context_features.dim, MAX_BILINEAR_CTX_DIM);
        let prog_proj =
            Self::select_top_variance_coords(&p_vecs, program_features.dim, MAX_BILINEAR_PROG_DIM);

        let model = Self {
            lambda,
            base_mult: 1.0,
            proj_mult: 1.0,
            inner_cv: CvEstimate::Unavailable {
                procedure: NESTED_PROCEDURE.into(),
                reason: CvUnavailableReason::EmptyUsableFolds,
                detail: "state only; not a fitted selector".into(),
                counts: CvCounts::default(),
            },
            inner_fold_fits: Vec::new(),
            prior_weight,
            k_neighbours,
            context_features,
            program_features,
            ridge,
            train_targets,
            tot_stats,
            parent_stats,
            sos_tot_stats,
            sos_parent_stats,
            external_prior: external_prior.map(|p| p.restrict_to(train)).transpose()?,
            ctx_proj,
            prog_proj,
            weights: vec![0.0; d],
        };

        Ok(model)
    }

    /// Refit target-derived state in each training-row complement, while sharing the enclosing
    /// training closure's target-free projection basis. No validation labels enter this matrix.
    fn training_equations(
        &self,
        train: &[&'a Pool],
        external: Option<&ExternalProgramPrior>,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let d = self.weights.len();
        let (mut xtx, mut xty) = (vec![0.0; d * d], vec![0.0; d]);
        let mut cross_fits = BTreeMap::new();
        for parent in evidence_parent_ids(train) {
            let complement: Vec<_> = train
                .iter()
                .copied()
                .filter(|p| p.parent_id != parent)
                .collect();
            ensure!(
                !complement.is_empty(),
                "base cross-fitting needs another parent"
            );
            let mut state = Self::fit_state(
                &complement,
                self.context_features,
                self.program_features,
                self.k_neighbours,
                self.prior_weight,
                self.lambda,
                external,
            )?;
            state.ctx_proj = self.ctx_proj.clone();
            state.prog_proj = self.prog_proj.clone();
            cross_fits.insert(parent, state);
        }
        for (pool, _, targets) in &self.train_targets {
            let state = &cross_fits[pool.parent_id.as_str()];
            let rows = state.view_design_rows(&pool.view())?;
            ensure!(
                rows.len() == targets.len(),
                "training prediction shape mismatch"
            );
            for (x, &t) in rows.iter().zip(targets) {
                ensure!(
                    x.len() == d && t.is_finite() && x.iter().all(|v| v.is_finite()),
                    "nonfinite training design"
                );
                for i in 0..d {
                    xty[i] += x[i] * t;
                    for j in 0..d {
                        xtx[i * d + j] += x[i] * x[j];
                    }
                }
            }
        }
        Ok((xtx, xty))
    }

    fn solve_weights(
        &self,
        mut xtx: Vec<f64>,
        xty: Vec<f64>,
        bm: f64,
        pm: f64,
    ) -> Result<Vec<f64>> {
        let d = self.weights.len();
        let lin_terms = (self.ctx_proj.len() + self.prog_proj.len()).max(1);
        let cross_terms = (self.ctx_proj.len() * self.prog_proj.len()).max(1);
        let lin_end = SURROGATE_BASE_FEATURES + self.ctx_proj.len() + self.prog_proj.len();
        xtx[0] += 1e-12;
        for i in 1..d {
            let penalty = if i < SURROGATE_BASE_FEATURES {
                self.lambda * bm
            } else if i < lin_end {
                self.lambda * pm * lin_terms as f64
            } else {
                self.lambda * pm * cross_terms as f64
            };
            xtx[i * d + i] += penalty;
        }
        ensure!(
            xtx.iter().chain(&xty).all(|v| v.is_finite()),
            "nonfinite normal equations"
        );
        let weights = cholesky_solve(xtx, xty, d)?;
        ensure!(weights.iter().all(|v| v.is_finite()), "nonfinite weights");
        Ok(weights)
    }

    fn state_sha256(&self) -> Result<String> {
        // Include every learned component, descriptor projection and filtered external prior.
        canonical_sha256(&(
            self.ridge.weights.clone(),
            &self.ctx_proj,
            &self.prog_proj,
            &self.tot_stats,
            self.parent_stats.iter().collect::<Vec<_>>(),
            &self.sos_tot_stats,
            self.sos_parent_stats.iter().collect::<Vec<_>>(),
            self.train_targets
                .iter()
                .map(|(p, t, s)| (&p.context_id, t, s))
                .collect::<Vec<_>>(),
            self.external_prior
                .as_ref()
                .map(|p| (&p.tot_stats, p.parent_stats.iter().collect::<Vec<_>>())),
        ))
    }

    /// Performs the fit with prior operation and returns an error if it cannot be completed.
    pub fn fit_with_prior(
        train: &[&'a Pool],
        context_features: &'a Features,
        program_features: &'a Features,
        k_neighbours: usize,
        prior_weight: f64,
        lambda: f64,
        external_prior: Option<&ExternalProgramPrior>,
    ) -> Result<Self> {
        let parents: BTreeSet<_> = train.iter().map(|p| p.parent_id.as_str()).collect();
        let folds: Vec<_> = train
            .iter()
            .map(|p| super::evaluate::fold_of(INNER_SALT, &p.parent_id, INNER_FOLDS))
            .collect();
        let mut counts = CvCounts {
            attempted_units: train.iter().map(|p| p.members.len()).sum(),
            parents: parents.len(),
            folds: folds.iter().copied().collect::<BTreeSet<_>>().len(),
            ..CvCounts::default()
        };
        if parents.len() < 3 {
            return Err(unavailable(
                CvUnavailableReason::InsufficientParents,
                "nested OOF needs at least three parents",
                counts,
            ));
        }
        if counts.attempted_units == 0 {
            return Err(unavailable(
                CvUnavailableReason::EmptyUsableFolds,
                "no member rows",
                counts,
            ));
        }
        if train
            .iter()
            .any(|p| p.members.len() < 2 || p.members.iter().all(|m| m.outcome().is_none()))
        {
            return Err(unavailable(
                CvUnavailableReason::EmptyUsableFolds,
                "tiny or wholly failed pool has no usable ranking target",
                counts,
            ));
        }
        if train
            .iter()
            .flat_map(|p| &p.members)
            .filter_map(|m| m.q)
            .any(|v| !v.is_finite())
        {
            return Err(unavailable(
                CvUnavailableReason::NonfiniteSystem,
                "nonfinite measured outcome",
                counts,
            ));
        }
        for features in [context_features, program_features] {
            if features.dim == 0
                || features.rows.is_empty()
                || features.rows.values().any(|v| v.len() != features.dim)
            {
                return Err(unavailable(
                    CvUnavailableReason::ShapeMismatch,
                    "feature width mismatch",
                    counts,
                ));
            }
            if features.rows.values().flatten().any(|v| !v.is_finite()) {
                return Err(unavailable(
                    CvUnavailableReason::NonfiniteSystem,
                    "nonfinite input features",
                    counts,
                ));
            }
        }
        context_features.verify_context_descriptor().map_err(|e| {
            unavailable(
                CvUnavailableReason::UnboundContextProvenance,
                e,
                counts.clone(),
            )
        })?;
        if let Some(prior) = external_prior {
            prior.restrict_to(train).map_err(|e| {
                unavailable(
                    CvUnavailableReason::ExternalPriorProvenance,
                    e,
                    counts.clone(),
                )
            })?;
        }
        if counts.folds < 2 {
            return Err(unavailable(
                CvUnavailableReason::EmptyUsableFolds,
                "all parents collide in one fold",
                counts,
            ));
        }
        // Recompute every component and both train/held design blocks inside each inner closure.
        let mut closures = Vec::new();
        let mut evidence = Vec::new();
        for fold in 0..INNER_FOLDS {
            let held: Vec<_> = train
                .iter()
                .copied()
                .zip(&folds)
                .filter_map(|(p, &f)| (f == fold).then_some(p))
                .collect();
            if held.is_empty() {
                continue;
            }
            let fitting: Vec<_> = train
                .iter()
                .copied()
                .zip(&folds)
                .filter_map(|(p, &f)| (f != fold).then_some(p))
                .collect();
            if fitting
                .iter()
                .map(|p| &p.parent_id)
                .collect::<BTreeSet<_>>()
                .len()
                < 2
            {
                return Err(unavailable(
                    CvUnavailableReason::InsufficientParents,
                    "inner training closure cannot cross-fit its base components",
                    counts,
                ));
            }
            let state = Self::fit_state(
                &fitting,
                context_features,
                program_features,
                k_neighbours,
                prior_weight,
                lambda,
                external_prior,
            )
            .map_err(|e| component_error(e, counts.clone()))?;
            let (xtx, xty) = state
                .training_equations(&fitting, external_prior)
                .map_err(|e| component_error(e, counts.clone()))?;
            let designs = held
                .iter()
                .map(|p| state.view_design_rows(&p.view()))
                .collect::<Result<Vec<_>>>()
                .map_err(|e| component_error(e, counts.clone()))?;
            evidence.push(InnerFoldFit {
                fold,
                training_parents: fitting
                    .iter()
                    .map(|p| p.parent_id.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect(),
                held_parents: held
                    .iter()
                    .map(|p| p.parent_id.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect(),
                training_input_sha256: canonical_sha256(
                    &fitting
                        .iter()
                        .map(|p| p.sha256())
                        .collect::<Result<Vec<_>>>()?,
                )?,
                training_features_sha256: canonical_sha256(
                    &fitting
                        .iter()
                        .map(|p| {
                            (
                                &context_features.rows[&p.context_id],
                                p.members
                                    .iter()
                                    .map(|m| &program_features.rows[&m.program_sha256])
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>(),
                )?,
                split_sha256: canonical_sha256(&(
                    NESTED_PROCEDURE,
                    fold,
                    &evidence_parent_ids(&fitting),
                    &evidence_parent_ids(&held),
                ))?,
                state_sha256: state.state_sha256()?,
                held_design_sha256: canonical_sha256(&designs)?,
                prediction_sha256: Vec::new(),
                selected_prediction_bits: Vec::new(),
            });
            closures.push((state, xtx, xty, held, designs));
        }
        let mut best: Option<(f64, f64, f64, usize)> = None;
        for bm in [1.0, 10.0, 100.0, 1000.0] {
            for pm in [1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0] {
                if pm < bm {
                    continue;
                }
                let mut sq_err = 0.0;
                let mut scored = 0;
                let mut candidate_bits = Vec::new();
                let mut valid = true;
                for (fold_index, (state, xtx, xty, held, designs)) in closures.iter().enumerate() {
                    let weights = match state.solve_weights(xtx.clone(), xty.clone(), bm, pm) {
                        Ok(w) => w,
                        Err(error) => {
                            counts.failure_details.push(format!(
                                "fold={},bm={bm},pm={pm}: {error}",
                                evidence[fold_index].fold
                            ));
                            valid = false;
                            continue;
                        }
                    };
                    let predictions: Vec<Vec<f64>> = designs
                        .iter()
                        .map(|rows| {
                            rows.iter()
                                .map(|x| x.iter().zip(&weights).map(|(x, w)| x * w).sum())
                                .collect()
                        })
                        .collect();
                    if predictions.iter().flatten().any(|v| !v.is_finite()) {
                        counts.failure_details.push(format!(
                            "fold={},bm={bm},pm={pm}: nonfinite held predictions",
                            evidence[fold_index].fold
                        ));
                        valid = false;
                        continue;
                    }
                    // Predictions are fixed before touching held outcomes.
                    evidence[fold_index]
                        .prediction_sha256
                        .push(canonical_sha256(&(format!("{bm},{pm}"), &predictions))?);
                    candidate_bits.push(
                        predictions
                            .iter()
                            .map(|row| {
                                row.iter()
                                    .map(|p| format!("{:016x}", p.to_bits()))
                                    .collect()
                            })
                            .collect(),
                    );
                    for (pool, preds) in held.iter().zip(predictions) {
                        let targets = rank_targets(pool);
                        if preds.len() != targets.len() {
                            return Err(unavailable(
                                CvUnavailableReason::ShapeMismatch,
                                "held prediction width",
                                counts,
                            ));
                        }
                        if preds.is_empty() {
                            continue;
                        }
                        let n = preds.len() as f64;
                        let pmean = preds.iter().sum::<f64>() / n;
                        let tmean = targets.iter().sum::<f64>() / n;
                        for (p, t) in preds.iter().zip(targets) {
                            sq_err += ((p - pmean) - (t - tmean)).powi(2);
                            scored += 1;
                        }
                    }
                }
                if !valid || scored == 0 || !sq_err.is_finite() {
                    counts.failed_candidates += 1;
                    continue;
                }
                let loss = sq_err / scored as f64;
                if best.is_none_or(|b| loss < b.2) {
                    best = Some((bm, pm, loss, scored));
                    for (record, bits) in evidence.iter_mut().zip(candidate_bits) {
                        record.selected_prediction_bits = bits;
                    }
                }
            }
        }
        let Some((bm, pm, loss, scored)) = best else {
            return Err(unavailable(
                CvUnavailableReason::NoFiniteCandidate,
                "no candidate produced a finite complete OOF loss",
                counts,
            ));
        };
        counts.scored_units = scored;
        counts.excluded_units = counts.attempted_units - scored;
        let mut model = Self::fit_state(
            train,
            context_features,
            program_features,
            k_neighbours,
            prior_weight,
            lambda,
            external_prior,
        )
        .map_err(|e| component_error(e, counts.clone()))?;
        let (xtx, xty) = model
            .training_equations(train, external_prior)
            .map_err(|e| component_error(e, counts.clone()))?;
        model.weights = model
            .solve_weights(xtx, xty, bm, pm)
            .map_err(|e| component_error(e, counts.clone()))?;
        model.base_mult = bm;
        model.proj_mult = pm;
        model.inner_cv = CvEstimate::Supported {
            procedure: NESTED_PROCEDURE.into(),
            metric: ExactCvLoss {
                f64_bits: format!("{:016x}", loss.to_bits()),
            },
            counts,
        };
        model.inner_fold_fits = evidence;
        Ok(model)
    }
}

impl Selector for ContextualSurrogateQHat<'_> {
    fn name(&self) -> &str {
        "contextual_surrogate_qhat"
    }
    fn class(&self) -> Class {
        Class::Candidate
    }
    fn rule(&self) -> String {
        format!(
            "contextual_surrogate_qhat/v3: strict parent-nested refit; context-id-bytes/v1; inner-CV LOPO SoS-debiased Empirical-Bayes + compact bilinear context x program ridge (lambda={}, base_mult={}, proj_mult={}, inner_cv={:?}, d_ctx={}->{}, d_prog={}->{}) + kNN/ridge stack (shadow-only candidate; no multiplicity adjustment)",
            self.lambda,
            self.base_mult,
            self.proj_mult,
            self.inner_cv,
            self.context_features.dim,
            self.ctx_proj.len(),
            self.program_features.dim,
            self.prog_proj.len()
        )
    }
    fn scores(&self, view: &PoolView<'_>) -> Result<Vec<f64>> {
        let rows = self.view_design_rows(view)?;
        let scores: Vec<f64> = rows
            .into_iter()
            .map(|x| x.iter().zip(&self.weights).map(|(xi, wi)| xi * wi).sum())
            .collect();
        ensure!(
            scores.len() == view.programs.len() && scores.iter().all(|s| s.is_finite()),
            "contextual surrogate produced nonfinite or malformed predictions"
        );
        Ok(scores)
    }
}

/// Two-stage proposal funnel (`K -> B`, e.g. `K=64 -> B=8`) that pre-screens candidate proposals
/// using a `Selector` before running expensive 2-fold CV evaluation and Rust parity compilation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunnelAllocation {
    /// Identifier for the associated context.
    pub context_id: String,
    /// Selector name recorded for this funnel allocation.
    pub selector_name: String,
    /// Proposal pool k recorded for this funnel allocation.
    pub proposal_pool_k: usize,
    /// Eval budget b recorded for this funnel allocation.
    pub eval_budget_b: usize,
    /// SHA-256 digest identifying the associated selected program.
    pub selected_program_sha256: Vec<String>,
    /// Collection of selected scores values carried by this funnel allocation.
    pub selected_scores: Vec<f64>,
}

/// Performs the pre screen funnel operation and returns an error if it cannot be completed.
pub fn pre_screen_funnel(
    view: &PoolView<'_>,
    selector: &dyn Selector,
    proposal_pool_k: usize,
    eval_budget_b: usize,
) -> Result<FunnelAllocation> {
    ensure!(
        eval_budget_b >= 1 && proposal_pool_k >= eval_budget_b,
        "funnel requires 1 <= eval_budget_b <= proposal_pool_k"
    );
    ensure!(
        !view.programs.is_empty(),
        "cannot pre-screen an empty candidate view"
    );
    let scores = selector.scores(view)?;
    ensure!(
        scores.len() == view.programs.len() && scores.iter().all(|s| s.is_finite()),
        "selector must return one finite score per view program"
    );
    let mut seen = BTreeSet::new();
    let mut ranked: Vec<(f64, &str)> = scores
        .into_iter()
        .zip(view.programs.iter().copied())
        .filter(|(_, prog_sha)| seen.insert(*prog_sha))
        .take(proposal_pool_k)
        .collect();
    let capped_k = ranked.len();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
    ranked.truncate(eval_budget_b);
    Ok(FunnelAllocation {
        context_id: view.context_id.to_owned(),
        selector_name: selector.name().to_owned(),
        proposal_pool_k: capped_k,
        eval_budget_b: ranked.len(),
        selected_program_sha256: ranked.iter().map(|(_, p)| (*p).to_owned()).collect(),
        selected_scores: ranked.iter().map(|(s, _)| *s).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::pool::{Member, Pool, Status, POOL_SCHEMA};
    use super::*;

    #[test]
    fn cholesky_solves_a_small_system() {
        // [[4,2],[2,3]] x = [2,1] -> x = [0.5, 0]
        let x = cholesky_solve(vec![4.0, 2.0, 2.0, 3.0], vec![2.0, 1.0], 2).unwrap();
        assert!((x[0] - 0.5).abs() < 1e-12 && x[1].abs() < 1e-12);
    }

    #[test]
    fn contextual_surrogate_qhat_learns_context_program_interactions_and_screens_funnel() {
        // Construct 4 training pools where contexts with z > 0 reward token 1 and contexts with z < 0 reward token 2.
        // Marginal popularity of token 1 and token 2 is identical (0.5 each), so context-blind selectors tie at 0.5,
        // while ContextualSurrogateQHat learns the bilinear z * phi interaction and ranks the held-out pool accurately.
        let mk_hex = |b: u8| format!("{b:02x}").repeat(32);
        let p_pos_a = mk_hex(0x11);
        let p_neg_a = mk_hex(0x22);
        let p_pos_b = mk_hex(0x33);
        let p_neg_b = mk_hex(0x44);

        let mk_pool = |cid: u8, parent: &str, pos_wins: bool, use_b: bool| -> Pool {
            let (p1, p2) = if use_b {
                (p_pos_b.clone(), p_neg_b.clone())
            } else {
                (p_pos_a.clone(), p_neg_a.clone())
            };
            let (q1, q2) = if pos_wins { (0.9, 0.1) } else { (0.1, 0.9) };
            Pool {
                schema: POOL_SCHEMA.into(),
                context_id: format!(
                    "{}{}",
                    if cid % 2 == 1 { "ff" } else { "00" },
                    &mk_hex(cid)[2..]
                ),
                parent_id: parent.into(),
                role: "train".into(),
                protocol_sha256: mk_hex(0xaa),
                incumbent_q: Some(0.0),
                members: vec![
                    Member {
                        program_sha256: p1,
                        status: Status::Complete,
                        q: Some(q1),
                        tokens: Some(vec![1, 1, 0]),
                    },
                    Member {
                        program_sha256: p2,
                        status: Status::Complete,
                        q: Some(q2),
                        tokens: Some(vec![2, 2, 0]),
                    },
                ],
            }
        };

        let all_pools = vec![
            mk_pool(1, "p1", true, false),
            mk_pool(2, "p2", false, false),
            mk_pool(3, "p3", true, true),
            mk_pool(4, "p4", false, true),
            mk_pool(5, "p5", true, true),
        ];
        let prog_feats = token_bag_features(&all_pools).unwrap();
        let ctx_feats = Features::context_id_descriptors(&all_pools, 1).unwrap();

        let train_refs = vec![&all_pools[0], &all_pools[1], &all_pools[2], &all_pools[3]];
        let qhat = ContextualSurrogateQHat::fit(&train_refs, &ctx_feats, &prog_feats, 2, 2.0, 1e-2)
            .unwrap();
        let (bm, pm) = qhat.selected_multipliers();
        assert!(
            [1.0, 10.0, 100.0, 1000.0].contains(&bm)
                && [1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0].contains(&pm)
                && pm >= bm
                && qhat
                    .inner_cv_loss()
                    .is_some_and(|v| v.is_finite() && v >= 0.0),
            "expected inner-CV selected multipliers in grid with pm >= bm, got ({bm}, {pm})"
        );
        assert!(qhat
            .rule()
            .contains(&format!("base_mult={bm}, proj_mult={pm}")));
        let single =
            ContextualSurrogateQHat::fit(&[&all_pools[0]], &ctx_feats, &prog_feats, 2, 2.0, 1e-2);
        assert!(single.is_err());

        let test_view = all_pools[4].view();
        let scores = qhat.scores(&test_view).unwrap();
        assert!(
            scores[0] > scores[1],
            "expected positive-z context to rank token-1 program higher: {scores:?}"
        );

        let funnel = pre_screen_funnel(&test_view, &qhat, 64, 1).unwrap();
        assert_eq!(funnel.selected_program_sha256, vec![p_pos_b.clone()]);

        // Verify Strength-of-Schedule (SoS) target calibration: when `p_pos_a` beats `p_neg_a` in two pools
        // (`t(p_pos_a) = 1.0`, `t(p_neg_a) = 0.0`), a program `p_pos_b` that ties `0.5` against elite `p_pos_a`
        // receives a strictly higher SoS-calibrated target than `p_neg_b` tying `0.5` against weak `p_neg_a`,
        // and non-tied pools with nonzero `ctx_offset` preserve their exact 1.0 winner-loser gap without clamping.
        let sos_pools = [
            mk_pool(10, "sp1", true, false),
            mk_pool(11, "sp2", true, false),
            Pool {
                schema: POOL_SCHEMA.into(),
                context_id: mk_hex(12),
                parent_id: "sp3".into(),
                role: "train".into(),
                protocol_sha256: mk_hex(0xaa),
                incumbent_q: Some(0.0),
                members: vec![
                    Member {
                        program_sha256: p_pos_a.clone(),
                        status: Status::Complete,
                        q: Some(0.85),
                        tokens: None,
                    },
                    Member {
                        program_sha256: p_pos_b.clone(),
                        status: Status::Complete,
                        q: Some(0.85),
                        tokens: None,
                    },
                ],
            },
            Pool {
                schema: POOL_SCHEMA.into(),
                context_id: mk_hex(13),
                parent_id: "sp4".into(),
                role: "train".into(),
                protocol_sha256: mk_hex(0xaa),
                incumbent_q: Some(0.0),
                members: vec![
                    Member {
                        program_sha256: p_neg_a.clone(),
                        status: Status::Complete,
                        q: Some(0.15),
                        tokens: None,
                    },
                    Member {
                        program_sha256: p_neg_b.clone(),
                        status: Status::Complete,
                        q: Some(0.15),
                        tokens: None,
                    },
                ],
            },
            Pool {
                schema: POOL_SCHEMA.into(),
                context_id: mk_hex(15),
                parent_id: "sp6".into(),
                role: "train".into(),
                protocol_sha256: mk_hex(0xaa),
                incumbent_q: Some(0.0),
                members: vec![
                    Member {
                        program_sha256: p_pos_a.clone(),
                        status: Status::Complete,
                        q: Some(0.95),
                        tokens: None,
                    },
                    Member {
                        program_sha256: p_pos_b.clone(),
                        status: Status::Complete,
                        q: Some(0.75),
                        tokens: None,
                    },
                ],
            },
        ];
        let raw_sos: Vec<(&Pool, Vec<f64>)> =
            sos_pools.iter().map(|p| (p, rank_targets(p))).collect();
        let calibrated = ContextualSurrogateQHat::compute_sos_targets(&raw_sos, 2.0);
        assert!(
            calibrated[2][1] > 0.5 && calibrated[3][1] < 0.5 && calibrated[2][1] > calibrated[3][1],
            "expected SoS calibration to boost target against elite opponent ({}) above weak opponent ({})",
            calibrated[2][1],
            calibrated[3][1]
        );
        assert!(
            (calibrated[2][0] - calibrated[2][1]).abs() < 1e-12,
            "expected equal-q members in the same context pool to retain identical calibrated targets"
        );
        assert!(
            calibrated[4][0] > 1.0 && (calibrated[4][0] - calibrated[4][1] - 1.0).abs() < 1e-12,
            "expected non-tied pool with positive ctx_offset to preserve exact 1.0 winner-loser gap: {:?}",
            calibrated[4]
        );
        let single_pool = Pool {
            schema: POOL_SCHEMA.into(),
            context_id: mk_hex(14),
            parent_id: "sp5".into(),
            role: "train".into(),
            protocol_sha256: mk_hex(0xaa),
            incumbent_q: Some(0.0),
            members: vec![Member {
                program_sha256: mk_hex(0x99),
                status: Status::Complete,
                q: Some(0.5),
                tokens: None,
            }],
        };
        let single_cal = ContextualSurrogateQHat::compute_sos_targets(
            &[(&single_pool, rank_targets(&single_pool))],
            2.0,
        );
        assert_eq!(single_cal, vec![vec![0.5]]);
        let seen_single_pool = Pool {
            schema: POOL_SCHEMA.into(),
            context_id: mk_hex(16),
            parent_id: "sp7".into(),
            role: "train".into(),
            protocol_sha256: mk_hex(0xaa),
            incumbent_q: Some(0.0),
            members: vec![Member {
                program_sha256: p_pos_a.clone(),
                status: Status::Complete,
                q: Some(0.5),
                tokens: None,
            }],
        };
        let seen_single_cal = ContextualSurrogateQHat::compute_sos_targets(
            &[
                (&sos_pools[0], rank_targets(&sos_pools[0])),
                (&seen_single_pool, rank_targets(&seen_single_pool)),
            ],
            2.0,
        );
        assert!(
            (seen_single_cal[1][0] - (1.0 + 0.5 * 2.0) / (1.0 + 2.0)).abs() < 1e-12,
            "expected single-member pool with out-of-parent history to equal mu_p=2/3, got {}",
            seen_single_cal[1][0]
        );
    }

    #[test]
    fn pre_screen_funnel_deduplicates_before_proposal_pool_truncation() {
        struct FixedScores(Vec<f64>);
        impl Selector for FixedScores {
            fn name(&self) -> &str {
                "fixed"
            }
            fn class(&self) -> Class {
                Class::Candidate
            }
            fn rule(&self) -> String {
                "fixed".into()
            }
            fn scores(&self, _view: &PoolView<'_>) -> Result<Vec<f64>> {
                Ok(self.0.clone())
            }
        }

        let mk_hex = |b: u8| format!("{b:02x}").repeat(32);
        let p1 = mk_hex(0x01);
        let p2 = mk_hex(0x02);
        let p3 = mk_hex(0x03);
        // Feed 3 duplicate entries of p1 first, followed by p2 (highest score) and p3.
        // With proposal_pool_k = 2 and eval_budget_b = 2, deduplication BEFORE .take(proposal_pool_k)
        // must retain p1 and p2 rather than starving the proposal pool to 1 distinct program.
        let programs = vec![
            p1.as_str(),
            p1.as_str(),
            p1.as_str(),
            p2.as_str(),
            p3.as_str(),
        ];
        let view = PoolView {
            context_id: &mk_hex(0xaa),
            parent_id: "parent",
            incumbent_q: None,
            programs,
            tokens: vec![None; 5],
        };
        let selector = FixedScores(vec![0.2, 0.2, 0.2, 0.9, 0.5]);
        let funnel = pre_screen_funnel(&view, &selector, 2, 2).unwrap();
        assert_eq!(funnel.proposal_pool_k, 2);
        assert_eq!(funnel.eval_budget_b, 2);
        assert_eq!(funnel.selected_program_sha256, vec![p2, p1]);
    }
}
