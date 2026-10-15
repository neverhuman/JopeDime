//! Selectors over a frozen pool (CLOVER §6 Experiment A).
//!
//! A selector scores the members of a `PoolView`, which carries no outcomes. Fitted selectors
//! learn only from the training pools of their cross-fitting fold. Every selector records its
//! exact rule; deterministic selectors are never given a propensity. The uniform baseline is not
//! a `Selector`: its expected metrics are computed exactly in `metrics::uniform_context`.

use super::pool::{Pool, PoolView};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Whether a selector is a simple baseline or a learned candidate under test. A candidate passes
/// only against the strongest baseline, never by beating uniform alone (CLOVER §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    Baseline,
    Candidate,
}

pub trait Selector {
    fn name(&self) -> &str;
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
    pub dim: usize,
    pub rows: BTreeMap<String, Vec<f64>>,
}

/// One line of a feature file.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureLine {
    pub id: String,
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
    Ok(b)
}

impl<'a> RidgeProgram<'a> {
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

pub const EXTERNAL_SCHEMA: &str = "jope.j9-external-scores/v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalScore {
    pub context_id: String,
    pub program_sha256: String,
    pub score: f64,
}

/// An externally produced score document: the PolicyLM q-head adapter, a native GP ranking
/// export, or any learned selector. `provenance` (checkpoint, source and input digests) is
/// carried into the receipt verbatim.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalDocument {
    pub schema: String,
    pub name: String,
    pub class: Class,
    pub rule: String,
    pub provenance: serde_json::Value,
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

/// Contextual + program-interaction ridge surrogate (`GB-09` / `J9`) trained on dense GP trajectories.
///
/// Combines intercept, program token-bag features `\phi(p)`, dataset context features `z(c)`,
/// bilinear context-program interactions `z_i(c) * \phi_j(p)`, and kNN retrieval score into a
/// closed-form ridge predictor capable of ranking unseen `N+1` candidate programs on new contexts.
pub struct ContextualSurrogateQHat<'a> {
    lambda: f64,
    context_features: &'a Features,
    program_features: &'a Features,
    knn: RetrievalKnn<'a>,
    weights: Vec<f64>,
}

pub const MAX_SURROGATE_DESIGN_DIM: usize = 512;

impl<'a> ContextualSurrogateQHat<'a> {
    fn design_row(z: &[f64], phi: &[f64], knn_score: f64) -> Vec<f64> {
        let mut x = Vec::with_capacity(2 + z.len() + phi.len() + z.len() * phi.len());
        x.push(1.0);
        x.push(knn_score);
        x.extend_from_slice(z);
        x.extend_from_slice(phi);
        for &zi in z {
            for &pj in phi {
                x.push(zi * pj);
            }
        }
        x
    }

    pub fn fit(
        train: &[&'a Pool],
        context_features: &'a Features,
        program_features: &'a Features,
        k_neighbours: usize,
        prior_weight: f64,
        lambda: f64,
    ) -> Result<Self> {
        ensure!(
            lambda > 0.0 && lambda.is_finite(),
            "lambda must be positive and finite"
        );
        ensure!(
            context_features.dim > 0 && program_features.dim > 0,
            "non-empty context and program feature dimensions required"
        );
        let cross = context_features
            .dim
            .checked_mul(program_features.dim)
            .context("bilinear feature dimension overflow")?;
        let d = 2usize
            .checked_add(context_features.dim)
            .and_then(|v| v.checked_add(program_features.dim))
            .and_then(|v| v.checked_add(cross))
            .context("design dimension overflow")?;
        ensure!(
            d <= MAX_SURROGATE_DESIGN_DIM,
            "bilinear design dimension {d} exceeds MAX_SURROGATE_DESIGN_DIM ({MAX_SURROGATE_DESIGN_DIM}); project context/program features first"
        );
        let knn = RetrievalKnn::fit(train, context_features, k_neighbours, prior_weight)?;
        let (mut xtx, mut xty) = (vec![0.0; d * d], vec![0.0; d]);
        let mut rows = 0usize;
        for pool in train {
            let z = context_features
                .rows
                .get(&pool.context_id)
                .with_context(|| format!("no context features for {}", pool.context_id))?;
            let view = pool.view();
            let knn_scores = knn.scores(&view)?;
            for ((m, t), knn_s) in pool.members.iter().zip(rank_targets(pool)).zip(knn_scores) {
                let phi = program_features
                    .rows
                    .get(&m.program_sha256)
                    .with_context(|| format!("no program features for {}", m.program_sha256))?;
                let x = Self::design_row(z, phi, knn_s);
                for i in 0..d {
                    xty[i] += x[i] * t;
                    for j in 0..d {
                        xtx[i * d + j] += x[i] * x[j];
                    }
                }
                rows += 1;
            }
        }
        ensure!(rows > 0, "contextual surrogate q_hat needs training rows");
        for i in 1..d {
            xtx[i * d + i] += lambda;
        }
        xtx[0] += 1e-12;
        Ok(Self {
            lambda,
            context_features,
            program_features,
            knn,
            weights: cholesky_solve(xtx, xty, d)?,
        })
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
            "contextual_surrogate_qhat/v1: closed-form bilinear context x program ridge (lambda={}, d_ctx={}, d_prog={}) + kNN prior",
            self.lambda, self.context_features.dim, self.program_features.dim
        )
    }
    fn scores(&self, view: &PoolView<'_>) -> Result<Vec<f64>> {
        let z = self
            .context_features
            .rows
            .get(view.context_id)
            .with_context(|| format!("no context features for {}", view.context_id))?;
        let knn_scores = self.knn.scores(view)?;
        view.programs
            .iter()
            .zip(knn_scores)
            .map(|(p, knn_s)| {
                let phi = self
                    .program_features
                    .rows
                    .get(*p)
                    .with_context(|| format!("no program features for {p}"))?;
                let x = Self::design_row(z, phi, knn_s);
                Ok(x.iter()
                    .zip(&self.weights)
                    .map(|(xi, wi)| xi * wi)
                    .sum::<f64>())
            })
            .collect()
    }
}

/// Two-stage proposal funnel (`K -> B`, e.g. `K=64 -> B=8`) that pre-screens candidate proposals
/// using a `Selector` before running expensive 2-fold CV evaluation and Rust parity compilation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunnelAllocation {
    pub context_id: String,
    pub selector_name: String,
    pub proposal_pool_k: usize,
    pub eval_budget_b: usize,
    pub selected_program_sha256: Vec<String>,
    pub selected_scores: Vec<f64>,
}

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
                context_id: mk_hex(cid),
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
        let mut ctx_rows = BTreeMap::new();
        ctx_rows.insert(mk_hex(1), vec![1.0]);
        ctx_rows.insert(mk_hex(2), vec![-1.0]);
        ctx_rows.insert(mk_hex(3), vec![1.0]);
        ctx_rows.insert(mk_hex(4), vec![-1.0]);
        ctx_rows.insert(mk_hex(5), vec![1.0]);
        let ctx_feats = Features {
            dim: 1,
            rows: ctx_rows,
        };

        let train_refs = vec![&all_pools[0], &all_pools[1], &all_pools[2], &all_pools[3]];
        let qhat = ContextualSurrogateQHat::fit(&train_refs, &ctx_feats, &prog_feats, 2, 2.0, 1e-2)
            .unwrap();

        let test_view = all_pools[4].view();
        let scores = qhat.scores(&test_view).unwrap();
        assert!(
            scores[0] > scores[1],
            "expected positive-z context to rank token-1 program higher: {scores:?}"
        );

        let funnel = pre_screen_funnel(&test_view, &qhat, 64, 1).unwrap();
        assert_eq!(funnel.selected_program_sha256, vec![p_pos_b]);
    }
}
