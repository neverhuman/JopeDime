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

/// Cross-family S3 `gp_distilled/v1` program prior statistics restricted to an outer training split.
#[derive(Clone, Debug, Default)]
pub struct ExternalProgramPrior {
    pub tot_stats: BTreeMap<String, (f64, usize)>,
    pub parent_stats: BTreeMap<(String, String), (f64, usize)>,
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
    prior_weight: f64,
    k_neighbours: usize,
    context_features: &'a Features,
    program_features: &'a Features,
    ridge: RidgeProgram<'a>,
    train_targets: Vec<(&'a Pool, Vec<f64>)>,
    tot_stats: BTreeMap<String, (f64, usize)>,
    parent_stats: BTreeMap<(String, String), (f64, usize)>,
    external_prior: Option<ExternalProgramPrior>,
    ctx_proj: Vec<(usize, f64, f64)>,
    prog_proj: Vec<(usize, f64, f64)>,
    weights: Vec<f64>,
}

pub const MAX_SURROGATE_DESIGN_DIM: usize = 512;
pub const MAX_BILINEAR_CTX_DIM: usize = 8;
pub const MAX_BILINEAR_PROG_DIM: usize = 8;
const SURROGATE_BASE_FEATURES: usize = 8;

impl<'a> ContextualSurrogateQHat<'a> {
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
        knn_s: f64,
        eb_prior: f64,
        eb_knn: f64,
        nov: f64,
    ) -> Vec<f64> {
        let mut x =
            Vec::with_capacity(SURROGATE_BASE_FEATURES + zt.len() + pt.len() + zt.len() * pt.len());
        x.push(1.0);
        x.push(r_s);
        x.push(prior_s);
        x.push(knn_s);
        x.push(eb_prior);
        x.push(eb_knn);
        x.push(nov);
        x.push(nov * r_s);
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
        let z = self
            .context_features
            .rows
            .get(view.context_id)
            .with_context(|| format!("no context features for {}", view.context_id))?;
        let zt = Self::project(z, &self.ctx_proj);

        let mut neighbours: Vec<(f64, &str, &Pool, &Vec<f64>)> = self
            .train_targets
            .iter()
            .filter(|(p, _)| p.parent_id != view.parent_id)
            .map(|(p, t)| {
                (
                    cosine(z, &self.context_features.rows[&p.context_id]),
                    p.context_id.as_str(),
                    *p,
                    t,
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
            let s3_backoff = if c_p == 0 {
                if let Some(ext) = &self.external_prior {
                    let (es, ec) = ext.tot_stats.get(prog_sha).copied().unwrap_or((0.0, 0));
                    let (eps, epc) = ext
                        .parent_stats
                        .get(&(view.parent_id.to_owned(), prog_sha.to_owned()))
                        .copied()
                        .unwrap_or((0.0, 0));
                    let sc = ec.saturating_sub(epc);
                    if sc > 0 {
                        let sm = ((es - eps) / (sc as f64)).clamp(0.0, 1.0);
                        0.5 + 0.05 * (sm - 0.5)
                    } else {
                        0.5
                    }
                } else {
                    0.5
                }
            } else {
                0.5
            };
            let prior_s = (s_p + s3_backoff * self.prior_weight) / (c_p as f64 + self.prior_weight);
            let eb_prior = (s_p + self.prior_weight * r_s) / (c_p as f64 + self.prior_weight);

            let (mut num, mut den) = (0.0, 0.0);
            for (sim, _, pool, targets) in &neighbours {
                let w = sim.max(0.0);
                if let Ok(i) = pool
                    .members
                    .binary_search_by(|m| m.program_sha256.as_str().cmp(prog_sha))
                {
                    num += w * targets[i];
                    den += w;
                }
            }
            let knn_s = if den > 0.0 { num / den } else { prior_s };
            let eb_knn = (num + 1.5 * eb_prior) / (den + 1.5);
            let nov = 1.0 / (1.0 + c_p as f64);
            rows.push(Self::design_row(
                &zt, &pt, r_s, prior_s, knn_s, eb_prior, eb_knn, nov,
            ));
        }
        Ok(rows)
    }

    fn view_design_rows(&self, view: &PoolView<'_>) -> Result<Vec<Vec<f64>>> {
        let ridge_scores = self.ridge.scores(view)?;
        self.view_design_rows_with_ridge(view, &ridge_scores)
    }

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

    pub fn fit_with_prior(
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
        // Compute parent-grouped 5-fold out-of-fold RidgeProgram predictions so the meta-ridge stack
        // does not over-weight in-sample r_s predictions relative to out-of-sample test contexts.
        const INNER_RIDGE_FOLDS: usize = 5;
        let inner_folds: Vec<usize> = train
            .iter()
            .map(|p| {
                super::evaluate::fold_of(
                    "jope.j9-inner-ridge-folds/v1",
                    &p.parent_id,
                    INNER_RIDGE_FOLDS,
                )
            })
            .collect();
        let mut oof_ridges: Vec<Option<RidgeProgram<'a>>> = Vec::with_capacity(INNER_RIDGE_FOLDS);
        for f in 0..INNER_RIDGE_FOLDS {
            let sub_train: Vec<&'a Pool> = train
                .iter()
                .copied()
                .zip(&inner_folds)
                .filter_map(|(p, &ifv)| (ifv != f).then_some(p))
                .collect();
            let sub_model = if !sub_train.is_empty() && sub_train.len() < train.len() {
                RidgeProgram::fit(&sub_train, program_features, lambda).ok()
            } else {
                None
            };
            oof_ridges.push(sub_model);
        }

        let mut train_targets = Vec::with_capacity(train.len());
        let mut tot_stats: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        let mut parent_stats: BTreeMap<(String, String), (f64, usize)> = BTreeMap::new();
        let mut z_vecs: Vec<&[f64]> = Vec::with_capacity(train.len());
        let mut p_vecs: Vec<&[f64]> = Vec::new();

        for &pool in train {
            let z = context_features
                .rows
                .get(&pool.context_id)
                .with_context(|| format!("no context features for {}", pool.context_id))?;
            z_vecs.push(z.as_slice());
            let targets = rank_targets(pool);
            for (m, &t) in pool.members.iter().zip(&targets) {
                let phi = program_features
                    .rows
                    .get(&m.program_sha256)
                    .with_context(|| format!("no program features for {}", m.program_sha256))?;
                p_vecs.push(phi.as_slice());
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
            }
            train_targets.push((pool, targets));
        }
        ensure!(
            !p_vecs.is_empty(),
            "contextual surrogate q_hat needs training rows"
        );

        let ctx_proj =
            Self::select_top_variance_coords(&z_vecs, context_features.dim, MAX_BILINEAR_CTX_DIM);
        let prog_proj =
            Self::select_top_variance_coords(&p_vecs, program_features.dim, MAX_BILINEAR_PROG_DIM);

        let mut model = Self {
            lambda,
            prior_weight,
            k_neighbours,
            context_features,
            program_features,
            ridge,
            train_targets,
            tot_stats,
            parent_stats,
            external_prior: external_prior.cloned(),
            ctx_proj,
            prog_proj,
            weights: vec![0.0; d],
        };

        let (mut xtx, mut xty) = (vec![0.0; d * d], vec![0.0; d]);
        let mut rows = 0usize;
        for (pool_idx, (pool, targets)) in model.train_targets.iter().enumerate() {
            let view = pool.view();
            let oof_ridge = oof_ridges[inner_folds[pool_idx]]
                .as_ref()
                .unwrap_or(&model.ridge);
            let ridge_scores = oof_ridge.scores(&view)?;
            let design_rows = model.view_design_rows_with_ridge(&view, &ridge_scores)?;
            for (x, &t) in design_rows.iter().zip(targets) {
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
        let cross_terms = (model.ctx_proj.len() * model.prog_proj.len()).max(1);
        let bilinear_lambda = lambda * (cross_terms as f64 / 2.5).max(1.0);
        for i in 1..SURROGATE_BASE_FEATURES.min(d) {
            xtx[i * d + i] += lambda;
        }
        for i in SURROGATE_BASE_FEATURES..d {
            xtx[i * d + i] += bilinear_lambda;
        }
        xtx[0] += 1e-12;
        model.weights = cholesky_solve(xtx, xty, d)?;
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
            "contextual_surrogate_qhat/v2: internal cross-fitted LOPO Empirical-Bayes + compact bilinear context x program ridge (lambda={}, d_ctx={}->{}, d_prog={}->{}) + kNN/ridge stack (shadow-only candidate; no multiplicity adjustment)",
            self.lambda,
            self.context_features.dim,
            self.ctx_proj.len(),
            self.program_features.dim,
            self.prog_proj.len()
        )
    }
    fn scores(&self, view: &PoolView<'_>) -> Result<Vec<f64>> {
        let rows = self.view_design_rows(view)?;
        Ok(rows
            .into_iter()
            .map(|x| {
                x.iter()
                    .zip(&self.weights)
                    .map(|(xi, wi)| xi * wi)
                    .sum::<f64>()
            })
            .collect())
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
