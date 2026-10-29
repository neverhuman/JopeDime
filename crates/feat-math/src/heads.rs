//! Model heads in the battle dialect. Semantics follow the engine
//! (newmath-core benchmark.rs / gbm.rs) with a Cholesky solver for ridge
//! (the λ ≥ 1e-3 diagonal makes the system SPD; plain Cholesky = exact
//! cross-language parity, only +,-,*,/,sqrt).

use crate::boost2::{HistGbm2Model, HistGbm2Params};
use crate::dsl::{ParamValue, Stage};
use crate::gpr::{parse_kernel, GprKernelModel, GprModel};
use crate::krr::KrrModel;
use crate::ksmooth::KsmoothModel;
use crate::stack::StackModel;
use crate::Rows;
use anyhow::{bail, Result};
use rayon::prelude::*;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum FittedHead {
    Ridge(RidgeModel),
    RidgeGcv(RidgeGcvModel),
    ElasticNet(ElasticNetModel),
    Gbm(GbmModel),
    Ensemble {
        ridge: RidgeModel,
        gbm: GbmModel,
        w: f64,
    },
    HistGbm2(HistGbm2Model),
    Krr(KrrModel),
    Gpr(GprModel),
    GprKernel(GprKernelModel),
    Ksmooth(KsmoothModel),
    Hurdle2(Hurdle2Model),
    Stack(Box<StackModel>),
    Portfolio(Box<crate::portfolio::PortfolioModel>),
}

pub fn is_head_op(op: &str) -> bool {
    matches!(
        op,
        "ridge"
            | "logistic"
            | "ridge_gcv"
            | "elasticnet"
            | "histgbm"
            | "gbm"
            | "ensemble"
            | "ensemble_ridge_gbm"
            | "histgbm2"
            | "krr"
            | "gpr"
            | "ksmooth"
            | "hurdle2"
            | "stack"
            | "portfolio"
    )
}

pub fn fit_head(stage: &Stage, x: &Rows, y: &[f64]) -> Result<FittedHead> {
    match stage.op.as_str() {
        // Battle is regression-only: a logistic terminal from compatibility genomes maps to ridge.
        "ridge" | "logistic" => Ok(FittedHead::Ridge(RidgeModel::fit(
            x,
            y,
            stage.get_f64("lambda_frac", 0.02),
        ))),
        // Self-tuning ridge: the penalty is chosen by generalized cross-validation
        // over a fixed grid, so the stage carries no parameters.
        "ridge_gcv" => Ok(FittedHead::RidgeGcv(RidgeGcvModel::fit(x, y))),
        "elasticnet" => Ok(FittedHead::ElasticNet(ElasticNetModel::fit(
            x,
            y,
            stage.get_f64("alpha", 0.001),
            stage.get_f64("l1_ratio", 0.5),
            stage.get_usize("sweeps", 60),
        ))),
        "histgbm" | "gbm" => Ok(FittedHead::Gbm(GbmModel::fit(
            x,
            y,
            &GbmParams::from(stage),
        ))),
        "histgbm2" => Ok(FittedHead::HistGbm2(HistGbm2Model::fit(
            x,
            y,
            &HistGbm2Params::from(stage),
        ))),
        "krr" => Ok(FittedHead::Krr(KrrModel::fit(
            x,
            y,
            stage.get_usize("m", crate::krr::DEFAULT_M).clamp(50, 2000),
        ))),
        "gpr" => {
            let m = stage.get_usize("m", crate::gpr::DEFAULT_M).clamp(50, 2000);
            let steps = stage
                .get_usize("steps", crate::gpr::DEFAULT_STEPS)
                .clamp(1, 200);
            let kernel_src = match stage.params.get("kernel") {
                Some(ParamValue::Str(s)) => s.as_str(),
                _ => "rbf",
            };
            // kernel="rbf" (or absent) keeps the dedicated ARD path so compatibility
            // genomes stay bit-identical; any other expression uses the
            // composite-kernel engine.
            if kernel_src == "rbf" {
                Ok(FittedHead::Gpr(GprModel::fit(x, y, m, steps)))
            } else {
                let spec = match parse_kernel(kernel_src) {
                    Ok(spec) => spec,
                    Err(e) => bail!("gpr kernel: {e}"),
                };
                Ok(FittedHead::GprKernel(GprKernelModel::fit(
                    x, y, m, steps, &spec,
                )))
            }
        }
        "ksmooth" => Ok(FittedHead::Ksmooth(KsmoothModel::fit(
            x,
            y,
            stage
                .get_usize("k", crate::ksmooth::DEFAULT_K)
                .clamp(1, 512),
            stage
                .get_f64("gamma", crate::ksmooth::DEFAULT_GAMMA)
                .clamp(1e-6, 1e6),
        ))),
        "hurdle2" => Ok(FittedHead::Hurdle2(Hurdle2Model::fit(
            x,
            y,
            stage.get_f64("q", 0.86),
            stage.get_f64("q_spread", 0.03),
            stage.get_f64("lambda_frac", 0.2),
            stage.get_f64("gate_l2", 0.05),
        ))),
        "stack" => Ok(FittedHead::Stack(Box::new(StackModel::fit(x, y)))),
        // Compact portfolio head: `preset=core` expands at fit time to the
        // pinned member registry (members.rs), optionally edited by
        // add1/add2/drop1; the shared transforms ahead of this head feed
        // every member. Full semantics live in portfolio.rs (spec §4).
        "portfolio" => {
            let roster = crate::members::resolve_compact(stage)?;
            let repeats_auto = crate::members::compact_repeats_auto(stage);
            let model = crate::portfolio::fit_portfolio(&roster, x, y, 5, repeats_auto)?;
            Ok(FittedHead::Portfolio(Box::new(model)))
        }
        "ensemble" | "ensemble_ridge_gbm" => {
            let params = GbmParams::from(stage);
            let lambda_frac = stage.get_f64("lambda_frac", 0.02);
            let n = x.len();
            if n < 40 {
                let gbm = GbmModel::fit(x, y, &params);
                let ridge = RidgeModel::fit(x, y, lambda_frac);
                return Ok(FittedHead::Ensemble { ridge, gbm, w: 0.0 });
            }
            // Contiguous 80/20 internal holdout picks the blend weight; both members
            // are then refit on the full training set (engine semantics).
            let split = (n * 4) / 5;
            let sub: Rows = x[..split].to_vec();
            let hold: Rows = x[split..].to_vec();
            let y_sub = &y[..split];
            let y_hold = &y[split..];
            let ridge_h = RidgeModel::fit(&sub, y_sub, lambda_frac).predict(&hold);
            let gbm_h = GbmModel::fit(&sub, y_sub, &params).predict(&hold);
            let mut best_w = 0.5;
            let mut best = f64::NEG_INFINITY;
            for w in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let blend: Vec<f64> = ridge_h
                    .iter()
                    .zip(&gbm_h)
                    .map(|(r, g)| w * r + (1.0 - w) * g)
                    .collect();
                let s = -crate::metrics::rmse(y_hold, &blend);
                if s > best {
                    best = s;
                    best_w = w;
                }
            }
            Ok(FittedHead::Ensemble {
                ridge: RidgeModel::fit(x, y, lambda_frac),
                gbm: GbmModel::fit(x, y, &params),
                w: best_w,
            })
        }
        other => bail!("unsupported head: {other}"),
    }
}

pub fn fit_head_and_predict_train(
    stage: &Stage,
    x: &Rows,
    y: &[f64],
) -> Result<(FittedHead, Vec<f64>)> {
    let mut cache = crate::ops::SortedColumnsCache::new();
    fit_head_and_predict_train_with_cache(stage, x, y, &mut cache)
}

pub(crate) fn fit_head_with_cache(
    stage: &Stage,
    x: &Rows,
    y: &[f64],
    cache: &mut crate::ops::SortedColumnsCache,
) -> Result<FittedHead> {
    crate::kernel::width::reject_zero_width_rows(x)?;
    if matches!(stage.op.as_str(), "histgbm" | "gbm") {
        let params = GbmParams::from(stage);
        return Ok(FittedHead::Gbm(GbmModel::fit_with_cache(
            x,
            y,
            &params,
            Some(cache),
        )));
    }
    fit_head(stage, x, y)
}

pub(crate) fn fit_head_and_predict_train_with_cache(
    stage: &Stage,
    x: &Rows,
    y: &[f64],
    cache: &mut crate::ops::SortedColumnsCache,
) -> Result<(FittedHead, Vec<f64>)> {
    crate::kernel::width::reject_zero_width_rows(x)?;
    if matches!(stage.op.as_str(), "histgbm" | "gbm") && x.len() <= y.len() {
        let cfg = GbmParams::from(stage);
        let (gbm, f_train) = GbmModel::fit_with_train_preds_and_cache(x, y, &cfg, Some(cache));
        return Ok((FittedHead::Gbm(gbm), f_train));
    }
    let head = fit_head_with_cache(stage, x, y, cache)?;
    let tr_preds = head.predict(x)?;
    Ok((head, tr_preds))
}

impl FittedHead {
    pub fn predict(
        &self,
        x: &Rows,
    ) -> std::result::Result<Vec<f64>, crate::kernel::width::WidthMismatch> {
        crate::kernel::width::reject_zero_width_rows(x)?;
        Ok(match self {
            FittedHead::Ridge(m) => m.predict(x),
            FittedHead::RidgeGcv(m) => m.predict(x),
            FittedHead::ElasticNet(m) => m.predict(x),
            FittedHead::Gbm(m) => m.predict(x),
            FittedHead::Ensemble { ridge, gbm, w } => {
                let r = ridge.predict(x);
                let g = gbm.predict(x);
                r.iter()
                    .zip(&g)
                    .map(|(a, b)| w * a + (1.0 - w) * b)
                    .collect()
            }
            FittedHead::HistGbm2(m) => m.predict(x),
            FittedHead::Krr(m) => m.predict(x),
            FittedHead::Gpr(m) => m.predict(x),
            FittedHead::GprKernel(m) => m.predict(x),
            FittedHead::Ksmooth(m) => m.predict(x),
            FittedHead::Hurdle2(m) => m.predict(x),
            FittedHead::Stack(m) => m.predict(x),
            FittedHead::Portfolio(m) => m.predict(x)?,
        })
    }
}

// ------------------------- two-component hurdle -------------------------

/// Fold-local two-component expected-value regression. Each threshold is a
/// quantile of the current fit target; the gate and both severity heads are
/// fitted only from the rows given to this fit. Three neighbouring quantiles
/// are averaged by default to remove a brittle threshold-selection degree of
/// freedom observed in the hard-dataset campaign.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Hurdle2Model {
    pub components: Vec<Hurdle2Component>,
    pub fallback_mean: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Hurdle2Component {
    pub threshold: f64,
    pub gate: LogisticGate,
    pub low: ConditionalRidge,
    pub high: ConditionalRidge,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ConditionalRidge {
    Ridge(RidgeModel),
    Mean(f64),
}

impl ConditionalRidge {
    fn fit(x: &Rows, y: &[f64], rows: &[usize], lambda_frac: f64, fallback: f64) -> Self {
        if rows.len() < 8 {
            let value = if rows.is_empty() {
                fallback
            } else {
                rows.iter().map(|row| y[*row]).sum::<f64>() / rows.len() as f64
            };
            return Self::Mean(value);
        }
        let subset_x = rows.iter().map(|row| x[*row].clone()).collect::<Rows>();
        let subset_y = rows.iter().map(|row| y[*row]).collect::<Vec<_>>();
        Self::Ridge(RidgeModel::fit(&subset_x, &subset_y, lambda_frac))
    }

    fn predict(&self, x: &Rows) -> Vec<f64> {
        match self {
            Self::Ridge(model) => model.predict_unclamped(x),
            Self::Mean(value) => vec![*value; x.len()],
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LogisticGate {
    pub means: Vec<f64>,
    pub scales: Vec<f64>,
    /// `[intercept, w_0, ..]` on standardized, ±20-clamped features.
    pub w: Vec<f64>,
}

impl LogisticGate {
    fn fit(x: &Rows, labels: &[f64], l2: f64) -> Self {
        let n = x.len().min(labels.len());
        let p = x[0].len();
        let (means, scales) = std_fit(x);
        let prior = labels[..n].iter().sum::<f64>() / n.max(1) as f64;
        let mut w = vec![0.0; p + 1];
        w[0] = ((prior + 1e-6) / (1.0 - prior + 1e-6)).ln();
        let l2 = l2.clamp(1e-6, 10.0);
        // The averaged logistic Hessian has spectral norm bounded by roughly
        // 0.25*p after per-column standardization. This fixed conservative
        // step makes the fit deterministic and keeps each iteration O(n*p).
        let step = 1.0 / (0.25 * p.max(1) as f64 + l2 + 0.25);
        for _ in 0..240 {
            let mut gradient = vec![0.0; p + 1];
            for row in 0..n {
                let mut logit = w[0];
                for col in 0..p {
                    let value = ((x[row][col] - means[col]) / scales[col]).clamp(-20.0, 20.0);
                    logit += w[col + 1] * value;
                }
                let probability = logistic(logit);
                let error = probability - labels[row];
                gradient[0] += error;
                for col in 0..p {
                    let value = ((x[row][col] - means[col]) / scales[col]).clamp(-20.0, 20.0);
                    gradient[col + 1] += error * value;
                }
            }
            gradient[0] /= n.max(1) as f64;
            w[0] -= step * gradient[0];
            let mut max_delta = (step * gradient[0]).abs();
            for col in 0..p {
                let gradient = gradient[col + 1] / n.max(1) as f64 + l2 * w[col + 1];
                let delta = step * gradient;
                w[col + 1] -= delta;
                max_delta = max_delta.max(delta.abs());
            }
            if max_delta <= 1e-12 {
                break;
            }
        }
        Self { means, scales, w }
    }

    fn predict(&self, x: &Rows) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut logit = self.w[0];
                // `col` indexes four arrays in lockstep (row, means, scales, and w at an
                // offset). Zipping them would need a three-way chain plus an offset skip to
                // say what one index says plainly.
                #[allow(clippy::needless_range_loop)]
                for col in 0..self.means.len().min(row.len()) {
                    let value =
                        ((row[col] - self.means[col]) / self.scales[col]).clamp(-20.0, 20.0);
                    logit += self.w[col + 1] * value;
                }
                logistic(logit)
            })
            .collect()
    }
}

impl Hurdle2Model {
    pub fn fit(x: &Rows, y: &[f64], q: f64, q_spread: f64, lambda_frac: f64, gate_l2: f64) -> Self {
        let n = x.len().min(y.len());
        let fallback_mean = y[..n].iter().sum::<f64>() / n.max(1) as f64;
        if n < 12 || x[0].is_empty() || y[..n].iter().any(|value| !value.is_finite()) {
            return Self {
                components: Vec::new(),
                fallback_mean,
            };
        }
        let center = q.clamp(0.5, 0.95);
        let spread = q_spread.clamp(0.0, 0.10);
        let mut quantiles = vec![center];
        if spread > 0.0 {
            quantiles = vec![
                (center - spread).clamp(0.5, 0.95),
                center,
                (center + spread).clamp(0.5, 0.95),
            ];
            quantiles.dedup_by(|left, right| left.to_bits() == right.to_bits());
        }
        let mut sorted = y[..n].to_vec();
        sorted.sort_by(f64::total_cmp);
        let mut components = Vec::with_capacity(quantiles.len());
        for quantile in quantiles {
            let index = ((n - 1) as f64 * quantile).round() as usize;
            let threshold = sorted[index.min(n - 1)];
            let labels = y[..n]
                .iter()
                .map(|value| if *value > threshold { 1.0 } else { 0.0 })
                .collect::<Vec<_>>();
            let low_rows = labels
                .iter()
                .enumerate()
                .filter_map(|(row, label)| (*label == 0.0).then_some(row))
                .collect::<Vec<_>>();
            let high_rows = labels
                .iter()
                .enumerate()
                .filter_map(|(row, label)| (*label == 1.0).then_some(row))
                .collect::<Vec<_>>();
            components.push(Hurdle2Component {
                threshold,
                gate: LogisticGate::fit(x, &labels, gate_l2),
                low: ConditionalRidge::fit(
                    x,
                    y,
                    &low_rows,
                    lambda_frac.clamp(1e-5, 2.0),
                    fallback_mean,
                ),
                high: ConditionalRidge::fit(
                    x,
                    y,
                    &high_rows,
                    lambda_frac.clamp(1e-5, 2.0),
                    fallback_mean,
                ),
            });
        }
        Self {
            components,
            fallback_mean,
        }
    }

    pub fn predict(&self, x: &Rows) -> Vec<f64> {
        if self.components.is_empty() {
            return vec![self.fallback_mean; x.len()];
        }
        let mut output = vec![0.0; x.len()];
        for component in &self.components {
            let probability = component.gate.predict(x);
            let low = component.low.predict(x);
            let high = component.high.predict(x);
            for row in 0..x.len() {
                output[row] += probability[row] * high[row] + (1.0 - probability[row]) * low[row];
            }
        }
        let scale = 1.0 / self.components.len() as f64;
        for value in &mut output {
            *value *= scale;
            if !value.is_finite() {
                *value = self.fallback_mean;
            }
        }
        output
    }
}

fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value.min(40.0)).exp())
    } else {
        let exp = value.max(-40.0).exp();
        exp / (1.0 + exp)
    }
}

// --------------------------------- ridge ---------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RidgeModel {
    pub means: Vec<f64>,
    pub scales: Vec<f64>,
    /// `[intercept, w_0, .., w_{p-1}]` on standardized features.
    pub w: Vec<f64>,
}

impl RidgeModel {
    pub fn fit(x: &Rows, y: &[f64], lambda_frac: f64) -> RidgeModel {
        let n = x.len().min(y.len());
        let p = x[0].len();
        let (means, scales) = std_fit(x);
        let lambda = (lambda_frac * n as f64).max(1e-3);
        let y_mean = y[..n].iter().sum::<f64>() / n.max(1) as f64;
        // Gram + rhs on standardized (clamped ±20) features, centered target. B11: each entry is
        // the same running sum as the former one-row-at-a-time loop (0.0, then its terms in
        // ascending row order), accumulated in cache-sized row panels and register tiles, so every
        // non-NaN result is unchanged bit for bit (certified in kernel::ridge_fast; Rust leaves
        // the payload of a NaN result unspecified in either build).
        let mut z = Vec::with_capacity(n * p);
        for row in &x[..n] {
            for j in 0..p {
                z.push(((row[j] - means[j]) / scales[j]).clamp(-20.0, 20.0));
            }
        }
        let centered: Vec<f64> = y[..n].iter().map(|value| value - y_mean).collect();
        let (mut ata, mut atb) = ridge_normal_equations(&z, &centered, n, p);
        for i in 0..p {
            ata[i * p + i] += lambda;
            for j in 0..i {
                ata[i * p + j] = ata[j * p + i];
            }
        }
        let coeffs = cholesky_solve(&mut ata, &mut atb, p);
        let mut w = Vec::with_capacity(p + 1);
        w.push(y_mean);
        w.extend(coeffs);
        RidgeModel { means, scales, w }
    }

    pub fn predict(&self, x: &Rows) -> Vec<f64> {
        self.predict_unclamped(x)
            .into_iter()
            .map(|prediction| prediction.clamp(-60.0, 60.0))
            .collect()
    }

    /// Explicit successor caller; retains legacy standardized-feature safety bounds.
    pub(crate) fn predict_numeric_v2(&self, x: &Rows) -> Vec<f64> {
        self.predict_unclamped(x)
    }

    /// Raw ridge prediction for scale-equivariant composite heads. The public
    /// legacy ridge path retains its historical absolute safety clamp.
    fn predict_unclamped(&self, x: &Rows) -> Vec<f64> {
        let p = self.means.len();
        if x.len() * p >= 4096 && crate::is_parallel_enabled() {
            x.par_iter()
                .map(|r| {
                    let mut z = self.w[0];
                    for (j, &value) in r.iter().take(p).enumerate() {
                        let xs = ((value - self.means[j]) / self.scales[j]).clamp(-20.0, 20.0);
                        z += self.w[j + 1] * xs;
                    }
                    z
                })
                .collect()
        } else {
            x.iter()
                .map(|r| {
                    let mut z = self.w[0];
                    for (j, &value) in r.iter().take(p).enumerate() {
                        let xs = ((value - self.means[j]) / self.scales[j]).clamp(-20.0, 20.0);
                        z += self.w[j + 1] * xs;
                    }
                    z
                })
                .collect()
        }
    }
}

pub fn std_fit(x: &Rows) -> (Vec<f64>, Vec<f64>) {
    let n = x.len().max(1) as f64;
    let p = x[0].len();
    // B11: one row-major pass of per-column accumulators. Each column still sums its rows in
    // ascending order from 0.0, so the moments are the former per-column passes bit for bit.
    let mut sums = vec![0.0f64; p];
    for r in x {
        for (sum, &value) in sums.iter_mut().zip(&r[..p]) {
            *sum += value;
        }
    }
    let means: Vec<f64> = sums.iter().map(|sum| sum / n).collect();
    let mut squares = vec![0.0f64; p];
    for r in x {
        for ((square, &value), &mean) in squares.iter_mut().zip(&r[..p]).zip(&means) {
            let d = value - mean;
            *square += d * d;
        }
    }
    let scales = squares.iter().map(|v| (v / n).sqrt().max(1e-9)).collect();
    (means, scales)
}

/// B11: rows of the standardized design per panel; a packed panel stays in cache while its tiles
/// run.
const RIDGE_PANEL: usize = 128;
/// B11: Gram rows and columns per register tile.
const RIDGE_TILE: usize = 4;

/// `ZᵀZ` (row-major `p x p`; the upper triangle is the Gram, entries below the diagonal are
/// scratch) and `Zᵀyc` from the row-major standardized design `z` of `n` rows. Every upper entry
/// and every right-hand-side entry is a running sum that starts at `0.0` and adds its terms in
/// ascending row order, exactly as a one-row-at-a-time rank-one update does.
pub(crate) fn ridge_normal_equations(
    z: &[f64],
    centered: &[f64],
    n: usize,
    p: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut ata = vec![0.0f64; p * p];
    let mut atb = vec![0.0f64; p];
    let blocks = p.div_ceil(RIDGE_TILE);
    // One panel, packed column block by column block: block `b` holds the panel's rows
    // contiguously, `RIDGE_TILE` lanes per row. Lanes past column `p` are never written, so they
    // stay zero.
    let mut packed = vec![0.0f64; blocks * RIDGE_PANEL * RIDGE_TILE];
    for start in (0..n).step_by(RIDGE_PANEL) {
        let rows = (start + RIDGE_PANEL).min(n) - start;
        for r in 0..rows {
            let row = &z[(start + r) * p..(start + r + 1) * p];
            let yr = centered[start + r];
            for (sum, &value) in atb.iter_mut().zip(row) {
                *sum += value * yr;
            }
            for (b, lanes) in row.chunks(RIDGE_TILE).enumerate() {
                packed[(b * RIDGE_PANEL + r) * RIDGE_TILE..][..lanes.len()].copy_from_slice(lanes);
            }
        }
        for bi in 0..blocks {
            let a = &packed[bi * RIDGE_PANEL * RIDGE_TILE..][..rows * RIDGE_TILE];
            for bj in bi..blocks {
                let b = &packed[bj * RIDGE_PANEL * RIDGE_TILE..][..rows * RIDGE_TILE];
                accumulate_ridge_tile(a, b, &mut ata, p, bi * RIDGE_TILE, bj * RIDGE_TILE);
            }
        }
    }
    (ata, atb)
}

/// Adds a packed panel's terms to the Gram tile at `(i0, j0)`, each entry onto its own running
/// sum, one term per row in ascending order. Lanes past column `p` accumulate zeros and are not
/// stored.
fn accumulate_ridge_tile(a: &[f64], b: &[f64], ata: &mut [f64], p: usize, i0: usize, j0: usize) {
    let (height, width) = (RIDGE_TILE.min(p - i0), RIDGE_TILE.min(p - j0));
    let mut acc = [[0.0f64; RIDGE_TILE]; RIDGE_TILE];
    for (ii, sums) in acc.iter_mut().enumerate().take(height) {
        sums[..width].copy_from_slice(&ata[(i0 + ii) * p + j0..][..width]);
    }
    for (zi, zj) in a.chunks_exact(RIDGE_TILE).zip(b.chunks_exact(RIDGE_TILE)) {
        let zj: [f64; RIDGE_TILE] = zj.try_into().expect("a whole tile row");
        for (sums, &x) in acc.iter_mut().zip(zi) {
            for (sum, y) in sums.iter_mut().zip(zj) {
                *sum += x * y;
            }
        }
    }
    for (ii, sums) in acc.iter().enumerate().take(height) {
        ata[(i0 + ii) * p + j0..][..width].copy_from_slice(&sums[..width]);
    }
}

/// Rows factored together by [`cholesky_factor`].
const CHOLESKY_ROW_BLOCK: usize = 4;

/// In-place row-oriented Cholesky factorization (`A = L Lᵀ`, `L` in the lower triangle), with
/// exactly the arithmetic of the textbook loop
///
/// ```text
/// for i in 0..p { for j in 0..=i {
///     s = a[i][j]; for k in 0..j { s -= a[i][k] * a[j][k] }
///     a[i][j] = if i == j { if s > 1e-12 { sqrt(s) } else { 1e-6 } } else { s / a[j][j] }
/// } }
/// ```
///
/// Each entry's `s -= ..` chain is one dependent floating-point sequence, so that loop runs at
/// FP latency. Entries of *different* rows left of a row block depend only on finished rows and
/// on their own row, so here [`CHOLESKY_ROW_BLOCK`] rows are factored together: their chains
/// interleave, which hides the latency, while every entry still sees the same operands in the
/// same order (bit-identical; Rust never contracts to FMA). The small triangle inside each block
/// is then finished in the original row-major order.
fn cholesky_factor(a: &mut [f64], p: usize) {
    let mut i0 = 0;
    while i0 < p {
        let rows = CHOLESKY_ROW_BLOCK.min(p - i0);
        match rows {
            4 => cholesky_block_left::<4>(a, p, i0),
            3 => cholesky_block_left::<3>(a, p, i0),
            2 => cholesky_block_left::<2>(a, p, i0),
            _ => cholesky_block_left::<1>(a, p, i0),
        }
        for i in i0..i0 + rows {
            for j in i0..=i {
                let mut s = a[i * p + j];
                for k in 0..j {
                    s -= a[i * p + k] * a[j * p + k];
                }
                a[i * p + j] = if i == j {
                    if s > 1e-12 {
                        s.sqrt()
                    } else {
                        1e-6
                    }
                } else {
                    s / a[j * p + j]
                };
            }
        }
        i0 += rows;
    }
}

/// Entries `(i0 + t, j)` for `t < R` and every `j < i0`: `R` independent chains per column.
fn cholesky_block_left<const R: usize>(a: &mut [f64], p: usize, i0: usize) {
    for j in 0..i0 {
        let mut s = [0.0f64; R];
        for (t, st) in s.iter_mut().enumerate() {
            *st = a[(i0 + t) * p + j];
        }
        for k in 0..j {
            let ajk = a[j * p + k];
            for (t, st) in s.iter_mut().enumerate() {
                *st -= a[(i0 + t) * p + k] * ajk;
            }
        }
        let d = a[j * p + j];
        for (t, st) in s.iter().enumerate() {
            a[(i0 + t) * p + j] = st / d;
        }
    }
}

/// In-place Cholesky solve of the SPD system A w = b (A carries λ ≥ 1e-3 on the
/// diagonal). Same algorithm, same order, in the emitted artifacts.
pub fn cholesky_solve(a: &mut [f64], b: &mut [f64], p: usize) -> Vec<f64> {
    // Decompose: A = L Lᵀ, L stored in the lower triangle of a.
    cholesky_factor(a, p);
    // Forward: L z = b.
    let mut z = vec![0.0f64; p];
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * p + k] * z[k];
        }
        z[i] = s / a[i * p + i];
    }
    // Backward: Lᵀ w = z.
    let mut w = vec![0.0f64; p];
    for i in (0..p).rev() {
        let mut s = z[i];
        for k in (i + 1)..p {
            s -= a[k * p + i] * w[k];
        }
        w[i] = s / a[i * p + i];
    }
    w
}

// ------------------------------ ridge (GCV) ------------------------------

/// The fixed generalized-cross-validation penalty grid: logspace(-5, 3, 25),
/// baked as literals so the engine and both emitted artifacts share bits.
pub const GCV_LAMBDAS: [f64; 25] = [
    1e-05,
    2.1544346900318823e-05,
    4.641588833612782e-05,
    0.0001,
    0.00021544346900318823,
    0.00046415888336127773,
    0.001,
    0.002154434690031882,
    0.004641588833612777,
    0.01,
    0.021544346900318822,
    0.046415888336127774,
    0.1,
    0.21544346900318823,
    0.46415888336127725,
    1.0,
    2.154434690031882,
    4.641588833612772,
    10.0,
    21.54434690031882,
    46.41588833612773,
    100.0,
    215.44346900318823,
    464.1588833612773,
    1000.0,
];

/// Widest eigenproblem the GCV head will solve directly. When both the row
/// and column counts exceed this, the fit first keeps the `GCV_SCREEN_CAP`
/// columns most correlated with the target (a train-only statistic, ties
/// toward the smaller column index), so the cyclic-Jacobi cost is bounded at
/// 384^3 per sweep regardless of block width.
pub const GCV_SCREEN_CAP: usize = 384;

/// Ridge regression whose penalty is chosen by generalized cross-validation
/// (Golub, Heath & Wahba, 1979) over [`GCV_LAMBDAS`]. One cyclic-Jacobi
/// eigendecomposition scores every grid penalty in closed form; the winner is
/// refit on all rows. The eigenproblem is solved on whichever side is
/// smaller: the p x p standardized Gram when n >= p, or the n x n
/// inner-product (kernel) matrix Xs Xs^T when n < p — the two share their
/// nonzero spectrum, so the GCV score and the final coefficients are the
/// same ridge algebra either way. Blocks wider than [`GCV_SCREEN_CAP`] on
/// both sides are first screened to the top-384 columns by |corr(x_j, y)|.
/// Deterministic, parameter-free, and built only from +,-,*,/,sqrt for exact
/// cross-language parity with the emitted artifacts.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RidgeGcvModel {
    /// Training-column indices the fit kept (identity unless the width
    /// screen fired); `means`/`scales`/`w` are indexed by position in here.
    pub cols: Vec<usize>,
    pub means: Vec<f64>,
    pub scales: Vec<f64>,
    /// `[intercept, w_0, .., w_{k-1}]` on the standardized kept columns.
    pub w: Vec<f64>,
}

/// Train-only pre-screen for very wide blocks: the `cap` columns with the
/// largest |dot(xs_j, yc)| — the |corr(x_j, y)| order, since standardized
/// columns have unit variance — ties broken toward the smaller column index,
/// returned ascending. Constant columns standardize to zero and sort last.
fn gcv_screen_columns(
    x: &Rows,
    y: &[f64],
    n: usize,
    means: &[f64],
    scales: &[f64],
    y_mean: f64,
    cap: usize,
) -> Vec<usize> {
    let p = means.len();
    let mut score = vec![0.0f64; p];
    for r in 0..n {
        let yr = y[r] - y_mean;
        for j in 0..p {
            score[j] += ((x[r][j] - means[j]) / scales[j]) * yr;
        }
    }
    for s in score.iter_mut() {
        if *s < 0.0 {
            *s = -*s;
        }
    }
    let mut used = vec![false; p];
    let mut sel = Vec::with_capacity(cap);
    for _ in 0..cap {
        let mut best_j = 0usize;
        let mut best_s = -1.0f64;
        for j in 0..p {
            if !used[j] && score[j] > best_s {
                best_s = score[j];
                best_j = j;
            }
        }
        used[best_j] = true;
        sel.push(best_j);
    }
    sel.sort_unstable();
    sel
}

impl RidgeGcvModel {
    pub fn fit(x: &Rows, y: &[f64]) -> RidgeGcvModel {
        Self::fit_inner(x, y, false)
    }

    /// `force_primal` replays the p x p Gram route even when n < p (test-only
    /// hook: the dual route must agree with it to floating-point roundoff).
    fn fit_inner(x: &Rows, y: &[f64], force_primal: bool) -> RidgeGcvModel {
        let n = x.len().min(y.len());
        let mut p = x[0].len();
        let (mut means, mut scales) = std_fit(x);
        // Constant columns standardize to zero: divide by 1 instead of the
        // 1e-9 floor (the reference RidgeGCV convention).
        for s in scales.iter_mut() {
            if *s <= 1e-9 {
                *s = 1.0;
            }
        }
        let mut y_mean = 0.0;
        for &target in y.iter().take(n) {
            y_mean += target;
        }
        y_mean /= n.max(1) as f64;
        let mut y_var = 0.0;
        for &target in y.iter().take(n) {
            let d = target - y_mean;
            y_var += d * d;
        }
        let mut cols: Vec<usize> = (0..p).collect();
        // Degenerate training data: fall back to the mean predictor.
        if n < 2 || p == 0 || (y_var / n.max(1) as f64).sqrt() < 1e-12 {
            let mut w = vec![0.0; p + 1];
            w[0] = y_mean;
            return RidgeGcvModel {
                cols,
                means,
                scales,
                w,
            };
        }
        // Width screen: cap the eigenproblem at GCV_SCREEN_CAP a side.
        if n.min(p) > GCV_SCREEN_CAP {
            cols = gcv_screen_columns(x, y, n, &means, &scales, y_mean, GCV_SCREEN_CAP);
            p = cols.len();
            let kept_means: Vec<f64> = cols.iter().map(|&j| means[j]).collect();
            let kept_scales: Vec<f64> = cols.iter().map(|&j| scales[j]).collect();
            means = kept_means;
            scales = kept_scales;
        }
        let nf = n as f64;
        if !force_primal && n < p {
            // Dual (kernel) route: K = Xs Xs^T = Q diag(e) Q^T shares its
            // nonzero eigenvalues with the Gram Xs^T Xs. With u = Q^T yc,
            // fitted values shrink per component by e/(e + lambda), so
            // rss = y'y + sum f^2 - 2 f u (f = e u/(e + lambda)) and
            // tr(S) = sum e/(e + lambda), all without the p x p Gram. The
            // primal coefficients come back via coef = Xs^T Q diag(1/(e +
            // lambda)) u; zero-eigenvalue directions satisfy Xs^T q_i = 0
            // and are excluded exactly as in the primal route.
            let mut xs = vec![0.0f64; n * p];
            for r in 0..n {
                for j in 0..p {
                    xs[r * p + j] = (x[r][cols[j]] - means[j]) / scales[j];
                }
            }
            let mut yc = vec![0.0f64; n];
            let mut yty = 0.0f64;
            for r in 0..n {
                yc[r] = y[r] - y_mean;
                yty += yc[r] * yc[r];
            }
            let mut km = vec![0.0f64; n * n];
            for r in 0..n {
                for s in r..n {
                    let mut acc = 0.0;
                    for j in 0..p {
                        acc += xs[r * p + j] * xs[s * p + j];
                    }
                    km[r * n + s] = acc;
                }
            }
            for i in 0..n {
                for j in 0..i {
                    km[i * n + j] = km[j * n + i];
                }
            }
            let (mut e, q) = jacobi_eigen(&mut km, n);
            let mut e_max = 0.0;
            for &value in e.iter().take(n) {
                if value > e_max {
                    e_max = value;
                }
            }
            for value in e.iter_mut().take(n) {
                if *value < e_max * 1e-12 {
                    *value = 0.0;
                }
            }
            let mut u = vec![0.0f64; n];
            for i in 0..n {
                let mut s = 0.0;
                for r in 0..n {
                    s += q[r * n + i] * yc[r];
                }
                u[i] = s;
            }
            let mut best_lambda = GCV_LAMBDAS[0];
            let mut best_gcv = f64::INFINITY;
            for &lambda in GCV_LAMBDAS.iter() {
                let mut rss = yty;
                let mut tr = 0.0;
                for i in 0..n {
                    if e[i] > 0.0 {
                        let c = u[i] / (e[i] + lambda);
                        let f = e[i] * c;
                        rss += f * f - 2.0 * f * u[i];
                        tr += e[i] / (e[i] + lambda);
                    }
                }
                if rss < 0.0 {
                    rss = 0.0;
                }
                let mut denom = 1.0 - tr / nf;
                if denom < 1e-3 {
                    denom = 1e-3;
                }
                let gcv = (rss / nf) / (denom * denom);
                if gcv < best_gcv {
                    best_gcv = gcv;
                    best_lambda = lambda;
                }
            }
            let mut t = vec![0.0f64; n];
            for r in 0..n {
                let mut s = 0.0;
                for i in 0..n {
                    if e[i] > 0.0 {
                        s += q[r * n + i] * (u[i] / (e[i] + best_lambda));
                    }
                }
                t[r] = s;
            }
            let mut w = Vec::with_capacity(p + 1);
            w.push(y_mean);
            for j in 0..p {
                let mut s = 0.0;
                for r in 0..n {
                    s += xs[r * p + j] * t[r];
                }
                w.push(s);
            }
            return RidgeGcvModel {
                cols,
                means,
                scales,
                w,
            };
        }
        // Primal route: Gram matrix, rhs, and target energy on standardized
        // kept columns.
        let mut g = vec![0.0f64; p * p];
        let mut b = vec![0.0f64; p];
        let mut yty = 0.0f64;
        let mut xs_row = vec![0.0f64; p];
        for r in 0..n {
            for j in 0..p {
                xs_row[j] = (x[r][cols[j]] - means[j]) / scales[j];
            }
            let yr = y[r] - y_mean;
            yty += yr * yr;
            for i in 0..p {
                let xi = xs_row[i];
                b[i] += xi * yr;
                for j in i..p {
                    g[i * p + j] += xi * xs_row[j];
                }
            }
        }
        for i in 0..p {
            for j in 0..i {
                g[i * p + j] = g[j * p + i];
            }
        }
        let (mut e, v) = jacobi_eigen(&mut g, p);
        // Clamp tiny/negative eigenvalues (Jacobi roundoff) to exactly zero;
        // those directions are excluded from the fit and the trace.
        let mut e_max = 0.0;
        for &value in e.iter().take(p) {
            if value > e_max {
                e_max = value;
            }
        }
        for value in e.iter_mut().take(p) {
            if *value < e_max * 1e-12 {
                *value = 0.0;
            }
        }
        // Rotate the rhs into the eigenbasis: bt = V^T b.
        let mut bt = vec![0.0f64; p];
        for i in 0..p {
            let mut s = 0.0;
            for j in 0..p {
                s += v[j * p + i] * b[j];
            }
            bt[i] = s;
        }
        // GCV(lambda) = mean squared residual / max(1 - tr(S)/n, 1e-3)^2 in
        // the eigenbasis: c_i = bt_i/(e_i + lambda), rss = y'y - 2 c'bt +
        // sum c_i^2 e_i, tr(S) = sum e_i/(e_i + lambda). No n-by-p work per
        // grid point; ties keep the smaller penalty (ascending grid).
        let mut best_lambda = GCV_LAMBDAS[0];
        let mut best_gcv = f64::INFINITY;
        for &lambda in GCV_LAMBDAS.iter() {
            let mut rss = yty;
            let mut tr = 0.0;
            for i in 0..p {
                if e[i] > 0.0 {
                    let c = bt[i] / (e[i] + lambda);
                    rss += c * c * e[i] - 2.0 * c * bt[i];
                    tr += e[i] / (e[i] + lambda);
                }
            }
            if rss < 0.0 {
                rss = 0.0;
            }
            let mut denom = 1.0 - tr / nf;
            if denom < 1e-3 {
                denom = 1e-3;
            }
            let gcv = (rss / nf) / (denom * denom);
            if gcv < best_gcv {
                best_gcv = gcv;
                best_lambda = lambda;
            }
        }
        // Final coefficients at the winning penalty, mapped back: w = V c.
        let mut w = Vec::with_capacity(p + 1);
        w.push(y_mean);
        for j in 0..p {
            let mut s = 0.0;
            for i in 0..p {
                if e[i] > 0.0 {
                    s += v[j * p + i] * (bt[i] / (e[i] + best_lambda));
                }
            }
            w.push(s);
        }
        RidgeGcvModel {
            cols,
            means,
            scales,
            w,
        }
    }

    pub fn predict(&self, x: &Rows) -> Vec<f64> {
        x.iter()
            .map(|r| {
                let mut z = self.w[0];
                for (k, &j) in self.cols.iter().enumerate() {
                    if j < r.len() {
                        z += self.w[k + 1] * ((r[j] - self.means[k]) / self.scales[k]);
                    }
                }
                z.clamp(-60.0, 60.0)
            })
            .collect()
    }
}

/// Eigendecomposition of the symmetric p x p matrix `a` (flattened row-major,
/// overwritten) by cyclic Jacobi rotations (Jacobi, 1846): returns
/// (eigenvalues, eigenvectors) with the eigenvectors flattened row-major so
/// column i holds the unit vector of eigenvalue i. Deterministic: fixed sweep
/// order, at most 20 sweeps, stopping as soon as a whole-matrix check finds
/// the off-diagonal mass under a 1e-12-of-trace threshold (cyclic Jacobi
/// converges quadratically, so well-conditioned inputs exit long before the
/// cap). Same algorithm, same order, in the emitted artifacts.
pub fn jacobi_eigen(a: &mut [f64], p: usize) -> (Vec<f64>, Vec<f64>) {
    let is_bitwise_symmetric =
        (1..p).all(|r| (0..r).all(|col| a[r * p + col].to_bits() == a[col * p + r].to_bits()));
    if !is_bitwise_symmetric {
        return jacobi_eigen_scalar_reference(a, p);
    }
    // The rotations update eigenvector columns i and j; accumulating V transposed
    // (`vt[col * p + k]`) makes both updates contiguous instead of stride-p. Each element sees
    // the same operations in the same order, and V is transposed back on return.
    let mut vt = vec![0.0f64; p * p];
    for i in 0..p {
        vt[i * p + i] = 1.0;
    }
    let mut scale = 0.0;
    for i in 0..p {
        scale += a[i * p + i];
    }
    let limit = (scale * 1e-12) * (scale * 1e-12);
    for _ in 0..20 {
        let mut off = 0.0;
        for i in 0..p {
            for j in (i + 1)..p {
                off += a[i * p + j] * a[i * p + j];
            }
        }
        if off <= limit {
            break;
        }
        for i in 0..p {
            for j in (i + 1)..p {
                let apq = a[i * p + j];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[j * p + j] - a[i * p + i]) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / ((theta * theta + 1.0).sqrt() - theta)
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                {
                    let (head_a, tail_a) = a.split_at_mut(j * p);
                    let ai = &mut head_a[i * p..(i + 1) * p];
                    let aj = &mut tail_a[..p];
                    for (aik, ajk) in ai[..i].iter_mut().zip(aj[..i].iter_mut()) {
                        let (old_i, old_j) = (*aik, *ajk);
                        *aik = c * old_i - s * old_j;
                        *ajk = s * old_i + c * old_j;
                    }
                    for (aik, ajk) in ai[(j + 1)..p].iter_mut().zip(aj[(j + 1)..p].iter_mut()) {
                        let (old_i, old_j) = (*aik, *ajk);
                        *aik = c * old_i - s * old_j;
                        *ajk = s * old_i + c * old_j;
                    }
                }
                for k in (i + 1)..j {
                    let old_i = a[i * p + k];
                    let old_j = a[k * p + j];
                    let new_i = c * old_i - s * old_j;
                    let new_j = s * old_i + c * old_j;
                    a[i * p + k] = new_i;
                    a[k * p + j] = new_j;
                }
                let aii = a[i * p + i];
                let ajj = a[j * p + j];
                a[i * p + i] = c * c * aii - 2.0 * s * c * apq + s * s * ajj;
                a[j * p + j] = s * s * aii + 2.0 * s * c * apq + c * c * ajj;
                a[i * p + j] = 0.0;
                a[j * p + i] = 0.0;
                let (head, tail) = vt.split_at_mut(j * p);
                let vi = &mut head[i * p..(i + 1) * p];
                let vj = &mut tail[..p];
                for (vki, vkj) in vi.iter_mut().zip(vj.iter_mut()) {
                    let (old_i, old_j) = (*vki, *vkj);
                    *vki = c * old_i - s * old_j;
                    *vkj = s * old_i + c * old_j;
                }
            }
            for col in (i + 1)..p {
                a[col * p + i] = a[i * p + col];
            }
        }
        for r in 1..p {
            for col in 0..r {
                a[col * p + r] = a[r * p + col];
            }
        }
    }
    let mut e = vec![0.0f64; p];
    for i in 0..p {
        e[i] = a[i * p + i];
    }
    let mut v = vec![0.0f64; p * p];
    for col in 0..p {
        for k in 0..p {
            v[k * p + col] = vt[col * p + k];
        }
    }
    (e, v)
}

fn jacobi_eigen_scalar_reference(a: &mut [f64], p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0f64; p * p];
    for i in 0..p {
        v[i * p + i] = 1.0;
    }
    let mut scale = 0.0;
    for i in 0..p {
        scale += a[i * p + i];
    }
    let limit = (scale * 1e-12) * (scale * 1e-12);
    for _ in 0..20 {
        let mut off = 0.0;
        for i in 0..p {
            for j in (i + 1)..p {
                off += a[i * p + j] * a[i * p + j];
            }
        }
        if off <= limit {
            break;
        }
        for i in 0..p {
            for j in (i + 1)..p {
                let apq = a[i * p + j];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[j * p + j] - a[i * p + i]) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / ((theta * theta + 1.0).sqrt() - theta)
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..p {
                    if k != i && k != j {
                        let aik = a[k * p + i];
                        let ajk = a[k * p + j];
                        let new_i = c * aik - s * ajk;
                        let new_j = s * aik + c * ajk;
                        a[k * p + i] = new_i;
                        a[i * p + k] = new_i;
                        a[k * p + j] = new_j;
                        a[j * p + k] = new_j;
                    }
                }
                let aii = a[i * p + i];
                let ajj = a[j * p + j];
                a[i * p + i] = c * c * aii - 2.0 * s * c * apq + s * s * ajj;
                a[j * p + j] = s * s * aii + 2.0 * s * c * apq + c * c * ajj;
                a[i * p + j] = 0.0;
                a[j * p + i] = 0.0;
                for k in 0..p {
                    let vki = v[k * p + i];
                    let vkj = v[k * p + j];
                    v[k * p + i] = c * vki - s * vkj;
                    v[k * p + j] = s * vki + c * vkj;
                }
            }
        }
    }
    let mut e = vec![0.0f64; p];
    for i in 0..p {
        e[i] = a[i * p + i];
    }
    (e, v)
}

// ------------------------------ elastic net ------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ElasticNetModel {
    pub means: Vec<f64>,
    pub scales: Vec<f64>,
    pub w0: f64,
    pub w: Vec<f64>,
}

impl ElasticNetModel {
    /// Coordinate descent on standardized features and centered target.
    /// Fixed sweep count (deterministic); objective is the sklearn elastic net
    /// with penalty alpha*(l1_ratio*|w| + (1-l1_ratio)/2*w²).
    pub fn fit(x: &Rows, y: &[f64], alpha: f64, l1_ratio: f64, sweeps: usize) -> ElasticNetModel {
        let n = x.len().min(y.len());
        let p = x[0].len();
        let (means, scales) = std_fit(x);
        // Standardized features stored column-major: coordinate descent reads one column at a
        // time, so each sweep streams contiguous memory instead of chasing a row pointer per
        // element. Every value, and every accumulation order below, is unchanged.
        let mut xc = vec![0.0f64; n * p];
        for (r, row) in x[..n].iter().enumerate() {
            for j in 0..p {
                xc[j * n + r] = ((row[j] - means[j]) / scales[j]).clamp(-20.0, 20.0);
            }
        }
        let y_mean = y[..n].iter().sum::<f64>() / n.max(1) as f64;
        let nf = n as f64;
        let l1 = alpha * l1_ratio;
        let l2 = alpha * (1.0 - l1_ratio);
        // Column norms (constant during CD); each still sums its rows in ascending order.
        let col_sq: Vec<f64> = (0..p)
            .map(|j| {
                let mut sq = 0.0f64;
                for &v in &xc[j * n..(j + 1) * n] {
                    sq += v * v;
                }
                sq
            })
            .collect();
        let mut w = vec![0.0f64; p];
        let mut resid: Vec<f64> = (0..n).map(|r| y[r] - y_mean).collect();
        for _ in 0..sweeps.max(1) {
            let mut max_delta = 0.0f64;
            for j in 0..p {
                if col_sq[j] <= 1e-12 {
                    continue;
                }
                let column = &xc[j * n..(j + 1) * n];
                let mut rho = 0.0;
                for (&v, &e) in column.iter().zip(&resid) {
                    rho += v * e;
                }
                rho = rho / nf + w[j] * col_sq[j] / nf;
                let denom = col_sq[j] / nf + l2;
                let new_w = soft_threshold(rho, l1) / denom;
                let delta = new_w - w[j];
                if delta != 0.0 {
                    for (e, &v) in resid.iter_mut().zip(column) {
                        *e -= delta * v;
                    }
                    w[j] = new_w;
                    let ad = delta.abs();
                    if ad > max_delta {
                        max_delta = ad;
                    }
                }
            }
            if max_delta < 1e-9 {
                break;
            }
        }
        ElasticNetModel {
            means,
            scales,
            w0: y_mean,
            w,
        }
    }

    pub fn predict(&self, x: &Rows) -> Vec<f64> {
        let p = self.means.len();
        x.iter()
            .map(|r| {
                let mut z = self.w0;
                for (j, &value) in r.iter().take(p).enumerate() {
                    let xs = ((value - self.means[j]) / self.scales[j]).clamp(-20.0, 20.0);
                    z += self.w[j] * xs;
                }
                z.clamp(-60.0, 60.0)
            })
            .collect()
    }
}

fn soft_threshold(v: f64, t: f64) -> f64 {
    if v > t {
        v - t
    } else if v < -t {
        v + t
    } else {
        0.0
    }
}

// ----------------------------------- gbm -----------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GbmParams {
    pub rounds: usize,
    pub depth: usize,
    pub lr: f64,
    pub lambda: f64,
    pub bins: usize,
    pub min_leaf: usize,
    pub max_features: usize,
}

impl Default for GbmParams {
    fn default() -> Self {
        GbmParams {
            rounds: 150,
            depth: 3,
            lr: 0.1,
            lambda: 1.0,
            bins: 64,
            min_leaf: 10,
            max_features: 256,
        }
    }
}

impl GbmParams {
    pub fn from(stage: &Stage) -> GbmParams {
        let d = GbmParams::default();
        GbmParams {
            rounds: stage.get_usize("rounds", d.rounds).clamp(1, 500),
            depth: stage.get_usize("depth", d.depth).clamp(1, 8),
            lr: stage.get_f64("lr", d.lr),
            lambda: stage.get_f64("lambda", d.lambda),
            bins: stage.get_usize("bins", d.bins).clamp(2, 255),
            min_leaf: stage.get_usize("min_leaf", d.min_leaf).max(1),
            max_features: stage.get_usize("max_features", d.max_features).max(1),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum GbmNode {
    Leaf(f64),
    Split {
        feature: usize,
        thr_bin: u16,
        left: usize,
        right: usize,
    },
}

/// Deterministic histogram gradient boosting, squared loss — a fit/predict split
/// of the engine's `gbm.rs` (identical trees for identical inputs).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GbmModel {
    pub edges: Vec<Vec<f64>>,
    pub trees: Vec<Vec<GbmNode>>,
    pub init: f64,
    pub lr: f64,
    pub n_feat: usize,
    pub constant_value: f64,
}

impl GbmModel {
    pub fn fit(x: &Rows, y: &[f64], cfg: &GbmParams) -> GbmModel {
        Self::fit_with_train_preds(x, y, cfg).0
    }

    pub(crate) fn fit_with_cache(
        x: &Rows,
        y: &[f64],
        cfg: &GbmParams,
        cache: Option<&mut crate::ops::SortedColumnsCache>,
    ) -> GbmModel {
        Self::fit_with_train_preds_and_cache(x, y, cfg, cache).0
    }

    pub fn fit_with_train_preds(x: &Rows, y: &[f64], cfg: &GbmParams) -> (GbmModel, Vec<f64>) {
        Self::fit_with_train_preds_and_cache(x, y, cfg, None)
    }

    pub(crate) fn fit_with_train_preds_and_cache(
        x: &Rows,
        y: &[f64],
        cfg: &GbmParams,
        mut cache: Option<&mut crate::ops::SortedColumnsCache>,
    ) -> (GbmModel, Vec<f64>) {
        let n = x.len().min(y.len());
        let cols = x[0].len();
        let n_feat = cols.min(cfg.max_features);
        if n < 4 || n_feat == 0 {
            let m = mean(&y[..n.max(1).min(y.len())]);
            let f_train = vec![m; x.len()];
            return (
                GbmModel {
                    edges: Vec::new(),
                    trees: Vec::new(),
                    init: m,
                    lr: cfg.lr,
                    n_feat: 0,
                    constant_value: m,
                },
                f_train,
            );
        }
        if n == x.len() {
            if let Some(c) = cache.as_deref_mut() {
                c.record_probe_counts(n_feat);
            }
        }
        let cache_ref = cache.as_deref();
        let edges_and_const: Vec<(Vec<f64>, bool)> =
            if n_feat >= 32 && n_feat * n >= 16_384 && crate::is_parallel_enabled() {
                (0..n_feat)
                    .into_par_iter()
                    .map(|f| {
                        if n == x.len() {
                            if let Some(vals) = cache_ref.and_then(|c| c.get_col_slice(f)) {
                                if vals.len() == n {
                                    return bin_edges_from_sorted(vals, x[0][f], cfg.bins);
                                }
                            }
                        }
                        bin_edges_with_const(x, f, n, cfg.bins)
                    })
                    .collect()
            } else {
                (0..n_feat)
                    .map(|f| {
                        if n == x.len() {
                            if let Some(vals) = cache_ref.and_then(|c| c.get_col_slice(f)) {
                                if vals.len() == n {
                                    return bin_edges_from_sorted(vals, x[0][f], cfg.bins);
                                }
                            }
                        }
                        bin_edges_with_const(x, f, n, cfg.bins)
                    })
                    .collect()
            };
        let mut edges: Vec<Vec<f64>> = Vec::with_capacity(n_feat);
        let mut is_const_col: Vec<bool> = Vec::with_capacity(n_feat);
        for (e, c) in edges_and_const {
            edges.push(e);
            is_const_col.push(c);
        }
        // Training bins are only read one feature at a time (histograms, split partition), so
        // store them feature-major: a node's histogram then walks one contiguous column instead
        // of striding `n_feat` entries per row. Same bin values, same per-bin summation order.
        let train_cols = bin_columns_filtered(x, n, n_feat, &edges, &is_const_col);
        let init = mean(&y[..n]);
        let mut f_train = vec![init; n];
        let nbins = cfg.bins as u16;
        // Exact dead/duplicate feature elimination:
        // 1. A feature column whose training bins are all equal has `hl == h` (`hr == 0 < 1.0`)
        //    on every node and can never split.
        // 2. A feature `f2 > f1` whose training bin column `train_cols[f2*n..(f2+1)*n]` is identical
        //    to an earlier feature `f1` produces the exact same histogram and gain on every node,
        //    and can never win the strict `gain > best_gain` tie against `f1`.
        let mut active_features: Vec<(usize, usize)> = Vec::with_capacity(n_feat);
        let mut active_hashes: Vec<u64> = Vec::with_capacity(n_feat);
        for f in 0..n_feat {
            if is_const_col[f] {
                continue;
            }
            let col = &train_cols[f * n..(f + 1) * n];
            let first = col[0];
            let mut all_same = true;
            let mut h64: u64 = 0xcbf29ce484222325;
            for &b in col {
                if b != first {
                    all_same = false;
                }
                h64 ^= b as u64;
                h64 = h64.wrapping_mul(0x100000001b3);
            }
            if all_same {
                continue;
            }
            let is_dup =
                active_features
                    .iter()
                    .zip(active_hashes.iter())
                    .any(|(&(f_prev, _), &h_prev)| {
                        h_prev == h64 && train_cols[f_prev * n..(f_prev + 1) * n] == *col
                    });
            if !is_dup {
                let f_nbins = (edges[f].len() + 1).min(nbins as usize).max(2);
                active_features.push((f, f_nbins));
                active_hashes.push(h64);
            }
        }
        let mut trees = Vec::with_capacity(cfg.rounds);
        // Reusable (grad, leaf map, histogram, row-partition) buffers, allocated once per fit
        // instead of once per round / per node and feature. Squared loss has a unit hessian, so
        // `build_tree` counts rows where a general booster would sum hessians.
        let mut grad = vec![0.0f64; n];
        let mut train_leaf = vec![0usize; n];
        let mut scratch = HistScratch::new(nbins as usize, cfg.depth, n);
        let mut rows_pool = vec![0usize; cfg.depth.max(1) * n];
        for _ in 0..cfg.rounds {
            for i in 0..n {
                grad[i] = f_train[i] - y[i];
            }
            train_leaf.fill(0);
            let mut tree: Vec<GbmNode> = Vec::with_capacity((1usize << (cfg.depth + 1)).min(64));
            build_tree(
                &train_cols,
                n,
                &grad,
                &[],
                &mut rows_pool,
                &active_features,
                nbins,
                cfg,
                0,
                &mut tree,
                &mut train_leaf,
                &mut scratch,
            );
            for i in 0..n {
                if let GbmNode::Leaf(v) = tree[train_leaf[i]] {
                    f_train[i] += cfg.lr * v;
                }
            }
            trees.push(tree);
        }
        (
            GbmModel {
                edges,
                trees,
                init,
                lr: cfg.lr,
                n_feat,
                constant_value: init,
            },
            f_train,
        )
    }

    pub fn predict(&self, x: &Rows) -> Vec<f64> {
        if self.n_feat == 0 {
            return vec![self.constant_value; x.len()];
        }
        let mut used = vec![false; self.n_feat];
        for tree in &self.trees {
            for node in tree {
                if let GbmNode::Split { feature, .. } = node {
                    if *feature < self.n_feat {
                        used[*feature] = true;
                    }
                }
            }
        }
        let used_features: Vec<usize> = (0..self.n_feat).filter(|&f| used[f]).collect();
        let predict_row = |row: &Vec<f64>| -> f64 {
            let mut row_bins = [0u16; 512];
            if self.n_feat <= 512 {
                for &f in &used_features {
                    let v = if f < row.len() { row[f] } else { 0.0 };
                    row_bins[f] = self.edges[f].partition_point(|e| *e < v) as u16;
                }
                let row_slice = &row_bins[..self.n_feat];
                let mut f = self.init;
                for tree in &self.trees {
                    f += self.lr * route(tree, row_slice, 0, self.n_feat);
                }
                f
            } else {
                let mut dyn_bins = vec![0u16; self.n_feat];
                for &f in &used_features {
                    let v = if f < row.len() { row[f] } else { 0.0 };
                    dyn_bins[f] = self.edges[f].partition_point(|e| *e < v) as u16;
                }
                let mut f = self.init;
                for tree in &self.trees {
                    f += self.lr * route(tree, &dyn_bins, 0, self.n_feat);
                }
                f
            }
        };
        if x.len() * self.trees.len().max(1) >= 32_768 && crate::is_parallel_enabled() {
            x.par_iter().map(predict_row).collect()
        } else {
            x.iter().map(predict_row).collect()
        }
    }
}

/// Reused per-feature histogram and row-partition scratch buffers for [`build_tree`].
struct HistScratch {
    gh: Vec<f64>,
    count: Vec<u32>,
    right_tmp: Vec<usize>,
}

impl HistScratch {
    fn new(nbins: usize, _max_depth: usize, n_rows: usize) -> Self {
        HistScratch {
            gh: vec![0.0; nbins],
            count: vec![0u32; nbins],
            right_tmp: vec![0usize; n_rows],
        }
    }
}

/// Direct feature-major training bins, avoiding a temporary row-major matrix.
#[cfg_attr(not(test), allow(dead_code))]
fn bin_columns(x: &Rows, rows: usize, n_feat: usize, edges: &[Vec<f64>]) -> Vec<u16> {
    let is_const_col = vec![false; n_feat];
    bin_columns_filtered(x, rows, n_feat, edges, &is_const_col)
}

/// Direct feature-major training bins, skipping constant columns before binary search.
fn bin_columns_filtered(
    x: &Rows,
    rows: usize,
    n_feat: usize,
    edges: &[Vec<f64>],
    is_const_col: &[bool],
) -> Vec<u16> {
    let mut out = vec![0u16; rows * n_feat];
    if n_feat >= 32 && rows * n_feat >= 16_384 && rows > 0 && crate::is_parallel_enabled() {
        out.par_chunks_mut(rows).enumerate().for_each(|(f, col)| {
            if is_const_col[f] {
                return;
            }
            let ef = &edges[f];
            for r in 0..rows {
                let v = if f < x[r].len() { x[r][f] } else { 0.0 };
                col[r] = ef.partition_point(|e| *e < v) as u16;
            }
        });
    } else {
        for r in 0..rows {
            let row = &x[r];
            let rlen = row.len();
            for f in 0..n_feat {
                if is_const_col[f] {
                    continue;
                }
                let v = if f < rlen { row[f] } else { 0.0 };
                out[f * rows + r] = edges[f].partition_point(|e| *e < v) as u16;
            }
        }
    }
    out
}

/// Grow one tree over feature-major training bins `cols[f * n_rows + i]`.
///
/// The squared-loss hessian is 1 for every row, so a node's hessian sum is its row count and a
/// bin's is its row count: both are exact in `f64` (far below 2^53), so they equal the sums of
/// unit hessians the booster used to accumulate, bit for bit.
#[allow(clippy::too_many_arguments)]
fn build_tree(
    cols: &[u16],
    n_rows: usize,
    grad: &[f64],
    rows: &[usize],
    rows_pool: &mut [usize],
    active_features: &[(usize, usize)],
    nbins: u16,
    cfg: &GbmParams,
    depth: usize,
    out: &mut Vec<GbmNode>,
    leaf_of: &mut [usize],
    scratch: &mut HistScratch,
) -> usize {
    let is_root = depth == 0;
    let node_len = if is_root { n_rows } else { rows.len() };
    assert!(
        node_len <= u32::MAX as usize,
        "GBM node row count {node_len} exceeds u32::MAX"
    );
    let g: f64 = if is_root {
        grad[..n_rows].iter().copied().sum()
    } else {
        rows.iter().map(|&i| grad[i]).sum()
    };
    let h: f64 = node_len as f64;
    let leaf_value = -g / (h + cfg.lambda);
    let make_leaf =
        |out: &mut Vec<GbmNode>, leaf_of: &mut [usize], rows: &[usize], is_root: bool| -> usize {
            let idx = out.len();
            out.push(GbmNode::Leaf(leaf_value));
            if is_root {
                leaf_of[..n_rows].fill(idx);
            } else {
                for &i in rows {
                    leaf_of[i] = idx;
                }
            }
            idx
        };
    if depth >= cfg.depth || node_len < 2 * cfg.min_leaf || active_features.is_empty() {
        return make_leaf(out, leaf_of, rows, is_root);
    }
    let parent = g * g / (h + cfg.lambda);
    let n_active = active_features.len();
    let (best_gain, best_feat, best_thr) = if n_active >= 16
        && n_active * node_len >= 8_192
        && (nbins as usize) <= 256
        && crate::is_parallel_enabled()
    {
        active_features
            .par_iter()
            .map(|&(f, f_nbins)| {
                let col = &cols[f * n_rows..(f + 1) * n_rows];
                let mut gh = [0.0f64; 256];
                let mut count = [0u32; 256];
                let gh_s = &mut gh[..f_nbins];
                let count_s = &mut count[..f_nbins];
                if is_root {
                    for i in 0..n_rows {
                        let b = col[i] as usize;
                        gh_s[b] += grad[i];
                        count_s[b] += 1;
                    }
                } else {
                    for &i in rows {
                        let b = col[i] as usize;
                        gh_s[b] += grad[i];
                        count_s[b] += 1;
                    }
                }
                let mut f_best_gain = 0.0f64;
                let mut f_best_thr = 0u16;
                let (mut gl, mut hl) = (0.0f64, 0.0f64);
                for b in 0..(f_nbins - 1) {
                    gl += gh_s[b];
                    let cb = count_s[b];
                    hl += cb as f64;
                    if cb == 0 {
                        continue;
                    }
                    let gr = g - gl;
                    let hr = h - hl;
                    if hl < 1.0 || hr < 1.0 {
                        continue;
                    }
                    let gain = gl * gl / (hl + cfg.lambda) + gr * gr / (hr + cfg.lambda) - parent;
                    if gain > f_best_gain {
                        f_best_gain = gain;
                        f_best_thr = b as u16;
                    }
                }
                if f_best_gain > 0.0 {
                    (f_best_gain, f, f_best_thr)
                } else {
                    (0.0f64, usize::MAX, 0u16)
                }
            })
            .reduce(
                || (0.0f64, usize::MAX, 0u16),
                |left, right| {
                    if right.0 > left.0 {
                        right
                    } else {
                        left
                    }
                },
            )
    } else {
        let mut best_gain = 0.0f64;
        let mut best_feat = usize::MAX;
        let mut best_thr = 0u16;
        for &(f, f_nbins) in active_features {
            let col = &cols[f * n_rows..(f + 1) * n_rows];
            let gh = &mut scratch.gh[..f_nbins];
            let count = &mut scratch.count[..f_nbins];
            gh.fill(0.0);
            count.fill(0);
            if is_root {
                for i in 0..n_rows {
                    let b = col[i] as usize;
                    gh[b] += grad[i];
                    count[b] += 1;
                }
            } else {
                for &i in rows {
                    let b = col[i] as usize;
                    gh[b] += grad[i];
                    count[b] += 1;
                }
            }
            let (mut gl, mut hl) = (0.0f64, 0.0f64);
            for b in 0..(f_nbins - 1) {
                gl += gh[b];
                let cb = count[b];
                hl += cb as f64;
                if cb == 0 {
                    continue;
                }
                let gr = g - gl;
                let hr = h - hl;
                if hl < 1.0 || hr < 1.0 {
                    continue;
                }
                let gain = gl * gl / (hl + cfg.lambda) + gr * gr / (hr + cfg.lambda) - parent;
                if gain > best_gain {
                    best_gain = gain;
                    best_feat = f;
                    best_thr = b as u16;
                }
            }
        }
        (best_gain, best_feat, best_thr)
    };
    if best_feat == usize::MAX || best_gain <= 1e-9 {
        return make_leaf(out, leaf_of, rows, is_root);
    }
    let (curr_dest, deeper_pool) = rows_pool.split_at_mut(n_rows);
    let split_col = &cols[best_feat * n_rows..(best_feat + 1) * n_rows];
    let mut n_left = 0usize;
    let mut n_right = 0usize;
    if is_root {
        for (i, &val) in split_col.iter().enumerate().take(n_rows) {
            if val <= best_thr {
                curr_dest[n_left] = i;
                n_left += 1;
            } else {
                scratch.right_tmp[n_right] = i;
                n_right += 1;
            }
        }
    } else {
        for &i in rows {
            if split_col[i] <= best_thr {
                curr_dest[n_left] = i;
                n_left += 1;
            } else {
                scratch.right_tmp[n_right] = i;
                n_right += 1;
            }
        }
    }
    if n_left < cfg.min_leaf || n_right < cfg.min_leaf {
        return make_leaf(out, leaf_of, rows, is_root);
    }
    curr_dest[n_left..n_left + n_right].copy_from_slice(&scratch.right_tmp[..n_right]);
    let (left_rows, right_rows) = curr_dest[..n_left + n_right].split_at(n_left);
    let idx = out.len();
    out.push(GbmNode::Leaf(0.0)); // temporary leaf, overwritten below
    let left = build_tree(
        cols,
        n_rows,
        grad,
        left_rows,
        deeper_pool,
        active_features,
        nbins,
        cfg,
        depth + 1,
        out,
        leaf_of,
        scratch,
    );
    let right = build_tree(
        cols,
        n_rows,
        grad,
        right_rows,
        deeper_pool,
        active_features,
        nbins,
        cfg,
        depth + 1,
        out,
        leaf_of,
        scratch,
    );
    out[idx] = GbmNode::Split {
        feature: best_feat,
        thr_bin: best_thr,
        left,
        right,
    };
    idx
}

fn route(tree: &[GbmNode], bins: &[u16], row: usize, n_feat: usize) -> f64 {
    let mut idx = 0usize;
    loop {
        match &tree[idx] {
            GbmNode::Leaf(v) => return *v,
            GbmNode::Split {
                feature,
                thr_bin,
                left,
                right,
            } => {
                idx = if bins[row * n_feat + feature] <= *thr_bin {
                    *left
                } else {
                    *right
                };
            }
        }
    }
}

fn bin_edges_from_sorted(vals: &[f64], first: f64, bins: usize) -> (Vec<f64>, bool) {
    if vals.is_empty() {
        return (Vec::new(), true);
    }
    if vals[0].to_bits() == vals[vals.len() - 1].to_bits() {
        if bins <= 1 {
            return (Vec::new(), true);
        }
        return (vec![first], true);
    }
    let mut edges = Vec::with_capacity(bins.saturating_sub(1));
    for q in 1..bins {
        let idx = ((vals.len() as f64) * (q as f64 / bins as f64)) as usize;
        edges.push(vals[idx.min(vals.len() - 1)]);
    }
    edges.dedup();
    let is_const = edges.is_empty();
    (edges, is_const)
}

fn bin_edges_with_const(x: &Rows, f: usize, n: usize, bins: usize) -> (Vec<f64>, bool) {
    if n == 0 {
        return (Vec::new(), true);
    }
    let first = x[0][f];
    let first_bits = first.to_bits();
    if (1..n).all(|r| x[r][f].to_bits() == first_bits) {
        if bins <= 1 {
            return (Vec::new(), true);
        }
        return (vec![first], true);
    }
    let mut vals: Vec<f64> = (0..n).map(|r| x[r][f]).collect();
    vals.sort_by(|a, b| a.total_cmp(b));
    bin_edges_from_sorted(&vals, first, bins)
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Pcg32;

    fn reference_row_major_jacobi(a: &mut [f64], p: usize) -> (Vec<f64>, Vec<f64>) {
        let mut v = vec![0.0f64; p * p];
        for i in 0..p {
            v[i * p + i] = 1.0;
        }
        let mut scale = 0.0;
        for i in 0..p {
            scale += a[i * p + i];
        }
        let limit = (scale * 1e-12) * (scale * 1e-12);
        for _ in 0..20 {
            let mut off = 0.0;
            for i in 0..p {
                for j in (i + 1)..p {
                    off += a[i * p + j] * a[i * p + j];
                }
            }
            if off <= limit {
                break;
            }
            for i in 0..p {
                for j in (i + 1)..p {
                    let apq = a[i * p + j];
                    if apq == 0.0 {
                        continue;
                    }
                    let theta = (a[j * p + j] - a[i * p + i]) / (2.0 * apq);
                    let t = if theta >= 0.0 {
                        1.0 / (theta + (theta * theta + 1.0).sqrt())
                    } else {
                        -1.0 / ((theta * theta + 1.0).sqrt() - theta)
                    };
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for k in 0..p {
                        if k != i && k != j {
                            let aki = a[k * p + i];
                            let akj = a[k * p + j];
                            a[k * p + i] = c * aki - s * akj;
                            a[i * p + k] = a[k * p + i];
                            a[k * p + j] = s * aki + c * akj;
                            a[j * p + k] = a[k * p + j];
                        }
                    }
                    let aii = a[i * p + i];
                    let ajj = a[j * p + j];
                    a[i * p + i] = c * c * aii - 2.0 * s * c * apq + s * s * ajj;
                    a[j * p + j] = s * s * aii + 2.0 * s * c * apq + c * c * ajj;
                    a[i * p + j] = 0.0;
                    a[j * p + i] = 0.0;
                    for k in 0..p {
                        let vki = v[k * p + i];
                        let vkj = v[k * p + j];
                        v[k * p + i] = c * vki - s * vkj;
                        v[k * p + j] = s * vki + c * vkj;
                    }
                }
            }
        }
        let mut e = vec![0.0f64; p];
        for i in 0..p {
            e[i] = a[i * p + i];
        }
        (e, v)
    }

    fn reference_row_major_elastic(
        x: &Rows,
        y: &[f64],
        alpha: f64,
        l1_ratio: f64,
        sweeps: usize,
    ) -> ElasticNetModel {
        let n = x.len().min(y.len());
        let p = x[0].len();
        let (means, scales) = std_fit(x);
        let mut xs: Rows = x[..n]
            .iter()
            .map(|r| {
                (0..p)
                    .map(|j| ((r[j] - means[j]) / scales[j]).clamp(-20.0, 20.0))
                    .collect()
            })
            .collect();
        let y_mean = y[..n].iter().sum::<f64>() / n.max(1) as f64;
        let nf = n as f64;
        let l1 = alpha * l1_ratio;
        let l2 = alpha * (1.0 - l1_ratio);
        // Column norms (constant during CD).
        let mut col_sq = vec![0.0f64; p];
        for r in xs.iter() {
            for j in 0..p {
                col_sq[j] += r[j] * r[j];
            }
        }
        let mut w = vec![0.0f64; p];
        let mut resid: Vec<f64> = (0..n).map(|r| y[r] - y_mean).collect();
        for _ in 0..sweeps.max(1) {
            let mut max_delta = 0.0f64;
            for j in 0..p {
                if col_sq[j] <= 1e-12 {
                    continue;
                }
                let mut rho = 0.0;
                for r in 0..n {
                    rho += xs[r][j] * resid[r];
                }
                rho = rho / nf + w[j] * col_sq[j] / nf;
                let denom = col_sq[j] / nf + l2;
                let new_w = soft_threshold(rho, l1) / denom;
                let delta = new_w - w[j];
                if delta != 0.0 {
                    for r in 0..n {
                        resid[r] -= delta * xs[r][j];
                    }
                    w[j] = new_w;
                    let ad = delta.abs();
                    if ad > max_delta {
                        max_delta = ad;
                    }
                }
            }
            if max_delta < 1e-9 {
                break;
            }
        }
        xs.clear();
        ElasticNetModel {
            means,
            scales,
            w0: y_mean,
            w,
        }
    }

    #[test]
    fn transposed_jacobi_matches_397_row_major_bits() {
        for p in 0..=13 {
            for case in 0..4 {
                let mut a = vec![0.0; p * p];
                for i in 0..p {
                    for j in 0..=i {
                        let value = if case == 0 {
                            0.0
                        } else {
                            ((i * 17 + j * 3 + case) as f64 / 11.0).sin()
                        };
                        a[i * p + j] = value;
                        a[j * p + i] = value;
                    }
                    a[i * p + i] += case as f64;
                }
                let mut b = a.clone();
                let (ae, av) = jacobi_eigen(&mut a, p);
                let (be, bv) = reference_row_major_jacobi(&mut b, p);
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&a), bits(&b), "matrix p={p} case={case}");
                assert_eq!(bits(&ae), bits(&be), "eigenvalues p={p} case={case}");
                assert_eq!(bits(&av), bits(&bv), "vectors p={p} case={case}");
            }
        }
    }

    #[test]
    fn column_major_elastic_matches_397_row_major_bits() {
        for n in [1, 7, 24] {
            for p in [0, 1, 3, 8] {
                let x: Rows = (0..n)
                    .map(|i| {
                        (0..p)
                            .map(|j| {
                                if j == 0 {
                                    -0.0
                                } else {
                                    ((i * 7 + j) as f64).sin()
                                }
                            })
                            .collect()
                    })
                    .collect();
                let y: Vec<f64> = (0..n).map(|i| i as f64 / 3.0 - 2.0).collect();
                for sweeps in [0, 1, 5] {
                    for ratio in [0.0, 0.5, 1.0] {
                        let a = ElasticNetModel::fit(&x, &y, 0.1, ratio, sweeps);
                        let b = reference_row_major_elastic(&x, &y, 0.1, ratio, sweeps);
                        assert_eq!(
                            serde_json::to_vec(&a).unwrap(),
                            serde_json::to_vec(&b).unwrap(),
                            "n={n} p={p} sweeps={sweeps} ratio={ratio}"
                        );
                        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                        assert_eq!(bits(&a.predict(&x)), bits(&b.predict(&x)));
                    }
                }
            }
        }
    }

    fn bin_rows(x: &Rows, rows: usize, n_feat: usize, edges: &[Vec<f64>]) -> Vec<u16> {
        let mut out = vec![0u16; rows * n_feat];
        for r in 0..rows {
            for (f, edge) in edges.iter().enumerate().take(n_feat) {
                let v = if f < x[r].len() { x[r][f] } else { 0.0 };
                out[r * n_feat + f] = edge.partition_point(|e| *e < v) as u16;
            }
        }
        out
    }

    #[test]
    fn direct_feature_major_bins_match_397_transpose() {
        let x = vec![vec![-0.0, f64::NAN], vec![f64::INFINITY], vec![-1.0, 4.0]];
        let edges = vec![vec![-1.0, 0.0, 2.0], vec![0.0, 3.0, 5.0]];
        let row_bins = bin_rows(&x, 3, 2, &edges);
        let direct = bin_columns(&x, 3, 2, &edges);
        for r in 0..3 {
            for f in 0..2 {
                assert_eq!(direct[f * 3 + r], row_bins[r * 2 + f]);
            }
        }
    }

    /// The textbook row-major loop `cholesky_factor` must reproduce bit for bit.
    fn reference_cholesky_factor(a: &mut [f64], p: usize) {
        for i in 0..p {
            for j in 0..=i {
                let mut s = a[i * p + j];
                for k in 0..j {
                    s -= a[i * p + k] * a[j * p + k];
                }
                if i == j {
                    a[i * p + j] = if s > 1e-12 { s.sqrt() } else { 1e-6 };
                } else {
                    a[i * p + j] = s / a[j * p + j];
                }
            }
        }
    }

    #[test]
    fn blocked_cholesky_is_bit_identical_to_the_row_major_loop() {
        let mut rng = Pcg32::new(0xc401, 3);
        let specials = [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.0,
            0.0,
            1e-300,
        ];
        // Every block remainder (p mod 4), SPD and non-SPD inputs (the 1e-6 branch), and
        // non-finite entries.
        for p in 0..=37usize {
            for case in 0..6 {
                let mut a = vec![0.0f64; p * p];
                match case {
                    0 | 1 => {
                        // SPD: Gram matrix plus a ridge, as the heads build it.
                        let m = p + 3;
                        let z: Vec<f64> = (0..m * p).map(|_| rng.gaussian()).collect();
                        for i in 0..p {
                            for j in 0..p {
                                a[i * p + j] = (0..m).map(|r| z[r * p + i] * z[r * p + j]).sum();
                            }
                            a[i * p + i] += if case == 0 { 1e-3 } else { 10.0 };
                        }
                    }
                    2 | 3 => {
                        // Symmetric indefinite: many non-positive pivots.
                        for i in 0..p {
                            for j in 0..=i {
                                let v = rng.next_f64() * 4.0 - 2.0;
                                a[i * p + j] = v;
                                a[j * p + i] = v;
                            }
                        }
                    }
                    _ => {
                        for v in a.iter_mut() {
                            *v = rng.gaussian();
                        }
                        for _ in 0..(p / 3 + 1).min(p * p) {
                            let at = (rng.next_f64() * (p * p) as f64) as usize % (p * p);
                            let pick = (rng.next_f64() * specials.len() as f64) as usize;
                            a[at] = specials[pick.min(specials.len() - 1)];
                        }
                    }
                }
                let mut want = a.clone();
                reference_cholesky_factor(&mut want, p);
                let mut got = a.clone();
                cholesky_factor(&mut got, p);
                for (w, g) in want.iter().zip(&got) {
                    assert_eq!(w.to_bits(), g.to_bits(), "p={p} case={case}");
                }
                let b: Vec<f64> = (0..p).map(|_| rng.gaussian()).collect();
                let (mut a1, mut b1) = (a.clone(), b.clone());
                let solved = cholesky_solve(&mut a1, &mut b1, p);
                let mut a2 = a.clone();
                reference_cholesky_factor(&mut a2, p);
                let mut z = vec![0.0f64; p];
                for i in 0..p {
                    let mut s = b[i];
                    for k in 0..i {
                        s -= a2[i * p + k] * z[k];
                    }
                    z[i] = s / a2[i * p + i];
                }
                let mut w = vec![0.0f64; p];
                for i in (0..p).rev() {
                    let mut s = z[i];
                    for k in (i + 1)..p {
                        s -= a2[k * p + i] * w[k];
                    }
                    w[i] = s / a2[i * p + i];
                }
                for (x, y) in solved.iter().zip(&w) {
                    assert_eq!(x.to_bits(), y.to_bits(), "solve p={p} case={case}");
                }
            }
        }
    }

    #[test]
    fn hurdle2_is_deterministic_and_learns_bimodal_expectation() {
        let mut rng = Pcg32::new(0x57_20_4b, 17);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..600 {
            let gate = rng.next_f64() * 2.0 - 1.0;
            let severity = rng.next_f64() * 2.0 - 1.0;
            let noise = 0.03 * rng.gaussian();
            let target = if gate > 0.72 {
                8.0 + 0.7 * severity + noise
            } else {
                1.0 - 0.4 * severity + noise
            };
            x.push(vec![gate, severity]);
            y.push(target);
        }
        let train_x = x[..400].to_vec();
        let test_x = x[400..].to_vec();
        let train_y = &y[..400];
        let test_y = &y[400..];
        let first = Hurdle2Model::fit(&train_x, train_y, 0.86, 0.03, 0.2, 0.05);
        let second = Hurdle2Model::fit(&train_x, train_y, 0.86, 0.03, 0.2, 0.05);
        let hurdle = first.predict(&test_x);
        assert_eq!(hurdle, second.predict(&test_x));
        assert!(hurdle.iter().all(|value| value.is_finite()));
        let ridge = RidgeModel::fit(&train_x, train_y, 0.2).predict(&test_x);
        let hurdle_rmse = crate::metrics::rmse(test_y, &hurdle);
        let ridge_rmse = crate::metrics::rmse(test_y, &ridge);
        assert!(
            hurdle_rmse < ridge_rmse * 0.95,
            "hurdle {hurdle_rmse} should materially beat ridge {ridge_rmse}"
        );
    }

    #[test]
    fn hurdle2_constant_target_uses_finite_mean_fallback() {
        let x = (0..40).map(|row| vec![row as f64, 1.0]).collect::<Rows>();
        let y = vec![4.25; x.len()];
        let model = Hurdle2Model::fit(&x, &y, 0.86, 0.03, 0.2, 0.05);
        for prediction in model.predict(&x) {
            assert!((prediction - 4.25).abs() < 1e-12, "prediction={prediction}");
        }
    }

    #[test]
    fn hurdle2_is_target_scale_equivariant() {
        let x = (0..120)
            .map(|row| {
                let gate = row as f64 / 119.0;
                vec![gate, (gate * 7.0).sin()]
            })
            .collect::<Rows>();
        let y = x
            .iter()
            .map(|row| {
                if row[0] > 0.82 {
                    4.0 + row[1]
                } else {
                    0.5 - 0.2 * row[1]
                }
            })
            .collect::<Vec<_>>();
        let scaled_y = y.iter().map(|value| value * 1_000.0).collect::<Vec<_>>();
        let base = Hurdle2Model::fit(&x, &y, 0.82, 0.03, 0.2, 0.05).predict(&x);
        let scaled = Hurdle2Model::fit(&x, &scaled_y, 0.82, 0.03, 0.2, 0.05).predict(&x);
        for (base, scaled) in base.iter().zip(scaled) {
            let delta = (scaled - 1_000.0 * base).abs();
            let tolerance = 1e-6 * scaled.abs().max(1.0);
            assert!(
                delta <= tolerance,
                "scaled={scaled} expected={} delta={delta} tolerance={tolerance}",
                1_000.0 * base
            );
        }
    }

    #[test]
    fn jacobi_eigen_recovers_known_spectrum() {
        // [[2, 1], [1, 2]] has eigenvalues {1, 3}.
        let mut a = vec![2.0, 1.0, 1.0, 2.0];
        let (mut e, _) = jacobi_eigen(&mut a, 2);
        e.sort_by(|x, y| x.total_cmp(y));
        assert!((e[0] - 1.0).abs() < 1e-12, "{e:?}");
        assert!((e[1] - 3.0).abs() < 1e-12, "{e:?}");

        // Reconstruction A = V diag(e) V^T on a seeded random symmetric PSD
        // 8x8 (built as B^T B) — catches any rotation sign error.
        let p = 8;
        let mut rng = Pcg32::new(11, 7);
        let b: Vec<f64> = (0..p * p).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
        let mut a = vec![0.0f64; p * p];
        for i in 0..p {
            for j in 0..p {
                let mut s = 0.0;
                for k in 0..p {
                    s += b[k * p + i] * b[k * p + j];
                }
                a[i * p + j] = s;
            }
        }
        let orig = a.clone();
        let (e, v) = jacobi_eigen(&mut a, p);
        for i in 0..p {
            for j in 0..p {
                let mut s = 0.0;
                for k in 0..p {
                    s += v[i * p + k] * e[k] * v[j * p + k];
                }
                assert!(
                    (s - orig[i * p + j]).abs() < 1e-9,
                    "reconstruction off at ({i}, {j}): {s} vs {}",
                    orig[i * p + j]
                );
            }
        }
    }

    #[test]
    fn ridge_gcv_beats_fixed_ridge_on_noisy_wide_data() {
        // Wide, heavily noisy fixture: the penalty that generalizes sits far
        // above ridge's default lambda = 0.02 * n, so the GCV pick should win
        // on held-out rows.
        let (n_train, n_test, p) = (60usize, 120usize, 40usize);
        let mut rng = Pcg32::new(4242, 9);
        let mut x: Rows = Vec::new();
        let mut y = Vec::new();
        for _ in 0..(n_train + n_test) {
            let row: Vec<f64> = (0..p).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
            let t = 0.2 * row[0] + 2.0 * rng.gaussian();
            x.push(row);
            y.push(t);
        }
        let x_tr: Rows = x[..n_train].to_vec();
        let x_te: Rows = x[n_train..].to_vec();
        let (y_tr, y_te) = (&y[..n_train], &y[n_train..]);
        let gcv = RidgeGcvModel::fit(&x_tr, y_tr).predict(&x_te);
        let fixed = RidgeModel::fit(&x_tr, y_tr, 0.02).predict(&x_te);
        let rmse_gcv = crate::metrics::rmse(y_te, &gcv);
        let rmse_fixed = crate::metrics::rmse(y_te, &fixed);
        assert!(
            rmse_gcv < rmse_fixed,
            "GCV ridge should beat fixed ridge here: {rmse_gcv} vs {rmse_fixed}"
        );
    }

    #[test]
    fn ridge_gcv_degenerate_constant_target() {
        let x: Rows = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![5.0, 5.0, 5.0];
        let m = RidgeGcvModel::fit(&x, &y);
        for v in m.predict(&x) {
            assert_eq!(v, 5.0);
        }
    }

    fn gcv_fixture(n: usize, p: usize, seed: u64) -> (Rows, Vec<f64>) {
        let mut rng = Pcg32::new(seed, 5);
        let mut x: Rows = Vec::new();
        let mut y = Vec::new();
        for _ in 0..n {
            let row: Vec<f64> = (0..p).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
            let t = 1.3 * row[0] - 0.7 * row[1] + 0.4 * row[2 % p] + 0.5 * rng.gaussian();
            x.push(row);
            y.push(t);
        }
        (x, y)
    }

    #[test]
    fn ridge_gcv_dual_matches_primal() {
        // n < p auto-routes to the dual (kernel) eigendecomposition; the
        // force_primal hook replays the p x p Gram route. Identical ridge
        // algebra, so weights and predictions agree to roundoff.
        for (n, p, seed) in [(25usize, 40usize, 101u64), (30, 90, 202)] {
            let (x, y) = gcv_fixture(n, p, seed);
            let dual = RidgeGcvModel::fit(&x, &y);
            let primal = RidgeGcvModel::fit_inner(&x, &y, true);
            assert_eq!(dual.cols, primal.cols);
            assert_eq!(dual.w.len(), primal.w.len());
            for (a, b) in dual.w.iter().zip(&primal.w) {
                assert!((a - b).abs() < 1e-9, "weights differ: {a} vs {b}");
            }
            let (xt, _) = gcv_fixture(50, p, seed + 1);
            for (a, b) in dual.predict(&xt).iter().zip(primal.predict(&xt)) {
                assert!((a - b).abs() < 1e-9, "predictions differ: {a} vs {b}");
            }
        }
        // n >= p: fit takes the primal route already, so the hook is a no-op
        // and the results are bit-identical.
        let (x, y) = gcv_fixture(80, 12, 303);
        let auto = RidgeGcvModel::fit(&x, &y);
        let primal = RidgeGcvModel::fit_inner(&x, &y, true);
        assert_eq!(auto.w, primal.w);
    }

    #[test]
    fn ridge_gcv_screen_ranks_by_abs_corr() {
        // Column strengths engineered so the |corr| ranking is known:
        // c0 = noise, c1 = y (corr 1), c2 = weak signal, c3 = -y (tie with
        // c1 on |corr|; both beat c2). cap=3 keeps {1, 2, 3}, ascending.
        let mut rng = Pcg32::new(9, 13);
        let n = 200usize;
        let mut x: Rows = Vec::new();
        let mut y = Vec::new();
        for _ in 0..n {
            let t = rng.gaussian();
            let noise = rng.gaussian();
            x.push(vec![noise, t, 0.3 * t + rng.gaussian(), -t]);
            y.push(t);
        }
        let (means, mut scales) = std_fit(&x);
        for s in scales.iter_mut() {
            if *s <= 1e-9 {
                *s = 1.0;
            }
        }
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let sel = gcv_screen_columns(&x, &y, n, &means, &scales, y_mean, 3);
        assert_eq!(sel, vec![1, 2, 3]);
        // With a tie between c1 and c3 and cap=1, the smaller index wins.
        let sel1 = gcv_screen_columns(&x, &y, n, &means, &scales, y_mean, 1);
        assert_eq!(sel1, vec![1]);
    }

    #[test]
    fn ridge_gcv_screen_path_end_to_end() {
        // Both sides above GCV_SCREEN_CAP: the fit must keep exactly 384
        // ascending unique columns and stay finite. Kept small enough (n =
        // p = 390) to run in debug while still crossing the 384 threshold.
        let (x, y) = gcv_fixture(390, 390, 404);
        let m = RidgeGcvModel::fit(&x, &y);
        assert_eq!(m.cols.len(), GCV_SCREEN_CAP);
        assert_eq!(m.w.len(), GCV_SCREEN_CAP + 1);
        assert!(m.cols.windows(2).all(|w| w[0] < w[1]), "not ascending");
        assert!(*m.cols.last().unwrap() < 390);
        for v in m.predict(&x) {
            assert!(v.is_finite());
        }
    }

    /// Perf probe for the wide-block fixes (dual route + width screen); run
    /// with: cargo test -p feat-math --release ridge_gcv_wide_perf -- \
    /// --ignored --nocapture. Budget: < 2 s per fit single-threaded.
    #[test]
    #[ignore = "perf probe: run under --release with --ignored --nocapture"]
    fn ridge_gcv_wide_perf() {
        // (a) 62 rows x 130 columns: the shape that took ~62 s primal.
        let (x, y) = gcv_fixture(62, 130, 51);
        let t0 = std::time::Instant::now();
        let m = RidgeGcvModel::fit(&x, &y);
        let secs_a = t0.elapsed().as_secs_f64();
        assert!(m.predict(&x).iter().all(|v| v.is_finite()));
        // (b) 1000 rows x 512 columns: screen to 384, then the primal route.
        let (x, y) = gcv_fixture(1000, 512, 52);
        let t1 = std::time::Instant::now();
        let m = RidgeGcvModel::fit(&x, &y);
        let secs_b = t1.elapsed().as_secs_f64();
        assert_eq!(m.cols.len(), GCV_SCREEN_CAP);
        assert!(m.predict(&x).iter().all(|v| v.is_finite()));
        println!("ridge_gcv perf: n=62/p=130 fit {secs_a:.3}s; n=1000/p=512 fit {secs_b:.3}s");
        assert!(secs_a < 2.0, "n=62/p=130 too slow: {secs_a:.3}s");
        assert!(secs_b < 2.0, "n=1000/p=512 too slow: {secs_b:.3}s");
    }

    #[test]
    fn jacobi_eigen_contiguous_range3_matches_scalar_reference_bits() {
        let mut rng = Pcg32::new(0x9a37_79b9, 19);
        for p in [0usize, 1, 2, 3, 8, 17, 36] {
            let mut a_sym = vec![0.0f64; p * p];
            for i in 0..p {
                for j in i..p {
                    let v = rng.gaussian();
                    a_sym[i * p + j] = v;
                    a_sym[j * p + i] = v;
                }
            }
            let mut a_got = a_sym.clone();
            let mut a_ref = a_sym.clone();
            let (e_got, v_got) = jacobi_eigen(&mut a_got, p);
            let (e_ref, v_ref) = jacobi_eigen_scalar_reference(&mut a_ref, p);
            for (x, y) in e_got.iter().zip(&e_ref) {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "eigenvalue bits mismatch at p={p}"
                );
            }
            for (x, y) in v_got.iter().zip(&v_ref) {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "eigenvector bits mismatch at p={p}"
                );
            }
            for (x, y) in a_got.iter().zip(&a_ref) {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "overwritten matrix bits mismatch at p={p}"
                );
            }

            // Asymmetric entry fallback check
            if p >= 3 {
                let mut a_asym = a_sym.clone();
                a_asym[p] += 1e-7; // break bitwise symmetry at (1, 0)
                let mut a_asym_got = a_asym.clone();
                let mut a_asym_ref = a_asym.clone();
                let (ea_got, va_got) = jacobi_eigen(&mut a_asym_got, p);
                let (ea_ref, va_ref) = jacobi_eigen_scalar_reference(&mut a_asym_ref, p);
                for (x, y) in ea_got.iter().zip(&ea_ref) {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "asym fallback e mismatch at p={p}"
                    );
                }
                for (x, y) in va_got.iter().zip(&va_ref) {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "asym fallback v mismatch at p={p}"
                    );
                }
            }
        }
    }

    #[test]
    fn gbm_zero_alloc_partition_and_const_skip_preserve_exact_bits() {
        let mut rng = Pcg32::new(0x7157_624d, 31);
        let n = 96usize;
        let mut x: Rows = Vec::with_capacity(n);
        let mut y: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let f0 = rng.gaussian();
            let f1 = 4.25; // constant column
            let f2 = (i as f64 * 0.13).sin();
            let f3 = f0; // duplicate of f0
            let f4 = rng.gaussian();
            x.push(vec![f0, f1, f2, f3, f4]);
            y.push(1.5 * f0 - 0.8 * f2 + 0.4 * f4 * f4 + 0.1 * rng.gaussian());
        }
        let cfg = GbmParams {
            rounds: 25,
            lr: 0.1,
            depth: 3,
            bins: 16,
            lambda: 1.0,
            min_leaf: 3,
            max_features: 64,
        };
        let (m1, tr1) = GbmModel::fit_with_train_preds(&x, &y, &cfg);
        let (m2, tr2) = GbmModel::fit_with_train_preds(&x, &y, &cfg);
        assert_eq!(
            serde_json::to_vec(&m1).unwrap(),
            serde_json::to_vec(&m2).unwrap()
        );
        for (a, b) in tr1.iter().zip(&tr2) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        // Constant column f1 has exact 1-element edge vector [4.25] and is never chosen as split
        assert_eq!(m1.edges[1], vec![4.25]);
        // Duplicate column f3 is never chosen over f0
        for tree in &m1.trees {
            for node in tree {
                if let GbmNode::Split { feature, .. } = node {
                    assert_ne!(*feature, 1);
                    assert_ne!(*feature, 3);
                }
            }
        }
        let preds = m1.predict(&x);
        assert_eq!(preds.len(), n);
        assert!(preds.iter().all(|v| v.is_finite()));
        for (a, b) in preds.iter().zip(&tr1) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
