use super::input::str_param;
use super::transforms::{fit_pls_weights, V3Op};
use super::types::{stratified_fold_assignments, TaskKind};
use crate::dsl::Stage;
use crate::heads::{fit_head, FittedHead, RidgeModel};
use crate::ops::quantile_sorted;
use crate::{Pcg32, Rows};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

// ------------------------------- v3 heads --------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum ClassWeight {
    None,
    Balanced,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct LogitRidgeModel {
    pub(super) means: Vec<f64>,
    pub(super) scales: Vec<f64>,
    pub(super) weights: Vec<f64>,
}

impl LogitRidgeModel {
    fn fit(x: &Rows, y: &[f64], l2: f64, class_weight: ClassWeight) -> Result<Self> {
        validate_binary_fit_target(y, x.len())?;
        let n = x.len();
        let cols = x[0].len();
        let mut means = vec![0.0; cols];
        for row in x {
            for j in 0..cols {
                means[j] += row[j] / n as f64;
            }
        }
        let mut scales = vec![0.0; cols];
        for row in x {
            for j in 0..cols {
                let delta = row[j] - means[j];
                scales[j] += delta * delta;
            }
        }
        for scale in &mut scales {
            *scale = (*scale / n as f64).sqrt().max(1e-12);
        }
        let positives = y.iter().filter(|v| **v == 1.0).count();
        let negatives = n - positives;
        let pos_weight = if class_weight == ClassWeight::Balanced {
            n as f64 / (2.0 * positives as f64)
        } else {
            1.0
        };
        let neg_weight = if class_weight == ClassWeight::Balanced {
            n as f64 / (2.0 * negatives as f64)
        } else {
            1.0
        };
        let mut weights = vec![0.0; cols + 1];
        weights[0] = ((positives as f64 + 0.5) / (negatives as f64 + 0.5)).ln();
        let l2 = l2.clamp(1e-9, 1e6);
        for _ in 0..100 {
            let mut gradient = vec![0.0; cols + 1];
            let mut diagonal = vec![1e-9; cols + 1];
            for (row, &label) in x.iter().zip(y) {
                let sample_weight = if label == 1.0 { pos_weight } else { neg_weight };
                let mut linear = weights[0];
                for j in 0..cols {
                    linear += weights[j + 1] * ((row[j] - means[j]) / scales[j]).clamp(-30.0, 30.0);
                }
                let probability = sigmoid(linear);
                let error = sample_weight * (probability - label);
                let curvature = sample_weight * probability * (1.0 - probability);
                gradient[0] += error;
                diagonal[0] += curvature;
                for j in 0..cols {
                    let value = ((row[j] - means[j]) / scales[j]).clamp(-30.0, 30.0);
                    gradient[j + 1] += error * value;
                    diagonal[j + 1] += curvature * value * value;
                }
            }
            let mut max_delta = 0.0f64;
            let intercept_delta = (gradient[0] / diagonal[0]).clamp(-2.0, 2.0);
            weights[0] -= intercept_delta;
            max_delta = max_delta.max(intercept_delta.abs());
            for j in 0..cols {
                let gradient = gradient[j + 1] + l2 * weights[j + 1];
                let hessian = diagonal[j + 1] + l2;
                let delta = (gradient / hessian).clamp(-1.0, 1.0);
                weights[j + 1] -= delta;
                max_delta = max_delta.max(delta.abs());
            }
            if max_delta < 1e-9 {
                break;
            }
        }
        Ok(Self {
            means,
            scales,
            weights,
        })
    }

    fn predict(&self, x: &Rows) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut linear = self.weights[0];
                for j in 0..self.means.len().min(row.len()) {
                    linear += self.weights[j + 1]
                        * ((row[j] - self.means[j]) / self.scales[j]).clamp(-30.0, 30.0);
                }
                sigmoid(linear)
            })
            .collect()
    }
}

pub(super) fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value.min(40.0)).exp())
    } else {
        let z = value.max(-40.0).exp();
        z / (1.0 + z)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct BoostStump {
    pub(super) feature: usize,
    pub(super) threshold: f64,
    pub(super) left: f64,
    pub(super) right: f64,
    pub(super) missing_left: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TabBoostModel {
    pub(super) base: f64,
    pub(super) learning_rate: f64,
    pub(super) binary: bool,
    pub(super) stumps: Vec<BoostStump>,
}

impl TabBoostModel {
    fn fit(x: &Rows, y: &[f64], stage: &Stage, binary: bool) -> Result<Self> {
        if binary {
            validate_binary_fit_target(y, x.len())?;
        }
        let rounds = stage.get_usize("rounds", 100).clamp(1, 500);
        let bins = stage.get_usize("bins", 32).clamp(8, 256);
        let min_leaf = stage.get_usize("min_leaf", 5).clamp(1, 100);
        let learning_rate = stage.get_f64("lr", 0.05).clamp(0.001, 1.0);
        let lambda = stage.get_f64("lambda", 1.0).clamp(0.0, 1e6);
        let loss = str_param(stage, "loss", if binary { "log_loss" } else { "squared" })?;
        if (binary && loss != "log_loss") || (!binary && !matches!(loss, "squared" | "huber")) {
            bail!("tab_boost: loss is log_loss for binary and squared|huber for regression");
        }
        let huber_delta = if loss == "huber" {
            stage.get_f64("delta", 1.0).clamp(1e-6, 1e6)
        } else {
            f64::INFINITY
        };
        let mean = y.iter().take(x.len()).sum::<f64>() / x.len() as f64;
        let base = if binary {
            ((mean.clamp(1e-6, 1.0 - 1e-6)) / (1.0 - mean.clamp(1e-6, 1.0 - 1e-6))).ln()
        } else {
            mean
        };
        let mut raw = vec![base; x.len()];
        let mut stumps = Vec::new();
        for _ in 0..rounds {
            let mut best: Option<(f64, BoostStump)> = None;
            for feature in 0..x[0].len() {
                let mut sorted: Vec<f64> = x
                    .iter()
                    .map(|row| row[feature])
                    .filter(|v| v.is_finite())
                    .collect();
                sorted.sort_by(f64::total_cmp);
                if sorted.len() < 2 * min_leaf {
                    continue;
                }
                for b in 1..bins {
                    let threshold = quantile_sorted(&sorted, b as f64 / bins as f64);
                    for missing_left in [false, true] {
                        let mut gl = 0.0;
                        let mut gr = 0.0;
                        let mut hl = 0.0;
                        let mut hr = 0.0;
                        let mut nl = 0usize;
                        let mut nr = 0usize;
                        for r in 0..x.len() {
                            let probability = sigmoid(raw[r]);
                            let (gradient, hessian) = if binary {
                                (y[r] - probability, probability * (1.0 - probability))
                            } else {
                                ((y[r] - raw[r]).clamp(-huber_delta, huber_delta), 1.0)
                            };
                            let left = if x[r][feature].is_finite() {
                                x[r][feature] <= threshold
                            } else {
                                missing_left
                            };
                            if left {
                                gl += gradient;
                                hl += hessian;
                                nl += 1;
                            } else {
                                gr += gradient;
                                hr += hessian;
                                nr += 1;
                            }
                        }
                        if nl < min_leaf || nr < min_leaf {
                            continue;
                        }
                        let left = gl / (hl + lambda).max(1e-12);
                        let right = gr / (hr + lambda).max(1e-12);
                        let gain =
                            gl * gl / (hl + lambda).max(1e-12) + gr * gr / (hr + lambda).max(1e-12);
                        if best.as_ref().is_none_or(|entry| gain > entry.0) {
                            best = Some((
                                gain,
                                BoostStump {
                                    feature,
                                    threshold,
                                    left,
                                    right,
                                    missing_left,
                                },
                            ));
                        }
                    }
                }
            }
            let Some((_, stump)) = best else { break };
            for (row, prediction) in x.iter().zip(&mut raw) {
                let left = if row[stump.feature].is_finite() {
                    row[stump.feature] <= stump.threshold
                } else {
                    stump.missing_left
                };
                *prediction += learning_rate * if left { stump.left } else { stump.right };
            }
            stumps.push(stump);
        }
        Ok(Self {
            base,
            learning_rate,
            binary,
            stumps,
        })
    }

    fn predict(&self, x: &Rows) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut value = self.base;
                for stump in &self.stumps {
                    let left = if row[stump.feature].is_finite() {
                        row[stump.feature] <= stump.threshold
                    } else {
                        stump.missing_left
                    };
                    value += self.learning_rate * if left { stump.left } else { stump.right };
                }
                if self.binary {
                    sigmoid(value)
                } else {
                    finite(value)
                }
            })
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) enum ExtraNode {
    Leaf(f64),
    Split {
        feature: usize,
        threshold: f64,
        missing_left: bool,
        left: Box<ExtraNode>,
        right: Box<ExtraNode>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ExtraTreesModel {
    pub(super) binary: bool,
    pub(super) trees: Vec<ExtraNode>,
}

impl ExtraTreesModel {
    fn fit(x: &Rows, y: &[f64], stage: &Stage, binary: bool) -> Result<Self> {
        if binary {
            validate_binary_fit_target(y, x.len())?;
        }
        let trees = stage.get_usize("trees", 128).clamp(1, 512);
        let depth = stage.get_usize("depth", 8).clamp(1, 20);
        let min_leaf = stage.get_usize("min_leaf", 2).clamp(1, 100);
        let max_features = stage
            .get_usize("max_features", (x[0].len() as f64).sqrt().ceil() as usize)
            .clamp(1, x[0].len());
        let seed = stage.get_u64("seed", 7);
        let mut rng = Pcg32::for_stage(seed, x[0].len(), trees);
        let rows: Vec<usize> = (0..x.len()).collect();
        let mut fitted = Vec::with_capacity(trees);
        for _ in 0..trees {
            fitted.push(fit_extra_node(
                x,
                y,
                &rows,
                depth,
                min_leaf,
                max_features,
                binary,
                &mut rng,
            ));
        }
        Ok(Self {
            binary,
            trees: fitted,
        })
    }

    fn predict(&self, x: &Rows) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let value = self
                    .trees
                    .iter()
                    .map(|tree| predict_extra(tree, row))
                    .sum::<f64>()
                    / self.trees.len().max(1) as f64;
                if self.binary {
                    value.clamp(0.0, 1.0)
                } else {
                    finite(value)
                }
            })
            .collect()
    }
}

fn fit_extra_node(
    x: &Rows,
    y: &[f64],
    rows: &[usize],
    depth: usize,
    min_leaf: usize,
    max_features: usize,
    binary: bool,
    rng: &mut Pcg32,
) -> ExtraNode {
    if depth == 0 || rows.len() < 2 * min_leaf {
        return ExtraNode::Leaf(extra_leaf(y, rows, binary));
    }
    let cols = x[0].len();
    let start = rng.next_u32() as usize % cols;
    let step = (rng.next_u32() as usize % cols).max(1);
    let mut best: Option<(f64, usize, f64, bool, Vec<usize>, Vec<usize>)> = None;
    let mut visited = BTreeSet::new();
    let mut feature = start;
    while visited.len() < max_features {
        if !visited.insert(feature) {
            feature = (feature + 1) % cols;
            continue;
        }
        let finite_values: Vec<f64> = rows
            .iter()
            .map(|&r| x[r][feature])
            .filter(|v| v.is_finite())
            .collect();
        if let (Some(lo), Some(hi)) = (
            finite_values.iter().copied().min_by(f64::total_cmp),
            finite_values.iter().copied().max_by(f64::total_cmp),
        ) {
            if hi > lo {
                let threshold = rng.range_f64(lo, hi);
                for missing_left in [false, true] {
                    let (left, right): (Vec<usize>, Vec<usize>) =
                        rows.iter().copied().partition(|&r| {
                            if x[r][feature].is_finite() {
                                x[r][feature] <= threshold
                            } else {
                                missing_left
                            }
                        });
                    if left.len() < min_leaf || right.len() < min_leaf {
                        continue;
                    }
                    let score = split_variance(y, &left) + split_variance(y, &right);
                    if best.as_ref().is_none_or(|entry| score < entry.0) {
                        best = Some((score, feature, threshold, missing_left, left, right));
                    }
                }
            }
        }
        feature = (feature + step) % cols;
    }
    let Some((_, feature, threshold, missing_left, left_rows, right_rows)) = best else {
        return ExtraNode::Leaf(extra_leaf(y, rows, binary));
    };
    ExtraNode::Split {
        feature,
        threshold,
        missing_left,
        left: Box::new(fit_extra_node(
            x,
            y,
            &left_rows,
            depth - 1,
            min_leaf,
            max_features,
            binary,
            rng,
        )),
        right: Box::new(fit_extra_node(
            x,
            y,
            &right_rows,
            depth - 1,
            min_leaf,
            max_features,
            binary,
            rng,
        )),
    }
}

fn split_variance(y: &[f64], rows: &[usize]) -> f64 {
    let mean = rows.iter().map(|&r| y[r]).sum::<f64>() / rows.len().max(1) as f64;
    rows.iter().map(|&r| (y[r] - mean).powi(2)).sum()
}

fn extra_leaf(y: &[f64], rows: &[usize], binary: bool) -> f64 {
    let sum = rows.iter().map(|&r| y[r]).sum::<f64>();
    if binary {
        (sum + 1.0) / (rows.len() as f64 + 2.0)
    } else {
        sum / rows.len().max(1) as f64
    }
}

fn predict_extra(node: &ExtraNode, row: &[f64]) -> f64 {
    match node {
        ExtraNode::Leaf(value) => *value,
        ExtraNode::Split {
            feature,
            threshold,
            missing_left,
            left,
            right,
        } => {
            let go_left = if row[*feature].is_finite() {
                row[*feature] <= *threshold
            } else {
                *missing_left
            };
            predict_extra(if go_left { left } else { right }, row)
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct GamTerm {
    pub(super) feature: usize,
    pub(super) edges: Vec<f64>,
    pub(super) values: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct GamBoostModel {
    pub(super) base: f64,
    pub(super) learning_rate: f64,
    pub(super) binary: bool,
    pub(super) terms: Vec<GamTerm>,
}

impl GamBoostModel {
    fn fit(x: &Rows, y: &[f64], stage: &Stage, binary: bool) -> Result<Self> {
        if binary {
            validate_binary_fit_target(y, x.len())?;
        }
        let rounds = stage.get_usize("rounds", 100).clamp(1, 500);
        let bins = stage.get_usize("bins", 16).clamp(4, 64);
        let learning_rate = stage.get_f64("lr", 0.05).clamp(0.001, 1.0);
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        let base = if binary {
            (mean.clamp(1e-6, 1.0 - 1e-6) / (1.0 - mean.clamp(1e-6, 1.0 - 1e-6))).ln()
        } else {
            mean
        };
        let mut raw = vec![base; x.len()];
        let mut terms = Vec::new();
        let mut all_edges = Vec::new();
        for feature in 0..x[0].len() {
            let mut sorted: Vec<f64> = x.iter().map(|row| row[feature]).collect();
            sorted.sort_by(f64::total_cmp);
            let edges: Vec<f64> = (1..bins)
                .map(|b| quantile_sorted(&sorted, b as f64 / bins as f64))
                .collect();
            all_edges.push(edges);
        }
        for _ in 0..rounds {
            let residuals: Vec<f64> = (0..x.len())
                .map(|r| {
                    if binary {
                        y[r] - sigmoid(raw[r])
                    } else {
                        y[r] - raw[r]
                    }
                })
                .collect();
            let mut best: Option<(f64, GamTerm)> = None;
            for (feature, edges) in all_edges.iter().enumerate() {
                let mut sums = vec![0.0; bins];
                let mut counts = vec![0usize; bins];
                for r in 0..x.len() {
                    let bin = edges
                        .partition_point(|edge| *edge < x[r][feature])
                        .min(bins - 1);
                    sums[bin] += residuals[r];
                    counts[bin] += 1;
                }
                let values: Vec<f64> = sums
                    .iter()
                    .zip(&counts)
                    .map(|(sum, count)| sum / (*count as f64 + 1.0))
                    .collect();
                let gain = (0..x.len())
                    .map(|r| {
                        let bin = edges
                            .partition_point(|edge| *edge < x[r][feature])
                            .min(bins - 1);
                        let before = residuals[r] * residuals[r];
                        let after = residuals[r] - values[bin];
                        before - after * after
                    })
                    .sum::<f64>();
                if best.as_ref().is_none_or(|entry| gain > entry.0) {
                    best = Some((
                        gain,
                        GamTerm {
                            feature,
                            edges: edges.clone(),
                            values,
                        },
                    ));
                }
            }
            let Some((gain, term)) = best else { break };
            if gain <= 0.0 {
                break;
            }
            for (row, prediction) in x.iter().zip(&mut raw) {
                let bin = term
                    .edges
                    .partition_point(|edge| *edge < row[term.feature])
                    .min(term.values.len() - 1);
                *prediction += learning_rate * term.values[bin];
            }
            terms.push(term);
        }
        Ok(Self {
            base,
            learning_rate,
            binary,
            terms,
        })
    }

    fn predict(&self, x: &Rows) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut prediction = self.base;
                for term in &self.terms {
                    let bin = term
                        .edges
                        .partition_point(|edge| *edge < row[term.feature])
                        .min(term.values.len() - 1);
                    prediction += self.learning_rate * term.values[bin];
                }
                if self.binary {
                    sigmoid(prediction)
                } else {
                    finite(prediction)
                }
            })
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct PlsHeadModel {
    pub(super) projection: V3Op,
    pub(super) ridge: RidgeModel,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ClassificationPortfolioModel {
    pub(super) members: Vec<V3Head>,
    pub(super) weights: Vec<f64>,
}

impl ClassificationPortfolioModel {
    fn fit(x: &Rows, y: &[f64]) -> Result<Self> {
        validate_binary_fit_target(y, x.len())?;
        let stages = [
            Stage::new("logit_ridge"),
            Stage::new("tab_boost"),
            Stage::new("gam_boost"),
        ];
        let positives = y.iter().filter(|value| **value == 1.0).count();
        let minority = positives.min(y.len() - positives);
        if minority < 2 {
            let members = stages
                .iter()
                .map(|stage| fit_v3_head(stage, x, y, TaskKind::Binary))
                .collect::<Result<Vec<_>>>()?;
            return Ok(Self {
                weights: vec![1.0 / members.len() as f64; members.len()],
                members,
            });
        }
        let folds = 5.min(minority);
        let assignments = stratified_fold_assignments(y, folds);
        let mut columns = vec![vec![0.5; x.len()]; stages.len()];
        for fold in 0..folds {
            let fit_rows: Vec<usize> = (0..x.len()).filter(|&r| assignments[r] != fold).collect();
            let valid_rows: Vec<usize> = (0..x.len()).filter(|&r| assignments[r] == fold).collect();
            let x_fit: Rows = fit_rows.iter().map(|&r| x[r].clone()).collect();
            let y_fit: Vec<f64> = fit_rows.iter().map(|&r| y[r]).collect();
            let x_valid: Rows = valid_rows.iter().map(|&r| x[r].clone()).collect();
            for (member, stage) in stages.iter().enumerate() {
                let fitted = fit_v3_head(stage, &x_fit, &y_fit, TaskKind::Binary)?;
                for (&row, value) in valid_rows.iter().zip(fitted.predict(&x_valid)) {
                    columns[member][row] = value;
                }
            }
        }
        let mut weights = vec![1.0 / stages.len() as f64; stages.len()];
        for _ in 0..200 {
            let mut gradient = vec![0.0; stages.len()];
            for row in 0..x.len() {
                let blend = (0..stages.len())
                    .map(|member| weights[member] * columns[member][row])
                    .sum::<f64>()
                    .clamp(1e-6, 1.0 - 1e-6);
                let common = (blend - y[row]) / (blend * (1.0 - blend) * x.len() as f64);
                for member in 0..stages.len() {
                    gradient[member] += common * columns[member][row];
                }
            }
            for member in 0..weights.len() {
                weights[member] *= (-0.05 * gradient[member].clamp(-20.0, 20.0)).exp();
            }
            let sum = weights.iter().sum::<f64>().max(1e-24);
            for weight in &mut weights {
                *weight /= sum;
            }
        }
        let members = stages
            .iter()
            .map(|stage| fit_v3_head(stage, x, y, TaskKind::Binary))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { members, weights })
    }

    fn predict(&self, x: &Rows) -> Vec<f64> {
        let columns: Vec<Vec<f64>> = self
            .members
            .iter()
            .map(|member| member.predict(x))
            .collect();
        (0..x.len())
            .map(|row| {
                self.weights
                    .iter()
                    .zip(&columns)
                    .map(|(weight, predictions)| weight * predictions[row])
                    .sum::<f64>()
                    .clamp(0.0, 1.0)
            })
            .collect()
    }
}

impl PlsHeadModel {
    fn fit(x: &Rows, y: &[f64], k: usize) -> Self {
        let (means, weights) = fit_pls_weights(x, y, k);
        let projection = V3Op::PlsProject {
            means,
            weights,
            append: false,
        };
        let projected = projection.apply(x);
        let ridge = RidgeModel::fit(&projected, y, 0.02);
        Self { projection, ridge }
    }

    fn predict(&self, x: &Rows) -> Vec<f64> {
        self.ridge.predict(&self.projection.apply(x))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) enum V3Head {
    Legacy(FittedHead),
    LogitRidge(LogitRidgeModel),
    TabBoost(TabBoostModel),
    ExtraTrees(ExtraTreesModel),
    GamBoost(GamBoostModel),
    Pls(PlsHeadModel),
    ClassificationPortfolio(ClassificationPortfolioModel),
}

impl V3Head {
    pub(super) fn predict(&self, x: &Rows) -> Vec<f64> {
        match self {
            V3Head::Legacy(head) => head.predict(x),
            V3Head::LogitRidge(head) => head.predict(x),
            V3Head::TabBoost(head) => head.predict(x),
            V3Head::ExtraTrees(head) => head.predict(x),
            V3Head::GamBoost(head) => head.predict(x),
            V3Head::Pls(head) => head.predict(x),
            V3Head::ClassificationPortfolio(head) => head.predict(x),
        }
    }
}

pub(super) fn fit_v3_head(stage: &Stage, x: &Rows, y: &[f64], task: TaskKind) -> Result<V3Head> {
    let binary = task == TaskKind::Binary;
    match stage.op.as_str() {
        "logit_ridge" | "logistic" if binary => {
            let class_weight = parse_class_weight(stage)?;
            Ok(V3Head::LogitRidge(LogitRidgeModel::fit(
                x,
                y,
                stage.get_f64("l2", stage.get_f64("lambda_frac", 0.02)),
                class_weight,
            )?))
        }
        "tab_boost" => Ok(V3Head::TabBoost(TabBoostModel::fit(x, y, stage, binary)?)),
        "extra_trees" => Ok(V3Head::ExtraTrees(ExtraTreesModel::fit(
            x, y, stage, binary,
        )?)),
        "gam_boost" => Ok(V3Head::GamBoost(GamBoostModel::fit(x, y, stage, binary)?)),
        "portfolio" if binary => Ok(V3Head::ClassificationPortfolio(
            ClassificationPortfolioModel::fit(x, y)?,
        )),
        "pls" if !binary => Ok(V3Head::Pls(PlsHeadModel::fit(
            x,
            y,
            stage.get_usize("k", 8).clamp(1, 32),
        ))),
        "ridge" if binary => bail!("binary advanced-v3 pipelines require a probability head"),
        "logit_ridge" => bail!("logit_ridge is only valid for binary tasks"),
        "pls" => bail!("pls is only valid for regression tasks"),
        _ if !binary => Ok(V3Head::Legacy(fit_head(stage, x, y)?)),
        other => bail!("head '{other}' is not probability-capable for a binary task"),
    }
}

fn parse_class_weight(stage: &Stage) -> Result<ClassWeight> {
    match str_param(stage, "class_weight", "none")? {
        "none" => Ok(ClassWeight::None),
        "balanced" => Ok(ClassWeight::Balanced),
        other => bail!(
            "{}: class_weight must be none|balanced, got '{other}'",
            stage.op
        ),
    }
}

pub(super) fn validate_binary_target(y: &[f64], rows: usize) -> Result<()> {
    if y.len() != rows || rows == 0 {
        bail!("binary target length differs from nonempty training rows");
    }
    let positives = y.iter().filter(|value| **value == 1.0).count();
    if y.iter().any(|value| *value != 0.0 && *value != 1.0) {
        bail!("binary targets must be exactly 0 or 1");
    }
    if positives < 2 || rows - positives < 2 {
        bail!("binary tasks require at least two rows in each class");
    }
    Ok(())
}

fn validate_binary_fit_target(y: &[f64], rows: usize) -> Result<()> {
    if y.len() != rows || rows == 0 {
        bail!("binary target length differs from nonempty training rows");
    }
    let positives = y.iter().filter(|value| **value == 1.0).count();
    if y.iter().any(|value| *value != 0.0 && *value != 1.0) {
        bail!("binary targets must be exactly 0 or 1");
    }
    if positives == 0 || positives == rows {
        bail!("binary fits require both classes");
    }
    Ok(())
}

pub(super) fn finite(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}
