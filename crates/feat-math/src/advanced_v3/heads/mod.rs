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

pub const MULTICLASS_MAX_TRANSFORMED_FEATURES: usize = 64;
const MULTICLASS_FEATURE_BUDGETS: [usize; 4] = [32, 64, 128, 256];

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
        Self::fit_binary_unchecked(x, y, l2, class_weight)
    }

    fn fit_ovr(x: &Rows, y: &[f64], l2: f64) -> Result<Self> {
        if y.len() != x.len() || x.is_empty() {
            bail!("multiclass OvR target length differs from nonempty training rows");
        }
        let positives = y.iter().filter(|value| **value == 1.0).count();
        if positives == 0
            || positives == y.len()
            || y.iter().any(|value| !matches!(*value, 0.0 | 1.0))
        {
            bail!("multiclass OvR targets must contain both zero and one");
        }
        Self::fit_binary_unchecked(x, y, l2, ClassWeight::Balanced)
    }

    fn fit_binary_unchecked(
        x: &Rows,
        y: &[f64],
        l2: f64,
        class_weight: ClassWeight,
    ) -> Result<Self> {
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
                // `j` indexes four arrays in lockstep (row, means, scales, and weights at an
                // offset). One index says this more plainly than a three-way zip plus a skip.
                #[allow(clippy::needless_range_loop)]
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

/// A candidate split held while searching: score, feature index, threshold, the branch
/// orientation flag, and the two row partitions it produces.
type ExtraSplit = (f64, usize, f64, bool, Vec<usize>, Vec<usize>);

// Recursive tree fitting: every argument is threaded to the child calls, so bundling them
// into a parameter struct would add a type and a construction per node without removing a
// single value from the recursion.
#[allow(clippy::too_many_arguments)]
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
    let mut best: Option<ExtraSplit> = None;
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
pub(super) struct MulticlassRidgeModel {
    classes: Vec<i64>,
    feature_count: usize,
    means: Vec<f64>,
    scales: Vec<f64>,
    weights: Vec<Vec<f64>>,
}

impl MulticlassRidgeModel {
    fn fit(x: &Rows, y: &[f64], stage: &Stage) -> Result<Self> {
        let classes = validate_multiclass_target(y, x.len())?;
        if x.first().is_none_or(Vec::is_empty) || x.iter().any(|row| row.len() != x[0].len()) {
            bail!("multiclass fits require a nonempty rectangular feature matrix");
        }
        let class_weight = str_param(stage, "class_weight", "balanced")?;
        if class_weight != "balanced" {
            bail!("multiclass_ridge: class_weight must be balanced");
        }
        let feature_count = x[0].len().min(multiclass_feature_budget(stage)?);
        let mut means = vec![0.0; feature_count];
        for row in x {
            for feature in 0..feature_count {
                means[feature] += finite(row[feature]) / x.len() as f64;
            }
        }
        let mut scales = vec![0.0; feature_count];
        for row in x {
            for feature in 0..feature_count {
                let delta = finite(row[feature]) - means[feature];
                scales[feature] += delta * delta;
            }
        }
        for scale in &mut scales {
            *scale = (*scale / x.len() as f64).sqrt().max(1e-12);
        }

        let mut class_counts = vec![0usize; classes.len()];
        let class_indices: Vec<usize> = y
            .iter()
            .map(|value| {
                classes
                    .binary_search(&(*value as i64))
                    .expect("validated class is present")
            })
            .collect();
        for &class in &class_indices {
            class_counts[class] += 1;
        }

        let width = feature_count + 1;
        let mut gram = vec![vec![0.0; width]; width];
        let mut rhs = vec![vec![0.0; width]; classes.len()];
        for (row, &class) in x.iter().zip(&class_indices) {
            let sample_weight = x.len() as f64 / (classes.len() * class_counts[class]) as f64;
            let mut values = Vec::with_capacity(width);
            values.push(1.0);
            values.extend((0..feature_count).map(|feature| {
                ((finite(row[feature]) - means[feature]) / scales[feature]).clamp(-30.0, 30.0)
            }));
            // Lower-triangular Gram accumulation: `right` runs to `left` inclusive, and both
            // index `values` as well as `gram`. The triangle is the point.
            #[allow(clippy::needless_range_loop)]
            for left in 0..width {
                for right in 0..=left {
                    gram[left][right] += sample_weight * values[left] * values[right];
                }
                rhs[class][left] += sample_weight * values[left];
            }
        }
        // Mirror the lower triangle into the upper one. Reading and writing two different
        // cells of the same matrix is what this does; an iterator form cannot express it.
        #[allow(clippy::needless_range_loop)]
        for left in 0..width {
            for right in 0..left {
                gram[right][left] = gram[left][right];
            }
        }
        let l2 = stage.get_f64("l2", 0.02).clamp(1e-9, 1e6) * x.len() as f64;
        gram[0][0] += 1e-9;
        for (feature, row) in gram.iter_mut().enumerate().take(width).skip(1) {
            row[feature] += l2;
        }
        let weights = rhs
            .iter()
            .map(|target| {
                solve_positive_definite(&gram, target).ok_or_else(|| {
                    anyhow::anyhow!("multiclass ridge normal equations are singular")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            classes,
            feature_count,
            means,
            scales,
            weights,
        })
    }

    fn probabilities(&self, x: &Rows) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| {
                let mut scores = Vec::with_capacity(self.classes.len());
                for weights in &self.weights {
                    let mut score = weights[0];
                    for feature in 0..self.feature_count.min(row.len()) {
                        let value = ((finite(row[feature]) - self.means[feature])
                            / self.scales[feature])
                            .clamp(-30.0, 30.0);
                        score += weights[feature + 1] * value;
                    }
                    scores.push(finite(score));
                }
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut probabilities: Vec<f64> = scores
                    .iter()
                    .map(|score| (score - maximum).clamp(-40.0, 0.0).exp())
                    .collect();
                let sum = probabilities.iter().sum::<f64>().max(1e-300);
                for probability in &mut probabilities {
                    *probability /= sum;
                }
                probabilities
            })
            .collect()
    }

    fn labels(&self, x: &Rows) -> Vec<i64> {
        self.probabilities(x)
            .iter()
            .map(|row| {
                let index = row
                    .iter()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map_or(0, |(index, _)| index);
                self.classes[index]
            })
            .collect()
    }
}

fn multiclass_feature_budget(stage: &Stage) -> Result<usize> {
    let budget = stage.get_usize("feature_budget", 64);
    if !MULTICLASS_FEATURE_BUDGETS.contains(&budget) {
        bail!("{}: feature_budget must be one of 32|64|128|256", stage.op);
    }
    Ok(budget)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct MulticlassOvrLogisticModel {
    classes: Vec<i64>,
    models: Vec<LogitRidgeModel>,
}

impl MulticlassOvrLogisticModel {
    fn fit(x: &Rows, y: &[f64], stage: &Stage) -> Result<Self> {
        let classes = validate_multiclass_target(y, x.len())?;
        let feature_count = x[0].len().min(multiclass_feature_budget(stage)?);
        let bounded = x
            .iter()
            .map(|row| row[..feature_count].to_vec())
            .collect::<Rows>();
        let l2 = stage.get_f64("l2", 0.02).clamp(1e-9, 1e6);
        let models = classes
            .iter()
            .map(|class| {
                let binary = y
                    .iter()
                    .map(|value| f64::from(*value as i64 == *class))
                    .collect::<Vec<_>>();
                LogitRidgeModel::fit_ovr(&bounded, &binary, l2)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { classes, models })
    }

    fn probabilities(&self, x: &Rows) -> Vec<Vec<f64>> {
        let columns = self
            .models
            .iter()
            .map(|model| model.predict(x))
            .collect::<Vec<_>>();
        (0..x.len())
            .map(|row| {
                let mut values = columns
                    .iter()
                    .map(|column| finite(column[row]).max(0.0))
                    .collect::<Vec<_>>();
                let sum = values.iter().sum::<f64>();
                if !sum.is_finite() || sum <= 0.0 {
                    let uniform = 1.0 / values.len() as f64;
                    values.fill(uniform);
                } else {
                    for value in &mut values {
                        *value /= sum;
                    }
                }
                values
            })
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct MulticlassDiagonalLdaModel {
    classes: Vec<i64>,
    feature_count: usize,
    means: Vec<Vec<f64>>,
    variances: Vec<f64>,
    log_priors: Vec<f64>,
}

impl MulticlassDiagonalLdaModel {
    fn fit(x: &Rows, y: &[f64], stage: &Stage) -> Result<Self> {
        let classes = validate_multiclass_target(y, x.len())?;
        let feature_count = x[0].len().min(multiclass_feature_budget(stage)?);
        let mut means = vec![vec![0.0; feature_count]; classes.len()];
        let mut counts = vec![0usize; classes.len()];
        for (row, label) in x.iter().zip(y) {
            let class = classes
                .binary_search(&(*label as i64))
                .expect("validated class");
            counts[class] += 1;
            for feature in 0..feature_count {
                means[class][feature] += finite(row[feature]);
            }
        }
        for (mean, count) in means.iter_mut().zip(&counts) {
            for value in mean {
                *value /= *count as f64;
            }
        }
        let mut variances = vec![0.0; feature_count];
        for (row, label) in x.iter().zip(y) {
            let class = classes
                .binary_search(&(*label as i64))
                .expect("validated class");
            for feature in 0..feature_count {
                let delta = finite(row[feature]) - means[class][feature];
                variances[feature] += delta * delta;
            }
        }
        let shrinkage = stage.get_f64("shrinkage", 0.01).clamp(0.0, 1.0);
        let global = (variances.iter().sum::<f64>()
            / (feature_count.max(1) * x.len().max(1)) as f64)
            .max(1e-12);
        for variance in &mut variances {
            *variance =
                ((1.0 - shrinkage) * (*variance / x.len() as f64) + shrinkage * global).max(1e-12);
        }
        let log_priors = counts
            .iter()
            .map(|count| (*count as f64 / x.len() as f64).ln())
            .collect();
        Ok(Self {
            classes,
            feature_count,
            means,
            variances,
            log_priors,
        })
    }

    fn probabilities(&self, x: &Rows) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| {
                let scores = self
                    .means
                    .iter()
                    .zip(&self.log_priors)
                    .map(|(mean, prior)| {
                        *prior
                            - 0.5
                                * (0..self.feature_count)
                                    .map(|feature| {
                                        let delta = finite(row[feature]) - mean[feature];
                                        delta * delta / self.variances[feature]
                                    })
                                    .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                softmax(&scores)
            })
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct MulticlassNearestCentroidModel {
    classes: Vec<i64>,
    feature_count: usize,
    centroids: Vec<Vec<f64>>,
    scales: Vec<f64>,
}

impl MulticlassNearestCentroidModel {
    fn fit(x: &Rows, y: &[f64], stage: &Stage) -> Result<Self> {
        let classes = validate_multiclass_target(y, x.len())?;
        let feature_count = x[0].len().min(multiclass_feature_budget(stage)?);
        let mut centroids = vec![vec![0.0; feature_count]; classes.len()];
        let mut counts = vec![0usize; classes.len()];
        for (row, label) in x.iter().zip(y) {
            let class = classes
                .binary_search(&(*label as i64))
                .expect("validated class");
            counts[class] += 1;
            for feature in 0..feature_count {
                centroids[class][feature] += finite(row[feature]);
            }
        }
        for (centroid, count) in centroids.iter_mut().zip(&counts) {
            for value in centroid {
                *value /= *count as f64;
            }
        }
        let mut global = vec![0.0; feature_count];
        for row in x {
            for feature in 0..feature_count {
                global[feature] += finite(row[feature]) / x.len() as f64;
            }
        }
        let mut scales = vec![0.0; feature_count];
        for row in x {
            for feature in 0..feature_count {
                let delta = finite(row[feature]) - global[feature];
                scales[feature] += delta * delta;
            }
        }
        for scale in &mut scales {
            *scale = (*scale / x.len() as f64).sqrt().max(1e-12);
        }
        Ok(Self {
            classes,
            feature_count,
            centroids,
            scales,
        })
    }

    fn probabilities(&self, x: &Rows) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| {
                let scores = self
                    .centroids
                    .iter()
                    .map(|centroid| {
                        -(0..self.feature_count)
                            .map(|feature| {
                                let delta = (finite(row[feature]) - centroid[feature])
                                    / self.scales[feature];
                                delta * delta
                            })
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                softmax(&scores)
            })
            .collect()
    }
}

fn softmax(scores: &[f64]) -> Vec<f64> {
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities = scores
        .iter()
        .map(|score| (finite(*score) - maximum).clamp(-40.0, 0.0).exp())
        .collect::<Vec<_>>();
    let sum = probabilities.iter().sum::<f64>().max(1e-300);
    for probability in &mut probabilities {
        *probability /= sum;
    }
    probabilities
}

fn solve_positive_definite(matrix: &[Vec<f64>], target: &[f64]) -> Option<Vec<f64>> {
    let size = matrix.len();
    if size == 0 || target.len() != size {
        return None;
    }
    for attempt in 0..6 {
        let jitter = if attempt == 0 {
            0.0
        } else {
            10f64.powi(attempt - 12)
        };
        let mut lower = vec![vec![0.0; size]; size];
        let mut valid = true;
        for row in 0..size {
            for column in 0..=row {
                let mut value = matrix[row][column];
                if row == column {
                    value += jitter;
                }
                // Cholesky inner product: `prior` indexes two different rows of the same
                // triangular factor. This is the textbook recurrence, written as such.
                #[allow(clippy::needless_range_loop)]
                for prior in 0..column {
                    value -= lower[row][prior] * lower[column][prior];
                }
                if row == column {
                    if !value.is_finite() || value <= 1e-18 {
                        valid = false;
                        break;
                    }
                    lower[row][column] = value.sqrt();
                } else {
                    lower[row][column] = value / lower[column][column];
                }
            }
            if !valid {
                break;
            }
        }
        if !valid {
            continue;
        }
        let mut forward = vec![0.0; size];
        for row in 0..size {
            let residual = target[row]
                - (0..row)
                    .map(|column| lower[row][column] * forward[column])
                    .sum::<f64>();
            forward[row] = residual / lower[row][row];
        }
        let mut solution = vec![0.0; size];
        for row in (0..size).rev() {
            let residual = forward[row]
                - ((row + 1)..size)
                    .map(|column| lower[column][row] * solution[column])
                    .sum::<f64>();
            solution[row] = residual / lower[row][row];
        }
        if solution.iter().all(|value| value.is_finite()) {
            return Some(solution);
        }
    }
    None
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
    MulticlassRidge(MulticlassRidgeModel),
    MulticlassOvrLogistic(MulticlassOvrLogisticModel),
    MulticlassDiagonalLda(MulticlassDiagonalLdaModel),
    MulticlassNearestCentroid(MulticlassNearestCentroidModel),
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
            V3Head::MulticlassRidge(head) => head
                .labels(x)
                .into_iter()
                .map(|label| label as f64)
                .collect(),
            V3Head::MulticlassOvrLogistic(head) => {
                probability_labels(&head.classes, head.probabilities(x))
            }
            V3Head::MulticlassDiagonalLda(head) => {
                probability_labels(&head.classes, head.probabilities(x))
            }
            V3Head::MulticlassNearestCentroid(head) => {
                probability_labels(&head.classes, head.probabilities(x))
            }
        }
    }

    pub(super) fn multiclass_probabilities(&self, x: &Rows) -> Option<(Vec<i64>, Vec<Vec<f64>>)> {
        match self {
            V3Head::MulticlassRidge(head) => Some((head.classes.clone(), head.probabilities(x))),
            V3Head::MulticlassOvrLogistic(head) => {
                Some((head.classes.clone(), head.probabilities(x)))
            }
            V3Head::MulticlassDiagonalLda(head) => {
                Some((head.classes.clone(), head.probabilities(x)))
            }
            V3Head::MulticlassNearestCentroid(head) => {
                Some((head.classes.clone(), head.probabilities(x)))
            }
            _ => None,
        }
    }
}

fn probability_labels(classes: &[i64], probabilities: Vec<Vec<f64>>) -> Vec<f64> {
    probabilities
        .iter()
        .map(|row| {
            let index = row
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map_or(0, |(index, _)| index);
            classes[index] as f64
        })
        .collect()
}

pub(super) fn fit_v3_head(stage: &Stage, x: &Rows, y: &[f64], task: TaskKind) -> Result<V3Head> {
    let binary = task == TaskKind::Binary;
    match stage.op.as_str() {
        "multiclass_ridge" if task == TaskKind::Multiclass => Ok(V3Head::MulticlassRidge(
            MulticlassRidgeModel::fit(x, y, stage)?,
        )),
        "multiclass_ovr_logistic" if task == TaskKind::Multiclass => Ok(
            V3Head::MulticlassOvrLogistic(MulticlassOvrLogisticModel::fit(x, y, stage)?),
        ),
        "multiclass_diagonal_lda" if task == TaskKind::Multiclass => Ok(
            V3Head::MulticlassDiagonalLda(MulticlassDiagonalLdaModel::fit(x, y, stage)?),
        ),
        "multiclass_nearest_centroid" if task == TaskKind::Multiclass => Ok(
            V3Head::MulticlassNearestCentroid(MulticlassNearestCentroidModel::fit(x, y, stage)?),
        ),
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
        _ if matches!(task, TaskKind::Regression | TaskKind::Timeseries) => {
            Ok(V3Head::Legacy(fit_head(stage, x, y)?))
        }
        "multiclass_ridge"
        | "multiclass_ovr_logistic"
        | "multiclass_diagonal_lda"
        | "multiclass_nearest_centroid" => {
            bail!("{} is only valid for multiclass tasks", stage.op)
        }
        other if task == TaskKind::Multiclass => {
            bail!("head '{other}' is not probability-capable for a multiclass task")
        }
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

pub(super) fn validate_multiclass_target(y: &[f64], rows: usize) -> Result<Vec<i64>> {
    if y.len() != rows || rows == 0 {
        bail!("multiclass target length differs from nonempty training rows");
    }
    let mut classes = BTreeSet::new();
    for value in y {
        if !value.is_finite() || value.fract() != 0.0 || *value < 0.0 || *value > i64::MAX as f64 {
            bail!("multiclass targets must be finite nonnegative integer class codes");
        }
        classes.insert(*value as i64);
    }
    if classes.len() < 3 {
        bail!("multiclass fits require at least three observed training classes");
    }
    let mut counts = vec![0usize; classes.len()];
    let classes: Vec<i64> = classes.into_iter().collect();
    for value in y {
        counts[classes
            .binary_search(&(*value as i64))
            .expect("class present")] += 1;
    }
    // One-shot observed classes are valid. The native heads all regularize or
    // shrink their estimates, and the corpus evaluator maps these contiguous
    // fitted classes back into the full declared metadata class space.
    Ok(classes)
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
