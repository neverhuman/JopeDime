use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::search_v3::validate_sha256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortfolioLaneV1 {
    ConfidenceLcb,
    CumulativeMean,
    WorstSplit,
    SemanticNovelty,
}

impl PortfolioLaneV1 {
    const ALL: [Self; 4] = [
        Self::ConfidenceLcb,
        Self::CumulativeMean,
        Self::WorstSplit,
        Self::SemanticNovelty,
    ];
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortfolioMemberV1 {
    pub semantic_sha256: String,
    pub lane: PortfolioLaneV1,
    pub oof_predictions: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortfolioFitV1 {
    pub semantic_sha256s: Vec<String>,
    pub weights: Vec<f64>,
    pub oof_rmse: f64,
    pub shrinkage: f64,
}

impl PortfolioFitV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.semantic_sha256s.is_empty()
            || self.semantic_sha256s.len() > 4
            || self.semantic_sha256s.len() != self.weights.len()
            || self.semantic_sha256s.iter().collect::<BTreeSet<_>>().len()
                != self.semantic_sha256s.len()
        {
            return Err("portfolio must contain one to four unique members".into());
        }
        for semantic in &self.semantic_sha256s {
            validate_sha256("portfolio semantic", semantic)?;
        }
        if self
            .weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
            || (self.weights.iter().sum::<f64>() - 1.0).abs() > 1e-10
            || !self.oof_rmse.is_finite()
            || self.oof_rmse < 0.0
            || (self.shrinkage - 0.05).abs() > f64::EPSILON
        {
            return Err("portfolio weights or objective violate the frozen simplex policy".into());
        }
        Ok(())
    }
}

fn mse(actual: &[f64], predicted: &[f64]) -> f64 {
    actual
        .iter()
        .zip(predicted)
        .map(|(actual, predicted)| (actual - predicted).powi(2))
        .sum::<f64>()
        / actual.len() as f64
}

fn project_simplex(values: &[f64]) -> Vec<f64> {
    // Euclidean projection onto {w >= 0, sum(w)=1}.
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));
    let mut cumulative = 0.0;
    let mut rho = 0usize;
    for (index, value) in sorted.iter().enumerate() {
        cumulative += value;
        if value - (cumulative - 1.0) / (index + 1) as f64 > 0.0 {
            rho = index + 1;
        }
    }
    let theta = (sorted.iter().take(rho).sum::<f64>() - 1.0) / rho as f64;
    values
        .iter()
        .map(|value| (value - theta).max(0.0))
        .collect()
}

fn fit_subset(actual: &[f64], members: &[&PortfolioMemberV1]) -> Vec<f64> {
    let width = members.len();
    let singleton_errors = members
        .iter()
        .map(|member| mse(actual, &member.oof_predictions))
        .collect::<Vec<_>>();
    let best_singleton = (0..width)
        .min_by(|left, right| {
            singleton_errors[*left]
                .total_cmp(&singleton_errors[*right])
                .then_with(|| {
                    members[*left]
                        .semantic_sha256
                        .cmp(&members[*right].semantic_sha256)
                })
        })
        .unwrap();
    let mut weights = vec![1.0 / width as f64; width];
    let max_column_norm = members
        .iter()
        .map(|member| {
            member
                .oof_predictions
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                / actual.len() as f64
        })
        .max_by(f64::total_cmp)
        .unwrap_or(1.0)
        .max(1e-9);
    let step = 0.25 / (max_column_norm * width as f64 + 0.05);
    for _ in 0..4_000 {
        let prediction = (0..actual.len())
            .map(|row| {
                members
                    .iter()
                    .zip(&weights)
                    .map(|(member, weight)| member.oof_predictions[row] * weight)
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let gradient = members
            .iter()
            .enumerate()
            .map(|(column, member)| {
                let data = 2.0
                    * prediction
                        .iter()
                        .zip(actual)
                        .zip(&member.oof_predictions)
                        .map(|((predicted, actual), feature)| (predicted - actual) * feature)
                        .sum::<f64>()
                    / actual.len() as f64;
                let target = if column == best_singleton { 1.0 } else { 0.0 };
                data + 2.0 * 0.05 * (weights[column] - target)
            })
            .collect::<Vec<_>>();
        let next = project_simplex(
            &weights
                .iter()
                .zip(gradient)
                .map(|(weight, gradient)| weight - step * gradient)
                .collect::<Vec<_>>(),
        );
        let movement = next
            .iter()
            .zip(&weights)
            .map(|(next, old)| (next - old).abs())
            .sum::<f64>();
        weights = next;
        if movement < 1e-13 {
            break;
        }
    }
    weights
}

fn combinations(
    width: usize,
    count: usize,
    start: usize,
    current: &mut Vec<usize>,
    output: &mut Vec<Vec<usize>>,
) {
    if current.len() == count {
        output.push(current.clone());
        return;
    }
    for index in start..=width - (count - current.len()) {
        current.push(index);
        combinations(width, count, index + 1, current, output);
        current.pop();
    }
}

/// Exhaustively searches every subset of one to four of the eight lane
/// finalists, fitting deterministic nonnegative simplex weights with a 0.05
/// quadratic shrink toward the best singleton inside that subset.
pub fn fit_sparse_portfolio(
    actual: &[f64],
    finalists: &[PortfolioMemberV1],
) -> Result<PortfolioFitV1, String> {
    if actual.is_empty() || actual.iter().any(|value| !value.is_finite()) {
        return Err("portfolio requires finite nonempty OOF targets".into());
    }
    if finalists.len() != 8 {
        return Err("portfolio requires the eight frozen lane finalists".into());
    }
    let mut semantics = BTreeSet::new();
    let mut lane_counts = BTreeMap::new();
    for finalist in finalists {
        validate_sha256("portfolio finalist", &finalist.semantic_sha256)?;
        if !semantics.insert(finalist.semantic_sha256.as_str())
            || finalist.oof_predictions.len() != actual.len()
            || finalist
                .oof_predictions
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err("portfolio finalist is duplicate or has invalid OOF predictions".into());
        }
        *lane_counts.entry(finalist.lane).or_insert(0usize) += 1;
    }
    if PortfolioLaneV1::ALL
        .iter()
        .any(|lane| lane_counts.get(lane) != Some(&2))
    {
        return Err("portfolio must retain two unique leaders from every specialist lane".into());
    }
    let mut subsets = Vec::new();
    for count in 1..=4 {
        combinations(finalists.len(), count, 0, &mut Vec::new(), &mut subsets);
    }
    let mut fits = Vec::with_capacity(subsets.len());
    for subset in subsets {
        let members = subset
            .iter()
            .map(|index| &finalists[*index])
            .collect::<Vec<_>>();
        let weights = fit_subset(actual, &members);
        let prediction = (0..actual.len())
            .map(|row| {
                members
                    .iter()
                    .zip(&weights)
                    .map(|(member, weight)| member.oof_predictions[row] * weight)
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let fit = PortfolioFitV1 {
            semantic_sha256s: members
                .iter()
                .map(|member| member.semantic_sha256.clone())
                .collect(),
            weights,
            oof_rmse: mse(actual, &prediction).sqrt(),
            shrinkage: 0.05,
        };
        fit.validate()?;
        fits.push(fit);
    }
    fits.into_iter()
        .min_by(|left, right| {
            left.oof_rmse
                .total_cmp(&right.oof_rmse)
                .then_with(|| {
                    left.semantic_sha256s
                        .len()
                        .cmp(&right.semantic_sha256s.len())
                })
                .then_with(|| left.semantic_sha256s.cmp(&right.semantic_sha256s))
        })
        .ok_or_else(|| "portfolio subset search produced no fit".into())
}
