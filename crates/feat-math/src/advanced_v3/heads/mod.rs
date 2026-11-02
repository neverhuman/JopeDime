mod extra_trees;
mod gam;
mod logistic;
mod multiclass;
mod portfolio;
mod tab_boost;

pub(super) use extra_trees::{ExtraNode, ExtraTreesModel};
pub(super) use gam::{GamBoostModel, GamTerm};
pub(super) use logistic::{sigmoid, LogitRidgeModel};
pub(super) use multiclass::{
    MulticlassDiagonalLdaModel, MulticlassNearestCentroidModel, MulticlassOvrLogisticModel,
    MulticlassRidgeModel,
};
pub(super) use portfolio::{ClassificationPortfolioModel, PlsHeadModel};
pub(super) use tab_boost::{BoostStump, TabBoostModel};

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
    pub(super) fn predict(&self, x: &Rows) -> Result<Vec<f64>> {
        Ok(match self {
            V3Head::Legacy(head) => head.predict(x)?,
            V3Head::LogitRidge(head) => head.predict(x),
            V3Head::TabBoost(head) => head.predict(x),
            V3Head::ExtraTrees(head) => head.predict(x),
            V3Head::GamBoost(head) => head.predict(x),
            V3Head::Pls(head) => head.predict(x),
            V3Head::ClassificationPortfolio(head) => head.predict(x)?,
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
        })
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
