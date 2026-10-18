//! Residual-guided proposals derived only from an identified training fit closure.
//!
//! This API deliberately accepts no held-out observations or score callback. Callers must
//! construct it from one fold's training rows, predictions, labels, and role ancestry.

use crate::kernel::ordered_axis::OrderedInputIdentity;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const RESIDUAL_CONTEXT_ABI: u16 = 1;
const PROPOSAL_SOURCE: &str = "fit_residual_v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowSemantics {
    Unordered,
    Ordered { identity: OrderedInputIdentity },
}

#[derive(Clone, Debug, PartialEq)]
pub struct FitClosure<'a> {
    pub producer: &'a str,
    pub fold_id: &'a str,
    pub row_semantics: RowSemantics,
    pub selected_features: &'a [usize],
    pub train_row_ids: &'a [u64],
    pub train_x: &'a [Vec<f64>],
    pub train_y: &'a [f64],
    pub train_predictions: &'a [f64],
    pub model_state_sha256: &'a str,
    pub role_ancestry_sha256: &'a str,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidualContext {
    pub abi: u16,
    pub producer: String,
    pub fold_id: String,
    pub row_semantics: RowSemantics,
    pub selected_features: Vec<usize>,
    pub train_row_ids: Vec<u64>,
    pub fit_data_sha256: String,
    pub model_state_sha256: String,
    pub role_ancestry_sha256: String,
    pub context_sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditKind {
    AddResidualFeature,
    ScaleFeatureByResidual,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidualEdit {
    pub feature_index: usize,
    pub kind: EditKind,
    pub correlation: f64,
    pub heteroscedasticity: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidualProposal {
    pub context: ResidualContext,
    pub edits: Vec<ResidualEdit>,
    pub source: String,
    pub charged_work: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagnosticBudget {
    remaining: u64,
    charged: u64,
}

impl DiagnosticBudget {
    pub fn new(limit: u64) -> Self {
        Self {
            remaining: limit,
            charged: 0,
        }
    }

    pub fn charged(&self) -> u64 {
        self.charged
    }

    fn charge(&mut self, work: u64) -> bool {
        if self.remaining < work {
            return false;
        }
        self.remaining -= work;
        self.charged += work;
        true
    }
}

/// Analyze one TRAIN fit closure and propose deterministic, typed feature edits.
/// Work is charged before any residual statistic is computed; refusal leaves the
/// incumbent untouched and records no partial proposal.
pub fn propose(
    closure: &FitClosure<'_>,
    budget: &mut DiagnosticBudget,
) -> Option<ResidualProposal> {
    if closure.producer.trim().is_empty()
        || closure.fold_id.trim().is_empty()
        || closure.model_state_sha256.trim().is_empty()
        || closure.role_ancestry_sha256.trim().is_empty()
        || closure.train_x.is_empty()
        || closure.train_x.len() != closure.train_y.len()
        || closure.train_x.len() != closure.train_predictions.len()
        || closure.train_x.len() != closure.train_row_ids.len()
        || closure.selected_features.is_empty()
        || closure
            .train_x
            .iter()
            .any(|row| closure.selected_features.iter().any(|&i| i >= row.len()))
        || closure
            .train_y
            .iter()
            .chain(closure.train_predictions)
            .any(|x| !x.is_finite())
        || closure.train_x.iter().flatten().any(|x| !x.is_finite())
    {
        return None;
    }
    if let RowSemantics::Ordered { identity } = &closure.row_semantics {
        identity.validate().ok()?;
    }

    let rows = closure.train_x.len() as u64;
    let feature_count = closure.selected_features.len() as u64;
    let work = rows.checked_mul(feature_count)?.checked_mul(2)?;
    if !budget.charge(work) {
        return None;
    }

    let residuals: Vec<_> = closure
        .train_y
        .iter()
        .zip(closure.train_predictions)
        .map(|(y, p)| y - p)
        .collect();
    let squared: Vec<_> = residuals.iter().map(|r| r * r).collect();
    let mut edits = Vec::with_capacity(closure.selected_features.len());
    for &feature_index in closure.selected_features {
        let values: Vec<_> = closure
            .train_x
            .iter()
            .map(|row| row[feature_index])
            .collect();
        let residual_correlation = correlation(&values, &residuals)?;
        let heteroscedasticity = correlation(&values, &squared)?.abs();
        let kind = if heteroscedasticity > residual_correlation.abs() {
            EditKind::ScaleFeatureByResidual
        } else {
            EditKind::AddResidualFeature
        };
        edits.push(ResidualEdit {
            feature_index,
            kind,
            correlation: residual_correlation,
            heteroscedasticity,
        });
    }
    edits.sort_by(|a, b| {
        b.correlation
            .abs()
            .max(b.heteroscedasticity)
            .total_cmp(&a.correlation.abs().max(a.heteroscedasticity))
            .then_with(|| a.feature_index.cmp(&b.feature_index))
    });
    let mut context = ResidualContext {
        abi: RESIDUAL_CONTEXT_ABI,
        producer: closure.producer.to_owned(),
        fold_id: closure.fold_id.to_owned(),
        row_semantics: closure.row_semantics.clone(),
        selected_features: closure.selected_features.to_vec(),
        train_row_ids: closure.train_row_ids.to_vec(),
        fit_data_sha256: fit_data_hash(closure)?,
        model_state_sha256: closure.model_state_sha256.to_owned(),
        role_ancestry_sha256: closure.role_ancestry_sha256.to_owned(),
        context_sha256: String::new(),
    };
    context.context_sha256 = context_hash(&context)?;
    Some(ResidualProposal {
        context,
        edits,
        source: PROPOSAL_SOURCE.into(),
        charged_work: work,
    })
}

fn context_hash(context: &ResidualContext) -> Option<String> {
    let bytes = serde_json::to_vec(&(
        context.abi,
        &context.producer,
        &context.fold_id,
        &context.row_semantics,
        &context.selected_features,
        &context.train_row_ids,
        &context.fit_data_sha256,
        &context.model_state_sha256,
        &context.role_ancestry_sha256,
    ))
    .ok()?;
    Some(format!("{:x}", Sha256::digest(bytes)))
}

fn fit_data_hash(closure: &FitClosure<'_>) -> Option<String> {
    let bytes =
        serde_json::to_vec(&(closure.train_x, closure.train_y, closure.train_predictions)).ok()?;
    Some(format!("{:x}", Sha256::digest(bytes)))
}

fn correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let mx = x.iter().sum::<f64>() / x.len() as f64;
    let my = y.iter().sum::<f64>() / y.len() as f64;
    let (mut covariance, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for (&a, &b) in x.iter().zip(y) {
        let (dx, dy) = (a - mx, b - my);
        covariance += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if !covariance.is_finite() || vx <= 0.0 || vy <= 0.0 {
        return Some(0.0);
    }
    let result = covariance / (vx * vy).sqrt();
    result.is_finite().then_some(result)
}
