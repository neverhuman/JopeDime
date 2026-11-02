//! Draft, versioned OOF stacking and residual composition contracts.
//!
//! Each `Candidate::fit` call receives only its declared training rows and target.
//! The fitted candidate owns its complete pipeline, including fitted transforms and
//! target transforms. Held rows are supplied only to `predict`; their labels never
//! cross the fit boundary. Public module/caller wiring is reserved for the integrator.

use std::collections::BTreeSet;

pub const COMBINATOR_ABI: u16 = 1;
pub type Rows = Vec<Vec<f64>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FitRole {
    InnerOof {
        outer_fold: Option<usize>,
        inner_fold: usize,
    },
    OuterRefit {
        fold: usize,
    },
    FullTraining,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FitContext {
    pub abi: u16,
    pub candidate_id: String,
    pub role: FitRole,
    pub seed: u64,
    pub train_row_ids: Vec<u64>,
}

pub struct FitInput<'a> {
    pub rows: &'a [Vec<f64>],
    pub target: &'a [f64],
    pub context: &'a FitContext,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FitError {
    InvalidPlan(String),
    Member(String),
    Prediction(String),
    UnsupportedEmission(String),
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

pub trait FittedCandidate {
    fn predict(&self, rows: &[Vec<f64>]) -> Result<Vec<f64>, FitError>;

    /// `predict` must return values in the original target space after applying
    /// the candidate's fold-fitted inverse target transform.
    fn prediction_target(&self) -> &str {
        "original"
    }

    /// `None` means the candidate has no portable fitted emitter. No implicit
    /// interpreter fallback is labelled as emitted execution.
    fn predict_emitted(&self, _rows: &[Vec<f64>]) -> Result<Option<Vec<f64>>, FitError> {
        Ok(None)
    }
}

pub trait Candidate {
    type Fitted: FittedCandidate;

    fn id(&self) -> &str;
    fn fit(&self, input: FitInput<'_>) -> Result<Self::Fitted, FitError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OOFFitPlan {
    pub abi: u16,
    pub fold_of_row: Vec<usize>,
    pub fold_count: usize,
    pub seed: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FitRecord {
    pub candidate_id: String,
    pub role: FitRole,
    pub seed: u64,
    pub train_row_ids: Vec<u64>,
    pub held_row_ids: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OOFOutput {
    pub predictions: Vec<f64>,
    pub records: Vec<FitRecord>,
}

impl OOFFitPlan {
    pub fn new(fold_of_row: Vec<usize>, fold_count: usize, seed: u64) -> Result<Self, FitError> {
        let plan = Self {
            abi: COMBINATOR_ABI,
            fold_of_row,
            fold_count,
            seed,
        };
        plan.validate_shape()?;
        Ok(plan)
    }

    fn validate_shape(&self) -> Result<(), FitError> {
        if self.abi != COMBINATOR_ABI || self.fold_count < 2 || self.fold_of_row.is_empty() {
            return Err(FitError::InvalidPlan(
                "unsupported ABI or empty fold plan".into(),
            ));
        }
        if self.fold_of_row.iter().any(|&fold| fold >= self.fold_count) {
            return Err(FitError::InvalidPlan(
                "fold id is outside the declared range".into(),
            ));
        }
        for fold in 0..self.fold_count {
            let held = self.fold_of_row.iter().filter(|&&f| f == fold).count();
            if held == 0 || held == self.fold_of_row.len() {
                return Err(FitError::InvalidPlan(format!(
                    "fold {fold} has an empty fit or hold set"
                )));
            }
        }
        Ok(())
    }

    pub fn cross_fit<C: Candidate>(
        &self,
        candidate: &C,
        rows: &[Vec<f64>],
        target: &[f64],
        row_ids: &[u64],
        outer_fold: Option<usize>,
    ) -> Result<OOFOutput, FitError> {
        let mut records = Vec::new();
        let predictions =
            self.cross_fit_into(candidate, rows, target, row_ids, outer_fold, &mut records)?;
        Ok(OOFOutput {
            predictions,
            records,
        })
    }

    fn cross_fit_into<C: Candidate>(
        &self,
        candidate: &C,
        rows: &[Vec<f64>],
        target: &[f64],
        row_ids: &[u64],
        outer_fold: Option<usize>,
        records: &mut Vec<FitRecord>,
    ) -> Result<Vec<f64>, FitError> {
        self.validate_inputs(rows, target, row_ids)?;
        if self.fold_of_row.len() != rows.len() {
            return Err(FitError::InvalidPlan(
                "fold map length differs from fit rows".into(),
            ));
        }
        let mut predictions = vec![f64::NAN; rows.len()];
        for fold in 0..self.fold_count {
            let train: Vec<_> = (0..rows.len())
                .filter(|&i| self.fold_of_row[i] != fold)
                .collect();
            let held: Vec<_> = (0..rows.len())
                .filter(|&i| self.fold_of_row[i] == fold)
                .collect();
            let train_rows = select_rows(rows, &train);
            let train_target = select_values(target, &train);
            let train_ids = select_values(row_ids, &train);
            let held_ids = select_values(row_ids, &held);
            let role = FitRole::InnerOof {
                outer_fold,
                inner_fold: fold,
            };
            let context = FitContext {
                abi: COMBINATOR_ABI,
                candidate_id: candidate.id().to_owned(),
                role: role.clone(),
                seed: fold_seed(self.seed, fold as u64),
                train_row_ids: train_ids.clone(),
            };
            records.push(FitRecord {
                candidate_id: candidate.id().to_owned(),
                role: role.clone(),
                seed: context.seed,
                train_row_ids: train_ids.clone(),
                held_row_ids: held_ids.clone(),
            });
            let fitted = candidate.fit(FitInput {
                rows: &train_rows,
                target: &train_target,
                context: &context,
            })?;
            let held_rows = select_rows(rows, &held);
            let held_predictions = checked_prediction(&fitted, &held_rows)?;
            if held_predictions.len() != held.len() {
                return Err(FitError::Prediction(format!(
                    "{} returned {} predictions for {} held rows",
                    candidate.id(),
                    held_predictions.len(),
                    held.len()
                )));
            }
            for (index, value) in held.into_iter().zip(held_predictions) {
                predictions[index] = value;
            }
        }
        if predictions.iter().any(|p| !p.is_finite()) {
            return Err(FitError::Prediction(
                "OOF plan left a row without one finite prediction".into(),
            ));
        }
        Ok(predictions)
    }

    fn validate_inputs(
        &self,
        rows: &[Vec<f64>],
        target: &[f64],
        row_ids: &[u64],
    ) -> Result<(), FitError> {
        self.validate_shape()?;
        if rows.len() != target.len() || rows.len() != row_ids.len() || rows.is_empty() {
            return Err(FitError::InvalidPlan(
                "rows, target, and row identities must align".into(),
            ));
        }
        let width = rows[0].len();
        if width == 0
            || rows
                .iter()
                .any(|r| r.len() != width || r.iter().any(|v| !v.is_finite()))
        {
            return Err(FitError::InvalidPlan(
                "input rows must have one finite, nonzero width".into(),
            ));
        }
        if target.iter().any(|v| !v.is_finite())
            || row_ids.iter().copied().collect::<BTreeSet<_>>().len() != row_ids.len()
        {
            return Err(FitError::InvalidPlan(
                "targets must be finite and row identities unique".into(),
            ));
        }
        Ok(())
    }
}

mod helpers;
use helpers::*;
mod stack;
pub use stack::{FitAudit, StackModel, StackOf, StackOutcome};
mod boost;
pub use boost::{BoostCombinator, BoostModel, BoostOutcome};
