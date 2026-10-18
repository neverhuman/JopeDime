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
            let role = match outer_fold {
                outer_fold => FitRole::InnerOof {
                    outer_fold,
                    inner_fold: fold,
                },
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FitAudit {
    pub abi: u16,
    pub fit_count: usize,
    pub records: Vec<FitRecord>,
}

pub struct StackOf {
    pub abi: u16,
    pub plan: OOFFitPlan,
    pub shrinkage: f64,
    pub weight_iterations: usize,
}

pub enum StackOutcome<M> {
    Composed {
        model: StackModel<M>,
        audit: FitAudit,
    },
    KeepIncumbent {
        incumbent: M,
        reason: FitError,
        audit: FitAudit,
    },
}

pub struct StackModel<M> {
    pub abi: u16,
    pub fit_plan: OOFFitPlan,
    pub fit_row_ids: Vec<u64>,
    pub members: Vec<M>,
    pub weights: Vec<f64>,
    pub shrinkage: f64,
    pub weight_iterations: usize,
}

impl StackOf {
    pub fn fit<C: Candidate>(
        &self,
        candidates: &[C],
        incumbent: C::Fitted,
        rows: &[Vec<f64>],
        target: &[f64],
        row_ids: &[u64],
    ) -> StackOutcome<C::Fitted> {
        let mut records = Vec::new();
        let fail = |incumbent, reason, records: Vec<FitRecord>| StackOutcome::KeepIncumbent {
            audit: FitAudit {
                abi: COMBINATOR_ABI,
                fit_count: records.len(),
                records,
            },
            incumbent,
            reason,
        };
        if self.abi != COMBINATOR_ABI || !valid_shrinkage(self.shrinkage) || candidates.is_empty() {
            return fail(
                incumbent,
                FitError::InvalidPlan("invalid stack ABI, shrinkage, or empty member set".into()),
                records,
            );
        }
        if let Err(error) = self.plan.validate_inputs(rows, target, row_ids) {
            return fail(incumbent, error, records);
        }
        let mut ids = BTreeSet::new();
        if candidates
            .iter()
            .any(|c| c.id().trim().is_empty() || !ids.insert(c.id()))
        {
            return fail(
                incumbent,
                FitError::InvalidPlan("candidate IDs must be nonempty and unique".into()),
                records,
            );
        }
        let mut oof_members = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            match self
                .plan
                .cross_fit_into(candidate, rows, target, row_ids, None, &mut records)
            {
                Ok(output) => oof_members.push(output),
                Err(error) => return fail(incumbent, error, records),
            }
        }
        let mut weights = fit_simplex_weights(&oof_members, target, self.weight_iterations.max(1));
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0)
            || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-9
        {
            return fail(
                incumbent,
                FitError::Prediction("OOF stack optimizer produced invalid weights".into()),
                records,
            );
        }
        let uniform = self.shrinkage / weights.len() as f64;
        for weight in &mut weights {
            *weight = (1.0 - self.shrinkage) * *weight + uniform;
        }
        let mut fitted_members = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let context = full_context(candidate.id(), self.plan.seed, row_ids);
            records.push(FitRecord {
                candidate_id: candidate.id().into(),
                role: FitRole::FullTraining,
                seed: context.seed,
                train_row_ids: row_ids.to_vec(),
                held_row_ids: vec![],
            });
            match candidate.fit(FitInput {
                rows,
                target,
                context: &context,
            }) {
                Ok(model) if model.prediction_target() == "original" => {
                    match checked_prediction(&model, rows) {
                        Ok(values) if values.len() == rows.len() => fitted_members.push(model),
                        Ok(values) => {
                            return fail(
                                incumbent,
                                FitError::Prediction(format!(
                                    "{} returned {} predictions for {} full-fit rows",
                                    candidate.id(),
                                    values.len(),
                                    rows.len()
                                )),
                                records,
                            )
                        }
                        Err(error) => return fail(incumbent, error, records),
                    }
                }
                Ok(_) => {
                    return fail(
                        incumbent,
                        FitError::Prediction(format!(
                            "{} does not predict in original target space",
                            candidate.id()
                        )),
                        records,
                    )
                }
                Err(error) => return fail(incumbent, error, records),
            }
        }
        let audit = FitAudit {
            abi: COMBINATOR_ABI,
            fit_count: records.len(),
            records,
        };
        StackOutcome::Composed {
            model: StackModel {
                abi: COMBINATOR_ABI,
                fit_plan: self.plan.clone(),
                fit_row_ids: row_ids.to_vec(),
                members: fitted_members,
                weights,
                shrinkage: self.shrinkage,
                weight_iterations: self.weight_iterations.max(1),
            },
            audit,
        }
    }
}

impl<M: FittedCandidate> StackModel<M> {
    pub fn predict(&self, rows: &[Vec<f64>]) -> Result<Vec<f64>, FitError> {
        combine_predictions(&self.members, &self.weights, rows, false)
    }

    pub fn predict_emitted(&self, rows: &[Vec<f64>]) -> Result<Vec<f64>, FitError> {
        combine_predictions(&self.members, &self.weights, rows, true)
    }
}

pub struct BoostCombinator {
    pub abi: u16,
    pub outer_plan: OOFFitPlan,
    pub inner_fold_count: usize,
    pub seed: u64,
    pub shrinkage: f64,
}

pub enum BoostOutcome<M> {
    Composed {
        model: BoostModel<M>,
        oof_predictions: Vec<f64>,
        audit: FitAudit,
    },
    KeepIncumbent {
        incumbent: M,
        reason: FitError,
        audit: FitAudit,
    },
}

pub struct BoostModel<M> {
    pub abi: u16,
    pub outer_plan: OOFFitPlan,
    pub fit_row_ids: Vec<u64>,
    pub seed: u64,
    pub inner_fold_count: usize,
    pub stages: Vec<M>,
    pub shrinkage: f64,
}

impl BoostCombinator {
    /// Sequential residual learning uses inner OOF predictions within each outer
    /// TRAIN closure. The outer held labels are never used to form residual targets.
    pub fn fit<C: Candidate>(
        &self,
        stages: &[C],
        incumbent: C::Fitted,
        rows: &[Vec<f64>],
        target: &[f64],
        row_ids: &[u64],
    ) -> BoostOutcome<C::Fitted> {
        let mut records = Vec::new();
        let fail = |incumbent, reason, records: Vec<FitRecord>| BoostOutcome::KeepIncumbent {
            audit: FitAudit {
                abi: COMBINATOR_ABI,
                fit_count: records.len(),
                records,
            },
            incumbent,
            reason,
        };
        if self.abi != COMBINATOR_ABI || !valid_shrinkage(self.shrinkage) || stages.is_empty() {
            return fail(
                incumbent,
                FitError::InvalidPlan("invalid boost ABI, shrinkage, or empty stages".into()),
                records,
            );
        }
        if let Err(error) = self.outer_plan.validate_inputs(rows, target, row_ids) {
            return fail(incumbent, error, records);
        }
        if self.outer_plan.fold_of_row.len() != rows.len() || self.inner_fold_count < 2 {
            return fail(
                incumbent,
                FitError::InvalidPlan("outer map mismatch or fewer than two inner folds".into()),
                records,
            );
        }
        let mut oof = vec![0.0; rows.len()];
        for outer in 0..self.outer_plan.fold_count {
            let train: Vec<_> = (0..rows.len())
                .filter(|&i| self.outer_plan.fold_of_row[i] != outer)
                .collect();
            let held: Vec<_> = (0..rows.len())
                .filter(|&i| self.outer_plan.fold_of_row[i] == outer)
                .collect();
            let train_x = select_rows(rows, &train);
            let train_y = select_values(target, &train);
            let train_ids = select_values(row_ids, &train);
            let held_x = select_rows(rows, &held);
            let held_ids = select_values(row_ids, &held);
            let inner = match generated_plan(
                &train_ids,
                self.inner_fold_count.min(train.len()),
                fold_seed(self.seed, outer as u64),
            ) {
                Ok(plan) => plan,
                Err(error) => return fail(incumbent, error, records),
            };
            let mut residual = train_y.clone();
            for (stage_index, candidate) in stages.iter().enumerate() {
                let inner_oof = match inner.cross_fit_into(
                    candidate,
                    &train_x,
                    &residual,
                    &train_ids,
                    Some(outer),
                    &mut records,
                ) {
                    Ok(output) => output,
                    Err(error) => return fail(incumbent, error, records),
                };
                let context = FitContext {
                    abi: COMBINATOR_ABI,
                    candidate_id: candidate.id().into(),
                    role: FitRole::OuterRefit { fold: outer },
                    seed: fold_seed(self.seed, ((outer as u64) << 32) | stage_index as u64),
                    train_row_ids: train_ids.clone(),
                };
                records.push(FitRecord {
                    candidate_id: candidate.id().into(),
                    role: context.role.clone(),
                    seed: context.seed,
                    train_row_ids: train_ids.clone(),
                    held_row_ids: held_ids.clone(),
                });
                let fitted = match candidate.fit(FitInput {
                    rows: &train_x,
                    target: &residual,
                    context: &context,
                }) {
                    Ok(model) if model.prediction_target() == "original" => model,
                    Ok(_) => {
                        return fail(
                            incumbent,
                            FitError::Prediction(format!(
                                "{} does not predict in original target space",
                                candidate.id()
                            )),
                            records,
                        )
                    }
                    Err(error) => return fail(incumbent, error, records),
                };
                let stage_predictions = match checked_prediction(&fitted, &held_x) {
                    Ok(predictions) if predictions.len() == held.len() => predictions,
                    Ok(predictions) => {
                        return fail(
                            incumbent,
                            FitError::Prediction(format!(
                                "{} returned {} predictions for {} held rows",
                                candidate.id(),
                                predictions.len(),
                                held.len()
                            )),
                            records,
                        )
                    }
                    Err(error) => return fail(incumbent, error, records),
                };
                for (&index, prediction) in held.iter().zip(stage_predictions) {
                    oof[index] += self.shrinkage * prediction;
                }
                for (residual, prediction) in residual.iter_mut().zip(inner_oof) {
                    *residual -= self.shrinkage * prediction;
                }
            }
        }
        if oof.iter().any(|value| !value.is_finite()) {
            return fail(
                incumbent,
                FitError::Prediction("outer OOF composition overflowed".into()),
                records,
            );
        }
        let full_inner =
            match generated_plan(row_ids, self.inner_fold_count.min(rows.len()), self.seed) {
                Ok(plan) => plan,
                Err(error) => return fail(incumbent, error, records),
            };
        let mut residual = target.to_vec();
        let mut fitted_stages = Vec::with_capacity(stages.len());
        for candidate in stages {
            let inner_oof = match full_inner.cross_fit_into(
                candidate,
                rows,
                &residual,
                row_ids,
                None,
                &mut records,
            ) {
                Ok(output) => output,
                Err(error) => return fail(incumbent, error, records),
            };
            let context = full_context(candidate.id(), self.seed, row_ids);
            records.push(FitRecord {
                candidate_id: candidate.id().into(),
                role: FitRole::FullTraining,
                seed: context.seed,
                train_row_ids: row_ids.to_vec(),
                held_row_ids: vec![],
            });
            match candidate.fit(FitInput {
                rows,
                target: &residual,
                context: &context,
            }) {
                Ok(model) if model.prediction_target() == "original" => {
                    match checked_prediction(&model, rows) {
                        Ok(values) if values.len() == rows.len() => fitted_stages.push(model),
                        Ok(values) => {
                            return fail(
                                incumbent,
                                FitError::Prediction(format!(
                                    "{} returned {} predictions for {} full-fit rows",
                                    candidate.id(),
                                    values.len(),
                                    rows.len()
                                )),
                                records,
                            )
                        }
                        Err(error) => return fail(incumbent, error, records),
                    }
                }
                Ok(_) => {
                    return fail(
                        incumbent,
                        FitError::Prediction(format!(
                            "{} does not predict in original target space",
                            candidate.id()
                        )),
                        records,
                    )
                }
                Err(error) => return fail(incumbent, error, records),
            }
            for (residual, prediction) in residual.iter_mut().zip(inner_oof) {
                *residual -= self.shrinkage * prediction;
            }
        }
        let audit = FitAudit {
            abi: COMBINATOR_ABI,
            fit_count: records.len(),
            records,
        };
        BoostOutcome::Composed {
            model: BoostModel {
                abi: COMBINATOR_ABI,
                outer_plan: self.outer_plan.clone(),
                fit_row_ids: row_ids.to_vec(),
                seed: self.seed,
                inner_fold_count: self.inner_fold_count,
                stages: fitted_stages,
                shrinkage: self.shrinkage,
            },
            oof_predictions: oof,
            audit,
        }
    }
}

impl<M: FittedCandidate> BoostModel<M> {
    pub fn predict(&self, rows: &[Vec<f64>]) -> Result<Vec<f64>, FitError> {
        let mut output = vec![0.0; rows.len()];
        for model in &self.stages {
            let predictions = checked_prediction(model, rows)?;
            if predictions.len() != rows.len() {
                return Err(FitError::Prediction(
                    "boost member prediction width differs from row count".into(),
                ));
            }
            for (sum, value) in output.iter_mut().zip(predictions) {
                *sum += self.shrinkage * value;
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(FitError::Prediction("boost output overflowed".into()));
        }
        Ok(output)
    }

    pub fn predict_emitted(&self, rows: &[Vec<f64>]) -> Result<Vec<f64>, FitError> {
        let mut output = vec![0.0; rows.len()];
        for model in &self.stages {
            let predictions = emitted_prediction(model, rows)?;
            if predictions.len() != rows.len() {
                return Err(FitError::Prediction(
                    "emitted boost member prediction width differs from row count".into(),
                ));
            }
            for (sum, value) in output.iter_mut().zip(predictions) {
                *sum += self.shrinkage * value;
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(FitError::Prediction(
                "emitted boost output overflowed".into(),
            ));
        }
        Ok(output)
    }
}

fn generated_plan(row_ids: &[u64], fold_count: usize, seed: u64) -> Result<OOFFitPlan, FitError> {
    if fold_count < 2 || row_ids.len() < fold_count {
        return Err(FitError::InvalidPlan(
            "not enough rows for the inner OOF plan".into(),
        ));
    }
    let mut order: Vec<_> = (0..row_ids.len()).collect();
    order.sort_by_key(|&i| mix64(row_ids[i] ^ seed));
    let mut assignments = vec![0; row_ids.len()];
    for (rank, index) in order.into_iter().enumerate() {
        assignments[index] = rank % fold_count;
    }
    OOFFitPlan::new(assignments, fold_count, seed)
}

fn full_context(candidate_id: &str, seed: u64, row_ids: &[u64]) -> FitContext {
    FitContext {
        abi: COMBINATOR_ABI,
        candidate_id: candidate_id.into(),
        role: FitRole::FullTraining,
        seed,
        train_row_ids: row_ids.to_vec(),
    }
}

fn checked_prediction<M: FittedCandidate>(
    model: &M,
    rows: &[Vec<f64>],
) -> Result<Vec<f64>, FitError> {
    if model.prediction_target() != "original" {
        return Err(FitError::Prediction(
            "candidate output is not in the original target space".into(),
        ));
    }
    let values = model.predict(rows)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(FitError::Prediction(
            "candidate produced nonfinite values".into(),
        ));
    }
    Ok(values)
}

fn emitted_prediction<M: FittedCandidate>(
    model: &M,
    rows: &[Vec<f64>],
) -> Result<Vec<f64>, FitError> {
    if model.prediction_target() != "original" {
        return Err(FitError::Prediction(
            "emitted candidate output is not in original target space".into(),
        ));
    }
    let values = model.predict_emitted(rows)?.ok_or_else(|| {
        FitError::UnsupportedEmission("candidate has no portable fitted emitter".into())
    })?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(FitError::Prediction(
            "emitted candidate produced nonfinite values".into(),
        ));
    }
    Ok(values)
}

fn combine_predictions<M: FittedCandidate>(
    members: &[M],
    weights: &[f64],
    rows: &[Vec<f64>],
    emitted: bool,
) -> Result<Vec<f64>, FitError> {
    if members.is_empty()
        || members.len() != weights.len()
        || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
    {
        return Err(FitError::InvalidPlan(
            "member and weight shapes differ or weights are invalid".into(),
        ));
    }
    if (weights.iter().sum::<f64>() - 1.0).abs() > 1e-9 {
        return Err(FitError::InvalidPlan(
            "stack weights must sum to one".into(),
        ));
    }
    let mut output = vec![0.0; rows.len()];
    for (member, &weight) in members.iter().zip(weights) {
        let values = if emitted {
            emitted_prediction(member, rows)?
        } else {
            checked_prediction(member, rows)?
        };
        if values.len() != rows.len() {
            return Err(FitError::Prediction(
                "member prediction width differs from input row count".into(),
            ));
        }
        for (sum, value) in output.iter_mut().zip(values) {
            *sum += weight * value;
        }
    }
    Ok(output)
}

fn fit_simplex_weights(predictions: &[Vec<f64>], target: &[f64], iterations: usize) -> Vec<f64> {
    let count = predictions.len();
    let mut weights = vec![1.0 / count as f64; count];
    let scale =
        predictions.iter().flatten().map(|x| x * x).sum::<f64>() / target.len().max(1) as f64;
    let step = 0.5 / (scale * count as f64 + 1e-12);
    for _ in 0..iterations {
        let blended: Vec<_> = (0..target.len())
            .map(|i| {
                predictions
                    .iter()
                    .zip(&weights)
                    .map(|(p, w)| p[i] * w)
                    .sum::<f64>()
            })
            .collect();
        let gradient: Vec<_> = (0..count)
            .map(|m| {
                2.0 * (0..target.len())
                    .map(|i| predictions[m][i] * (blended[i] - target[i]))
                    .sum::<f64>()
                    / target.len() as f64
            })
            .collect();
        weights = project_simplex(
            weights
                .iter()
                .zip(gradient)
                .map(|(w, g)| w - step * g)
                .collect(),
        );
    }
    weights
}

fn project_simplex(mut values: Vec<f64>) -> Vec<f64> {
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let mut cumulative = 0.0;
    let mut theta = 0.0;
    for (index, value) in sorted.iter().enumerate() {
        cumulative += value;
        let candidate = (cumulative - 1.0) / (index + 1) as f64;
        if index + 1 == sorted.len() || sorted[index + 1] <= candidate {
            theta = candidate;
            break;
        }
    }
    for value in &mut values {
        *value = (*value - theta).max(0.0);
    }
    values
}

fn valid_shrinkage(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}
fn select_rows(rows: &[Vec<f64>], indices: &[usize]) -> Rows {
    indices.iter().map(|&i| rows[i].clone()).collect()
}
fn select_values<T: Copy>(values: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&i| values[i]).collect()
}
fn fold_seed(seed: u64, part: u64) -> u64 {
    mix64(seed ^ part.wrapping_mul(0x9e3779b97f4a7c15))
}
fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}
