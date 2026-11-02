use super::*;

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
