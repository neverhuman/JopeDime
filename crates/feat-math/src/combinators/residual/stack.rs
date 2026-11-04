use super::*;

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
