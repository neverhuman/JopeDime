use super::*;

impl<'a> ContextualSurrogateQHat<'a> {
    /// Performs the fit operation and returns an error if it cannot be completed.
    pub fn fit(
        train: &[&'a Pool],
        context_features: &'a Features,
        program_features: &'a Features,
        k_neighbours: usize,
        prior_weight: f64,
        lambda: f64,
    ) -> Result<Self> {
        Self::fit_with_prior(
            train,
            context_features,
            program_features,
            k_neighbours,
            prior_weight,
            lambda,
            None,
        )
    }

    /// Performs the fit with prior operation and returns an error if it cannot be completed.
    pub fn fit_with_prior(
        train: &[&'a Pool],
        context_features: &'a Features,
        program_features: &'a Features,
        k_neighbours: usize,
        prior_weight: f64,
        lambda: f64,
        external_prior: Option<&ExternalProgramPrior>,
    ) -> Result<Self> {
        let parents: BTreeSet<_> = train.iter().map(|p| p.parent_id.as_str()).collect();
        let folds: Vec<_> = train
            .iter()
            .map(|p| super::evaluate::fold_of(INNER_SALT, &p.parent_id, INNER_FOLDS))
            .collect();
        let mut counts = CvCounts {
            attempted_units: train.iter().map(|p| p.members.len()).sum(),
            parents: parents.len(),
            folds: folds.iter().copied().collect::<BTreeSet<_>>().len(),
            ..CvCounts::default()
        };
        if parents.len() < 3 {
            return Err(unavailable(
                CvUnavailableReason::InsufficientParents,
                "nested OOF needs at least three parents",
                counts,
            ));
        }
        if counts.attempted_units == 0 {
            return Err(unavailable(
                CvUnavailableReason::EmptyUsableFolds,
                "no member rows",
                counts,
            ));
        }
        if train
            .iter()
            .any(|p| p.members.len() < 2 || p.members.iter().all(|m| m.outcome().is_none()))
        {
            return Err(unavailable(
                CvUnavailableReason::EmptyUsableFolds,
                "tiny or wholly failed pool has no usable ranking target",
                counts,
            ));
        }
        if train
            .iter()
            .flat_map(|p| &p.members)
            .filter_map(|m| m.q)
            .any(|v| !v.is_finite())
        {
            return Err(unavailable(
                CvUnavailableReason::NonfiniteSystem,
                "nonfinite measured outcome",
                counts,
            ));
        }
        for features in [context_features, program_features] {
            if features.dim == 0
                || features.rows.is_empty()
                || features.rows.values().any(|v| v.len() != features.dim)
            {
                return Err(unavailable(
                    CvUnavailableReason::ShapeMismatch,
                    "feature width mismatch",
                    counts,
                ));
            }
            if features.rows.values().flatten().any(|v| !v.is_finite()) {
                return Err(unavailable(
                    CvUnavailableReason::NonfiniteSystem,
                    "nonfinite input features",
                    counts,
                ));
            }
        }
        context_features.verify_context_descriptor().map_err(|e| {
            unavailable(
                CvUnavailableReason::UnboundContextProvenance,
                e,
                counts.clone(),
            )
        })?;
        if let Some(prior) = external_prior {
            prior.restrict_to(train).map_err(|e| {
                unavailable(
                    CvUnavailableReason::ExternalPriorProvenance,
                    e,
                    counts.clone(),
                )
            })?;
        }
        if counts.folds < 2 {
            return Err(unavailable(
                CvUnavailableReason::EmptyUsableFolds,
                "all parents collide in one fold",
                counts,
            ));
        }
        // Recompute every component and both train/held design blocks inside each inner closure.
        let mut closures = Vec::new();
        let mut evidence = Vec::new();
        for fold in 0..INNER_FOLDS {
            let held: Vec<_> = train
                .iter()
                .copied()
                .zip(&folds)
                .filter_map(|(p, &f)| (f == fold).then_some(p))
                .collect();
            if held.is_empty() {
                continue;
            }
            let fitting: Vec<_> = train
                .iter()
                .copied()
                .zip(&folds)
                .filter_map(|(p, &f)| (f != fold).then_some(p))
                .collect();
            if fitting
                .iter()
                .map(|p| &p.parent_id)
                .collect::<BTreeSet<_>>()
                .len()
                < 2
            {
                return Err(unavailable(
                    CvUnavailableReason::InsufficientParents,
                    "inner training closure cannot cross-fit its base components",
                    counts,
                ));
            }
            let state = Self::fit_state(
                &fitting,
                context_features,
                program_features,
                k_neighbours,
                prior_weight,
                lambda,
                external_prior,
            )
            .map_err(|e| component_error(e, counts.clone()))?;
            let (xtx, xty) = state
                .training_equations(&fitting, external_prior)
                .map_err(|e| component_error(e, counts.clone()))?;
            let designs = held
                .iter()
                .map(|p| state.view_design_rows(&p.view()))
                .collect::<Result<Vec<_>>>()
                .map_err(|e| component_error(e, counts.clone()))?;
            evidence.push(InnerFoldFit {
                fold,
                training_parents: fitting
                    .iter()
                    .map(|p| p.parent_id.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect(),
                held_parents: held
                    .iter()
                    .map(|p| p.parent_id.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect(),
                training_input_sha256: canonical_sha256(
                    &fitting
                        .iter()
                        .map(|p| p.sha256())
                        .collect::<Result<Vec<_>>>()?,
                )?,
                training_features_sha256: canonical_sha256(
                    &fitting
                        .iter()
                        .map(|p| {
                            (
                                &context_features.rows[&p.context_id],
                                p.members
                                    .iter()
                                    .map(|m| &program_features.rows[&m.program_sha256])
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>(),
                )?,
                split_sha256: canonical_sha256(&(
                    NESTED_PROCEDURE,
                    fold,
                    &evidence_parent_ids(&fitting),
                    &evidence_parent_ids(&held),
                ))?,
                state_sha256: state.state_sha256()?,
                held_design_sha256: canonical_sha256(&designs)?,
                prediction_sha256: Vec::new(),
                selected_prediction_bits: Vec::new(),
            });
            closures.push((state, xtx, xty, held, designs));
        }
        let mut best: Option<(f64, f64, f64, usize)> = None;
        for bm in [1.0, 10.0, 100.0, 1000.0] {
            for pm in [1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0] {
                if pm < bm {
                    continue;
                }
                let mut sq_err = 0.0;
                let mut scored = 0;
                let mut candidate_bits = Vec::new();
                let mut valid = true;
                for (fold_index, (state, xtx, xty, held, designs)) in closures.iter().enumerate() {
                    let weights = match state.solve_weights(xtx.clone(), xty.clone(), bm, pm) {
                        Ok(w) => w,
                        Err(error) => {
                            counts.failure_details.push(format!(
                                "fold={},bm={bm},pm={pm}: {error}",
                                evidence[fold_index].fold
                            ));
                            valid = false;
                            continue;
                        }
                    };
                    let predictions: Vec<Vec<f64>> = designs
                        .iter()
                        .map(|rows| {
                            rows.iter()
                                .map(|x| x.iter().zip(&weights).map(|(x, w)| x * w).sum())
                                .collect()
                        })
                        .collect();
                    if predictions.iter().flatten().any(|v| !v.is_finite()) {
                        counts.failure_details.push(format!(
                            "fold={},bm={bm},pm={pm}: nonfinite held predictions",
                            evidence[fold_index].fold
                        ));
                        valid = false;
                        continue;
                    }
                    // Predictions are fixed before touching held outcomes.
                    evidence[fold_index]
                        .prediction_sha256
                        .push(canonical_sha256(&(format!("{bm},{pm}"), &predictions))?);
                    candidate_bits.push(
                        predictions
                            .iter()
                            .map(|row| {
                                row.iter()
                                    .map(|p| format!("{:016x}", p.to_bits()))
                                    .collect()
                            })
                            .collect(),
                    );
                    for (pool, preds) in held.iter().zip(predictions) {
                        let targets = rank_targets(pool);
                        if preds.len() != targets.len() {
                            return Err(unavailable(
                                CvUnavailableReason::ShapeMismatch,
                                "held prediction width",
                                counts,
                            ));
                        }
                        if preds.is_empty() {
                            continue;
                        }
                        let n = preds.len() as f64;
                        let pmean = preds.iter().sum::<f64>() / n;
                        let tmean = targets.iter().sum::<f64>() / n;
                        for (p, t) in preds.iter().zip(targets) {
                            sq_err += ((p - pmean) - (t - tmean)).powi(2);
                            scored += 1;
                        }
                    }
                }
                if !valid || scored == 0 || !sq_err.is_finite() {
                    counts.failed_candidates += 1;
                    continue;
                }
                let loss = sq_err / scored as f64;
                if best.is_none_or(|b| loss < b.2) {
                    best = Some((bm, pm, loss, scored));
                    for (record, bits) in evidence.iter_mut().zip(candidate_bits) {
                        record.selected_prediction_bits = bits;
                    }
                }
            }
        }
        let Some((bm, pm, loss, scored)) = best else {
            return Err(unavailable(
                CvUnavailableReason::NoFiniteCandidate,
                "no candidate produced a finite complete OOF loss",
                counts,
            ));
        };
        counts.scored_units = scored;
        counts.excluded_units = counts.attempted_units - scored;
        let mut model = Self::fit_state(
            train,
            context_features,
            program_features,
            k_neighbours,
            prior_weight,
            lambda,
            external_prior,
        )
        .map_err(|e| component_error(e, counts.clone()))?;
        let (xtx, xty) = model
            .training_equations(train, external_prior)
            .map_err(|e| component_error(e, counts.clone()))?;
        model.weights = model
            .solve_weights(xtx, xty, bm, pm)
            .map_err(|e| component_error(e, counts.clone()))?;
        model.base_mult = bm;
        model.proj_mult = pm;
        model.inner_cv = CvEstimate::Supported {
            procedure: NESTED_PROCEDURE.into(),
            metric: ExactCvLoss {
                f64_bits: format!("{:016x}", loss.to_bits()),
            },
            counts,
        };
        model.inner_fold_fits = evidence;
        Ok(model)
    }
}
