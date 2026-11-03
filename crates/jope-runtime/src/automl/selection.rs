use super::*;

/// One scored candidate's evidence, kept until the winner is known.
struct Scored {
    loss: f64,
    calibrated: bool,
    predictions: Vec<f64>,
}

/// Where the scored rows sit in a forward-validated table, so a fold's output map obeys the
/// same embargo as its model.
pub(super) struct ForwardRows<'a> {
    /// Table row of each scored row.
    pub(super) positions: &'a [usize],
    /// First table row of each fold.
    pub(super) starts: Vec<usize>,
    pub(super) gap: usize,
}

/// Scored rows whose out-of-fold predictions may teach the map applied to `fold`: every other
/// fold, or with forward validation only the rows the fold's model could have trained on (rows
/// more than `gap` before the fold starts; none for the first fold).
pub(super) fn calibration_rows(
    folds_of_scored: &[usize],
    fold: usize,
    forward: Option<&ForwardRows>,
) -> Vec<usize> {
    (0..folds_of_scored.len())
        .filter(|&i| match forward {
            Some(rows) => rows.positions[i] + rows.gap < rows.starts[fold],
            None => folds_of_scored[i] != fold,
        })
        .collect()
}

/// The objective's score of one candidate's out-of-fold predictions: raw, and cross-fitted
/// calibrated when a calibration family is in force. Calibration stops at the time limit.
#[allow(clippy::too_many_arguments)]
fn objective_score(
    fitness: &Fitness,
    family: Calibration,
    y: &[f64],
    p: &[f64],
    columns: &Columns,
    folds_of_scored: &[usize],
    forward: Option<&ForwardRows>,
    budget: &Budget,
) -> Result<(Option<f64>, Option<f64>)> {
    let raw = fitness.evaluate_with(y, p, columns).map(|s| s.loss);
    if family == Calibration::None {
        return Ok((Some(raw?), None));
    }
    budget.check("output calibration")?;
    let fit_rows = |fold: usize| calibration_rows(folds_of_scored, fold, forward);
    let crossed = calibration::cross_fit(
        fitness,
        family,
        y,
        p,
        columns,
        folds_of_scored,
        &fit_rows,
        &|| budget.exhausted(),
    )?;
    let calibrated = fitness.evaluate_with(y, &crossed, columns).map(|s| s.loss);
    match (raw, calibrated) {
        (Err(raw), Err(_)) => Err(raw),
        (raw, calibrated) => Ok((raw.ok(), calibrated.ok())),
    }
}

/// Selection with a portable regression objective, forward validation and output calibration.
/// `None` objective with IID folds and no calibration is exactly [`select`] (`v1`).
pub fn select_with_fitness(
    training: &Training,
    candidates: &[RustMethod],
    requested_folds: usize,
    budget: &Budget,
    fitness: Option<&Fitness>,
    validation: &Validation,
    calibration: Calibration,
) -> Result<(Selection, RustMethod)> {
    let family = calibration.resolve(fitness.is_some());
    if let Some(fitness) = fitness {
        fitness.validate()?;
        anyhow::ensure!(
            training.kind == TaskKind::Regression,
            "custom fitness is regression-only"
        );
        anyhow::ensure!(
            !fitness.requires_order() || matches!(validation, Validation::Forward { .. }),
            "{} depends on row order and requires forward validation",
            fitness.name()
        );
        for column in fitness.columns() {
            anyhow::ensure!(
                training.auxiliary.contains_key(&column),
                "the training table did not supply objective column {column:?}"
            );
        }
    } else {
        anyhow::ensure!(
            family == Calibration::None,
            "output calibration needs an objective (--fitness)"
        );
    }
    anyhow::ensure!(
        matches!(validation, Validation::Iid {}) || training.kind == TaskKind::Regression,
        "forward validation is regression-only"
    );
    let k = effective_folds(training, requested_folds)?;
    let n = training.targets.len();
    let assignment = match validation {
        Validation::Iid {} => folds(training.kind, &training.targets, k),
        Validation::Forward { gap } => {
            let block = n / (k + 1);
            if block.saturating_sub(*gap) < 2 {
                return Err(Failure::input(format!(
                    "forward validation needs more rows or a smaller fold count/gap \
                     ({n} rows, {k} folds, gap {gap})"
                )));
            }
            (0..n)
                .map(|r| {
                    if r < block {
                        usize::MAX
                    } else {
                        ((r - block) / block).min(k - 1)
                    }
                })
                .collect()
        }
    };
    let forward = matches!(validation, Validation::Forward { .. });
    let warmup_rows = assignment.iter().filter(|f| **f == usize::MAX).count();
    let scored_rows: Vec<usize> = (0..n).filter(|r| assignment[*r] != usize::MAX).collect();
    let forward_rows = match validation {
        Validation::Forward { gap } => Some(ForwardRows {
            positions: &scored_rows,
            starts: (0..k)
                .map(|f| {
                    assignment
                        .iter()
                        .position(|a| *a == f)
                        .expect("every forward fold has rows")
                })
                .collect(),
            gap: *gap,
        }),
        Validation::Iid {} => None,
    };
    let scored_targets: Vec<f64> = scored_rows.iter().map(|r| training.targets[*r]).collect();
    let scored_folds: Vec<usize> = scored_rows.iter().map(|r| assignment[*r]).collect();
    let scored_columns = calibration::subset_columns(&training.auxiliary, &scored_rows);
    let mut outcomes = Vec::with_capacity(candidates.len());
    let mut best: Option<(f64, usize)> = None;
    let mut evidence: Vec<Option<Scored>> = Vec::with_capacity(candidates.len());
    for (index, method) in candidates.iter().enumerate() {
        let started = Instant::now();
        let mut outcome = CandidateOutcome {
            id: method.id.clone(),
            status: "skipped",
            loss: None,
            fitness_value: None,
            uncalibrated_loss: None,
            calibrated_loss: None,
            calibrated: false,
            folds_completed: 0,
            reason: None,
            seconds: 0.0,
        };
        evidence.push(None);
        if budget.exhausted() {
            outcome.reason = Some("time limit reached".into());
            outcomes.push(outcome);
            continue;
        }
        let mut completed = 0;
        let result = crate::error::contain(&method.id, || {
            out_of_fold(
                method,
                training,
                &assignment,
                k,
                budget,
                &mut completed,
                validation,
            )
        })
        .and_then(|(p, scored)| {
            anyhow::ensure!(
                scored == scored_rows,
                "scored rows differ between candidates"
            );
            let Some(fitness) = fitness else {
                let value = loss(training.kind, &scored_targets, &p)?;
                return Ok((value, None, None, false, Vec::new()));
            };
            let Predictions::Values(values) = p else {
                anyhow::bail!("regression fitness needs numeric predictions");
            };
            let (raw, calibrated) = objective_score(
                fitness,
                family,
                &scored_targets,
                &values,
                &scored_columns,
                &scored_folds,
                forward_rows.as_ref(),
                budget,
            )?;
            let use_calibrated = match (raw, calibrated) {
                (Some(r), Some(c)) => c < r,
                (None, Some(_)) => true,
                _ => false,
            };
            let value =
                if use_calibrated { calibrated } else { raw }.expect("one score is present");
            Ok((value, raw, calibrated, use_calibrated, values))
        });
        outcome.seconds = started.elapsed().as_secs_f64();
        outcome.folds_completed = completed;
        match result {
            Ok((value, raw, calibrated, use_calibrated, predictions)) => {
                if best.is_none_or(|(b, _)| value < b) {
                    best = Some((value, index));
                }
                let natural = match fitness.map(Fitness::direction) {
                    Some(Direction::Maximize) => -value,
                    _ => value,
                };
                outcome.status = "scored";
                outcome.loss = Some(value);
                outcome.fitness_value = fitness.is_some().then_some(natural);
                if family != Calibration::None {
                    outcome.uncalibrated_loss = raw;
                    outcome.calibrated_loss = calibrated;
                    outcome.calibrated = use_calibrated;
                }
                evidence[index] = Some(Scored {
                    loss: value,
                    calibrated: use_calibrated,
                    predictions,
                });
            }
            Err(error) => {
                let skipped = crate::error::kind_of(&error) == crate::FailureKind::Budget;
                outcome.status = if skipped { "skipped" } else { "failed" };
                outcome.reason = Some(format!("{error:#}"));
            }
        }
        outcomes.push(outcome);
    }
    let Some((_, winner)) = best else {
        let budget_ran_out = outcomes.iter().any(|o| o.status == "skipped");
        let detail: Vec<String> = outcomes
            .iter()
            .map(|o| format!("{}: {}", o.id, o.reason.as_deref().unwrap_or(o.status)))
            .collect();
        let message = format!("no candidate completed selection ({})", detail.join("; "));
        return Err(if budget_ran_out {
            Failure::budget(message)
        } else {
            anyhow::anyhow!(message)
        });
    };
    let calibration_record = match (fitness, family) {
        (Some(fitness), family) if family != Calibration::None => {
            let won = evidence[winner].as_ref().expect("the winner was scored");
            debug_assert!(won.loss.is_finite());
            // The final map honours the clock like the cross-fitted ones; a budget that runs out
            // here fails the selection. A winner that kept its raw predictions keeps the floor.
            let output_transform = if won.calibrated {
                Some(calibration::fit_until(
                    fitness,
                    family,
                    &scored_targets,
                    &won.predictions,
                    &scored_columns,
                    &|| budget.exhausted(),
                )?)
            } else {
                calibration::floor_only(fitness, &scored_targets)
            };
            Some(CalibrationRecord {
                family,
                method: calibration::CALIBRATION_METHOD,
                output_transform,
            })
        }
        _ => None,
    };
    let v2 = fitness.is_some() || forward;
    let selection = Selection {
        policy: if v2 {
            SELECTION_POLICY_V2
        } else {
            SELECTION_POLICY
        },
        fold_policy: if forward {
            FORWARD_FOLD_POLICY
        } else {
            FOLD_POLICY
        },
        folds: k,
        metric: fitness.map_or_else(|| task::metric(training.kind).to_owned(), Fitness::name),
        loss: if fitness.is_some() {
            "direction_normalized_fitness".into()
        } else {
            loss_name(training.kind).into()
        },
        fitness: fitness.cloned(),
        fitness_sha256: fitness.map(Fitness::identity).transpose()?,
        validation: v2.then(|| validation.clone()),
        scored_rows: v2.then_some(n - warmup_rows),
        warmup_rows: v2.then_some(warmup_rows),
        calibration: calibration_record,
        selected: candidates[winner].id.clone(),
        candidates: outcomes,
    };
    Ok((selection, candidates[winner].clone()))
}
