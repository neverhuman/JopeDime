use super::*;

pub(super) fn check_refinements(
    s: &Search,
    measurements: &[Measurement],
    records: &[RefinementRecord],
) -> Result<()> {
    let mut trials = 0usize;
    let mut rejected = 0usize;
    let mut measured = BTreeSet::new();
    for record in records {
        let refinement = &record.refinement;
        ensure!(
            refinement.evaluations <= s.refinement.trials_per_program
                && refinement.rejected_trials <= refinement.evaluations
                && s.refinement.top_k > 0,
            "invalid refinement allowance"
        );
        trials = trials
            .checked_add(refinement.evaluations)
            .context("trial count overflow")?;
        rejected = rejected
            .checked_add(refinement.rejected_trials)
            .context("rejected count overflow")?;
        let parents: Vec<_> = measurements
            .iter()
            .filter(|m| {
                m.observation.program_sha256 == record.parent_sha256
                    && m.observation.status == crate::search::MEASURED
            })
            .collect();
        ensure!(!parents.is_empty(), "refinement parent was not measured");
        // laneD:B17-lm-refinement begin
        check_trace(s, refinement, &parents[0].observation.genome)?;
        // laneD:B17-lm-refinement end
        ensure!(
            refinement.changed
                == parents
                    .iter()
                    .all(|p| p.observation.genome != refinement.program),
            "refinement change flag disagrees"
        );
        if let Some(index) = record.measurement {
            let child = measurements
                .get(index)
                .context("refinement measurement index out of range")?;
            // Publication assigns children by occurrence in the order the search
            // measured them, so the recorded child indices strictly increase.
            ensure!(
                measured.last().is_none_or(|&last| index > last)
                    && measured.insert(index)
                    && child.observation.origin == super::super::REFINE
                    && child.observation.genome == refinement.program
                    && refinement.evaluations > 0
                    && refinement.failure.is_none()
                    && record.measurement_not_started_reason.is_none()
                    && measurements[..index]
                        .iter()
                        .any(|p| p.observation.program_sha256 == record.parent_sha256
                            && p.observation.status == crate::search::MEASURED
                            && p.observation.generation <= child.observation.generation),
                "refinement child does not re-derive"
            );
        } else {
            ensure!(
                (refinement.failure.is_some() && record.measurement_not_started_reason.is_none())
                    || (record.measurement_not_started_reason.as_deref()
                        == Some(crate::search::STOPPED_BY_SECONDS)
                        && s.stopped_by == crate::search::STOPPED_BY_SECONDS),
                "missing child measurement lacks a stop/failure reason"
            );
        }
    }
    ensure!(
        trials == s.objective_trials
            && rejected == s.rejected_trials
            && measured
                == measurements
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| m.observation.origin == super::super::REFINE)
                    .map(|(i, _)| i)
                    .collect(),
        "refinement accounting does not re-derive"
    );
    Ok(())
}

// laneD:B17-lm-refinement begin
/// A refinement carries a trace exactly when the damped least-squares method was selected and it
/// spent a trial. The trace's stages, each checked against its own counts and bound to the
/// `parent` program's stages and the refinement allowance, account for every charged trial and
/// refusal.
pub(super) fn check_trace(s: &Search, refinement: &TrainingRefinement, parent: &str) -> Result<()> {
    let selected = s
        .engine_selection
        .as_ref()
        .and_then(super::super::EngineRecord::damped_ls);
    let expected = selected.is_some() && refinement.evaluations > 0;
    ensure!(
        refinement.trace.is_some() == expected,
        "refinement trace presence disagrees with the selected method"
    );
    if let (Some(trace), Some(selected)) = (&refinement.trace, selected) {
        let (evaluations, rejected) =
            trace.check(parent, &selected.jacobian, s.refinement.trials_per_program)?;
        ensure!(
            evaluations == refinement.evaluations && rejected == refinement.rejected_trials,
            "refinement trace does not account for its trials"
        );
    }
    Ok(())
}
// laneD:B17-lm-refinement end
