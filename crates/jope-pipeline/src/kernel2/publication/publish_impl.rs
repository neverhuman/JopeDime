use super::*;

pub fn publish(
    cut: &crate::prepare::Cut,
    searched: &CutSearch,
    seeding: Seeding,
    root: &Path,
) -> Result<(Search, String)> {
    let measurements: Vec<_> = searched
        .observations
        .iter()
        .map(Measurement::record)
        .collect::<Result<_>>()?;
    // Two parents can refine to one program in a round, giving equal child
    // measurements. Match by occurrence: the k-th child equal to a measurement
    // takes the k-th equal observation, in the order the search recorded them.
    let mut claimed = vec![false; searched.observations.len()];
    let refinements = searched
        .refinements
        .iter()
        .map(|r| {
            let measurement = r
                .measurement
                .as_ref()
                .map(|m| -> Result<usize> {
                    let index = searched
                        .observations
                        .iter()
                        .zip(&claimed)
                        .position(|(o, &taken)| !taken && o == m)
                        .context("refined measurement missing from observations")?;
                    claimed[index] = true;
                    Ok(index)
                })
                .transpose()?;
            let reason = (measurement.is_none() && r.refinement.failure.is_none())
                .then(|| searched.stopped_by.to_owned());
            Ok(RefinementRecord {
                parent_sha256: r.parent_sha256.clone(),
                refinement: r.refinement.clone(),
                measurement,
                measurement_not_started_reason: reason,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let dir = root.join("search/cut-0");
    std::fs::create_dir_all(&dir)?;
    let write = |name: &str, value: &serde_json::Value| -> Result<String> {
        let bytes = serde_json::to_vec_pretty(value)?;
        std::fs::write(dir.join(name), &bytes)?;
        Ok(sha256_hex(&bytes))
    };
    let observations_sha256 = write("observations.json", &serde_json::to_value(&measurements)?)?;
    let refinements_sha256 = write("refinements.json", &serde_json::to_value(&refinements)?)?;
    // laneD:B18-pareto-archive begin
    let engine_selection = super::super::archive::publish(searched, root)?;
    // laneD:B18-pareto-archive end
    // laneD:B20-rewrite-simplify begin
    let engine_selection = super::super::search::moves::publish(searched, root, engine_selection)?;
    // laneD:B20-rewrite-simplify end
    let record = Search {
        engine: receipt::ENGINE.into(),
        profile_id: searched.profile_id.into(),
        profile_sha256: searched.profile_sha256.clone(),
        fault_policy: searched.fault_policy.clone(),
        rng_seed: searched.rng_seed,
        evaluations: searched.budget_evaluations,
        seconds: searched.budget_seconds,
        refinement: searched.refinement,
        cut: cut.index,
        train_rows: cut.train_rows(),
        hold_rows: cut.hold_rows(),
        generations: searched.generations,
        completed_evaluations: searched.completed_evaluations,
        candidate_measurements: searched.candidate_measurements,
        objective_trials: searched.objective_trials,
        rejected_trials: searched.rejected_trials,
        successful_measurements: searched.successful_measurements,
        stopped_by: searched.stopped_by.into(),
        active_seconds: searched.seconds,
        deadline_overrun: searched.seconds > searched.budget_seconds as f64,
        arm_counts: searched
            .arm_counts
            .iter()
            .map(|(name, count)| ((*name).into(), *count))
            .collect(),
        seeding,
        selection: receipt::SELECTION.into(),
        best: Measurement::record(&searched.best)?,
        observations_sha256,
        refinements_sha256,
        engine_selection,
    };
    let digest = write("search.json", &serde_json::to_value(&record)?)?;
    Ok((record, digest))
}
