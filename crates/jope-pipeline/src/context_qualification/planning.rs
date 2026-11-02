use super::*;

pub(super) fn make_jobs(
    experiment: &Experiment,
    references: &MaterializedReferences,
    program: &Artifact,
    assets: &EncoderAssets,
    journal: &Path,
) -> Result<Vec<WorkerJob>> {
    let bytes = read_artifact(&references.manifest, 128 << 20)?;
    let expected_inputs = context_producer::case_inputs(&bytes, &references.manifest.sha256)?;
    ensure!(
        references.cases.len() == 8 && references.job_identities.len() == 8,
        "incomplete qualification case inventory"
    );
    let manifest: context_producer::ReferenceManifest = serde_json::from_slice(&bytes)?;
    let mut jobs = Vec::new();
    for (index, expected_input) in expected_inputs.iter().enumerate() {
        let input = &references.cases[index];
        ensure!(
            read_artifact(input, 128 << 20)? == *expected_input,
            "case payload differs from predeclared manifest order"
        );
        let identity = &references.job_identities[index];
        let reference = &manifest.cases[index];
        let native: serde_json::Value = serde_json::from_slice(&reference.context_receipt)?;
        ensure!(
            identity.fitting_rows == reference.prepared_rows.len()
                && identity.fold as u64
                    == native["cut"].as_u64().context("native reference cut")?
                && native["prepared_sha256"] == identity.split_sha256
                && native["selected_training_rows_sha256"] == identity.fitting_rows_sha256,
            "qualification diagnostic identity differs from actual reference rows"
        );
        let reference_coordinates = reference.components["hyperion"]
            .vector
            .iter()
            .chain(&reference.components["foundation"].vector)
            .chain(&reference.components["target"].vector)
            .chain(&reference.context_vector)
            .copied()
            .collect::<Vec<_>>();
        ensure!(
            reference_coordinates.len() == 5120 * 4,
            "native diagnostic coordinate count changed"
        );
        let request = EvaluationRequest {
            schema: REQUEST_SCHEMA.into(),
            experiment_sha256: experiment.identity()?,
            dataset_sha256: identity.dataset_sha256.clone(),
            split_sha256: identity.split_sha256.clone(),
            fitting_rows_sha256: identity.fitting_rows_sha256.clone(),
            scoring_rows_sha256: sha256_hex(&reference_coordinates),
            fitting_rows: identity.fitting_rows,
            scoring_rows: 5120,
            recipe_sha256: sha256_hex(&serde_json::to_vec(&(
                QUALIFICATION_PROTOCOL,
                &program.sha256,
                &assets.configuration.sha256,
                &input.sha256,
            ))?),
            evaluator_sha256: context_producer::verification_policy_sha256(),
            checkpoint_sha256: Some(context_producer::TRANSFORMS_SHA256.into()),
            task: jope_datasets::catalog::Task::Regression,
            metric: Metric::Rmse,
            classes: vec![],
            fold: identity.fold,
            stage: EvaluationStage::Diagnostic,
            seed: experiment.seed,
            numerical_policy: context_producer::verification_policy_sha256(),
        };
        let directory = journal.join("jobs").join(request.identity()?);
        let invocation = WorkerInvocation {
            mode: WorkerMode::QualifyCase,
            input: policy_artifact(input),
            assets: policy_artifact(&assets.configuration),
            source_sha40: experiment.source_sha.clone(),
            producer_sha256: program.sha256.clone(),
            output: directory.join("case-output.json"),
        };
        let mut inputs = vec![input.clone(), assets.configuration.clone()];
        inputs.extend(assets.files.clone());
        let job = WorkerJob {
            request,
            command: WorkerCommand {
                program: program.clone(),
                args: invocation.arguments(),
                inputs,
                environment: BTreeMap::new(),
            },
            directory,
            outputs: vec!["case-output.json".into()],
            arm: None,
        };
        job.identity()?;
        jobs.push(job);
    }
    Ok(jobs)
}
