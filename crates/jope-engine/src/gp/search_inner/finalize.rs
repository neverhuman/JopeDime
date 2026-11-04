    write_json(&output.join("observations.json"), &observations)?;
    experiment_failpoint("observations_snapshot", experiment.is_some());
    drop(journal);
    // Retain the durable journal until search publication; restart must never lose committed work.
    let observations_sha256 = file_hash(&output.join("observations.json"))?;
    let successful_measurements = observations
        .iter()
        .filter(|row| row["failure"].is_null() && row["metrics"].is_object())
        .count();
    let search_request = relative_artifact(&completion_root, &request_path)?;
    write_json(
        &output.join("search.json"),
        &json!({"schema":"parallel.search.v1","search_request":search_request,"search_id":file_hash(&request_path)?,"random_protocol":RANDOM_PROTOCOL,"execution_accounting":execution_accounting(&observations),"terminal":true,"stop_reason":stop_reason,"successful_measurements":successful_measurements,"failed_or_unavailable_measurements":observations.len()-successful_measurements,"observations_sha256":observations_sha256,"completed_at":crate::telemetry::now(),"evaluator_sha256":evaluator_sha,"evaluator_protocol":crate::label_admission::PROTOCOL,"vocabulary_sha256":vocabulary_sha256,"prepared_sha256":prepared_sha,"cut":cut,"completed_evaluations":observations.len(),"wall_seconds":start.elapsed().as_secs_f64()+previous_seconds,"lifecycle_seconds":lifecycle_start.elapsed().as_secs_f64(),"search_clock":clock,"execution_order_protocol":execution_protocol,"execution_order":execution_order,"resumed":resuming,"rng_seed":rng_seed,"proposal_attempts":proposal_attempts,"rejected_proposals":rejected_proposals,"unique_programs":evaluated_programs.len().saturating_sub(usize::from(native.is_some_and(|(_, task, _, _)| task.stage == crate::native_genome::Stage::Continuation))),"inherited_parent_excluded":native.is_some_and(|(_, task, _, _)| task.stage == crate::native_genome::Stage::Continuation),"best_validation_rmse":ranked.first().map(|x|x.0),"best_program":ranked.first().map(|x|&x.1),"seed_population_supplied":supplied.is_some(),"neural_seeded":neural_seeded,"seeds_sha256":seeds_sha256,"vocabulary_sha256":vocabulary_sha256,"final_tests_opened":false}),
    )?;
    experiment_failpoint("search_summary", experiment.is_some());
    // Keep the original journal after publication for historical reconciliation.
    Ok(())
