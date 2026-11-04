        if let Some(r) = &result {
            experiment_failpoint("candidate_metric", experiment.is_some());
            let rmse = r["rmse"].as_f64().context("invalid evaluator RMSE")? * pp.target_scale;
            ensure!(rmse.is_finite() && rmse >= 0.0, "invalid evaluator score");
            let signature = r["semantic_sha256"]
                .as_str()
                .context("missing prediction signature")?;
            if semantic.insert(signature.to_string()) {
                ranked.push((rmse, genome.clone(), i as i64));
                ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
                ranked.truncate(80);
                if i + 1 == 200 {
                    let _ = write_json(
                        &output.join("rerank-200.json"),
                        &json!({"schema":"parallel.gp_rerank.v1","evaluation_index":200,
                            "survivors":ranked.len(),"order":"validation_rmse_then_policy",
                            "parent_checkpoint_used":supplied.as_ref().map(|s|&s.checkpoint_sha256)}),
                    );
                }
            }
        }
        use std::io::Read;
        let mut diagnostic = Vec::new();
        if let Ok(log) = fs::File::open(output.join(format!("candidate-{i}.log"))) {
            log.take(8192).read_to_end(&mut diagnostic)?;
        }
        let diagnostic = String::from_utf8_lossy(&diagnostic).to_string();
        let origin = if let Some((_, task, _, _)) = native {
            match task.stage {
                crate::native_genome::Stage::Direct => "native_genome_direct_v1",
                crate::native_genome::Stage::Continuation => {
                    "native_budgeted_seeded_continuation_v1"
                }
            }
        } else if initial_proposal {
            supplied
                .as_ref()
                .map(|s| s.candidates[execution_order[i]].origin.as_str())
                .unwrap_or("generic_bootstrap")
        } else {
            "gp_evolution"
        };
        let observation = json!({"schema":"parallel.gp_observation.v1","dataset":p.dataset,"family":p.family,"partition":p.meta_partition,"cut":cut,"prepared_sha256":prepared_sha,"training_sha256":c.training_sha256,"validation_sha256":c.validation_sha256,"genome":genome,"program_sha256":file_hash(&genome_path)?,"evaluator_sha256":evaluator_sha,"evaluation_bias":"adaptive_cut_heldout_teacher","family_partition_frozen":p.family_partition_frozen,"qualification":qualification,"independent_evaluation":false,"target_access":"cut heldout optimized repeatedly; original final test sealed","origin":origin,"initial_seed_slot":if initial_proposal{Some(execution_order[i])}else{None},"execution_order_protocol":execution_protocol,"seeds_sha256":seeds_sha256,"vocabulary_sha256":vocabulary_sha256,"proposal_before_projection":if raw_genome!=genome{Some(&raw_genome)}else{None},"evaluation_index":i,"rng_seed":rng_seed,"proposal_attempts":proposal_attempts,"rejected_proposals":rejected_proposals,"recovered_completed_artifact":recovered_metric,"runtime_seconds":measured_runtime,"completed_search_seconds":start.elapsed().as_secs_f64()+previous_seconds,"search_clock":SEARCH_CLOCK,"search_clock_accuracy":clock_accuracy,"failure":failure,"worker_diagnostic":diagnostic,"metrics":result,"target_scale":pp.target_scale});
        let mut observation = observation;
        if let Some((reference, task, candidate, _)) = native {
            observation["native_genome_task"] = serde_json::to_value(reference)?;
            observation["native_candidate"] = serde_json::to_value(&task.candidate)?;
            observation["comparison_arm"] = serde_json::to_value(candidate.arm)?;
            observation["continuation_parent"] = serde_json::to_value(&task.direct_parent)?;
        }
        observation["task"] = json!("regression");
        // The pre-dispatch lineage envelope, repeated so the ledger is self-contained.
        observation["lineage"] = lineage;
        // The estimate beside its observed outcome (full stage bounds stay in
        // the attempt): verdict and reasons only.
        observation["feasibility"] = json!({"protocol":feasibility["protocol"],
            "verdict":feasibility["verdict"],"reasons":feasibility["reasons"]});
        observation["clean_completion"] = json!(completion_reference);
        if let Some(reference) = &completion_reference {
            let completion = crate::label_admission::artifact(&completion_root, reference)?;
            if completion["fit_completion"].is_object() {
                let native = crate::label_admission::artifact(
                    &completion_root,
                    &completion["fit_completion"],
                )?;
                observation["numerical_runtime_seconds"] = native["elapsed_seconds"].clone();
                observation["fit_outcome"] = native["outcome"].clone();
            }
        }
        observation["evaluation_origin"] = result
            .as_ref()
            .map(|r| r["evaluation_origin"].clone())
            .unwrap_or(serde_json::Value::Null);
        observation["evaluator_protocol"] = json!(crate::label_admission::PROTOCOL);
        observation["resource_protocol"] = json!(crate::label_admission::RESOURCE);
        observation["preprocessing_sha256"] = json!(c.preprocessing_sha256);
        observation["metric_units"] = json!("cut_normalized_target");
        observation["evaluation_seed"] = json!(0);
        observation["feature_view_sha256"] = json!(crate::label_admission::raw_view(
            &c.training_sha256,
            &c.preprocessing_sha256
        ));
        observation["evaluation_identity_sha256"] = json!(hash(&serde_json::to_vec(
            &crate::label_admission::identity(&observation)
        )?));
        use std::io::Write;
        let mut line = serde_json::to_vec(&observation)?;
        line.push(b'\n');
        // A journaled observation is acknowledged work. Commit a strict winner
        // first so recovery cannot skip a journal row whose winner was lost.
        if let Some(root) = &champion_root {
            crate::champions::record(root, &observation, &output.join("observations.jsonl"))?;
        }
        journal.write_all(&line)?;
        journal.sync_data()?;
        experiment_failpoint("observation_journal", experiment.is_some());
        observations.push(observation);
        write_json(
            &output.join("progress.json"),
            &json!({"prepared_sha256":prepared_sha,"cut":cut,"seeds_sha256":seeds_sha256,"vocabulary_sha256":vocabulary_sha256,"execution_order_protocol":execution_protocol,"execution_order":execution_order,"wall_seconds":start.elapsed().as_secs_f64()+previous_seconds,"lifecycle_seconds":lifecycle_start.elapsed().as_secs_f64(),"search_clock":clock,"evaluations":observations.len(),"rng_seed":rng_seed,"proposal_attempts":proposal_attempts,"rejected_proposals":rejected_proposals,"best_validation_rmse":ranked.first().map(|x|x.0),"timestamp":crate::telemetry::now()}),
        )?;
        experiment_failpoint("observation_progress", experiment.is_some());
        // Original attempts, programs, outputs and completions are durable
        // admission/recovery evidence. Artifact eviction never permits refits.
