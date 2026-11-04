    let SearchOptions {
        evaluations,
        seconds,
        qualification,
        seeds,
        vocabulary,
    } = options;

    let lifecycle_start = Instant::now();
    let completion_root = artifact_root
        .canonicalize()
        .context("search artifact root unavailable")?;
    ensure!(
        completion_root.is_dir(),
        "search artifact root must be a directory"
    );
    let prepared_path = search_artifact_path(&completion_root, prepared)?;
    let output_path = search_artifact_path(&completion_root, output)?;
    let authority_path = std::env::var_os("JOPE_FIT_AUTHORITY")
        .map(|path| search_artifact_path(&completion_root, Path::new(&path)))
        .transpose()
        .context("fit authority artifacts must share the declared search root")?;
    // A neural search process may rank candidates, but CPU fits use the
    // explicitly bound authority evaluator. Authority still verifies its loaded
    // executable and mappings before releasing a start token.
    let executable = cpu_evaluator_binding(std::env::var_os("JOPE_CPU_EVALUATOR"))?;
    let prepared = prepared_path.as_path();
    let output = output_path.as_path();
    let experiment = std::env::var_os("JOPE_EXPERIMENT_DIR").map(std::path::PathBuf::from);
    if let Some(directory) = &experiment {
        verify_experiment_output(output, directory)?;
        ensure!(
            qualification && evaluations <= 200 && seconds <= 180,
            "isolated comparison requires qualification and 200-evaluation / 180-second limits"
        );
    }
    let p: Prepared = read_json(&prepared.join("prepared.json"))?;
    crate::task_contract::require_task(&p.task)?;
    let native_overlay = if let Some((_, task, candidate, _)) = native {
        if let Some(overlay) = &task.preparation_overlay {
            ensure!(!qualification, "overlay cannot admit qualification search");
            crate::preparation_overlay::verify(&completion_root, overlay, candidate, &p)?;
            true
        } else {
            false
        }
    } else {
        false
    };
    ensure!(
        p.schema == "parallel.prepared.v1"
            && (p.clean_evaluation || (qualification && p.qualification_only && p.clean_status == "raw_training_reconstructed_qualification_only"))
            && (p.family_partition_frozen || qualification || native_overlay),
        "reconstructed cut preparation is required; limited evaluation needs explicit qualification admission"
    );
    // W1-T04: a search is task-level development search. Its observations are
    // teacher/learning inputs, so a sealed meta_test family never reaches it,
    // including under qualification admission; nothing has been loaded yet.
    ensure!(
        p.meta_partition != "meta_test",
        "sealed meta_test families are not released to search, including qualification"
    );
    ensure!(
        p.meta_partition == "meta_train" || qualification,
        "bootstrap/search training observations require meta-training families"
    );
    // Signed input verification is lifecycle overhead, outside the numerical budget.
    let start = if native.is_some() {
        Instant::now()
    } else {
        lifecycle_start
    };
    let c = p
        .cuts
        .iter()
        .find(|c| c.index == cut)
        .context("unknown cut")?;
    let dir = prepared.join(format!("cut-{cut}"));
    let training_view = crate::prepare::load_view_in_root(&completion_root, prepared, cut, true)?;
    let validation_view =
        crate::prepare::load_view_in_root(&completion_root, prepared, cut, false)?;
    // W5-T04: the cut's declared shape for the pre-dispatch feasibility
    // diagnostic (rows fitted or applied, raw feature width).
    let feasibility_shape = feat_math::feasibility::Shape {
        rows: training_view.len().max(validation_view.len()) as u64,
        cols: training_view
            .first()
            .map_or(0, |row| row.len().saturating_sub(1)) as u64,
    };
    let memory = memory_view(&training_view, &validation_view)?;
    drop((training_view, validation_view));
    ensure!(
        evaluations > 0 && evaluations <= 1000 && seconds > 0 && seconds <= 600,
        "pilot budget exceeds 1000 evaluations or 600 seconds"
    );
    fs::create_dir_all(output)?;
    // W5-T05: reserve the output before reading any resume state, so two
    // searchers can never dispatch the same reservation or both publish.
    let _lease = search_lease(output)?;
    ensure!(
        !output.join("search.json").exists(),
        "search is already complete"
    );
    let resuming = output.join("progress.json").exists();
    // Route R8: parse exactly the verified preprocessing bytes.
    let pp: Preprocessing = serde_json::from_slice(
        &read_verified(
            &dir.join("preprocessing.json"),
            &c.preprocessing_sha256,
            "preprocessing",
        )
        .context("preprocessing changed")?,
    )?;
    // W5-T01: the seed population and vocabulary are read once each; the search parses and
    // pins exactly those bytes, and freezes them beside its request (below).
    let read_input =
        |path: &Path| fs::read(path).with_context(|| format!("open {}", path.display()));
    let vocabulary_bytes = vocabulary.map(read_input).transpose()?;
    let seeds_bytes = seeds.map(read_input).transpose()?;
    let parse = |path: Option<&Path>| format!("parse {}", path.unwrap_or(Path::new("")).display());
    let vocab: Option<crate::grammar::Vocab> = vocabulary_bytes
        .as_deref()
        .map(serde_json::from_slice)
        .transpose()
        .with_context(|| parse(vocabulary))?;
    let vocabulary_sha256 = vocabulary_bytes.as_deref().map(hash);
    let supplied: Option<SeedPopulation> = seeds_bytes
        .as_deref()
        .map(serde_json::from_slice)
        .transpose()
        .with_context(|| parse(seeds))?;
    let seeds_sha256 = seeds_bytes.as_deref().map(hash);
    let neural_seeded = supplied
        .as_ref()
        .is_some_and(|s| s.candidates.iter().any(|c| c.origin == "neural_policy"));
    if let Some(s) = &supplied {
        ensure!(
            s.schema == "parallel.seed_population.v1"
                && s.prepared_sha256 == file_hash(&prepared.join("prepared.json"))?
                && s.cut == cut
                && s.training_sha256 == c.training_sha256
                && s.candidates.len() == 80,
            "neural population identity mismatch"
        );
        for candidate in &s.candidates {
            GenomeIr::parse(&candidate.genome)?;
        }
    }
    let mut population = supplied
        .as_ref()
        .map(|s| {
            s.candidates
                .iter()
                .map(|c| c.genome.clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec!["(pipe X (ridge))".to_string()]);
    if let Some((_, task, candidate, _)) = native {
        // Direct means precisely one declared genotype. Continuation starts from
        // its original measured parent and spends its budget on new variation.
        population = match task.stage {
            crate::native_genome::Stage::Direct => vec![candidate.genome.clone()],
            crate::native_genome::Stage::Continuation => Vec::new(),
        };
    }
    let mut seen = BTreeSet::new();
    seen.extend(population.iter().cloned());
    ensure!(
        seen.len() == population.len(),
        "duplicate initial proposals"
    );
    // Original proposal and lineage source of every initial slot (W5-T01).
    // A critic weight estimates the original proposal, never its adaptation.
    let mut initial_sources: Vec<(String, serde_json::Value)> =
        if let Some((_, task, _, _)) = native {
            population
                .iter()
                .map(|genome| {
                    Ok((
                        genome.clone(),
                        json!({"kind":"native_candidate","slot":0,
                            "candidate":serde_json::to_value(&task.candidate)?}),
                    ))
                })
                .collect::<Result<_>>()?
        } else if let Some(s) = &supplied {
            s.candidates
                .iter()
                .enumerate()
                .map(|(slot, c)| {
                    let mut source = json!({"kind":"seed_population_slot","slot":slot,
                        "seeds_sha256":seeds_sha256,"origin":c.origin,
                        "rank_log_weight":c.policy_logprob,
                        "score_subject":"original_proposal"});
                    // W4-T07 join: the slot's proposal-ledger record, when the file names it.
                    if let Some(record) = &c.proposal_record_sha256 {
                        source["proposal_record_sha256"] = json!(record);
                    }
                    (c.genome.clone(), source)
                })
                .collect()
        } else {
            vec![(
                population[0].clone(),
                json!({"kind":"fixed_reference","slot":0}),
            )]
        };
    if native.is_none() {
        let first = initial_sources.len();
        let originals =
            fill_initial_population(&mut population, &mut seen, rng_seed, vocab.as_ref())?;
        initial_sources.extend(originals.into_iter().enumerate().map(
            |(offset, (raw, generator_seed))| {
                (
                    raw,
                    json!({"kind":"generic_bootstrap","slot":first + offset,
                        "generator":"feat_math.random_canonical_pipeline.depth6",
                        "generator_seed":generator_seed}),
                )
            },
        ));
    }
    ensure!(
        initial_sources.len() == population.len(),
        "initial proposal sources are incomplete"
    );
    let prepared_sha = file_hash(&prepared.join("prepared.json"))?;
    let evaluator_sha = file_hash(&executable)?;
    let champion_root = if p.storage.is_some() && experiment.is_none() {
        Some(completion_root.clone())
    } else {
        None
    };
    let request_path = output.join("search-request.json");
    let mut request = json!({"schema":"parallel.search_request.v1", "artifact_root":completion_root,
        "fit_authority":authority_path,
        "prepared":prepared.strip_prefix(&completion_root)?, "output":output.strip_prefix(&completion_root)?,
        "prepared_sha256":prepared_sha,"evaluator_sha256":evaluator_sha,"cut":cut,
        "training_sha256":c.training_sha256,"validation_sha256":c.validation_sha256,
        "preprocessing_sha256":c.preprocessing_sha256,"seeds_sha256":seeds_sha256,
        "vocabulary_sha256":vocabulary_sha256,"evaluations":evaluations,"seconds":seconds,
        "qualification":qualification,"rng_seed":rng_seed,"random_protocol":RANDOM_PROTOCOL,
        // W5-T02: the objective, evidence, tie and stop rules are frozen with
        // the request, so a resume under other rules is refused, not restarted.
        "selection_contract":selection_replay::CONTRACT,
        "proposal_cap_per_slot":selection_replay::PROPOSAL_CAP_PER_SLOT});
    if let Some((reference, task, _, _)) = native {
        request["native_genome_task"] = serde_json::to_value(reference)?;
        request["native_genome_stage"] = serde_json::to_value(task.stage)?;
    }
    // Check the recorded resume identity before comparing or recording the request.
    // Legacy outputs predate search-request.json: recording a refused request first
    // would make the refusal permanent and block the original identity's resume.
    let saved_progress: Option<serde_json::Value> = if resuming {
        let progress: serde_json::Value = read_json(&output.join("progress.json"))?;
        ensure!(
            progress["prepared_sha256"] == prepared_sha && progress["cut"] == cut,
            "resume identity mismatch"
        );
        ensure!(
            progress["seeds_sha256"] == json!(seeds_sha256),
            "resume neural seed identity mismatch"
        );
        ensure!(
            progress["rng_seed"].as_u64().unwrap_or(SEED) == rng_seed,
            "resume RNG seed mismatch"
        );
        Some(progress)
    } else {
        None
    };
    if request_path.exists() {
        ensure!(
            read_json::<serde_json::Value>(&request_path)? == request,
            "accepted search artifact root or request changed"
        );
    } else {
        write_json(&request_path, &request)?;
    }
    // W5-T01: the exact bytes the request pins, frozen beside it, so the lineage verifies
    // from the output alone (`lineage::verify_search_recorded`).
    for (name, bytes) in [
        (SEARCH_SEEDS, &seeds_bytes),
        (SEARCH_VOCABULARY, &vocabulary_bytes),
    ] {
        if let Some(bytes) = bytes {
            freeze_search_input(&output.join(SEARCH_INPUTS).join(name), bytes)?;
        }
    }
    let journal_path = output.join("observations.jsonl");
    let mut observations: Vec<serde_json::Value> = if journal_path.exists() {
        read_journal(&journal_path)?
    } else if resuming {
        read_json(&output.join("observations.json"))?
    } else {
        Vec::new()
    };
    ensure!(
        resuming || observations.is_empty(),
        "observation ledger has no resume identity"
    );
    // The journal is fsynced before progress.json, so it may contain the most
    // recent completed work after a crash. Never restore a smaller budget from
    // the older progress receipt. Old rows remain untouched: their per-row
    // runtimes cannot reconstruct exact cumulative search wall time.
    const SEARCH_CLOCK: &str = "parallel.gp_search_clock.v1";
    let mut last_completed_seconds = 0.0_f64;
    let mut legacy_clock_rows = 0usize;
    for observation in &observations {
        if observation["completed_search_seconds"].is_null() {
            legacy_clock_rows += 1;
            continue;
        }
        let completed = observation["completed_search_seconds"]
            .as_f64()
            .context("invalid completed search clock")?;
        ensure!(
            completed.is_finite() && completed >= last_completed_seconds,
            "completed search clock is negative or not monotonic"
        );
        ensure!(
            observation["search_clock"] == SEARCH_CLOCK,
            "unknown search clock definition"
        );
        last_completed_seconds = completed;
    }
    if !journal_path.exists() {
        let mut bytes = Vec::new();
        for observation in &observations {
            bytes.extend(serde_json::to_vec(observation)?);
            bytes.push(b'\n');
        }
        atomic_write(&journal_path, &bytes)?;
    }
    let mut journal = fs::OpenOptions::new().append(true).open(&journal_path)?;
    // (measured objective, executable program, ledger position; -1 = inherited).
    let mut ranked: Vec<(f64, String, i64)> = Vec::new();
    if let Some((_, task, candidate, Some(parent))) = native {
        ensure!(
            task.stage == crate::native_genome::Stage::Continuation,
            "direct native search cannot carry a parent score"
        );
        if parent["failure"].is_null() {
            let rmse = parent["metrics"]["rmse"]
                .as_f64()
                .context("original parent RMSE absent")?;
            ensure!(
                rmse.is_finite() && rmse >= 0.,
                "original parent score invalid"
            );
            ranked.push((rmse * pp.target_scale, candidate.genome.clone(), -1));
        }
    }
    let mut semantic = BTreeSet::new();
    let mut proposal_attempts = 0u64;
    let mut rejected_proposals = 0u64;
    // Existing journals retain their original slot order on recovery. New
    // searches protect known teachers from being starved by 64 slow neural fits.
    let teacher_first = saved_progress
        .as_ref()
        .is_none_or(|progress| progress["execution_order_protocol"] == "teacher_first_v1");
    let execution_order =
        initial_execution_order(supplied.as_ref(), population.len(), teacher_first);
    let execution_protocol = if teacher_first {
        "teacher_first_v1"
    } else {
        "legacy_seed_order_v1"
    };
    let previous_seconds = if let Some(progress) = &saved_progress {
        proposal_attempts = progress["proposal_attempts"]
            .as_u64()
            .unwrap_or(observations.len() as u64);
        rejected_proposals = progress["rejected_proposals"].as_u64().unwrap_or(0);
        let saved_seconds = progress["wall_seconds"]
            .as_f64()
            .context("resume elapsed time absent")?;
        ensure!(
            saved_seconds.is_finite() && saved_seconds >= 0.0,
            "invalid resume elapsed time"
        );
        saved_seconds.max(last_completed_seconds) + 30.0
    } else {
        0.0
    };
    if let Some(last) = observations.last() {
        proposal_attempts = last["proposal_attempts"]
            .as_u64()
            .unwrap_or(proposal_attempts);
        rejected_proposals = last["rejected_proposals"]
            .as_u64()
            .unwrap_or(rejected_proposals);
    }
    let clock_accuracy = if legacy_clock_rows > 0 {
        "conservative_resume_with_legacy_prefix"
    } else if resuming {
        "conservative_resume_offset"
    } else {
        "exact_active_elapsed"
    };
    let clock = json!({"schema":SEARCH_CLOCK,"definition":"monotonic active process elapsed from search entry through completed observation; excludes process downtime; resumed searches add a conservative 30-second allowance for unobserved interrupted work","accuracy":clock_accuracy,"resume_offset_seconds":previous_seconds,"resume_unobserved_allowance_seconds":if resuming {30.0} else {0.0},"legacy_observations_without_clock":legacy_clock_rows,"exact_wall_prefix_available":!resuming && legacy_clock_rows==0});
    let mut evaluated_programs = BTreeSet::new();
    if let Some((_, task, candidate, _)) = native {
        if task.stage == crate::native_genome::Stage::Continuation {
            evaluated_programs.insert(candidate.genome.clone());
        }
    }
    for (index, o) in observations.iter().enumerate() {
        ensure!(
            o["prepared_sha256"] == prepared_sha
                && o["cut"] == cut
                && o["evaluation_index"] == index
                && o["evaluator_sha256"] == evaluator_sha
                && o["training_sha256"] == c.training_sha256
                && o["validation_sha256"] == c.validation_sha256
                && o["seeds_sha256"] == json!(seeds_sha256)
                && o["vocabulary_sha256"] == json!(vocabulary_sha256)
                && o["rng_seed"].as_u64().unwrap_or(SEED) == rng_seed,
            "observation ledger identity mismatch"
        );
        evaluated_programs
            .insert(GenomeIr::parse(o["genome"].as_str().context("missing genome")?)?.canonical());
        if o["failure"].is_null() {
            let r = &o["metrics"];
            // Resumed rows obey the live admission rule and must still be the
            // exact output their original completion attests (W5-T06).
            let rmse = r["rmse"].as_f64().context("missing RMSE")? * pp.target_scale;
            ensure!(
                rmse.is_finite() && rmse >= 0.0,
                "journaled evaluator score invalid"
            );
            if !o["search_clock"].is_null() {
                journaled_measurement_attested(&completion_root, o)?;
            }
            if semantic.insert(
                r["semantic_sha256"]
                    .as_str()
                    .context("missing semantic hash")?
                    .to_string(),
            ) {
                ranked.push((
                    rmse,
                    o["genome"].as_str().context("missing genome")?.to_string(),
                    index as i64,
                ));
            }
        }
    }
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    ranked.truncate(80);
    if !resuming {
        write_json(&output.join("observations.json"), &observations)?;
    }
    // Persist the reserved resume allowance before doing more work, including
    // resumes that have already consumed their entire search budget.
    write_json(
        &output.join("progress.json"),
        &json!({"prepared_sha256":prepared_sha,"cut":cut,"seeds_sha256":seeds_sha256,"vocabulary_sha256":vocabulary_sha256,"execution_order_protocol":execution_protocol,"execution_order":execution_order,"wall_seconds":start.elapsed().as_secs_f64()+previous_seconds,"lifecycle_seconds":lifecycle_start.elapsed().as_secs_f64(),"search_clock":clock,"evaluations":observations.len(),"rng_seed":rng_seed,"proposal_attempts":proposal_attempts,"rejected_proposals":rejected_proposals}),
    )?;
    // W5-T05 terminal disposition, cross-checked by the independent replay.
    let mut stop_reason = "evaluation_budget";
