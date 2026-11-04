fn search_inner(
    artifact_root: &Path,
    prepared: &Path,
    cut: usize,
    output: &Path,
    options: SearchOptions<'_>,
    rng_seed: u64,
    native: Option<(
        &crate::native_graph::Artifact,
        &crate::native_genome::GenomeTask,
        &crate::native_genome::Candidate,
        Option<&serde_json::Value>,
    )>,
) -> Result<()> {
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
    for i in observations.len()..evaluations {
        if start.elapsed().as_secs_f64() + previous_seconds >= seconds as f64 {
            stop_reason = "search_deadline";
            break;
        }
        // Projection can collapse distinct native mutations onto one legal
        // program. Reject that duplication before paying for a GP fit.
        let attempt_path = output.join(format!("candidate-{i}.attempt.json"));
        let saved_attempt: Option<serde_json::Value> = if attempt_path.exists() {
            let a: serde_json::Value = read_json(&attempt_path)?;
            ensure!(
                a["prepared_sha256"] == prepared_sha
                    && a["evaluator_sha256"] == evaluator_sha
                    && a["cut"] == cut
                    && a["evaluation_index"] == i
                    && a["seeds_sha256"] == json!(seeds_sha256)
                    && a["vocabulary_sha256"] == json!(vocabulary_sha256)
                    && a["evaluations"] == evaluations
                    && a["seconds"] == seconds
                    && a["rng_seed"] == rng_seed
                    && a["random_protocol"] == RANDOM_PROTOCOL,
                "incomplete candidate identity mismatch"
            );
            Some(a)
        } else {
            None
        };
        let slot = i as u64;
        let (raw_genome, genome, initial_proposal, lineage, slot_draw) = if let Some(a) =
            &saved_attempt
        {
            proposal_attempts = a["proposal_attempts"]
                .as_u64()
                .context("missing proposal counter")?;
            rejected_proposals = a["rejected_proposals"]
                .as_u64()
                .context("missing rejection counter")?;
            let slot_draw = a["slot_draw"].as_u64().context("missing slot draw")?;
            ensure!(
                a["random_streams"]
                    == json!({
                "operator":random_word(rng_seed,slot,slot_draw,"operator"),
                "left_parent":random_word(rng_seed,slot,slot_draw,"left_parent"),
                "right_parent":random_word(rng_seed,slot,slot_draw,"right_parent"),
                "variation":random_word(rng_seed,slot,slot_draw,"variation")}),
                "original candidate random streams changed"
            );
            let genome = a["genome"]
                .as_str()
                .context("missing candidate genome")?
                .to_string();
            ensure!(
                evaluated_programs.insert(GenomeIr::parse(&genome)?.canonical()),
                "pending candidate already committed"
            );
            (
                a["raw_genome"]
                    .as_str()
                    .context("missing raw genome")?
                    .to_string(),
                genome,
                a["initial_proposal"]
                    .as_bool()
                    .context("missing candidate origin")?,
                // Attempts recorded before the lineage envelope keep null.
                a.get("lineage").cloned().unwrap_or(serde_json::Value::Null),
                slot_draw,
            )
        } else {
            let mut selected = None;
            let mut out_of_time = false;
            for attempt in 0..selection_replay::PROPOSAL_CAP_PER_SLOT {
                if start.elapsed().as_secs_f64() + previous_seconds >= seconds as f64 {
                    out_of_time = true;
                    break;
                }
                proposal_attempts += 1;
                let initial = attempt == 0 && i < population.len();
                // Streams are keyed by (slot, draw within the slot), never by
                // the search-wide counter (W5-T02).
                let proposal_seed = random_word(rng_seed, slot, attempt, "variation");
                let operator = random_word(rng_seed, slot, attempt, "operator") % 16;
                // An explicit variation event: operator, parents and stream words.
                let variation = |kind: &str, parents: Vec<serde_json::Value>| {
                    json!({"kind":"variation","operator":kind,"parents":parents,
                        "slot":slot,"slot_draw":attempt,
                        "proposal_attempt":proposal_attempts,"variation_seed":proposal_seed,
                        "operator_word":operator})
                };
                let parent = |entry: &(f64, String, i64)| json!({"position":entry.2,"program_sha256":hash(entry.1.as_bytes())});
                let (raw, source) = if initial {
                    // The original proposal, never its already adapted form.
                    initial_sources[execution_order[i]].clone()
                } else {
                    let keep = ranked.len().min(10);
                    if keep == 0 && native.is_some() {
                        let (_, _, candidate, _) =
                            native.context("native continuation parent absent")?;
                        (
                            feat_math::evolve::mutate_canonical_pipeline(
                                &candidate.genome,
                                proposal_seed,
                                6,
                            )
                            .map_err(anyhow::Error::msg)?,
                            variation(
                                "continuation_mutate",
                                vec![json!({"position":null,"source":"native_candidate",
                                    "program_sha256":hash(candidate.genome.as_bytes())})],
                            ),
                        )
                    } else if keep == 0 || operator < 2 {
                        (
                            feat_math::evolve::random_canonical_pipeline(proposal_seed, 6),
                            variation("fresh_random", Vec::new()),
                        )
                    } else {
                        let ranked_genomes: Vec<String> = ranked
                            .iter()
                            .take(keep)
                            .map(|(_, genome, _)| genome.clone())
                            .collect();
                        let left = &ranked[policy_weighted_parent(
                            keep,
                            random_word(rng_seed, slot, attempt, "left_parent"),
                            supplied.as_ref(),
                            Some(ranked_genomes.as_slice()),
                        )];
                        if operator < 9 {
                            (
                                feat_math::evolve::mutate_canonical_pipeline(
                                    &left.1,
                                    proposal_seed,
                                    6,
                                )
                                .map_err(anyhow::Error::msg)?,
                                variation("mutate", vec![parent(left)]),
                            )
                        } else {
                            let right =
                                &ranked[(random_word(rng_seed, slot, attempt, "right_parent")
                                    % keep as u64)
                                    as usize];
                            (
                                feat_math::evolve::crossover_canonical_pipelines(
                                    &left.1,
                                    &right.1,
                                    proposal_seed,
                                    6,
                                )
                                .map_err(anyhow::Error::msg)?,
                                variation("crossover", vec![parent(left), parent(right)]),
                            )
                        }
                    }
                };
                let projected = if let Some(v) = &vocab {
                    v.project_proposal(&raw)
                } else {
                    Ok(raw.clone())
                };
                if let Ok(genome) = projected {
                    let canonical = GenomeIr::parse(&genome)?.canonical();
                    if evaluated_programs.insert(canonical) {
                        // Original → one explicit adaptation → executable recipe.
                        let lineage = json!({"schema":"parallel.gp_candidate_lineage.v1",
                            "original":{"program":raw,"program_sha256":hash(raw.as_bytes())},
                            "source":source,
                            "adaptation":if vocab.is_some() {
                                json!({"kind":"vocabulary_projection","applications":1,
                                    "vocabulary_sha256":vocabulary_sha256,
                                    "input_sha256":hash(raw.as_bytes()),
                                    "output_sha256":hash(genome.as_bytes()),
                                    "changed":raw != genome})
                            } else {
                                json!({"kind":"none","applications":0})
                            },
                            "executable":{"program":genome,"program_sha256":hash(genome.as_bytes())}});
                        selected = Some((raw, genome, initial, lineage, attempt));
                        break;
                    }
                }
                rejected_proposals += 1;
            }
            let Some(selected) = selected else {
                stop_reason = if out_of_time {
                    "search_deadline"
                } else {
                    "proposal_space_exhausted"
                };
                break;
            };
            selected
        };
        let genome_path = output.join(format!("candidate-{i}.dsl"));
        let mut candidate_identity = json!({"program_sha256":hash(genome.as_bytes()),
            "family":p.family,"partition":p.meta_partition,"prepared_sha256":prepared_sha,"cut":cut,
            "training_sha256":c.training_sha256,"validation_sha256":c.validation_sha256,
            "evaluator_sha256":evaluator_sha,"evaluator_protocol":crate::label_admission::PROTOCOL,
            "preprocessing_sha256":c.preprocessing_sha256,"target_scale":pp.target_scale,
            "metric_units":"cut_normalized_target","vocabulary_sha256":vocabulary_sha256,
            "feature_view_sha256":crate::label_admission::raw_view(&c.training_sha256,&c.preprocessing_sha256),
            "evaluation_seed":0,"resource_protocol":crate::label_admission::RESOURCE});
        candidate_identity["evaluation_identity_sha256"] = json!(hash(&serde_json::to_vec(
            &crate::label_admission::identity(&candidate_identity)
        )?));
        // W5-T04 pre-dispatch diagnostic under identical rules for every arm.
        // It is an estimate recorded beside the observed outcome; it does not
        // change dispatch (a refusal policy needs its own protocol identity).
        let feasibility = if let Some(a) = &saved_attempt {
            a.get("feasibility")
                .cloned()
                .unwrap_or(serde_json::Value::Null)
        } else {
            serde_json::to_value(feat_math::feasibility::classify(
                &GenomeIr::parse(&genome)?,
                feasibility_shape,
                &feat_math::feasibility::Limits::default(),
            ))?
        };
        if saved_attempt.is_some() {
            ensure!(
                fs::read(&genome_path)? == genome.as_bytes(),
                "pending candidate program changed"
            );
        } else {
            atomic_write(&genome_path, genome.as_bytes())?;
            write_json(
                &attempt_path,
                &json!({"prepared_sha256":prepared_sha,"evaluator_sha256":evaluator_sha,
                "cut":cut,"evaluation_index":i,"seeds_sha256":seeds_sha256,"vocabulary_sha256":vocabulary_sha256,
                "evaluations":evaluations,"seconds":seconds,"raw_genome":raw_genome,"genome":genome,
                "program_sha256":candidate_identity["program_sha256"],"evaluation_identity_sha256":candidate_identity["evaluation_identity_sha256"],
                "initial_proposal":initial_proposal,"rng_seed":rng_seed,"random_protocol":RANDOM_PROTOCOL,
                "slot_draw":slot_draw,
                "random_streams":{"operator":random_word(rng_seed,slot,slot_draw,"operator"),
                    "left_parent":random_word(rng_seed,slot,slot_draw,"left_parent"),
                    "right_parent":random_word(rng_seed,slot,slot_draw,"right_parent"),
                    "variation":random_word(rng_seed,slot,slot_draw,"variation")},
                "proposal_attempts":proposal_attempts,"rejected_proposals":rejected_proposals,
                "lineage":lineage,"feasibility":feasibility}),
            )?;
        }
        let result_path = output.join(format!("candidate-{i}.json"));
        experiment_failpoint("candidate_intent", experiment.is_some());
        let completion_path = output.join(format!("candidate-{i}.completion.json"));
        let saved_completion: Option<serde_json::Value> =
            if saved_attempt.is_some() && completion_path.exists() {
                Some(read_json(&completion_path)?)
            } else {
                None
            };
        let recovered_metric = saved_completion.is_some();
        let begun = Instant::now();
        let deadline_seconds =
            (seconds as f64 - start.elapsed().as_secs_f64() - previous_seconds).clamp(0.001, 30.);
        let deadline_monotonic_ns =
            crate::fit_boundary::deadline_after(Duration::from_secs_f64(deadline_seconds))?;
        let mut failure = None;
        let mut exit_code = None;
        let mut exit_signal = None;
        let mut exit_observed = false;
        let mut deadline_exceeded = false;
        let mut process_group = serde_json::Value::Null;
        if let Some(completion) = &saved_completion {
            failure = completion["failure"].as_str().map(String::from);
        } else if saved_attempt.is_some() {
            // An intent/result is not an observed successful exit. It might
            // already have fitted; never dispatch this candidate again.
            failure = Some("unknown_original_execution_without_completion".into());
        } else {
            let log = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(output.join(format!("candidate-{i}.log")))?;
            let mut command = evaluator_command(&executable);
            use std::os::fd::AsRawFd;
            command
                .args(["evaluate-memory", "--fd"])
                .arg(memory.as_raw_fd().to_string())
                .arg("--prepared")
                .arg(prepared)
                .arg("--cut")
                .arg(cut.to_string())
                .arg("--genome")
                .arg(&genome_path)
                .arg("--output")
                .arg(&result_path)
                .env(
                    "JOPE_FIT_DEADLINE_MONOTONIC_NS",
                    deadline_monotonic_ns.to_string(),
                )
                .env(
                    "OMP_NUM_THREADS",
                    std::env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "1".into()),
                )
                .env("OPENBLAS_NUM_THREADS", "1")
                .env("MKL_NUM_THREADS", "1")
                .env(
                    "RAYON_NUM_THREADS",
                    std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "1".into()),
                )
                .stdout(Stdio::from(log.try_clone()?))
                .stderr(Stdio::from(log));
            use std::os::unix::process::CommandExt;
            let controller_pid = std::process::id();
            // W5-T05: the evaluator's guard kills its whole process group when this
            // controller dies, even by SIGKILL (`guard_evaluator_group`).
            command.env(EVALUATOR_GROUP_GUARD, controller_pid.to_string());
            // SAFETY: the hook runs between fork and exec and makes only async-signal-safe calls
            // (setpgid, prctl, getppid, setrlimit); its errors are errno codes, so nothing
            // allocates in the child.
            unsafe {
                command.pre_exec(move || {
                    // SIGTERM, not SIGKILL: the guard turns it into SIGKILL for the group;
                    // an evaluator without the guard still ends by its default action.
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    // The controller exited: refuse with an errno, which does not allocate
                    // (an io::Error carrying a message would box it in the forked child).
                    if libc::getppid() as u32 != controller_pid {
                        return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
                    }
                    let limit = libc::rlimit {
                        rlim_cur: 8 * 1024 * 1024 * 1024,
                        rlim_max: 8 * 1024 * 1024 * 1024,
                    };
                    // W5-T05: the evaluator leads its own process group so that
                    // every process it starts is drained with it.
                    if libc::setrlimit(libc::RLIMIT_AS, &limit) != 0 || libc::setpgid(0, 0) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            let mut child = command.spawn()?;
            let group = i32::try_from(child.id()).context("evaluator pid out of range")?;
            loop {
                let exited = evaluator_exited(group)?;
                let expired = !exited && begun.elapsed().as_secs_f64() >= deadline_seconds;
                if exited || expired {
                    // The leader is still unreaped (a zombie when it exited), so
                    // its group id cannot be reused while the group is drained.
                    process_group = drain_process_group(group);
                    let status = child.wait()?;
                    use std::os::unix::process::ExitStatusExt;
                    exit_observed = true;
                    exit_code = status.code();
                    exit_signal = status.signal();
                    deadline_exceeded = expired || begun.elapsed().as_secs_f64() > deadline_seconds;
                    if deadline_exceeded {
                        failure = Some("candidate_or_view_timeout".into());
                    } else if !status.success() {
                        failure = Some(format!("worker_exit:{status}"));
                    }
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        let result: Option<serde_json::Value> =
            if take_clean_result(&mut failure, result_path.exists()) {
                Some(read_json(&result_path)?)
            } else {
                None
            };
        let measured_runtime = if let Some(completion) = &saved_completion {
            completion["runtime_seconds"].as_f64()
        } else if saved_attempt.is_some() {
            None
        } else {
            Some(begun.elapsed().as_secs_f64())
        };
        let completion_reference = if completion_path.exists() {
            Some(relative_artifact(&completion_root, &completion_path)?)
        } else if saved_attempt.is_none() {
            let mut completion = candidate_identity.clone();
            completion["schema"] = json!(crate::label_admission::COMPLETION_SCHEMA);
            completion["attempt"] = relative_artifact(&completion_root, &attempt_path)?;
            completion["exit_observed"] = json!(exit_observed);
            completion["exit_code"] = json!(exit_code);
            completion["signal"] = json!(exit_signal);
            completion["deadline_seconds"] = json!(deadline_seconds);
            completion["deadline_exceeded"] = json!(deadline_exceeded);
            completion["runtime_seconds"] = json!(measured_runtime);
            completion["failure"] = json!(failure);
            completion["process_group"] = process_group;
            let log_path = output.join(format!("candidate-{i}.log"));
            if log_path.exists() {
                fs::File::open(&log_path)?.sync_all()?;
                completion["worker_log"] = relative_artifact(&completion_root, &log_path)?;
            }
            completion["outcome"] = json!(if failure.is_none() {
                "success"
            } else if deadline_exceeded {
                "timeout"
            } else {
                "infrastructure_failure"
            });
            completion["output"] = if result.is_some() {
                relative_artifact(&completion_root, &result_path)?
            } else {
                serde_json::Value::Null
            };
            if let Some(r) = &result {
                completion["fit_completion"] =
                    relative_existing_reference(&completion_root, &r["fit_completion"])?;
                completion["model"] =
                    relative_existing_reference(&completion_root, &r["fitted_model"])?;
            } else {
                let terminal_path = result_path.with_extension("fit-outcome.json");
                if terminal_path.exists() {
                    let terminal: serde_json::Value = read_json(&terminal_path)?;
                    completion["fit_completion"] =
                        relative_existing_reference(&completion_root, &terminal["fit_completion"])?;
                    if !deadline_exceeded && terminal["outcome"] == "deterministic_failure" {
                        completion["outcome"] = json!("fit_failure");
                    }
                    completion["original_fit_outcome"] = terminal["outcome"].clone();
                }
            }
            write_json(&completion_path, &completion)?;
            experiment_failpoint("candidate_completion", experiment.is_some());
            Some(relative_artifact(&completion_root, &completion_path)?)
        } else {
            None
        };
        if recovered_metric {
            let mut recovered = candidate_identity.clone();
            recovered["clean_completion"] = json!(completion_reference);
            recovered["failure"] = json!(failure);
            recovered["metrics"] = json!(result);
            recovered["runtime_seconds"] = json!(measured_runtime);
            let completion = saved_completion
                .as_ref()
                .context("original completion absent")?;
            ensure!(
                completion["schema"] == crate::label_admission::COMPLETION_SCHEMA
                    && completion["attempt"] == relative_artifact(&completion_root, &attempt_path)?,
                "recovered completion attempt changed"
            );
            for key in [
                "evaluation_identity_sha256",
                "program_sha256",
                "evaluator_sha256",
            ] {
                ensure!(
                    completion[key] == candidate_identity[key],
                    "recovered completion identity changed: {key}"
                );
            }
            if completion["fit_completion"].is_object() {
                let native = crate::label_admission::artifact(
                    &completion_root,
                    &completion["fit_completion"],
                )?;
                recovered["numerical_runtime_seconds"] = native["elapsed_seconds"].clone();
                crate::label_admission::validate_completion(&completion_root, &recovered)?;
            } else {
                // A denied dispatch has no numerical receipt and cannot pass
                // quality admission. It can still recover its original failure.
                ensure!(
                    result.is_none()
                        && failure.as_ref().is_some_and(|s| !s.is_empty())
                        && (completion["outcome"] == "infrastructure_failure"
                            || completion["outcome"] == "timeout")
                        && completion["exit_observed"] == true
                        && (completion["exit_code"]
                            .as_i64()
                            .is_some_and(|code| code != 0)
                            || completion["signal"]
                                .as_i64()
                                .is_some_and(|signal| signal > 0)
                            || completion["deadline_exceeded"] == true),
                    "recovered failure lacks original non-success process evidence"
                );
                crate::label_admission::artifact_bytes(
                    &completion_root,
                    &completion["worker_log"],
                )?;
            }
        }
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
    }
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
}
