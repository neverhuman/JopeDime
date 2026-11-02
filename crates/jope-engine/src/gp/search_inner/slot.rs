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
