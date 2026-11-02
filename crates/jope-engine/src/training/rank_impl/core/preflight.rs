use super::*;

pub(super) fn prepare(input: RankInput<'_>) -> Result<RankPreflight> {
    let RankInput {
        root,
        corpus,
        run,
        launch,
        rank,
        world,
        micro,
        steps,
        checkpoint_steps,
        resume,
        initialize,
        global_batch,
        batch_qualification,
        replay_policy,
        native_stop,
        verified_native_continuation,
        declared_mode,
        assignment,
        admitted,
        partition_version,
        bounded_training,
    } = input;
    let (communicator, execution) = super::super::rank_plan::validate_rank_plan(
        assignment,
        admitted,
        rank,
        world,
        micro,
        steps,
        checkpoint_steps,
        launch,
        partition_version,
        global_batch,
        batch_qualification,
    )?;
    crate::neural::configure_runtime();
    let policy: Option<Value> = replay_policy.map(read_json).transpose()?;
    let policy_sha = replay_policy.map(file_hash).transpose()?;
    let replay = policy
        .as_ref()
        .map(|p| crate::checkpoint_recovery::continuation_replay(root, p))
        .transpose()?
        .flatten();
    let seed = match &policy {
        Some(p) => p["sampling_seed"]
            .as_u64()
            .context("replay sampling seed")?,
        None => SEED,
    };
    let bounded_inputs = bounded_training
        .map(|t| t.verified_input_bytes())
        .transpose()?;
    let index_bytes = match bounded_inputs {
        Some((index, _)) => index.to_vec(),
        None => fs::read(corpus.join("index.json"))?,
    };
    let corpus_hash = hash(&index_bytes);
    let verified_index: Value = serde_json::from_slice(&index_bytes)?;
    crate::corpus_projection::verify_admitted_context_closure(&verified_index)?;
    // W6 follow-up: the resident view is loaded from the verified bytes, so a
    // file substituted after verification cannot be loaded; T01's binding
    // stays as a second check.
    let resident = crate::teaching_store::load_resident_verified(root, corpus, &index_bytes)?;
    let (index, records) = resident.as_ref();
    inputs::bind_resident(&verified_index, index)?;
    ensure!(
        ["parallel.teaching_corpus.v1", crate::teaching_store::SCHEMA]
            .contains(&index.schema.as_str())
            && (index.qualification || (native_stop.is_some() && verified_native_continuation)),
        "training currently requires explicitly qualified real corpus"
    );
    // Parse the vocabulary from the bytes that were verified.
    let vocabulary: crate::grammar::Vocab = serde_json::from_slice(&inputs::read_verified(
        &corpus.join("vocabulary.json"),
        &index.vocabulary_sha256,
    )?)?;
    let frozen_inputs = inputs::freeze(index, &corpus_hash, &index.vocabulary_sha256, None)?;
    let frozen_inputs_sha = inputs::digest(&frozen_inputs)?;
    let mut families = BTreeMap::<String, Vec<usize>>::new();
    let mut validation = BTreeMap::<String, Vec<usize>>::new();
    let mut teacher_family = BTreeMap::<usize, String>::new();
    let mut train_records = 0;
    for (i, g) in index.groups.iter().enumerate() {
        if g.partition == "meta_train" {
            families.entry(g.family.clone()).or_default().push(i);
            train_records += g.records.len();
            if let Some(&teacher) = g.records.first() {
                ensure!(
                    teacher_family
                        .insert(teacher, g.family.clone())
                        .is_none_or(|family| family == g.family),
                    "comparison teacher belongs to two training families"
                );
            }
        } else if g.partition == "meta_validation" {
            validation.entry(g.family.clone()).or_default().push(i);
        }
    }
    ensure!(
        families.len() >= 2,
        "insufficient independent training families"
    );
    let mut fresh = BTreeMap::<String, Vec<usize>>::new();
    let mut fresh_by_teacher = BTreeMap::<usize, Vec<usize>>::new();
    let mut fresh_records = std::collections::BTreeSet::<usize>::new();
    let balanced = replay
        .as_ref()
        .is_some_and(|p| p["replay"] == "fresh_balanced");
    if let Some(p) = &replay {
        ensure!(
            p["schema"] == "parallel.replay_policy.v2"
                && p["eligible"] == true
                && p["corpus_sha256"] == corpus_hash
                && p["vocabulary_sha256"] == index.vocabulary_sha256
                && p["final_tests_opened"] == false,
            "replay policy identity or eligibility changed"
        );
        ensure!(
            ["broad", "fresh_balanced"]
                .iter()
                .any(|v| p["replay"] == *v)
                && ["reset", "carry"].iter().any(|v| p["optimizer"] == *v),
            "unknown replay/optimizer mode"
        );
        let mut seen = std::collections::BTreeSet::new();
        for value in p["fresh_groups"]
            .as_array()
            .context("fresh group indices")?
        {
            let i = value.as_u64().context("fresh group index")? as usize;
            ensure!(seen.insert(i), "duplicate fresh group");
            let g = index.groups.get(i).context("fresh group outside corpus")?;
            ensure!(
                g.partition == "meta_train" && g.records.len() >= 8,
                "fresh replay includes validation/test or insufficient comparators"
            );
            fresh.entry(g.family.clone()).or_default().push(i);
            let records = p["fresh_records"][i.to_string()]
                .as_array()
                .context("fresh program records absent")?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|id| id as usize)
                        .context("invalid fresh program record")
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                !records.is_empty()
                    && records.iter().all(|id| g.records.contains(id))
                    && records
                        .iter()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                        == records.len(),
                "fresh records must be distinct members of the selected training group"
            );
            fresh_records.extend(records.iter().copied());
            fresh_by_teacher.insert(g.records[0], records);
        }
    }
    let epoch_steps = train_records.div_ceil(global_batch).max(1);
    let fixed_panel = if validation.is_empty() {
        None
    } else {
        Some(match execution.plan() {
            None => ValidationPanel::V1(validation_panel(&index, world, SEED)?),
            Some(_) => {
                // W7-T06: one fixed global panel split by the plan, with its
                // global denominators from labels alone.
                let plan = partition::Plan::new(world, partition::VALIDATION_PANEL_ROWS, micro)?;
                let per_rank = validation_panel_partitioned(&index, &plan, SEED)?;
                let global_ids = per_rank
                    .iter()
                    .flatten()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>();
                let labels = global_ids
                    .iter()
                    .map(|&id| objective::RowLabels::from_record(id, &records.get(id)))
                    .collect::<Vec<_>>();
                let denominators = partition::denominators(&labels)?;
                ValidationPanel::V2 {
                    per_rank,
                    global_ids,
                    denominators,
                }
            }
        })
    };
    let panel_sha256 = fixed_panel
        .as_ref()
        .map(|panel| {
            match panel {
                ValidationPanel::V1(ids) => serde_json::to_vec(ids),
                ValidationPanel::V2 { global_ids, .. } => serde_json::to_vec(global_ids),
            }
            .map(|bytes| hash(&bytes))
        })
        .transpose()?;
    let config = Config::regression(vocabulary.tokens.len() as i64);
    let recovered_continuation = policy
        .as_ref()
        .filter(|p| p["schema"] == crate::checkpoint_recovery::CONTINUATION_SCHEMA)
        .map(|p| {
            crate::checkpoint_recovery::verify_continuation_partition(
                root,
                p,
                &corpus_hash,
                &index.vocabulary_sha256,
                execution.record(),
                steps,
            )
        })
        .transpose()?;
    ensure!(
        !(resume && initialize.is_some()),
        "resume and initialize are mutually exclusive"
    );
    // W6-T06: each attempt is exactly one of the four continuation modes.
    let mode = continuation::classify(resume, initialize.is_some(), policy.as_ref())?;
    // W6 follow-up: an explicit lineage-mode declaration must match the
    // request (or, on resume, the recorded lineage); none is recorded implicit.
    let declaration = continuation::declared_lineage(
        declared_mode,
        resume,
        initialize.is_some(),
        policy.as_ref(),
        if resume {
            Some(read_json::<Value>(&run.join("initialization.json"))?)
        } else {
            None
        }
        .as_ref(),
    )?;
    // The objective is the one the partition's reductions define.
    let objective_sha = execution.objective_sha256();
    let mut continuity = Value::Null;
    let resume_selection: Option<Value> = if resume {
        Some(read_json(&launch.join("resume-input.json"))?)
    } else {
        None
    };
    let selected_checkpoint = if let Some(latest) = &resume_selection {
        Some(resolve(
            root,
            Path::new(latest["checkpoint"].as_str().context("checkpoint path")?),
        )?)
    } else {
        initialize.map(|path| resolve(root, path)).transpose()?
    };
    // Read once: ancestry, the compatibility checks and the digests that bind
    // every later checkpoint load all come from these retained state bytes.
    let selected_state_bytes = selected_checkpoint
        .as_ref()
        .map(|cp| -> Result<Vec<u8>> {
            match bounded_inputs {
                Some((_, Some(state))) => Ok(state.to_vec()),
                Some((_, None)) => anyhow::bail!("bounded parent state missing"),
                None => Ok(fs::read(cp.join("state.json"))?),
            }
        })
        .transpose()?;
    let selected_state_sha = selected_state_bytes.as_deref().map(hash);
    if let Some(latest) = &resume_selection {
        ensure!(
            selected_state_sha.as_deref() == latest["state_sha256"].as_str(),
            "checkpoint selection changed"
        );
    }
    let selected_state: Option<Value> = selected_state_bytes
        .as_deref()
        .map(serde_json::from_slice)
        .transpose()?;
    // All compatibility and serialized-content checks precede CUDA/NCCL. The
    // configuration and optimizer metadata they read come through W2-T04's
    // read-once `read_bound`, bound by the retained state: the checked bytes
    // are the consumed bytes. Model and optimizer tensors are digest-checked
    // here and, after CUDA, loaded only through `load_bound_model` and
    // `AdamW::load_bound`, which verify exactly the bytes they load.
    if let (Some(cp), Some(state)) = (&selected_checkpoint, &selected_state) {
        let bound = |name: &str, field: &str| {
            crate::manifest::read_bound(
                &cp.join(name),
                &state[field],
                &format!("checkpoint {name} changed before CUDA"),
            )
        };
        let parent_config = bound("config.json", "config_sha256")?;
        if resume {
            // W6-T05: exact resume loads only a complete, internally
            // consistent checkpoint of this corpus and topology.
            let resume_input = resume_selection.as_ref();
            let verified_checkpoint = checkpoint_commit::verify_dir(
                root,
                cp,
                state["step"].as_u64().context("checkpoint step")?,
                &checkpoint_commit::Expectation {
                    corpus_sha256: Some(&corpus_hash),
                    vocabulary_sha256: Some(&index.vocabulary_sha256),
                    world: Some(world),
                    global_batch: Some(global_batch),
                    require_complete: false,
                    lineage_sha256: resume_input
                        .and_then(|r| r["lineage_sha256"].as_str())
                        .or_else(|| state["lineage"]["sha256"].as_str()),
                    intent_sha256: resume_input
                        .and_then(|r| r["intent_sha256"].as_str())
                        .or_else(|| state["intent"]["sha256"].as_str()),
                    attempt_id: resume_input
                        .and_then(|r| r["attempt_id"].as_str())
                        .or_else(|| state["attempt_id"].as_str()),
                    window_id: resume_input
                        .and_then(|r| r["window_id"].as_str())
                        .or_else(|| state["window_id"].as_str()),
                },
            )?;
            ensure!(
                Some(verified_checkpoint.state_sha256.as_str()) == selected_state_sha.as_deref(),
                "verified resume checkpoint differs from the retained parent state"
            );
            ensure!(
                state
                    .get("training_inputs_sha256")
                    .is_none_or(|frozen| *frozen == frozen_inputs_sha.as_str()),
                "exact resume requires the same frozen training inputs"
            );
        }
        // Full objective identity, partition, vocabulary and architecture
        // against the requested mode; undeclared changes are refused.
        continuity = continuation::check_parent_partition(
            mode,
            state,
            &serde_json::from_slice::<Value>(&parent_config)?,
            &continuation::Current {
                corpus_sha256: &corpus_hash,
                vocabulary_sha256: &index.vocabulary_sha256,
                world,
                microbatch: micro,
                global_batch,
                seed,
                sampler_id: active_sampler_id(replay.as_ref()),
                config: &serde_json::to_value(&config)?,
                objective_sha256: &objective_sha,
            },
            execution.record(),
            policy.as_ref(),
        )?;
        ensure!(
            state["autocast_cache_lifetime"] == "scoped_v2"
                && state["vocabulary_sha256"] == index.vocabulary_sha256,
            "checkpoint autocast or vocabulary mismatch before CUDA"
        );
        let previous: Config = serde_json::from_slice(&parent_config)?;
        ensure!(
            serde_json::to_value(previous)? == serde_json::to_value(&config)?,
            "checkpoint architecture mismatch before CUDA"
        );
        ensure!(
            file_hash(&cp.join("model.ot"))? == state["model_sha256"],
            "checkpoint model.ot changed before CUDA"
        );
        if resume || policy.as_ref().is_some_and(|p| p["optimizer"] == "carry") {
            ensure!(
                file_hash(&cp.join("optimizer.ot"))? == state["optimizer_sha256"],
                "checkpoint optimizer.ot changed before CUDA"
            );
            let optimizer: Value =
                serde_json::from_slice(&bound("optimizer.json", "optimizer_metadata_sha256")?)?;
            let expected = if resume {
                json!({"execution_partition":execution.record(),
                    "sampler_id":active_sampler_id(replay.as_ref()),"corpus_sha256":corpus_hash,
                    "world":world,"global_batch":global_batch,"seed":seed,
                    "ranking_backward":crate::neural::RANKING_BACKWARD_ID,
                    "runtime_objective":"huber_log1p_seconds_v2"})
            } else {
                json!({"ranking_backward":crate::neural::RANKING_BACKWARD_ID,"runtime_objective":"huber_log1p_seconds_v2"})
            };
            crate::task_contract::verify_resume_metadata(state, &expected, &optimizer)?;
        }
        if resume {
            let lineage: Value = read_json(&run.join("initialization.json"))?;
            ensure!(
                state["initialization"] == lineage,
                "resume initialization lineage changed before CUDA"
            );
            if policy.is_some() {
                ensure!(
                    lineage["replay_policy_sha256"] == json!(policy_sha)
                        && lineage["replay_policy"] == json!(policy),
                    "resume replay policy changed before CUDA"
                );
            } else {
                ensure!(
                    lineage.get("replay_policy_sha256").is_none(),
                    "resume original replay policy absent before CUDA"
                );
            }
            let target = steps
                .checked_add(lineage["optimizer_parent_step"].as_u64().unwrap_or(0) as usize)
                .context("training step overflow")?;
            ensure!(
                state["step"].as_u64().context("checkpoint step")? <= target as u64,
                "checkpoint exceeds requested training window before CUDA"
            );
        } else if let Some(p) = &policy {
            ensure!(
                p["parent_checkpoint_sha256"]
                    == selected_state_sha
                        .as_deref()
                        .context("parent state digest")?,
                "replay initialization changed before CUDA"
            );
        }
    }
    let exposure_baseline = if resume {
        crate::training_exposure::checkpoint_baseline_rows(
            selected_state
                .as_ref()
                .context("resume exposure state absent")?,
            rank,
            &execution.rows_per_rank(),
        )?
    } else {
        Some(crate::training_exposure::Counts::default())
    };
    let (family_baseline, family_baseline_known) = if resume {
        crate::training_exposure::checkpoint_family_baseline(
            selected_state
                .as_ref()
                .context("resume exposure state absent")?,
            rank,
            exposure_baseline.as_ref(),
        )?
    } else {
        (crate::training_exposure::FamilyGroups::new(), true)
    };
    Ok(RankPreflight {
        communicator,
        execution,
        policy,
        policy_sha,
        replay,
        seed,
        corpus_hash,
        resident,
        vocabulary,
        frozen_inputs,
        frozen_inputs_sha,
        families,
        validation,
        teacher_family,
        fresh,
        fresh_by_teacher,
        fresh_records,
        balanced,
        epoch_steps,
        fixed_panel,
        panel_sha256,
        config,
        recovered_continuation,
        mode,
        declaration,
        objective_sha,
        continuity,
        selected_checkpoint,
        selected_state_sha,
        selected_state,
        exposure_baseline,
        family_baseline,
        family_baseline_known,
    })
}
