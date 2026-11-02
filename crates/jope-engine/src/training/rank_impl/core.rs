use super::*;

#[macro_use]
#[path = "training_loop.rs"]
mod training_loop;

pub(super) fn rank_impl(
    root: &Path,
    corpus: &Path,
    run: &Path,
    launch: &Path,
    rank: usize,
    world: usize,
    micro: usize,
    steps: usize,
    checkpoint_steps: usize,
    resume: bool,
    initialize: Option<&Path>,
    global_batch: usize,
    batch_qualification: bool,
    replay_policy: Option<&Path>,
    native_stop: Option<&std::sync::atomic::AtomicBool>,
    verified_native_continuation: bool,
    declared_mode: Option<continuation::Mode>,
    assignment: Option<&topology::Assignment>,
    admitted: Option<&crate::native_loop::bounded::launch::AdmittedBootstrap>,
    partition_version: partition::Version,
    bounded_training: Option<&crate::native_loop::bounded::training_run::BoundedTraining>,
) -> Result<()> {
    let (communicator, execution) = super::rank_plan::validate_rank_plan(
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
    // The communicator rank is not the CUDA ordinal: bind this worker to its
    // owned device and check visibility before the communicator exists.
    let binding = topology::resolve(assignment, rank, world, crate::neural::local_devices)?;
    let comm = match &communicator {
        CommunicatorSource::LaunchFile(path) => Nccl::init(&binding, path)?,
        CommunicatorSource::Admitted => Nccl::init_admitted(
            &binding,
            admitted.context("admitted communicator bootstrap")?,
        )?,
    };
    let device = Device::Cuda(binding.device_index());
    tch::manual_seed(seed as i64);
    let mut vs = nn::VarStore::new(device);
    let model = Policy::new(vs.root(), config.clone())?;
    let mut optimizer = AdamW::new(1e-4, 0.01);
    ensure!(
        !(resume && initialize.is_some()),
        "resume and initialize are mutually exclusive"
    );
    let initialization_path = run.join("initialization.json");
    let mut initialization = json!({"kind":"fresh","seed":seed,"initialization":"pytorch_linear_mha_embedding_v2","autocast_cache_lifetime":"scoped_v2"});
    if initialize.is_some() {
        let checkpoint = selected_checkpoint
            .as_ref()
            .context("initialization selection absent")?;
        let state = selected_state
            .as_ref()
            .context("initialization state absent")?;
        ensure!(
            state["autocast_cache_lifetime"] == "scoped_v2",
            "initialization checkpoint predates corrected autocast cache lifetime"
        );
        ensure!(
            state["vocabulary_sha256"] == index.vocabulary_sha256,
            "warm-start vocabulary mismatch; no token remapping is implicit"
        );
        // The architecture was checked before CUDA against the configuration
        // bytes the retained state binds; it is not re-read here.
        crate::neural::load_bound_model(
            &mut vs,
            &checkpoint.join("model.ot"),
            &state["model_sha256"],
            "initialization model changed",
        )?;
        initialization = json!({"kind":"warm_start","checkpoint":checkpoint.strip_prefix(root)?,
            "state_sha256":selected_state_sha.as_deref().context("parent state digest")?,"model_sha256":state["model_sha256"],
            "parent_step":state["step"],"parent_corpus_sha256":state["corpus_sha256"],
            "optimizer":crate::native_loop::warm_start::FRESH_OPTIMIZER, "sampler":"new corpus starts at step zero"});
        if let Some(p) = &policy {
            ensure!(
                p["parent_checkpoint_sha256"]
                    == selected_state_sha
                        .as_deref()
                        .context("parent state digest")?,
                "replay initialization checkpoint changed"
            );
            initialization["replay_policy_sha256"] = json!(policy_sha);
            initialization["replay_policy"] = p.clone();
            if p["optimizer"] == "carry" {
                ensure!(
                    state["ranking_backward"] == crate::neural::RANKING_BACKWARD_ID
                        && state["runtime_objective"] == "huber_log1p_seconds_v2",
                    "optimizer warm-start requires compatible objective identity"
                );
                optimizer = AdamW::load_bound(
                    &checkpoint.join("optimizer.ot"),
                    &state["optimizer_sha256"],
                    &state["optimizer_metadata_sha256"],
                    device,
                    "warm-start optimizer missing or changed",
                )?;
                ensure!(
                    optimizer.step == state["step"].as_u64().context("parent optimizer step")?,
                    "warm-start optimizer/model step mismatch"
                );
                optimizer.validate_parameters(&vs)?;
                initialization["optimizer"] = json!("carried FP32 AdamW with bias-correction step");
                initialization["optimizer_parent_step"] = json!(optimizer.step);
            }
            initialization["sampler"] =
                json!("lineage-bound counter; local block step starts at zero");
            initialization["resolved_sampler_id"] = json!(active_sampler_id(replay.as_ref()));
            if let Some(recovered) = &recovered_continuation {
                initialization["continuation"] = recovered.clone();
                initialization["exact_same_experiment_resume"] = json!(false);
            }
        }
    }
    if !resume {
        // The lineage record names its mode; a new lineage never inherits an
        // exact-resume claim from its parent.
        initialization["continuation_mode"] = json!(mode.as_str());
        initialization["continuation_mode_declaration"] = declaration["declaration"].clone();
        initialization["objective_sha256"] = json!(objective_sha);
        initialization["exact_same_experiment_resume"] = json!(false);
        if !continuity.is_null() {
            initialization["continuity"] = continuity.clone();
        }
    }
    if resume {
        let state = selected_state.as_ref().context("resume state absent")?;
        ensure!(state["execution_partition"] == *execution.record(),
            "exact resume requires the original microbatch, accumulation and RNG/objective partition");
        ensure!(state["sampler_id"] == active_sampler_id(replay.as_ref()),
                "exact resume requires the same sampler; use an explicit warm-start across sampler versions");
        ensure!(state["ranking_backward"]==crate::neural::RANKING_BACKWARD_ID&&state["runtime_objective"]=="huber_log1p_seconds_v2","exact resume requires the same ranking/runtime objectives; use an explicit model warm-start across versions");
        ensure!(
            state["autocast_cache_lifetime"] == "scoped_v2",
            "cannot resume a checkpoint trained with stale autocast parameter casts"
        );
        ensure!(
            state["corpus_sha256"] == corpus_hash
                && state["vocabulary_sha256"] == index.vocabulary_sha256
                && state["world"] == world
                && state["global_batch"] == global_batch
                && state["seed"] == seed,
            "resume identity mismatch"
        );
        let cp = selected_checkpoint
            .as_ref()
            .context("resume selection absent")?;
        // W2-T04 (G-10c): the configuration, model and optimizer consumed are
        // exactly the bytes this checkpoint's (retained) state binds, each read
        // once; a separate digest pass over the paths first would add nothing.
        let previous: Config = serde_json::from_slice(&crate::manifest::read_bound(
            &cp.join("config.json"),
            &state["config_sha256"],
            "checkpoint optimizer or configuration changed",
        )?)?;
        ensure!(
            serde_json::to_value(previous)? == serde_json::to_value(&config)?,
            "resume architecture mismatch"
        );
        crate::neural::load_bound_model(
            &mut vs,
            &cp.join("model.ot"),
            &state["model_sha256"],
            "checkpoint model changed",
        )?;
        optimizer = AdamW::load_bound(
            &cp.join("optimizer.ot"),
            &state["optimizer_sha256"],
            &state["optimizer_metadata_sha256"],
            device,
            "checkpoint optimizer or configuration changed",
        )?;
        ensure!(
            optimizer.step == state["step"].as_u64().context("checkpoint step")?,
            "optimizer/model step mismatch"
        );
        initialization = read_json(&initialization_path)?;
        ensure!(
            state["initialization"] == initialization,
            "initialization lineage mismatch"
        );
        if policy.is_some() {
            ensure!(
                initialization["replay_policy_sha256"] == json!(policy_sha)
                    && initialization["replay_policy"] == json!(policy),
                "exact resume replay policy changed"
            );
        } else {
            ensure!(
                initialization.get("replay_policy_sha256").is_none(),
                "exact resume requires original replay policy"
            );
        }
    }
    ensure!(
        policy.is_none() || initialize.is_some() || resume,
        "replay experiment requires explicit initialization or exact resume"
    );
    let sampling_offset = initialization["optimizer_parent_step"]
        .as_u64()
        .unwrap_or(0) as usize;
    let steps = steps
        .checked_add(sampling_offset)
        .context("training step overflow")?;
    ensure!(
        optimizer.step as usize <= steps,
        "checkpoint exceeds the requested training window"
    );
    if rank == 0 && !resume {
        checkpoint_commit::replace_json(&initialization_path, &initialization)?;
    }
    if rank == 0 {
        checkpoint_commit::replace_json(&launch.join("training-inputs.json"), &frozen_inputs)?;
    }
    ensure!(
        index
            .contexts
            .iter()
            .all(|c| crate::corpus_projection::admitted_partition(&c.partition)),
        "non-admitted context partition in training closure"
    );
    // W6-T01: cached device staging is keyed by the content it was built
    // from; a cold load verifies every byte through the same functions.
    // W2-T04 (G-11a/b): a staged tensor lives on one device; the device is part
    // of every resident key.
    let context_key = format!("{device:?}:{}", inputs::context_identity(index)?);
    let context_tensor = RESIDENT_CONTEXT.with(|cache| -> Result<Tensor> {
        if let Some((identity, tensor)) = cache.borrow().as_ref() {
            if *identity == context_key {
                return Ok(tensor.shallow_clone());
            }
        }
        // Release an obsolete epoch before allocating its replacement.
        cache.borrow_mut().take();
        let contexts = inputs::stage_contexts(root, index)?;
        // Small immutable artifacts are staged once on each GPU; no CSV loading in training steps.
        let tensor = Tensor::from_slice(&contexts)
            .view([-1, 2048])
            .to_device(device);
        // Device tensors own their copied storage; release host staging before training.
        drop(contexts);
        *cache.borrow_mut() = Some((context_key.clone(), tensor.shallow_clone()));
        Ok(tensor)
    })?;
    let tensor_key = format!("{device:?}:{corpus_hash}:{frozen_inputs_sha}");
    let (resident_cache_hit, tensors) =
        RESIDENT_TENSORS.with(|cache| -> Result<(bool, Vec<Tensor>)> {
            if let Some((identity, tensors)) = cache.borrow().as_ref() {
                if *identity == tensor_key {
                    return Ok((true, tensors.iter().map(Tensor::shallow_clone).collect()));
                }
            }
            cache.borrow_mut().take();
            let staged = inputs::stage_records(index, vocabulary.tokens.len(), |i| records.get(i))?;
            let token_tensor = Tensor::from_slice(&staged.tokens)
                .view([-1, SEQUENCE as i64])
                .to_device(device);
            let legal_id_tensor = Tensor::from_slice(&staged.legal)
                .view([-1, SEQUENCE as i64])
                .to_device(device);
            let context_id_tensor = Tensor::from_slice(&staged.context_ids).to_device(device);
            let mask_table = Tensor::from_slice(&staged.masks)
                .view([-1, vocabulary.tokens.len() as i64])
                .to_kind(Kind::Bool)
                .to_device(device);
            drop(staged);
            let tensors = vec![token_tensor, legal_id_tensor, context_id_tensor, mask_table];
            *cache.borrow_mut() = Some((
                tensor_key.clone(),
                tensors.iter().map(Tensor::shallow_clone).collect(),
            ));
            Ok((false, tensors))
        })?;
    let [token_tensor, legal_id_tensor, context_id_tensor, mask_table]: [Tensor; 4] =
        tensors.try_into().ok().unwrap();
    // Validate every sampled comparison against the frozen index before its
    // records are read or any loss is formed (W6-T03).
    let comparisons = objective::SampledComparisons::new(index)?;
    // `global` selects the objective: `None` is v1's microbatch-local
    // composite; partition v2 passes the step's global denominators.
    let loss_for = |ids: &[usize],
                    train: bool,
                    global: Option<&partition::Denominators>|
     -> Result<(Tensor, [f64; 5])> {
        let corpus_partition = if train {
            "meta_train"
        } else {
            "meta_validation"
        };
        comparisons.check_ids(index, ids, corpus_partition)?;
        let rows = ids.iter().map(|&i| records.get(i)).collect::<Vec<_>>();
        let labels = ids
            .iter()
            .zip(&rows)
            .map(|(&id, r)| objective::RowLabels::from_record(id, r))
            .collect::<Vec<_>>();
        comparisons.check_rows(index, &labels)?;
        let length = rows.iter().map(|r| r.length).max().unwrap().div_ceil(8) * 8;
        let length = length.min(SEQUENCE) as i64;
        let id = Tensor::from_slice(&ids.iter().map(|&i| i as i64).collect::<Vec<_>>())
            .to_device(device);
        let tokens = token_tensor
            .index_select(0, &id)
            .narrow(1, 0, length)
            .to_kind(Kind::Int64)
            .bitwise_and(0xffffi64);
        let pad = tokens.ne(0);
        let context = context_tensor.index_select(0, &context_id_tensor.index_select(0, &id));
        let (logits, rewards, runtime, stability, ablation) = model.forward(
            &context.narrow(1, 0, 1536),
            &context.narrow(1, 1536, 512),
            &tokens,
            &pad,
            train,
        )?;
        let legal = mask_table
            .index_select(
                0,
                &legal_id_tensor
                    .index_select(0, &id)
                    .narrow(1, 1, length - 1)
                    .reshape([-1])
                    .to_kind(Kind::Int64)
                    .bitwise_and(0xffffi64),
            )
            .view([ids.len() as i64, length - 1, vocabulary.tokens.len() as i64]);
        let outputs = objective::Outputs {
            logits: &logits,
            rewards: &rewards,
            runtime: &runtime,
            stability: &stability,
            ablation: &ablation,
        };
        match global {
            None => objective::composite(
                &outputs,
                model.loss_log_vars.values(),
                &tokens,
                &pad,
                &legal,
                &labels,
            ),
            Some(global) => partition::composite(
                &outputs,
                model.loss_log_vars.values(),
                &tokens,
                &pad,
                &legal,
                &labels,
                global,
            ),
        }
    };
    let accumulation = execution.accumulation(rank)?;
    let started = Instant::now();
    let initial_step = optimizer.step as usize;
    let mut last_live_write = Instant::now() - Duration::from_secs(1);
    let mut measured_step_seconds = 0.0;
    let mut telemetry_seconds = 0.0;
    let mut coordination_seconds = 0.0;
    let mut checkpoint_seconds = 0.0;
    let mut metrics = if rank == 0 {
        Some(BufWriter::new(
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(run.join("metrics.jsonl"))?,
        ))
    } else {
        None
    };
    #[cfg(feature = "telemetry")]
    let mut telemetry = if rank == 0 {
        std::env::var_os("JOPE_TELEMETRY_RUN_DIR").and_then(|run_dir| {
            let metadata = jope_telemetry::manifest::RunMetadata {
                metric_groups: std::collections::BTreeMap::from([(
                    jope_telemetry::train::TRAIN_LOSS_COMPONENT,
                    std::collections::BTreeMap::from([
                        (0, "sequence_nll".to_owned()),
                        (1, "ranking_bt".to_owned()),
                        (2, "ranking_listwise".to_owned()),
                        (3, "runtime_huber".to_owned()),
                        (4, "contextfree_bt".to_owned()),
                    ]),
                )]),
                metadata: [
                    ("trainer".to_owned(), json!("jope-engine")),
                    ("corpus_sha256".to_owned(), json!(corpus_hash)),
                    ("world_size".to_owned(), json!(world)),
                    ("global_batch".to_owned(), json!(global_batch)),
                    ("microbatch".to_owned(), json!(micro)),
                    ("seed".to_owned(), json!(seed)),
                    ("device".to_owned(), json!(format!("{:?}", device))),
                ]
                .into_iter()
                .collect(),
                ..Default::default()
            };
            let mut builder = jope_telemetry::core::RecorderBuilder::new(run_dir, metadata);
            if let Err(error) = builder.add_producer("jope-engine") {
                eprintln!("jope-telemetry producer setup failed: {error:#}");
                return None;
            }
            match builder.start() {
                Ok((recorder, mut producers)) => {
                    producers.pop().map(|producer| (recorder, producer))
                }
                Err(error) => {
                    eprintln!("jope-telemetry initialization failed: {error:#}");
                    None
                }
            }
        })
    } else {
        None
    };
    let mut last_checkpoint = Instant::now();
    let mut validation_metrics = Value::Null;
    let mut latest: Value = if run.join("latest.json").is_file() {
        read_json(&run.join("latest.json"))?
    } else {
        Value::Null
    };
    let mut observed_exposure = crate::training_exposure::Counts::default();
    let mut committed_exposure = crate::training_exposure::Counts::default();
    let mut observed_families = crate::training_exposure::FamilyGroups::new();
    write_json(
        &launch.join(format!("rank-{rank}.json")),
        &json!({"schema":"parallel.neural_training.v1",
        "rank":rank,"world":world,"status":"active","step":initial_step,"steps":steps,
        "initial_optimizer_step":initial_step,"corpus_sha256":corpus_hash,"timestamp":now(),
        "exposure":crate::training_exposure::attempt_snapshot(&observed_exposure,&committed_exposure,false)?}),
    )?;
    // W6-T04: every consumed microbatch, attempted update and durable commit
    // of this rank and attempt, bound into each checkpoint receipt.
    let mut journal = consumption::Journal::create_with_shard(
        &launch.join(format!("consumption-rank-{rank}.jsonl")),
        consumption::Header {
            schema: consumption::SCHEMA.into(),
            rank,
            world,
            microbatch: micro,
            global_batch,
            accumulation,
            seed,
            sampler_id: active_sampler_id(replay.as_ref()).into(),
            rng: consumption::RNG_ID.into(),
            sampling_offset: sampling_offset as u64,
            initial_optimizer_step: initial_step as u64,
            target_optimizer_step: steps as u64,
            corpus_sha256: corpus_hash.clone(),
            replay_policy_sha256: policy_sha.clone(),
            resume_state_sha256: if resume {
                Some(file_hash(
                    &selected_checkpoint
                        .as_ref()
                        .context("resume selection absent")?
                        .join("state.json"),
                )?)
            } else {
                None
            },
        },
        execution.shard(rank)?,
    )?;
    let journal_path = journal.path().strip_prefix(root)?.to_path_buf();
    // The optimizer step whose update is in flight; cleared once applied.
    let mut attempting: Option<usize> = None;
    let (
        outcome,
        optimizer,
        mut journal,
        attempting,
        observed_exposure,
        committed_exposure,
        observed_families,
        corpus_hash,
    ) = training_loop!((
        root,
        run,
        launch,
        rank,
        world,
        micro,
        steps,
        checkpoint_steps,
        resume,
        global_batch,
        batch_qualification,
        native_stop,
        execution,
        policy,
        policy_sha,
        replay,
        seed,
        corpus_hash,
        index,
        records,
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
        exposure_baseline,
        family_baseline,
        family_baseline_known,
        binding,
        comm,
        device,
        vs,
        optimizer,
        initialization,
        sampling_offset,
        resident_cache_hit,
        comparisons,
        loss_for,
        accumulation,
        started,
        initial_step,
        last_live_write,
        measured_step_seconds,
        telemetry_seconds,
        coordination_seconds,
        checkpoint_seconds,
        metrics,
        telemetry,
        last_checkpoint,
        validation_metrics,
        latest,
        observed_exposure,
        committed_exposure,
        observed_families,
        journal,
        journal_path,
        attempting,
    ));
    if let Err(error) = &outcome {
        // An update that was attempted but not applied is recorded as
        // rejected; the journal refuses anything after it. Best effort: the
        // original error is what this rank returns.
        if let Some(step) = attempting {
            let _ = journal.rejected(step as u64, &format!("{error:#}"));
        }
        let _ = journal.sync();
    }
    if outcome.is_err() {
        // Normal error returns preserve all observed completed forwards. A
        // hard process death can leave an unreported tail and must remain
        // unknown during reconciliation; never infer zero lost exposure.
        if let Ok(mut exposure) = crate::training_exposure::attempt_snapshot(
            &observed_exposure,
            &committed_exposure,
            false,
        ) {
            exposure["failure_returned"] = json!(true);
            exposure["aborted_exposure_requires_stopped_rank_checkpoint_reconciliation"] =
                json!(true);
            let _ = write_json(
                &launch.join(format!("rank-{rank}.json")),
                &json!({
                "schema":"parallel.neural_training.v1","rank":rank,"world":world,"status":"failed",
                "initial_optimizer_step":initial_step,"step":optimizer.step,"corpus_sha256":corpus_hash,
                "exposure":exposure,"timestamp":now()}),
            );
        }
    }
    outcome
}
