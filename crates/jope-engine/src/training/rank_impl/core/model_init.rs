use super::*;

pub(super) fn initialize(
    input: RankInput<'_>,
    preflight: RankPreflight,
) -> Result<RankInitialized> {
    let RankInput {
        root,
        run,
        launch,
        rank,
        world,
        micro: _,
        steps,
        resume,
        initialize,
        global_batch,
        native_stop: _,
        declared_mode: _,
        assignment,
        admitted,
        ..
    } = input;
    let RankPreflight {
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
    } = preflight;
    let (index, records) = resident.as_ref();
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
    Ok(RankInitialized {
        preflight: RankPreflight {
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
        },
        device,
        vs,
        model,
        optimizer,
        initialization_path,
        initialization,
        sampling_offset,
        steps,
    })
}
