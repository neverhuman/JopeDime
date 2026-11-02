use super::*;

impl Loop {
    pub fn open(root: &Path, state: &Path, graph_state: &Path, config: &Artifact) -> Result<Self> {
        Self::open_mode(root, state, graph_state, config, false)
    }
    pub(super) fn open_mode(
        root: &Path,
        state: &Path,
        graph_state: &Path,
        config: &Artifact,
        read_only: bool,
    ) -> Result<Self> {
        ensure!(
            root.is_absolute() && fs::canonicalize(root)? == root,
            "loop requires canonical physical root"
        );
        let cfg: Config = native_genome::load(root, config)?;
        cfg.validate()?;
        let policy: ScientificPolicy = native_genome::load(root, &cfg.scientific_policy)?;
        ensure!(
            policy.schema == "jope.native_learning_policy.v1"
                && policy.training_target == "continuation_only_v1"
                && policy.sealed_partition == "meta_test"
                && policy.proposal_seed_protocol == "generation_sha256_v1"
                && policy.matched_arms == BTreeSet::from([Arm::Model, Arm::Generic, Arm::Shuffled])
                && !policy.training_arms.is_empty(),
            "scientific objective, matched arms or sealed policy not explicitly frozen"
        );
        let _: Value = native_genome::load(root, &policy.family_partition_receipt)?;
        let _: Value = native_genome::load(root, &policy.exclusion_receipt)?;
        let accepted: Value = native_genome::load(root, &cfg.accepted_decision)?;
        ensure!(
            accepted["schema"] == "jope.initial_proposal_parent_decision.v1"
                && accepted["checkpoint"] == serde_json::to_value(&cfg.accepted_checkpoint)?
                && accepted["corpus"] == serde_json::to_value(&cfg.initial_corpus)?
                && accepted["accepted"] == true
                && accepted["optimizer_step"] == cfg.initial_optimizer_step
                && (cfg.predecessor_run.is_none()
                    || accepted["predecessor_run"] == serde_json::to_value(&cfg.predecessor_run)?),
            "initial accepted decision does not bind original checkpoint/corpus"
        );
        if let Some(a) = &cfg.preparation_overlay {
            ensure!(
                !cfg.qualification
                    && cfg.initial_optimizer_step == 60000
                    && cfg.cut_start == 0
                    && cfg.cuts == 2,
                "first-window overlay cannot extend or relabel its reviewed scope"
            );
            crate::preparation_overlay::load(root, a, &cfg.run)?;
        }
        let checkpoint: Value = native_genome::load(root, &cfg.accepted_checkpoint)?;
        let _: Value = native_genome::load_corpus_index(root, &cfg.initial_corpus)?;
        ensure!(
            checkpoint["step"] == cfg.initial_optimizer_step
                && checkpoint["corpus_sha256"] == cfg.initial_corpus.sha256,
            "accepted checkpoint/corpus/window changed"
        );
        let cp = cfg
            .accepted_checkpoint
            .path
            .parent()
            .context("checkpoint parent")?;
        for (file, key) in [
            ("model.ot", "model_sha256"),
            ("optimizer.ot", "optimizer_sha256"),
        ] {
            ensure!(
                crate::manifest::file_hash(&crate::manifest::resolve(root, &cp.join(file))?)?
                    == checkpoint[key],
                "accepted model or optimizer bytes changed"
            );
        }
        if read_only {
            ensure!(
                fs::canonicalize(state)? == state
                    && state.starts_with(root)
                    && fs::canonicalize(graph_state)? == graph_state
                    && graph_state.starts_with(root),
                "existing physical read-only loop state required"
            );
            let frozen_config: Config = serde_json::from_slice(&read(&state.join("config.json"))?)?;
            ensure!(
                serde_json::to_vec(&frozen_config)? == serde_json::to_vec(&cfg)?,
                "read-only proof config differs from original frozen loop"
            );
            return Ok(Self {
                root: root.into(),
                state: state.into(),
                graph_state: graph_state.into(),
                config: cfg,
                policy,
                _lock: fs::File::open(state)?,
                read_only: true,
            });
        }
        physical_directory(root, state)?;
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(state.join("loop.lock"))?;
        ensure!(
            lock.try_lock().is_ok(),
            "another learning planner owns this run"
        );
        frozen(root, &state.join("config.json"), &cfg)?;
        Ok(Self {
            root: root.into(),
            state: state.into(),
            graph_state: graph_state.into(),
            config: cfg,
            policy,
            _lock: lock,
            read_only: false,
        })
    }
    pub(super) fn request(
        &self,
        generation: u64,
        id: String,
        role: NativeLoopRole,
        parameters: Value,
        predecessors: BTreeMap<String, Artifact>,
    ) -> Result<(Artifact, Request)> {
        let r = Request {
            schema: "jope.native_loop_request.v1".into(),
            run: self.config.run.clone(),
            generation,
            id,
            role,
            parameters,
            predecessors,
        };
        let key = identity(&r.run, generation, &r.id)?;
        Ok((
            frozen(
                &self.root,
                &self.state.join("requests").join(format!("{key}.json")),
                &r,
            )?,
            r,
        ))
    }
    /// Materializers freeze actual executable TaskManifests; this never fabricates an authority grant.
    pub fn bind(&self, binding: &Binding) -> Result<()> {
        ensure!(
            binding.schema == "jope.native_loop_binding.v1",
            "loop binding schema"
        );
        let request: Request = native_genome::load(&self.root, &binding.request)?;
        let key = identity(&request.run, request.generation, &request.id)?;
        ensure!(
            binding.request.path
                == self
                    .state
                    .join("requests")
                    .join(format!("{key}.json"))
                    .strip_prefix(&self.root)?,
            "binding is not original loop request"
        );
        self.validate_binding(&request, binding)?;
        frozen(
            &self.root,
            &self.state.join("bindings").join(format!("{key}.json")),
            binding,
        )?;
        Ok(())
    }
    /// Record an operator-reviewed quality decision, separate from training commitment and serving.
    pub fn bind_parent_decision(&self, reference: &Artifact) -> Result<()> {
        let decision: ProposalParentDecision = native_genome::load(&self.root, reference)?;
        ensure!(
            decision.schema == "jope.proposal_parent_decision.v1"
                && decision.run == self.config.run
                && decision.generation < self.config.cycles,
            "proposal parent decision scope differs"
        );
        for a in [
            &decision.checkpoint,
            &decision.previous_parent,
            &decision.training_result,
            &decision.comparison,
            &decision.assessment,
        ] {
            let _: Value = native_genome::load(&self.root, a)?;
        }
        frozen(
            &self.root,
            &self
                .state
                .join("parent-decisions")
                .join(format!("g{}.json", decision.generation)),
            reference,
        )?;
        Ok(())
    }
    pub(super) fn validate_binding(&self, r: &Request, b: &Binding) -> Result<TaskManifest> {
        let original_path = self
            .state
            .join("requests")
            .join(format!("{}.json", identity(&r.run, r.generation, &r.id)?));
        ensure!(
            b.schema == "jope.native_loop_binding.v1"
                && b.request == artifact(&self.root, &original_path)?
                && native_genome::checked(&self.root, &b.request, 1024 * 1024)?
                    == serde_json::to_vec(r)?,
            "binding does not preserve the exact frozen loop request"
        );
        ensure!(
            r.run == self.config.run
                && r.generation <= self.config.cycles
                && b.admission.predecessors == r.predecessors,
            "binding changes frozen predecessor graph"
        );
        let (task, input) = crate::smartcluster_worker::verify_task_inputs(
            &self.root,
            &crate::manifest::resolve(&self.root, &b.admission.manifest.path)?,
            &b.admission.manifest.sha256,
        )?;
        ensure!(
            input.input_artifacts.get(&b.request.path) == Some(&b.request.sha256),
            "task input closure omits original scientific loop request"
        );
        ensure!(
            task.run == r.run
                && task.generation == r.generation
                && task.task_id == r.id
                && task.deadline == self.config.task_deadline,
            "binding changes run/task/original deadline"
        );
        match (&r.role, &input.operation) {
            (
                NativeLoopRole::ProposalModel,
                Operation::Proposal {
                    corpus,
                    checkpoint,
                    shuffled: false,
                    cuts,
                    cut_start,
                    output,
                    preparation_overlay,
                    ..
                },
            )
            | (
                NativeLoopRole::ProposalShuffled,
                Operation::Proposal {
                    corpus,
                    checkpoint,
                    shuffled: true,
                    cuts,
                    cut_start,
                    output,
                    preparation_overlay,
                    ..
                },
            ) => {
                ensure!(
                    b.admission.delivery
                        == Delivery::Resident {
                            job_id: self.config.resident_job
                        }
                        && task.kind == TaskKind::ResidentGpu
                        && *preparation_overlay == self.config.preparation_overlay
                        && *cuts == self.config.cuts
                        && *cut_start == self.config.cut_start
                        && task.parent_checkpoint_sha256.as_ref()
                            == r.parameters["checkpoint"]["sha256"]
                                .as_str()
                                .map(|s| s.to_string())
                                .as_ref()
                        && task.corpus_sha256.as_ref()
                            == r.parameters["corpus"]["sha256"]
                                .as_str()
                                .map(|s| s.to_string())
                                .as_ref(),
                    "proposal frozen parent/resource binding changed"
                );
                ensure!(
                    corpus.join("index.json")
                        == *r.parameters["corpus"]["path"]
                            .as_str()
                            .context("corpus path")?
                        && checkpoint.join("state.json")
                            == *r.parameters["checkpoint"]["path"]
                                .as_str()
                                .context("parent path")?,
                    "proposal path changed"
                );
                let run: Value = crate::manifest::read_json(&crate::manifest::resolve(
                    &self.root,
                    &output.join("run.json"),
                )?)?;
                ensure!(
                    run["proposal_seed"] == r.parameters["proposal_seed"],
                    "proposal RNG is not generation-derived"
                );
            }
            (NativeLoopRole::Direct, Operation::DirectEvaluation { manifest, sha256 })
            | (NativeLoopRole::Continuation, Operation::SeededContinuation { manifest, sha256 }) => {
                ensure!(
                    b.admission.delivery == Delivery::NativeJob
                        && r.parameters["genome_task"] == json!({"path":manifest,"sha256":sha256}),
                    "genome task changed"
                );
            }
            (NativeLoopRole::Aggregate, Operation::Verify) => {
                ensure!(
                    b.admission.delivery == Delivery::NativeJob
                        && task.kind == TaskKind::Verification
                        && r.parameters["members"] == serde_json::to_value(&r.predecessors)?,
                    "aggregation changed predecessor membership"
                );
            }
            (
                NativeLoopRole::PackAndPublish,
                Operation::PackAndPublish {
                    manifest, sha256, ..
                },
            ) => {
                ensure!(
                    b.admission.delivery == Delivery::NativeJob && task.kind == TaskKind::Packing,
                    "corpus must be native packing job"
                );
                let packet: Value = native_genome::load(
                    &self.root,
                    &Artifact {
                        path: manifest.clone(),
                        sha256: sha256.clone(),
                    },
                )?;
                ensure!(
                    packet["operation"]["kind"] == "pack_continuation_and_publish"
                        && packet["operation"]["target_set"] == r.parameters["target_set"]
                        && packet["operation"]["closure"] == r.parameters["closure"],
                    "ordinary direct corpus pack is not continuation-target publication"
                );
            }
            (
                NativeLoopRole::Train,
                Operation::Train {
                    corpus,
                    initialize,
                    initial_optimizer_step,
                    requested_step_increment,
                    target_optimizer_step,
                    micro,
                    replay_policy,
                    continuation_mode,
                    ..
                },
            ) => {
                let window = self.config.train_window(r.generation)?;
                ensure!(
                    b.admission.delivery
                        == Delivery::Resident {
                            job_id: self.config.resident_job
                        }
                        && *initial_optimizer_step == window.initial
                        && *requested_step_increment == 3000
                        && *target_optimizer_step == window.target
                        && *micro == self.config.microbatch
                        && initialize.join("state.json")
                            == *r.parameters["checkpoint"]["path"]
                                .as_str()
                                .context("train parent")?
                        && corpus.join("index.json")
                            == *r.parameters["corpus"]["path"]
                                .as_str()
                                .context("train corpus")?,
                    "original optimizer window/corpus changed"
                );
                let replay: Value = crate::manifest::read_json(&crate::manifest::resolve(
                    &self.root,
                    replay_policy,
                )?)?;
                let expected = materialize::training_eligibility(
                    &self.root,
                    &serde_json::from_value(r.parameters["corpus"].clone())?,
                    &serde_json::from_value(r.parameters["checkpoint"].clone())?,
                    &self.config.scientific_policy,
                    r,
                )?;
                ensure!(
                    replay == expected,
                    "external training replay differs from independently reconstructed eligibility"
                );
                // A v2 input's declaration is the mode its replay policy
                // classifies to; a v1 input keeps its implicit meaning.
                ensure!(
                    continuation_mode.is_none_or(|declared| {
                        crate::task_contract::classify_continuation(false, true, Some(&expected))
                            .is_ok_and(|mode| mode == declared)
                    }),
                    "declared continuation mode differs from the frozen replay policy"
                );
            }
            _ => anyhow::bail!("materialized operation differs from frozen scientific request"),
        }
        Ok(task)
    }
    pub(super) fn binding(&self, r: &Request) -> Result<Option<Binding>> {
        let key = identity(&r.run, r.generation, &r.id)?;
        let path = self.state.join("bindings").join(format!("{key}.json"));
        if !path.exists() {
            return Ok(None);
        };
        let b: Binding = serde_json::from_slice(&read(&path)?)?;
        self.validate_binding(r, &b)?;
        Ok(Some(b))
    }
}
