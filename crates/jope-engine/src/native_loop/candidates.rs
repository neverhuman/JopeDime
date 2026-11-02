use super::*;

impl Loop {
    pub(super) async fn aggregate(
        &self,
        graph: &Graph,
        client: &Client,
        generation: u64,
        mut members: BTreeMap<String, Artifact>,
        progress: &mut Progress,
    ) -> Result<Option<BTreeMap<String, Artifact>>> {
        while members.len() > 1024 {
            let rows: Vec<_> = members.into_iter().collect();
            let mut next = BTreeMap::new();
            let mut ready = true;
            for chunk in rows.chunks(1024) {
                let predecessors: BTreeMap<_, _> = chunk.iter().cloned().collect();
                let id = format!(
                    "g{generation}-aggregate-{}",
                    hash(&serde_json::to_vec(&predecessors)?)
                );
                let (request_ref, r) = self.request(
                    generation,
                    id.clone(),
                    NativeLoopRole::Aggregate,
                    json!({"members":predecessors}),
                    predecessors.clone(),
                )?;
                let base = self.state.join("aggregates").join(&id);
                let input = InputManifest {
                    schema: "jope.smartcluster_input.v1".into(),
                    root: self.root.clone(),
                    operation: Operation::Verify,
                    input_artifacts: predecessors
                        .values()
                        .map(|a| (a.path.clone(), a.sha256.clone()))
                        .chain(std::iter::once((
                            request_ref.path.clone(),
                            request_ref.sha256.clone(),
                        )))
                        .collect(),
                    closure: None,
                    output_receipt: base.join("result.json").strip_prefix(&self.root)?.into(),
                };
                let input_ref = frozen(&self.root, &base.join("task.input.json"), &input)?;
                let manifest = TaskManifest {
                    schema: "jope.smartcluster_task.v1".into(),
                    run: self.config.run.clone(),
                    generation,
                    task_id: id.clone(),
                    kind: TaskKind::Verification,
                    input_manifest_sha256: input_ref.sha256,
                    dependencies: input
                        .input_artifacts
                        .iter()
                        .map(|(p, h)| (p.to_string_lossy().into_owned(), h.clone()))
                        .collect(),
                    parent_checkpoint_sha256: None,
                    corpus_sha256: None,
                    idempotency_key: IdempotencyKey::new(format!("{}/{id}", self.config.run))?,
                    deadline: self.config.task_deadline,
                    retry_allowance: 1,
                    resources: self.config.cpu_resources,
                    expected_receipt_id: format!("{}/{id}/complete", self.config.run),
                    checkpoint_allowance_ms: None,
                };
                let manifest_ref = frozen(&self.root, &base.join("task.json"), &manifest)?;
                self.bind(&Binding {
                    schema: "jope.native_loop_binding.v1".into(),
                    request: request_ref,
                    admission: Admission {
                        schema: "jope.native_graph_admission.v1".into(),
                        manifest: manifest_ref,
                        predecessors,
                        delivery: Delivery::NativeJob,
                    },
                })?;
                if let Some((reference, result)) = self.result(graph, client, &r).await? {
                    next.insert(result.task_id, reference);
                } else {
                    self.offer(graph, client, &r, progress).await?;
                    ready = false;
                }
            }
            if !ready {
                progress
                    .idle_reasons
                    .insert("awaiting_complete_dependency_aggregation".into());
                return Ok(None);
            }
            members = next;
        }
        Ok(Some(members))
    }

    pub(super) fn recorded_candidate(&self, path: &Path, bytes: &[u8]) -> Result<Artifact> {
        if self.read_only {
            let a = artifact(&self.root, path)?;
            ensure!(
                native_genome::checked(&self.root, &a, 1024 * 1024)? == bytes,
                "original candidate differs from deterministic proposal expansion"
            );
            Ok(a)
        } else {
            immutable(&self.root, path, bytes)
        }
    }
    /// Expand every valid recorded neural program, including those outside the ranked64 pool.
    /// Matched generic controls use the same cut/slot budget and independent deterministic RNG.
    pub(super) fn candidates(
        &self,
        generation: u64,
        arm: Arm,
        proposal: &Artifact,
        result: &VerifiedResult,
        checkpoint: &Artifact,
        corpus: &Artifact,
    ) -> Result<Vec<Artifact>> {
        let value: Value = native_genome::load(&self.root, &result.output)?;
        let outputs: BTreeMap<PathBuf, String> =
            serde_json::from_value(value["artifacts"].clone())?;
        let binding_path = self.state.join("bindings").join(format!(
            "{}.json",
            identity(&self.config.run, generation, &result.task_id)?
        ));
        let binding: Binding = serde_json::from_slice(&read(&binding_path)?)?;
        let (_, input) = crate::smartcluster_worker::verify_task_inputs(
            &self.root,
            &self.root.join(&binding.admission.manifest.path),
            &binding.admission.manifest.sha256,
        )?;
        let Operation::Proposal { output, .. } = input.operation else {
            anyhow::bail!("candidate source is not a verified proposal")
        };
        let jobs_file = Artifact {
            path: output.join("jobs-0.json"),
            sha256: outputs
                .get(Path::new("jobs-0.json"))
                .context("proposal jobs absent")?
                .clone(),
        };
        let jobs: crate::service::GpJobs = native_genome::load(&self.root, &jobs_file)?;
        let corpus_root = corpus.path.parent().context("corpus directory")?;
        let (index, vocab) = crate::teaching_store::read_index_and_vocabulary(
            &self.root,
            &self.root.join(corpus_root),
        )?;
        let mut result_refs = Vec::new();
        for job in jobs.jobs {
            let c = index
                .contexts
                .iter()
                .find(|c| c.prepared_sha256 == job.prepared_sha256 && c.cut == job.cut)
                .context("proposal context absent from parent corpus")?
                .clone();
            ensure!(
                c.partition == "meta_train" && !self.policy.excluded_families.contains(&c.family),
                "proposal selected excluded or sealed family"
            );
            let seed = job.seeds.context("proposal seed parent absent")?;
            let capture = seed
                .parent()
                .context("seed parent")?
                .join("all-valid-neural.json");
            let suffix = capture.strip_prefix(&output)?;
            let a = Artifact {
                path: capture.clone(),
                sha256: outputs
                    .get(suffix)
                    .context("all valid model proposals were not preserved")?
                    .clone(),
            };
            let all: Value = native_genome::load(&self.root, &a)?;
            ensure!(
                all["schema"] == "jope.all_valid_neural.v1" && all["final_tests_opened"] == false,
                "invalid all-valid capture"
            );
            let rows = all["rows"]
                .as_array()
                .context("valid candidate rows absent")?;
            ensure!(
                rows.len() <= 1024 && all["valid"] == rows.len(),
                "valid candidate coverage differs"
            );
            let mut unique = BTreeSet::new();
            let mut generic_unique = BTreeSet::new();
            let mut matched_slot = 0u64;
            for (origin_row, row) in rows.iter().enumerate() {
                let genome = row["genome"].as_str().context("candidate genome absent")?;
                ensure!(
                    row["program_sha256"] == hash(genome.as_bytes()),
                    "captured genotype hash differs"
                );
                if !unique.insert(genome.to_string()) {
                    continue;
                }
                let cand = Candidate {
                    schema: "jope.native_candidate.v1".into(),
                    run: self.config.run.clone(),
                    generation,
                    arm,
                    matched_slot,
                    context: c.clone(),
                    genome: genome.into(),
                    genome_sha256: hash(genome.as_bytes()),
                    proposal: proposal.clone(),
                    parent_checkpoint: checkpoint.clone(),
                    origin_capture: a.clone(),
                    origin_row,
                    generic_draw: None,
                    sampling_seed: all["sampling_seed"]
                        .as_u64()
                        .context("proposal RNG absent")?,
                };
                let bytes = serde_json::to_vec(&cand)?;
                result_refs.push(
                    self.recorded_candidate(
                        &self
                            .state
                            .join("candidates")
                            .join(format!("{}.json", hash(&bytes))),
                        &bytes,
                    )?,
                );
                if arm == Arm::Model {
                    let rng = generation_seed(
                        &self.config.run,
                        generation,
                        &format!("generic:{}:{}:{matched_slot}", c.prepared_sha256, c.cut),
                    )?;
                    let mut generated = None;
                    for offset in 0..10000u64 {
                        let raw = feat_math::evolve::random_canonical_pipeline(
                            rng.wrapping_add(offset),
                            6,
                        );
                        if let Ok(program) = vocab.project_proposal(&raw) {
                            if generic_unique.insert(program.clone()) {
                                generated = Some((program, offset));
                                break;
                            }
                        }
                    }
                    let (genome, draw) = generated.context("matched generic genome unavailable")?;
                    let generic = Candidate {
                        arm: Arm::Generic,
                        genome_sha256: hash(genome.as_bytes()),
                        genome,
                        sampling_seed: rng,
                        generic_draw: Some(draw),
                        ..cand
                    };
                    let bytes = serde_json::to_vec(&generic)?;
                    result_refs.push(
                        self.recorded_candidate(
                            &self
                                .state
                                .join("candidates")
                                .join(format!("{}.json", hash(&bytes))),
                            &bytes,
                        )?,
                    );
                }
                matched_slot += 1;
            }
        }
        ensure!(
            !result_refs.is_empty() && result_refs.len() <= 100000,
            "empty or unbounded candidate expansion"
        );
        Ok(result_refs)
    }
    pub(super) async fn verify_completed_predecessor(
        &self,
        graph: &Graph,
        client: &Client,
        completed: &RunCompletion,
        lineage: Option<&lineage::VerifiedLineage>,
    ) -> Result<()> {
        let old_state = self.root.join(
            completed
                .config
                .path
                .parent()
                .context("old config parent")?,
        );
        let previous = Self::open_mode(
            &self.root,
            &old_state,
            &self.graph_state,
            &completed.config,
            true,
        )?;
        let mut parent = previous.config.accepted_checkpoint.clone();
        if let Some(lineage) = lineage {
            if previous.config.schema == warm_start::LOOP_V2 {
                let state: Value = native_genome::load(&self.root, &parent)?;
                let anchor = &lineage.lineage().anchor;
                ensure!(
                    state["model_sha256"] == anchor.weights_sha256
                        && state["step"].as_u64() == Some(anchor.optimizer_step)
                        && anchor
                            .objective_sha256
                            .as_ref()
                            .is_none_or(|d| state["objective_sha256"] == *d),
                    "native lineage anchor differs from actual accepted manifest"
                );
            }
        }
        let mut corpus = previous.config.initial_corpus.clone();
        let mut acceptance = previous.config.accepted_decision.clone();
        for generation in 0..2 {
            let pack = previous
                .original_request(generation, &format!("g{generation}-pack-and-publish"))?;
            ensure!(
                pack.role == NativeLoopRole::PackAndPublish
                    && pack.parameters["previous_corpus"] == serde_json::to_value(&corpus)?,
                "predecessor corpus lineage changed"
            );
            let (pack_ref, packed) = previous
                .result(graph, client, &pack)
                .await?
                .context("predecessor publication absent")?;
            let published = native_genome::load_dataset_receipt(&self.root, &packed.output)?;
            ensure!(
                published["local_corpus_committed"] == true
                    && (!previous.policy.require_remote_publication
                        || published["remote_verified"] == true),
                "predecessor corpus publication not verified"
            );
            let next_corpus: Artifact = serde_json::from_value(published["corpus_index"].clone())?;
            let train = previous.original_request(generation, &format!("g{generation}-train"))?;
            let window = previous.config.train_window(generation)?;
            let initial = window.initial;
            ensure!(
                train.role == NativeLoopRole::Train
                    && train.parameters["checkpoint"] == serde_json::to_value(&parent)?
                    && train.parameters["corpus"] == serde_json::to_value(&next_corpus)?
                    && train.parameters["initial_step"] == initial
                    && train.parameters["requested_increment"] == 3000
                    && train.parameters["target_step"] == initial + 3000
                    && train.parameters["global_batch"] == 2048
                    && train.parameters
                        == warm_start::train_parameters(&parent, &next_corpus, &window)?
                    && train.predecessors == BTreeMap::from([(packed.task_id, pack_ref)]),
                "predecessor original optimizer/publication edge differs"
            );
            let (train_ref, trained) = previous
                .result(graph, client, &train)
                .await?
                .context("predecessor Train result absent")?;
            let numerical: Value = native_genome::load(&self.root, &trained.output)?;
            let checkpoint_path = PathBuf::from(
                numerical["latest"]["checkpoint"]
                    .as_str()
                    .context("predecessor checkpoint absent")?,
            );
            let next_parent = artifact(
                &self.root,
                &self.root.join(checkpoint_path).join("state.json"),
            )?;
            let comparison = previous
                .prove_comparison(graph, client, generation, &parent, &corpus, &acceptance)
                .await?;
            let decision_ref: Artifact = serde_json::from_slice(&read(
                &old_state
                    .join("parent-decisions")
                    .join(format!("g{generation}.json")),
            )?)?;
            let decision: ProposalParentDecision = native_genome::load(&self.root, &decision_ref)?;
            let assessment: Value = native_genome::load(&self.root, &decision.assessment)?;
            verify_parent_decision(
                &decision,
                &assessment,
                &completed.run,
                generation,
                crate::native_loop::ParentEvidence {
                    cp: &next_parent,
                    previous: &parent,
                    train: &train_ref,
                    comparison: &comparison,
                },
            )?;
            ensure!(decision.accepted, "predecessor checkpoint was not accepted");
            let state: Value = native_genome::load(&self.root, &next_parent)?;
            if let Some(lineage) = lineage {
                let edges = lineage.edges();
                let index = edges
                    .len()
                    .checked_sub(2)
                    .context("missing native lineage block edges")?
                    + generation as usize;
                lineage::check_checkpoint(
                    &state,
                    edges
                        .get(index)
                        .context("missing native lineage block edge")?,
                )?;
            }
            ensure!(
                state["step"] == initial + 3000 && state["corpus_sha256"] == next_corpus.sha256,
                "predecessor final optimizer/corpus bytes differ"
            );
            parent = next_parent;
            corpus = next_corpus;
            acceptance = decision_ref;
        }
        ensure!(
            parent == completed.checkpoint
                && corpus == completed.corpus
                && acceptance == completed.accepted_parent_decision,
            "successor completion does not derive from actual final Train/Pack"
        );
        for (arm, reference) in &completed.final_proposals {
            let id = format!(
                "g2-proposal-{}",
                if *arm == Arm::Model {
                    "model"
                } else {
                    "shuffled"
                }
            );
            let request = previous.original_request(2, &id)?;
            ensure!(
                request.parameters["checkpoint"] == serde_json::to_value(&parent)?
                    && request.parameters["corpus"] == serde_json::to_value(&corpus)?
                    && request.parameters["accepted_parent_decision"]
                        == serde_json::to_value(&acceptance)?
                    && request.parameters["proposal_seed"]
                        == generation_seed(&completed.run, 2, &parent.sha256)?,
                "final successor proposal parent changed"
            );
            let (actual, _) = previous
                .result(graph, client, &request)
                .await?
                .context("final predecessor proposal absent")?;
            ensure!(
                actual == *reference,
                "successor proposal differs from original bound result"
            );
        }
        Ok(())
    }
}
