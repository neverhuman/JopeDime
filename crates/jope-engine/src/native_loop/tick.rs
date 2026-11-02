use super::*;

impl Loop {
    pub async fn tick(&self, graph: &Graph, client: &Client) -> Result<Progress> {
        ensure!(!self.read_only, "read-only loop cannot dispatch work");
        self.config.validate()?;
        if let Some(a) = &self.config.preparation_overlay {
            crate::preparation_overlay::load(&self.root, a, &self.config.run)?
                .require_live(now()?, self.config.task_deadline.as_unix_millis())?;
        }
        let approval = self.verify_run_approval()?;
        graph.verify_transport(client, &approval)?;
        ensure!(
            self.config.task_deadline.as_unix_millis() > now()?,
            "original loop deadline elapsed"
        );
        if let Some(previous) = &self.config.predecessor_run {
            let proof: Option<lineage::LineageSuccessorEvidence> =
                if self.config.schema == warm_start::LOOP_V3 {
                    let proof: lineage::LineageSuccessorEvidence =
                        native_genome::load(&self.root, previous)?;
                    ensure!(
                        proof.schema == lineage::SUCCESSOR_EVIDENCE_SCHEMA,
                        "unknown lineage successor adapter"
                    );
                    Some(proof)
                } else {
                    None
                };
            let completed: RunCompletion = native_genome::load(
                &self.root,
                proof.as_ref().map_or(previous, |p| &p.completion),
            )?;
            let previous_config: Config = native_genome::load(&self.root, &completed.config)?;
            let prior: Option<lineage::LineageSuccessorEvidence> =
                if previous_config.schema == warm_start::LOOP_V3 {
                    let p: lineage::LineageSuccessorEvidence = native_genome::load(
                        &self.root,
                        previous_config
                            .predecessor_run
                            .as_ref()
                            .context("previous lineage adapter absent")?,
                    )?;
                    ensure!(
                        p.schema == lineage::SUCCESSOR_EVIDENCE_SCHEMA,
                        "unknown prior lineage adapter"
                    );
                    Some(p)
                } else {
                    None
                };
            let prior_verified = prior.as_ref().map(|p| p.bundle.verify()).transpose()?;
            let verified_lineage = verify_versioned_successor(
                &self.config,
                &previous_config,
                &completed,
                proof.as_ref().map(|p| &p.bundle),
                prior_verified.as_ref(),
            )?;
            self.verify_completed_predecessor(graph, client, &completed, verified_lineage.as_ref())
                .await?;
        }
        let mut p = Progress::default();
        let mut checkpoint = self.config.accepted_checkpoint.clone();
        let mut corpus = self.config.initial_corpus.clone();
        let mut parent_predecessors = BTreeMap::new();
        let mut parent_acceptance = self.config.accepted_decision.clone();
        let mut all_targets = Vec::new();
        for generation in 0..=self.config.cycles {
            let mut proposals = Vec::new();
            for (arm, role) in [
                (Arm::Model, NativeLoopRole::ProposalModel),
                (Arm::Shuffled, NativeLoopRole::ProposalShuffled),
            ] {
                let id = format!(
                    "g{generation}-proposal-{}",
                    if arm == Arm::Model {
                        "model"
                    } else {
                        "shuffled"
                    }
                );
                let seed = generation_seed(&self.config.run, generation, &checkpoint.sha256)?;
                let (_,r)=self.request(generation,id,role,json!({"checkpoint":checkpoint,"corpus":corpus,"proposal_seed":seed,"scientific_policy":self.config.scientific_policy,"accepted_parent_decision":parent_acceptance}),parent_predecessors.clone())?;
                if let Some(result) = self.result(graph, client, &r).await? {
                    proposals.push((arm, result));
                } else {
                    self.offer(graph, client, &r, &mut p).await?;
                    p.waiting_execution += 1;
                }
            }
            if proposals.len() != 2 {
                break;
            }
            if generation == self.config.cycles {
                // A v2 run's window is its fresh optimizer's (`warm_start`), under its own
                // completion schema, which no successor accepts.
                let completed = RunCompletion {
                    schema: match self.config.schema.as_str() {
                        warm_start::LOOP_V2 => warm_start::COMPLETION_V2,
                        warm_start::LOOP_V3 => warm_start::COMPLETION_V3,
                        _ => "jope.native_loop_completion.v1",
                    }
                    .into(),
                    run: self.config.run.clone(),
                    config: artifact(&self.root, &self.state.join("config.json"))?,
                    initial_optimizer_step: self.config.train_window(0)?.initial,
                    target_optimizer_step: self.config.train_window(1)?.target,
                    checkpoint: checkpoint.clone(),
                    corpus: corpus.clone(),
                    accepted_parent_decision: parent_acceptance.clone(),
                    final_proposals: proposals
                        .iter()
                        .map(|(arm, (a, _))| (*arm, a.clone()))
                        .collect(),
                };
                frozen(&self.root, &self.state.join("completion.json"), &completed)?;
                p.complete = true;
                break;
            }
            let mut pending_direct = Vec::new();
            let mut pending_continuation = Vec::new();
            let mut inflight_direct = 0usize;
            let mut inflight_continuation = 0usize;
            let mut targets = Vec::new();
            let mut target_predecessors = BTreeMap::new();
            let mut expected_targets = 0usize;
            let mut refused = BTreeMap::new();
            let mut comparison_candidates = Vec::new();
            let vocabulary = artifact(
                &self.root,
                &self
                    .root
                    .join(corpus.path.parent().context("corpus parent")?)
                    .join("vocabulary.json"),
            )?;
            for (arm, (proposal_ref, proposal_result)) in &proposals {
                for candidate in self.candidates(
                    generation,
                    *arm,
                    proposal_ref,
                    proposal_result,
                    &checkpoint,
                    &corpus,
                )? {
                    comparison_candidates.push(candidate.clone());
                    expected_targets += 1;
                    let predecessors =
                        BTreeMap::from([(proposal_result.task_id.clone(), proposal_ref.clone())]);
                    let direct = self.freeze_genome(
                        generation,
                        &candidate,
                        None,
                        predecessors,
                        vocabulary.clone(),
                    )?;
                    let Some((direct_ref, direct_result)) =
                        self.result(graph, client, &direct).await?
                    else {
                        if let Some(failure) = self.failed(graph, client, &direct).await? {
                            refused.insert(candidate.sha256.clone(), failure);
                            continue;
                        }
                        if self.submitted(&direct)? {
                            inflight_direct += 1
                        } else {
                            pending_direct.push(direct);
                        }
                        continue;
                    };
                    let continuation = self.freeze_genome(
                        generation,
                        &candidate,
                        Some(direct_result.output),
                        BTreeMap::from([(direct_result.task_id, direct_ref)]),
                        vocabulary.clone(),
                    )?;
                    if let Some((result_ref, result)) =
                        self.result(graph, client, &continuation).await?
                    {
                        let target = native_genome::target(&self.root, &result.output)?;
                        let a = frozen(
                            &self.root,
                            &self
                                .state
                                .join("targets")
                                .join(format!("{}.json", hash(&serde_json::to_vec(&target)?))),
                            &target,
                        )?;
                        targets.push(a);
                        target_predecessors.insert(result.task_id, result_ref);
                    } else if let Some(failure) = self.failed(graph, client, &continuation).await? {
                        refused.insert(candidate.sha256.clone(), failure);
                    } else if self.submitted(&continuation)? {
                        inflight_continuation += 1
                    } else {
                        pending_continuation.push(continuation);
                    }
                }
            }
            let comparisons = matched_comparisons(&self.root, generation, &comparison_candidates)?;
            frozen(
                &self.root,
                &self
                    .state
                    .join("comparisons")
                    .join(format!("g{generation}.json")),
                &comparisons,
            )?;
            p.pending_direct = pending_direct.len() + inflight_direct;
            p.pending_continuation = pending_continuation.len() + inflight_continuation;
            let selected = replenish(
                self.config.workers,
                inflight_direct,
                inflight_continuation,
                pending_direct,
                pending_continuation,
            );
            for request in selected {
                self.offer(graph, client, &request, &mut p).await?;
            }
            if targets.len() + refused.len() != expected_targets {
                p.idle_reasons
                    .insert("awaiting_original_direct_and_matched_continuation_outcomes".into());
                break;
            }
            let refusal_ref = frozen(
                &self.root,
                &self
                    .state
                    .join("refusals")
                    .join(format!("g{generation}.json")),
                &json!({"schema":"jope.native_generation_refusals.v1","run":self.config.run,"generation":generation,
                    "refusals":refused,"original_failure_receipts_retained":true}),
            )?;
            let comparison = frozen(
                &self.root,
                &self
                    .state
                    .join("comparisons")
                    .join(format!("g{generation}-outcomes.json")),
                &comparison_outcomes(
                    &self.root,
                    generation,
                    &targets,
                    &comparison_candidates,
                    &refusal_ref,
                )?,
            )?;
            all_targets.extend(targets);
            all_targets.sort_by(|a, b| a.sha256.cmp(&b.sha256));
            all_targets.dedup();
            let set = crate::continuation_corpus::TargetSet {
                preparation_overlay: self.config.preparation_overlay.clone(),
                schema: "jope.continuation_target_set.v1".into(),
                run: self.config.run.clone(),
                generation,
                source_graph_state: self.graph_state.strip_prefix(&self.root)?.into(),
                qualification: self.config.qualification,
                parent_corpus: corpus.clone(),
                vocabulary,
                admission_policy: self.config.label_admission_policy.clone(),
                frozen_scientific_policy: self.config.scientific_policy.clone(),
                targets: all_targets.clone(),
                training_arms: self.policy.training_arms.clone(),
            };
            let set_ref = frozen(
                &self.root,
                &self
                    .state
                    .join("corpora")
                    .join(format!("g{generation}-targets.json")),
                &set,
            )?;
            let mut source_closure =
                crate::continuation_corpus::input_closure(&self.root, &set_ref, &self.graph_state)?;
            source_closure.insert(refusal_ref.path.clone(), refusal_ref.sha256.clone());
            source_closure.insert(comparison.path.clone(), comparison.sha256.clone());
            for failure in refused.values() {
                source_closure.insert(failure.path.clone(), failure.sha256.clone());
            }
            let mut closure_bytes = Vec::new();
            for (path, sha256) in source_closure {
                serde_json::to_writer(&mut closure_bytes, &Artifact { path, sha256 })?;
                closure_bytes.push(b'\n');
            }
            let closure_ref = immutable(
                &self.root,
                &self
                    .state
                    .join("corpora")
                    .join(format!("g{generation}-source-closure.jsonl")),
                &closure_bytes,
            )?;
            let Some(target_predecessors) = self
                .aggregate(graph, client, generation, target_predecessors, &mut p)
                .await?
            else {
                break;
            };
            let (_, pack) = self.request(
                generation,
                format!("g{generation}-pack-and-publish"),
                NativeLoopRole::PackAndPublish,
                json!({"target_set":set_ref,"previous_corpus":corpus,"closure":closure_ref}),
                target_predecessors,
            )?;
            let Some((pack_ref, pack_result)) = self.result(graph, client, &pack).await? else {
                self.offer(graph, client, &pack, &mut p).await?;
                p.waiting_execution += 1;
                break;
            };
            let publication: Value =
                native_genome::load_dataset_receipt(&self.root, &pack_result.output)?;
            ensure!(
                publication["local_corpus_committed"] == true,
                "changed corpus not locally committed before training"
            );
            if self.policy.require_remote_publication {
                ensure!(
                    publication["remote_verified"] == true,
                    "changed corpus remote restoration not verified"
                );
            }
            let next_corpus: Artifact = serde_json::from_value(publication["corpus_index"].clone())
                .context("published corpus exact index receipt absent")?;
            let _: Value = native_genome::load_corpus_index(&self.root, &next_corpus)?;
            ensure!(
                next_corpus.sha256 != corpus.sha256,
                "training requires a changed committed corpus"
            );
            let predecessors = BTreeMap::from([(pack_result.task_id, pack_ref)]);
            // v1 parameters keep their exact historical bytes; a v2 warm start adds its mode.
            let window = self.config.train_window(generation)?;
            let (_, train) = self.request(
                generation,
                format!("g{generation}-train"),
                NativeLoopRole::Train,
                warm_start::train_parameters(&checkpoint, &next_corpus, &window)?,
                predecessors,
            )?;
            let Some((train_ref, train_result)) = self.result(graph, client, &train).await? else {
                self.offer(graph, client, &train, &mut p).await?;
                p.waiting_execution += 1;
                break;
            };
            let output: Value = native_genome::load(&self.root, &train_result.output)?;
            let cp = PathBuf::from(
                output["latest"]["checkpoint"]
                    .as_str()
                    .context("trained checkpoint path absent")?,
            );
            let candidate_checkpoint =
                artifact(&self.root, &self.root.join(cp).join("state.json"))?;
            let state: Value = native_genome::load(&self.root, &candidate_checkpoint)?;
            ensure!(
                state["step"] == window.target && state["corpus_sha256"] == next_corpus.sha256,
                "optimizer window or changed corpus differs"
            );
            let decision_path = self
                .state
                .join("parent-decisions")
                .join(format!("g{generation}.json"));
            if !decision_path.exists() {
                p.idle_reasons
                    .insert("awaiting_independent_proposal_parent_quality_decision".into());
                break;
            }
            let decision_ref: Artifact = serde_json::from_slice(&read(&decision_path)?)?;
            let decision: ProposalParentDecision = native_genome::load(&self.root, &decision_ref)?;
            let assessment: Value = native_genome::load(&self.root, &decision.assessment)?;
            verify_parent_decision(
                &decision,
                &assessment,
                &self.config.run,
                generation,
                crate::native_loop::ParentEvidence {
                    cp: &candidate_checkpoint,
                    previous: &checkpoint,
                    train: &train_ref,
                    comparison: &comparison,
                },
            )?;
            if !decision.accepted {
                p.idle_reasons
                    .insert("trained_checkpoint_rejected_as_proposal_parent".into());
                break;
            }
            checkpoint = candidate_checkpoint;
            parent_acceptance = decision_ref;
            corpus = next_corpus;
            parent_predecessors = BTreeMap::from([(train_result.task_id, train_ref)]);
            p.completed_transitions += 1;
        }
        let status = json!({"schema":"jope.native_loop_health.v1","timestamp_unix_ms":now()?,"stale_after_ms":30000,"progress":p,"model_improvement_claimed":false});
        crate::manifest::atomic_write(
            &self.state.join("health.json"),
            &serde_json::to_vec(&status)?,
        )?;
        Ok(p)
    }
    pub(super) fn submitted(&self, r: &Request) -> Result<bool> {
        let key = identity(&r.run, r.generation, &r.id)?;
        Ok(self
            .graph_state
            .join("tasks")
            .join(key)
            .join("submitted.json")
            .exists())
    }
}
