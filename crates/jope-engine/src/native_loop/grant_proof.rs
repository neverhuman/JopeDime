//! Read-only scientific eligibility proof for an independently owned provisioner.
use super::*;
use jain_smartcluster::{ProjectId, TenantId};
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrantProofInputs {
    pub root: PathBuf,
    pub loop_state: PathBuf,
    pub graph_state: PathBuf,
    pub worker_state: PathBuf,
    pub tenant: String,
    pub project: String,
    pub config: Artifact,
    pub request: Artifact,
    pub admission: Artifact,
    pub task_manifest: Artifact,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeGrantEligibilityProof {
    pub schema: String,
    pub inputs: GrantProofInputs,
    pub run: String,
    pub generation: u64,
    pub role: Role,
    pub scientific_policy: Artifact,
    pub candidate: Artifact,
    pub accepted_parent: Artifact,
    pub original_proposal: Artifact,
    pub evaluations: usize,
    pub seconds: u64,
    pub task_deadline: Timestamp,
    pub task_manifest_sha256: String,
    pub predecessor_receipts: BTreeMap<String, Artifact>,
    pub verifier_executable_sha256: String,
    pub verified_unix_ms: u64,
    pub authorization_issued: bool,
}
impl Loop {
    /// Does not materialize, claim, admit, dispatch, sign, or acquire a writer lock.
    pub async fn prove_native_grant_request(
        inputs: &GrantProofInputs,
        client: &Client,
    ) -> Result<NativeGrantEligibilityProof> {
        let controller = Self::open_mode(
            &inputs.root,
            &inputs.loop_state,
            &inputs.graph_state,
            &inputs.config,
            true,
        )?;
        let approval = controller.verify_run_approval()?;
        let graph = Graph::read_only(
            &inputs.root,
            &inputs.graph_state,
            &inputs.worker_state,
            TenantId::new(inputs.tenant.clone())?,
            ProjectId::new(inputs.project.clone())?,
        )?;
        graph.verify_transport(client, &approval)?;
        let request: Request = native_genome::load(&inputs.root, &inputs.request)?;
        ensure!(
            matches!(request.role, Role::Direct | Role::Continuation)
                && request.generation < controller.config.cycles,
            "only eligible direct or continuation tasks can request a provisioned grant"
        );
        let binding = controller
            .binding(&request)?
            .context("original loop binding absent")?;
        ensure!(
            binding.request == inputs.request
                && binding.admission.manifest == inputs.task_manifest
                && serde_json::to_vec(&binding.admission)?
                    == native_genome::checked(&inputs.root, &inputs.admission, 1024 * 1024)?,
            "provisioner inputs differ from exact original loop binding/admission"
        );
        let task = controller.validate_binding(&request, &binding)?;
        task.validate_at(Timestamp::from(now()?))?;
        if let Some(a) = &controller.config.preparation_overlay {
            crate::preparation_overlay::load(&controller.root, a, &controller.config.run)?
                .require_live(now()?, task.deadline.as_unix_millis())?;
        }
        for (id, reference) in &request.predecessors {
            let result = graph.verified(client, reference).await?;
            ensure!(
                result.task_id == *id
                    && result.run == request.run
                    && result.generation <= request.generation,
                "provisioner predecessor identity changed"
            );
        }
        let numerical_ref: Artifact =
            serde_json::from_value(request.parameters["genome_task"].clone())?;
        let numerical: GenomeTask = native_genome::load(&inputs.root, &numerical_ref)?;
        let candidate = numerical.candidate(&inputs.root)?;
        let expected_stage = if request.role == Role::Direct {
            Stage::Direct
        } else {
            Stage::Continuation
        };
        let expected_id = format!(
            "g{}-{}-{}",
            request.generation,
            if expected_stage == Stage::Direct {
                "direct"
            } else {
                "continuation"
            },
            numerical.candidate.sha256
        );
        ensure!(
            request.id == expected_id
                && candidate.run == controller.config.run
                && candidate.generation == request.generation
                && numerical.stage == expected_stage
                && numerical.admission_policy == controller.config.label_admission_policy
                && numerical.qualification == controller.config.qualification
                && numerical.preparation_overlay == controller.config.preparation_overlay
                && !candidate.context.partition.eq("meta_test")
                && !controller
                    .policy
                    .excluded_families
                    .contains(&candidate.context.family)
                && candidate.context.cut >= controller.config.cut_start
                && candidate.context.cut
                    < controller
                        .config
                        .cut_start
                        .checked_add(controller.config.cuts)
                        .context("cut range overflow")?
                && numerical.evaluations
                    == if expected_stage == Stage::Direct {
                        1
                    } else {
                        controller.config.continuation_evaluations
                    }
                && numerical.seconds
                    == if expected_stage == Stage::Direct {
                        controller.config.direct_seconds
                    } else {
                        controller.config.continuation_seconds
                    }
                && numerical.rng_seed
                    == generation_seed(
                        &candidate.run,
                        candidate.generation,
                        &format!(
                            "{}:{}:{}",
                            candidate.context.prepared_sha256,
                            candidate.context.cut,
                            candidate.matched_slot
                        )
                    )?,
            "native genome differs from frozen scientific identity, context, seed, or budget"
        );
        ensure!(
            numerical.output
                == controller
                    .state
                    .join("numerical")
                    .join(&expected_id)
                    .join("output")
                    .strip_prefix(&inputs.root)?,
            "native numerical output identity changed"
        );
        let proposal = graph.verified(client, &candidate.proposal).await?;
        let proposal_binding_path = controller.state.join("bindings").join(format!(
            "{}.json",
            identity(&candidate.run, candidate.generation, &proposal.task_id)?
        ));
        let proposal_binding: Binding = serde_json::from_slice(&read(&proposal_binding_path)?)?;
        let proposal_request: Request =
            native_genome::load(&inputs.root, &proposal_binding.request)?;
        controller.validate_binding(&proposal_request, &proposal_binding)?;
        verify_result_binding(&proposal_request, &proposal_binding.admission, &proposal)?;
        let arm = if candidate.arm == Arm::Generic {
            Arm::Model
        } else {
            candidate.arm
        };
        ensure!(
            proposal_request.id
                == format!(
                    "g{}-proposal-{}",
                    candidate.generation,
                    if arm == Arm::Model {
                        "model"
                    } else {
                        "shuffled"
                    }
                )
                && proposal_request.role
                    == if arm == Arm::Model {
                        Role::ProposalModel
                    } else {
                        Role::ProposalShuffled
                    }
                && proposal_request.parameters["proposal_seed"]
                    == generation_seed(
                        &candidate.run,
                        candidate.generation,
                        &candidate.parent_checkpoint.sha256
                    )?
                && proposal_request.parameters["checkpoint"]
                    == serde_json::to_value(&candidate.parent_checkpoint)?,
            "candidate original proposal parent/arm/generation seed changed"
        );
        let corpus: Artifact =
            serde_json::from_value(proposal_request.parameters["corpus"].clone())?;
        let vocabulary = artifact(
            &inputs.root,
            &inputs
                .root
                .join(corpus.path.parent().context("proposal corpus parent")?)
                .join("vocabulary.json"),
        )?;
        ensure!(
            numerical.vocabulary == vocabulary,
            "native genome vocabulary differs from original corpus"
        );
        let accepted: Artifact = serde_json::from_value(
            proposal_request.parameters["accepted_parent_decision"].clone(),
        )?;
        if request.generation == 0 {
            ensure!(
                candidate.parent_checkpoint == controller.config.accepted_checkpoint
                    && corpus == controller.config.initial_corpus
                    && accepted == controller.config.accepted_decision,
                "initial proposal parent/corpus differs from owner-approved config"
            );
        } else {
            let original: Artifact = serde_json::from_slice(&read(
                &controller
                    .state
                    .join("parent-decisions")
                    .join(format!("g{}.json", request.generation - 1)),
            )?)?;
            ensure!(
                accepted == original,
                "proposal parent acceptance differs from frozen decision"
            );
            let decision: ProposalParentDecision = native_genome::load(&inputs.root, &accepted)?;
            let assessment: Value = native_genome::load(&inputs.root, &decision.assessment)?;
            let prior_generation = request.generation - 1;
            let train_request = controller
                .original_request(prior_generation, &format!("g{prior_generation}-train"))?;
            ensure!(
                train_request.role == Role::Train
                    && prior_generation == 0
                    && train_request.parameters["checkpoint"]
                        == serde_json::to_value(&controller.config.accepted_checkpoint)?
                    && train_request.parameters["initial_step"]
                        == controller.config.initial_optimizer_step
                    && train_request.parameters["requested_increment"] == 3000
                    && train_request.parameters["target_step"]
                        == controller.config.initial_optimizer_step + 3000
                    && train_request.parameters["global_batch"] == 2048,
                "original parent training differs from frozen accepted optimizer window"
            );
            let pack_request = controller.original_request(
                prior_generation,
                &format!("g{prior_generation}-pack-and-publish"),
            )?;
            ensure!(
                pack_request.role == Role::PackAndPublish
                    && pack_request.parameters["previous_corpus"]
                        == serde_json::to_value(&controller.config.initial_corpus)?,
                "original parent corpus publication differs from initial corpus"
            );
            let (pack_ref, pack_result) =
                controller
                    .result(&graph, client, &pack_request)
                    .await?
                    .context("original parent corpus publication is not independently verified")?;
            let publication =
                native_genome::load_dataset_receipt(&inputs.root, &pack_result.output)?;
            ensure!(
                publication["local_corpus_committed"] == true
                    && (!controller.policy.require_remote_publication
                        || publication["remote_verified"] == true)
                    && publication["corpus_index"] == serde_json::to_value(&corpus)?
                    && train_request.parameters["corpus"] == publication["corpus_index"]
                    && train_request.predecessors
                        == BTreeMap::from([(pack_result.task_id, pack_ref)]),
                "training parent lacks the exact original committed corpus publication edge"
            );
            let (train_ref, training) = controller
                .result(&graph, client, &train_request)
                .await?
                .context("original loop training result is not verified")?;
            let numerical: Value = native_genome::load(&inputs.root, &training.output)?;
            let checkpoint_path = PathBuf::from(
                numerical["latest"]["checkpoint"]
                    .as_str()
                    .context("original trained checkpoint absent")?,
            );
            let committed_checkpoint = artifact(
                &inputs.root,
                &inputs.root.join(checkpoint_path).join("state.json"),
            )?;
            ensure!(
                committed_checkpoint == candidate.parent_checkpoint
                    && train_request.parameters["corpus"] == serde_json::to_value(&corpus)?,
                "selected parent differs from original bound Train output/corpus"
            );
            let comparison = controller
                .prove_comparison(
                    &graph,
                    client,
                    prior_generation,
                    &controller.config.accepted_checkpoint,
                    &controller.config.initial_corpus,
                    &controller.config.accepted_decision,
                )
                .await?;
            verify_parent_decision(
                &decision,
                &assessment,
                &request.run,
                prior_generation,
                &committed_checkpoint,
                &controller.config.accepted_checkpoint,
                &train_ref,
                &comparison,
            )?;
            ensure!(
                decision.accepted,
                "rejected checkpoint is not an authorized proposal parent"
            );
            let state: Value = native_genome::load(&inputs.root, &candidate.parent_checkpoint)?;
            ensure!(
                state["corpus_sha256"] == corpus.sha256
                    && state["step"]
                        == controller.config.initial_optimizer_step + request.generation * 3000,
                "accepted proposal parent optimizer/corpus binding changed"
            );
        }
        // This invokes the exact existing proposal expansion in read-only mode:
        // original candidate files must exist and match, including generic draws.
        let original_candidates = controller.candidates(
            candidate.generation,
            arm,
            &candidate.proposal,
            &proposal,
            &candidate.parent_checkpoint,
            &corpus,
        )?;
        ensure!(
            original_candidates.contains(&numerical.candidate),
            "candidate was not in original complete proposal expansion"
        );
        if expected_stage == Stage::Direct {
            ensure!(
                numerical.direct_parent.is_none()
                    && request.predecessors
                        == BTreeMap::from([(proposal.task_id, candidate.proposal.clone())]),
                "direct task predecessor differs from original proposal"
            );
        } else {
            ensure!(
                request.predecessors.len() == 1,
                "continuation must have one exact direct predecessor"
            );
            let (id, reference) = request.predecessors.iter().next().unwrap();
            let direct = graph.verified(client, reference).await?;
            ensure!(
                *id == format!(
                    "g{}-direct-{}",
                    request.generation, numerical.candidate.sha256
                ) && direct.generation == request.generation
                    && numerical.direct_parent.as_ref() == Some(&direct.output),
                "continuation original direct result differs"
            );
            let direct_request = controller.original_request(request.generation, id)?;
            ensure!(
                direct_request.role == Role::Direct,
                "continuation parent was not an original direct request"
            );
            let (original_ref, original_result) = controller
                .result(&graph, client, &direct_request)
                .await?
                .context("original direct loop binding/result absent")?;
            ensure!(
                original_ref == *reference && original_result.output == direct.output,
                "continuation direct graph result differs from original loop binding"
            );
            let direct_task_ref: Artifact =
                serde_json::from_value(direct_request.parameters["genome_task"].clone())?;
            let direct_task: GenomeTask = native_genome::load(&inputs.root, &direct_task_ref)?;
            controller.verify_direct_budget(
                &direct_request,
                &direct_task,
                &numerical.candidate,
                &candidate,
            )?;
            native_genome::direct_parent(&inputs.root, &numerical, &candidate)?;
        }
        // Re-read original frozen inputs after all potentially long verification.
        ensure!(
            serde_json::to_vec(&request)?
                == native_genome::checked(&inputs.root, &inputs.request, 1024 * 1024)?
                && serde_json::to_vec(&binding.admission)?
                    == native_genome::checked(&inputs.root, &inputs.admission, 1024 * 1024)?,
            "grant inputs changed during read-only verification"
        );
        task.validate_at(Timestamp::from(now()?))?;
        if let Some(a) = &controller.config.preparation_overlay {
            crate::preparation_overlay::load(&controller.root, a, &controller.config.run)?
                .require_live(now()?, task.deadline.as_unix_millis())?;
        }
        Ok(NativeGrantEligibilityProof {
            schema: "jope.native_grant_eligibility_proof.v1".into(),
            inputs: inputs.clone(),
            run: request.run,
            generation: request.generation,
            role: request.role,
            scientific_policy: controller.config.scientific_policy,
            candidate: numerical.candidate,
            accepted_parent: candidate.parent_checkpoint,
            original_proposal: candidate.proposal,
            evaluations: numerical.evaluations,
            seconds: numerical.seconds,
            task_deadline: task.deadline,
            task_manifest_sha256: inputs.task_manifest.sha256.clone(),
            predecessor_receipts: request.predecessors,
            verifier_executable_sha256: crate::manifest::file_hash(Path::new("/proc/self/exe"))?,
            verified_unix_ms: now()?,
            authorization_issued: false,
        })
    }
    pub(super) fn original_request(&self, generation: u64, id: &str) -> Result<Request> {
        let path = self.state.join("requests").join(format!(
            "{}.json",
            identity(&self.config.run, generation, id)?
        ));
        let request: Request = serde_json::from_slice(&read(&path)?)?;
        ensure!(
            request.run == self.config.run && request.generation == generation && request.id == id,
            "original request filename/scope mismatch"
        );
        Ok(request)
    }
    fn verify_direct_budget(
        &self,
        request: &Request,
        task: &GenomeTask,
        reference: &Artifact,
        candidate: &Candidate,
    ) -> Result<()> {
        ensure!(
            request.role == Role::Direct
                && task.stage == Stage::Direct
                && task.candidate == *reference
                && task.direct_parent.is_none()
                && task.evaluations == 1
                && task.seconds == self.config.direct_seconds
                && task.admission_policy == self.config.label_admission_policy
                && task.qualification == self.config.qualification
                && task.preparation_overlay == self.config.preparation_overlay
                && task.rng_seed
                    == generation_seed(
                        &self.config.run,
                        request.generation,
                        &format!(
                            "{}:{}:{}",
                            candidate.context.prepared_sha256,
                            candidate.context.cut,
                            candidate.matched_slot
                        )
                    )?
                && task.output
                    == self
                        .state
                        .join("numerical")
                        .join(&request.id)
                        .join("output")
                        .strip_prefix(&self.root)?,
            "original direct parent budget/seed/output differs from frozen loop"
        );
        Ok(())
    }
    pub(super) async fn prove_comparison(
        &self,
        graph: &Graph,
        client: &Client,
        generation: u64,
        expected_checkpoint: &Artifact,
        expected_corpus: &Artifact,
        expected_acceptance: &Artifact,
    ) -> Result<Artifact> {
        let mut targets = Vec::new();
        let mut candidates = Vec::new();
        let mut refused = BTreeMap::new();
        for arm in [Arm::Model, Arm::Shuffled] {
            let id = format!(
                "g{generation}-proposal-{}",
                if arm == Arm::Model {
                    "model"
                } else {
                    "shuffled"
                }
            );
            let request = self.original_request(generation, &id)?;
            let (reference, result) = self
                .result(graph, client, &request)
                .await?
                .context("comparison proposal not independently verified")?;
            let checkpoint: Artifact =
                serde_json::from_value(request.parameters["checkpoint"].clone())?;
            let corpus: Artifact = serde_json::from_value(request.parameters["corpus"].clone())?;
            ensure!(
                generation < self.config.cycles
                    && checkpoint == *expected_checkpoint
                    && corpus == *expected_corpus
                    && request.parameters["accepted_parent_decision"]
                        == serde_json::to_value(expected_acceptance)?
                    && request.parameters["proposal_seed"]
                        == generation_seed(&self.config.run, generation, &checkpoint.sha256)?,
                "comparison proposal differs from original two-cycle accepted starting parent"
            );
            for candidate_ref in
                self.candidates(generation, arm, &reference, &result, &checkpoint, &corpus)?
            {
                candidates.push(candidate_ref.clone());
                let direct_id = format!("g{generation}-direct-{}", candidate_ref.sha256);
                let direct = self.original_request(generation, &direct_id)?;
                if let Some(failure) = self.failed(graph, client, &direct).await? {
                    refused.insert(candidate_ref.sha256.clone(), failure);
                    continue;
                }
                let id = format!("g{generation}-continuation-{}", candidate_ref.sha256);
                let request = self.original_request(generation, &id)?;
                ensure!(
                    request.role == Role::Continuation,
                    "comparison is not a continuation"
                );
                if let Some(failure) = self.failed(graph, client, &request).await? {
                    refused.insert(candidate_ref.sha256.clone(), failure);
                    continue;
                }
                let (_, result) = self
                    .result(graph, client, &request)
                    .await?
                    .context("comparison continuation not independently verified")?;
                let target = native_genome::target(&self.root, &result.output)?;
                ensure!(
                    target.candidate == candidate_ref,
                    "comparison outcome belongs to a different candidate"
                );
                let bytes = serde_json::to_vec(&target)?;
                let reference = artifact(
                    &self.root,
                    &self
                        .state
                        .join("targets")
                        .join(format!("{}.json", hash(&bytes))),
                )?;
                ensure!(
                    native_genome::checked(&self.root, &reference, 1024 * 1024)? == bytes,
                    "comparison target differs from verified original continuation"
                );
                targets.push(reference);
            }
        }
        let refusal_ref = artifact(
            &self.root,
            &self
                .state
                .join("refusals")
                .join(format!("g{generation}.json")),
        )?;
        let refusal_bytes = serde_json::to_vec(
            &json!({"schema":"jope.native_generation_refusals.v1","run":self.config.run,
            "generation":generation,"refusals":refused,"original_failure_receipts_retained":true}),
        )?;
        ensure!(
            native_genome::checked(&self.root, &refusal_ref, 16 * 1024 * 1024)? == refusal_bytes,
            "recorded refusals differ from authenticated original native failures"
        );
        let bytes = serde_json::to_vec(&comparison_outcomes(
            &self.root,
            generation,
            &targets,
            &candidates,
            &refusal_ref,
        )?)?;
        let reference = artifact(
            &self.root,
            &self
                .state
                .join("comparisons")
                .join(format!("g{generation}-outcomes.json")),
        )?;
        ensure!(
            native_genome::checked(&self.root, &reference, 16 * 1024 * 1024)? == bytes,
            "comparison differs from independently reconstructed complete original outcomes"
        );
        Ok(reference)
    }
}
