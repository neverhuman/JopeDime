use super::super::decisions::comparison_outcomes;
use super::*;

impl Loop {
    pub(crate) fn verify_direct_budget(
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
    pub(crate) async fn prove_comparison(
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
