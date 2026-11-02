use super::*;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CohortProposalTaskScope {
    pub task: Task,
    pub accepted_checkpoint_sha256: Sha256Digest,
    pub context_count: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CohortProposalPointerV1 {
    pub schema: String,
    pub task_scopes: Vec<CohortProposalTaskScope>,
    pub cohort_sha256: Sha256Digest,
    pub catalog_sha256: Sha256Digest,
    pub decoder_configuration_sha256: Sha256Digest,
    pub seed: u64,
    pub proposal_sha256: Sha256Digest,
    pub proposal_job_id: String,
}

impl CohortProposalPointerV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == COHORT_PROPOSAL_POINTER_SCHEMA,
            "cohort proposal pointer schema is not v1"
        );
        ensure!(
            !self.proposal_job_id.trim().is_empty(),
            "proposal job identity is empty"
        );
        ensure!(
            matches!(self.task_scopes.len(), 1 | 2),
            "a proposal pointer must scope one or two task cohorts"
        );
        let mut tasks = self
            .task_scopes
            .iter()
            .map(|scope| scope.task)
            .collect::<Vec<_>>();
        tasks.sort_unstable();
        tasks.dedup();
        ensure!(
            tasks.len() == self.task_scopes.len(),
            "proposal pointer contains a duplicate task scope"
        );
        ensure!(
            self.task_scopes
                .iter()
                .all(|scope| scope.context_count == super::DATASETS_PER_COHORT),
            "each proposal task scope must contain exactly 64 contexts"
        );
        if self.task_scopes.len() == 2 {
            ensure!(
                self.task_scopes[0].accepted_checkpoint_sha256
                    == self.task_scopes[1].accepted_checkpoint_sha256,
                "task cohorts may share a proposal job only while their accepted checkpoints match"
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_scope(
        &self,
        task: Task,
        accepted_checkpoint_sha256: &Sha256Digest,
        cohort_sha256: &Sha256Digest,
        catalog_sha256: &Sha256Digest,
        decoder_configuration_sha256: &Sha256Digest,
        seed: u64,
    ) -> Result<()> {
        self.validate()?;
        ensure!(
            &self.cohort_sha256 == cohort_sha256
                && &self.catalog_sha256 == catalog_sha256
                && &self.decoder_configuration_sha256 == decoder_configuration_sha256
                && self.seed == seed,
            "proposal pointer immutable cohort scope mismatch"
        );
        let scope = self
            .task_scopes
            .iter()
            .find(|scope| scope.task == task)
            .context("proposal pointer does not contain the requested task")?;
        ensure!(
            &scope.accepted_checkpoint_sha256 == accepted_checkpoint_sha256,
            "proposal pointer accepted checkpoint mismatch"
        );
        Ok(())
    }

    pub fn id(&self) -> Result<Sha256Digest> {
        self.validate()?;
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(self)?))
    }
}
