use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CohortSupplyStage {
    FreezeCohort,
    SubmitCohortProposer,
    FetchValidateProposals,
    CacheFilterResample,
    BuildWave,
    KeeperAcknowledgment,
}

impl CohortSupplyStage {
    fn ordinal(self) -> u8 {
        match self {
            Self::FreezeCohort => 0,
            Self::SubmitCohortProposer => 1,
            Self::FetchValidateProposals => 2,
            Self::CacheFilterResample => 3,
            Self::BuildWave => 4,
            Self::KeeperAcknowledgment => 5,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CohortSupplyStateV1 {
    pub schema: String,
    pub task: Task,
    pub accepted_checkpoint_sha256: Sha256Digest,
    pub cohort_sha256: Sha256Digest,
    pub catalog_sha256: Sha256Digest,
    pub decoder_configuration_sha256: Sha256Digest,
    pub seed: u64,
    pub stage: CohortSupplyStage,
    pub revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposal_job_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposal_pointer_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wave_id: Option<String>,
}

impl CohortSupplyStateV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        task: Task,
        accepted_checkpoint_sha256: Sha256Digest,
        cohort_sha256: Sha256Digest,
        catalog_sha256: Sha256Digest,
        decoder_configuration_sha256: Sha256Digest,
        seed: u64,
    ) -> Self {
        Self {
            schema: SUPPLY_STATE_SCHEMA.to_owned(),
            task,
            accepted_checkpoint_sha256,
            cohort_sha256,
            catalog_sha256,
            decoder_configuration_sha256,
            seed,
            stage: CohortSupplyStage::FreezeCohort,
            revision: 0,
            proposal_job_id: None,
            proposal_pointer_sha256: None,
            wave_id: None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SUPPLY_STATE_SCHEMA,
            "cohort supply state schema is not v1"
        );
        if self.stage >= CohortSupplyStage::SubmitCohortProposer {
            ensure!(
                self.proposal_job_id
                    .as_deref()
                    .is_some_and(|value| !value.trim().is_empty()),
                "submitted supply state is missing its proposer job identity"
            );
        }
        if self.stage >= CohortSupplyStage::FetchValidateProposals {
            ensure!(
                self.proposal_pointer_sha256.is_some(),
                "validated supply state is missing its proposal pointer"
            );
        }
        if self.stage >= CohortSupplyStage::BuildWave {
            ensure!(
                self.wave_id
                    .as_deref()
                    .is_some_and(|value| !value.trim().is_empty()),
                "built supply state is missing its wave identity"
            );
        }
        Ok(())
    }

    pub fn advance(
        &mut self,
        next: CohortSupplyStage,
        proposal_job_id: Option<String>,
        proposal_pointer_sha256: Option<Sha256Digest>,
        wave_id: Option<String>,
    ) -> Result<bool> {
        ensure!(
            next.ordinal() == self.stage.ordinal() || next.ordinal() == self.stage.ordinal() + 1,
            "cohort supply stage must advance exactly once"
        );
        set_once(&mut self.proposal_job_id, proposal_job_id, "proposal job")?;
        set_once(
            &mut self.proposal_pointer_sha256,
            proposal_pointer_sha256,
            "proposal pointer",
        )?;
        set_once(&mut self.wave_id, wave_id, "wave")?;
        if next == self.stage {
            self.validate()?;
            return Ok(false);
        }
        self.stage = next;
        self.revision += 1;
        self.validate()?;
        Ok(true)
    }
}

fn set_once<T: Eq>(slot: &mut Option<T>, value: Option<T>, label: &str) -> Result<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if let Some(existing) = slot {
        ensure!(existing == &value, "{label} identity cannot change");
    } else {
        *slot = Some(value);
    }
    Ok(())
}
