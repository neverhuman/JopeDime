use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CampaignLayerV1 {
    Development,
    Confirmation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestAccessPolicyV1 {
    SearchTrainMetadataOnlySealedFinalizerOwnsTest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchDatasetBindingV1 {
    pub dataset_id: String,
    pub dataset_sha256: String,
    pub train_sha256: String,
    pub metadata_sha256: String,
}

impl SearchDatasetBindingV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.dataset_id.len() != 16 || !self.dataset_sha256.starts_with(&self.dataset_id) {
            return Err("search dataset ID does not prefix its full digest".into());
        }
        for (name, digest) in [
            ("dataset", self.dataset_sha256.as_str()),
            ("training", self.train_sha256.as_str()),
            ("metadata", self.metadata_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetFirstCampaignFreezeV1 {
    pub schema: String,
    pub campaign_id: String,
    pub development: Vec<SearchDatasetBindingV1>,
    pub confirmation: Vec<SearchDatasetBindingV1>,
    pub comparator_snapshot_sha256: String,
    pub code_revision: String,
    pub source_tree_sha256: String,
    pub search_binary_sha256: String,
    pub evaluator_sha256: String,
    pub proposal_checkpoint_sha256: String,
    pub proposal_inference_protocol_sha256: String,
    pub metric_sha256: String,
    pub budget_rungs: Vec<u32>,
    pub development_seeds: Vec<u64>,
    pub confirmation_seeds: Vec<u64>,
    pub test_access_policy: TestAccessPolicyV1,
    pub license_tracks: BTreeSet<LicenseTrackV1>,
}

impl BudgetFirstCampaignFreezeV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBudgetFirstCampaignFreeze.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.campaign_id != BUDGET_CAMPAIGN_ID {
            return Err("budget-first campaign schema or ID drift".into());
        }
        for (name, digest) in [
            (
                "comparator snapshot",
                self.comparator_snapshot_sha256.as_str(),
            ),
            ("source tree", self.source_tree_sha256.as_str()),
            ("search binary", self.search_binary_sha256.as_str()),
            ("evaluator", self.evaluator_sha256.as_str()),
            (
                "proposal checkpoint",
                self.proposal_checkpoint_sha256.as_str(),
            ),
            (
                "proposal inference protocol",
                self.proposal_inference_protocol_sha256.as_str(),
            ),
            ("metric", self.metric_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        if self.code_revision.trim().is_empty()
            || self.proposal_checkpoint_sha256 != FROZEN_CHECKPOINT_SHA256
            || self.budget_rungs != BUDGET_RUNGS
            || self.development_seeds != SEARCH_SEEDS
            || self.confirmation_seeds != CONFIRMATION_SEEDS
            || self.license_tracks
                != BTreeSet::from([
                    LicenseTrackV1::ProductionCompatible,
                    LicenseTrackV1::ResearchOnly,
                ])
        {
            return Err(
                "budget-first frozen code, checkpoint, seeds, rungs, or tracks drifted".into(),
            );
        }
        validate_layer(&self.development, &DEVELOPMENT_DATASET_IDS)?;
        validate_layer(&self.confirmation, &CONFIRMATION_DATASET_IDS)?;
        let development = self
            .development
            .iter()
            .map(|dataset| dataset.dataset_sha256.as_str())
            .collect::<BTreeSet<_>>();
        if self
            .confirmation
            .iter()
            .any(|dataset| development.contains(dataset.dataset_sha256.as_str()))
        {
            return Err("development and confirmation cohorts overlap".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

fn validate_layer(bindings: &[SearchDatasetBindingV1], expected: &[&str]) -> Result<(), String> {
    for binding in bindings {
        binding.validate()?;
    }
    let actual = bindings
        .iter()
        .map(|binding| binding.dataset_id.as_str())
        .collect::<BTreeSet<_>>();
    if bindings.len() != expected.len() || actual != expected.iter().copied().collect() {
        return Err("campaign cohort differs from its frozen dataset set".into());
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedTestBindingV1 {
    pub dataset_sha256: String,
    pub test_sha256: String,
    pub test_relative_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedTestInventoryV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub sealed_finalizer_sha256: String,
    pub dataset_claim_registry_sha256: String,
    pub tests: Vec<SealedTestBindingV1>,
}

impl SealedTestInventoryV1 {
    pub const SCHEMA: &'static str = "JopeDlgpSealedTestInventory.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.tests.len() != 42 {
            return Err("sealed test inventory schema or cardinality drift".into());
        }
        for digest in [
            &self.campaign_freeze_sha256,
            &self.sealed_finalizer_sha256,
            &self.dataset_claim_registry_sha256,
        ] {
            validate_sha256("sealed test inventory", digest)?;
        }
        let mut datasets = BTreeSet::new();
        for test in &self.tests {
            validate_sha256("sealed dataset", &test.dataset_sha256)?;
            validate_sha256("sealed test", &test.test_sha256)?;
            let path = Path::new(&test.test_relative_path);
            if path.is_absolute()
                || path.as_os_str().is_empty()
                || path.components().any(|part| part == Component::ParentDir)
                || !datasets.insert(test.dataset_sha256.as_str())
            {
                return Err("sealed test binding is unsafe or duplicated".into());
            }
        }
        Ok(())
    }
}
