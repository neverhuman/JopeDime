use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PanelPoolKindV2 {
    Training,
    Development,
    Promotion,
    Audit,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SealedPanelCandidateV2 {
    pub task: Task,
    pub context_identity: String,
    pub context_sha256: Sha256Digest,
    pub source_family_sha256: Sha256Digest,
    pub near_duplicate_cluster_sha256: Sha256Digest,
    pub origin: DatasetOriginV2,
    pub base_seal_sha256: Sha256Digest,
    pub catalog_sha256: Sha256Digest,
    pub train_sha256: Sha256Digest,
    pub test_sha256: Sha256Digest,
    pub context_facts_sha256: Sha256Digest,
    pub prior_use_index_sha256: Sha256Digest,
    pub zero_prior_use_proof_sha256: Sha256Digest,
    pub admission_watermark: u64,
}

pub fn sealed_zero_prior_use_proof_v2(
    context_sha256: &Sha256Digest,
    base_seal_sha256: &Sha256Digest,
    catalog_sha256: &Sha256Digest,
    prior_use_index_sha256: &Sha256Digest,
    admission_watermark: u64,
) -> Sha256Digest {
    Sha256Digest::of_bytes(
        format!(
            "SealedPanelZeroPriorUseProof.v2\0{}\0{}\0{}\0{}\0{admission_watermark}",
            context_sha256.as_str(),
            base_seal_sha256.as_str(),
            catalog_sha256.as_str(),
            prior_use_index_sha256.as_str(),
        )
        .as_bytes(),
    )
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SealedPanelCandidateSetV2 {
    pub schema: String,
    pub base_seal_sha256: Sha256Digest,
    pub catalog_sha256: Sha256Digest,
    pub prior_use_index_sha256: Sha256Digest,
    pub admission_watermark: u64,
    pub entries: Vec<SealedPanelCandidateV2>,
    pub produced_at_ms: u128,
}

impl SealedPanelCandidateSetV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SEALED_PANEL_CANDIDATE_SET_V2_SCHEMA,
            "sealed panel candidate set schema is not v2"
        );
        ensure!(
            !self.entries.is_empty(),
            "sealed panel candidate set is empty"
        );
        let mut contexts = BTreeSet::new();
        for entry in &self.entries {
            ensure!(
                !entry.context_identity.trim().is_empty() && contexts.insert(&entry.context_sha256),
                "sealed panel candidate repeats or empties a context"
            );
            ensure!(
                entry.base_seal_sha256 == self.base_seal_sha256
                    && entry.catalog_sha256 == self.catalog_sha256
                    && entry.prior_use_index_sha256 == self.prior_use_index_sha256
                    && entry.admission_watermark == self.admission_watermark,
                "sealed panel candidate authority drift"
            );
            ensure!(
                entry.zero_prior_use_proof_sha256
                    == sealed_zero_prior_use_proof_v2(
                        &entry.context_sha256,
                        &entry.base_seal_sha256,
                        &entry.catalog_sha256,
                        &entry.prior_use_index_sha256,
                        entry.admission_watermark,
                    ),
                "sealed panel zero-prior-use proof drift"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SealedPanelEntryV2 {
    pub task: Task,
    pub context_identity: String,
    pub context_sha256: Sha256Digest,
    pub source_family_sha256: Sha256Digest,
    pub near_duplicate_cluster_sha256: Sha256Digest,
    pub origin: DatasetOriginV2,
    pub pool: PanelPoolKindV2,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub promotion_panel: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SealedPanelPoolV2 {
    pub schema: String,
    pub entries: Vec<SealedPanelEntryV2>,
    pub created_at_ms: u128,
}

impl SealedPanelPoolV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SEALED_PANEL_POOL_V2_SCHEMA,
            "sealed panel pool schema is not v2"
        );
        ensure!(!self.entries.is_empty(), "sealed panel pool is empty");
        let mut contexts = BTreeSet::new();
        let mut family_pool = BTreeMap::new();
        let mut cluster_pool = BTreeMap::new();
        let mut family_panel = BTreeMap::new();
        let mut cluster_panel = BTreeMap::new();
        let mut panel_counts: BTreeMap<(Task, &str), (usize, usize)> = BTreeMap::new();
        for entry in &self.entries {
            ensure!(
                !entry.context_identity.trim().is_empty() && contexts.insert(&entry.context_sha256),
                "sealed panel repeats or empties a context"
            );
            if let Some(existing) = family_pool.insert(&entry.source_family_sha256, entry.pool) {
                ensure!(
                    existing == entry.pool,
                    "a source family crosses sealed panel pools"
                );
            }
            if let Some(existing) =
                cluster_pool.insert(&entry.near_duplicate_cluster_sha256, entry.pool)
            {
                ensure!(
                    existing == entry.pool,
                    "a near-duplicate cluster crosses sealed panel pools"
                );
            }
            match entry.pool {
                PanelPoolKindV2::Promotion => {
                    let panel = entry
                        .promotion_panel
                        .as_deref()
                        .context("promotion entry lacks a panel identity")?;
                    let panel_identity = (entry.task, panel);
                    if let Some(existing) =
                        family_panel.insert(&entry.source_family_sha256, panel_identity)
                    {
                        ensure!(
                            existing == panel_identity,
                            "a source family crosses promotion panels"
                        );
                    }
                    if let Some(existing) =
                        cluster_panel.insert(&entry.near_duplicate_cluster_sha256, panel_identity)
                    {
                        ensure!(
                            existing == panel_identity,
                            "a near-duplicate cluster crosses promotion panels"
                        );
                    }
                    let counts = panel_counts.entry((entry.task, panel)).or_default();
                    counts.0 += 1;
                    counts.1 += usize::from(entry.origin == DatasetOriginV2::Real);
                }
                _ => ensure!(
                    entry.promotion_panel.is_none(),
                    "non-promotion entry claims a promotion panel"
                ),
            }
        }
        let total = self.entries.len() as f64;
        let fraction = |pool| {
            self.entries
                .iter()
                .filter(|entry| entry.pool == pool)
                .count() as f64
                / total
        };
        ensure!(
            (fraction(PanelPoolKindV2::Training) - 0.70).abs() <= 0.01
                && (fraction(PanelPoolKindV2::Development) - 0.10).abs() <= 0.01
                && (fraction(PanelPoolKindV2::Promotion) - 0.15).abs() <= 0.01
                && (fraction(PanelPoolKindV2::Audit) - 0.05).abs() <= 0.01,
            "sealed panel allocation is not 70/10/15/5"
        );
        for ((task, panel), (count, real)) in panel_counts {
            ensure!(
                real * 5 >= count * 4,
                "{task} promotion panel {panel} is less than 80% real-world"
            );
        }
        Ok(())
    }

    pub fn promotion_panel(&self, task: Task, panel: &str) -> Vec<&SealedPanelEntryV2> {
        self.entries
            .iter()
            .filter(|entry| {
                entry.task == task
                    && entry.pool == PanelPoolKindV2::Promotion
                    && entry.promotion_panel.as_deref() == Some(panel)
            })
            .collect()
    }
}

mod builder;
pub use builder::build_sealed_panel_pool_v2;

#[path = "sealed_panels/context_facts.rs"]
mod context_facts;
pub use context_facts::*;
