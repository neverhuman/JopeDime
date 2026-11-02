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

#[derive(Debug)]
struct PanelComponent {
    indexes: Vec<usize>,
    key: Sha256Digest,
    real: usize,
}

fn panel_find(parents: &mut [usize], index: usize) -> usize {
    if parents[index] != index {
        parents[index] = panel_find(parents, parents[index]);
    }
    parents[index]
}

fn panel_union(parents: &mut [usize], left: usize, right: usize) {
    let left = panel_find(parents, left);
    let right = panel_find(parents, right);
    if left != right {
        parents[right] = left;
    }
}

pub fn build_sealed_panel_pool_v2(
    candidates: Vec<SealedPanelCandidateV2>,
    created_at_ms: u128,
) -> Result<SealedPanelPoolV2> {
    ensure!(
        candidates.len() >= 5_120,
        "sealed panel builder needs at least 5120 eligible contexts"
    );
    let mut contexts = BTreeSet::new();
    ensure!(
        candidates.iter().all(|candidate| {
            !candidate.context_identity.trim().is_empty()
                && contexts.insert(&candidate.context_sha256)
        }),
        "sealed panel candidates repeat or empty a context"
    );
    let mut parents = (0..candidates.len()).collect::<Vec<_>>();
    let mut family_owner = BTreeMap::new();
    let mut cluster_owner = BTreeMap::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if let Some(owner) = family_owner.insert(&candidate.source_family_sha256, index) {
            panel_union(&mut parents, owner, index);
        }
        if let Some(owner) = cluster_owner.insert(&candidate.near_duplicate_cluster_sha256, index) {
            panel_union(&mut parents, owner, index);
        }
    }
    let mut grouped = BTreeMap::<usize, Vec<usize>>::new();
    for index in 0..candidates.len() {
        let parent = panel_find(&mut parents, index);
        grouped.entry(parent).or_default().push(index);
    }
    let mut components = grouped
        .into_values()
        .map(|mut indexes| {
            indexes.sort_by(|left, right| {
                candidates[*left]
                    .context_sha256
                    .cmp(&candidates[*right].context_sha256)
            });
            let key = candidates[indexes[0]].context_sha256.clone();
            let real = indexes
                .iter()
                .filter(|index| candidates[**index].origin == DatasetOriginV2::Real)
                .count();
            PanelComponent { indexes, key, real }
        })
        .collect::<Vec<_>>();
    let mut component_key_by_index = vec![None; candidates.len()];
    for component in &components {
        let tasks = component
            .indexes
            .iter()
            .map(|index| candidates[*index].task)
            .collect::<BTreeSet<_>>();
        ensure!(
            tasks.len() == 1,
            "a family/near-duplicate component crosses tasks"
        );
        for index in &component.indexes {
            component_key_by_index[*index] = Some(component.key.clone());
        }
    }
    let total = candidates.len();
    let promotion_target = total * 15 / 100;
    let development_target = total * 10 / 100;
    let audit_target = total * 5 / 100;
    components.sort_by(|left, right| {
        (right.real * left.indexes.len())
            .cmp(&(left.real * right.indexes.len()))
            .then_with(|| left.key.cmp(&right.key))
    });
    let mut assignment = vec![PanelPoolKindV2::Training; candidates.len()];
    let mut promotion_count = 0;
    let mut remaining = Vec::new();
    for component in components {
        if promotion_count < promotion_target {
            for index in &component.indexes {
                assignment[*index] = PanelPoolKindV2::Promotion;
            }
            promotion_count += component.indexes.len();
        } else {
            remaining.push(component);
        }
    }
    remaining.sort_by(|left, right| left.key.cmp(&right.key));
    let mut development_count = 0;
    let mut audit_count = 0;
    for component in remaining {
        let pool = if development_count < development_target {
            development_count += component.indexes.len();
            PanelPoolKindV2::Development
        } else if audit_count < audit_target {
            audit_count += component.indexes.len();
            PanelPoolKindV2::Audit
        } else {
            PanelPoolKindV2::Training
        };
        for index in component.indexes {
            assignment[index] = pool;
        }
    }
    let mut panel_by_index = BTreeMap::<usize, String>::new();
    for task in [Task::Binary, Task::Regression] {
        let promotion = candidates
            .iter()
            .enumerate()
            .filter(|(index, candidate)| {
                candidate.task == task && assignment[*index] == PanelPoolKindV2::Promotion
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        ensure!(
            promotion.len() >= 3 * PROMOTION_MIN_CONTEXTS,
            "{task} promotion pool cannot supply three independent panels"
        );
        let panel_count = 3.max(promotion.len().div_ceil(PROMOTION_MAX_CONTEXTS));
        ensure!(
            promotion.len() >= panel_count * PROMOTION_MIN_CONTEXTS,
            "{task} promotion panels cannot all reach 128 contexts"
        );
        let mut by_component = BTreeMap::<Sha256Digest, Vec<usize>>::new();
        for index in promotion {
            by_component
                .entry(
                    component_key_by_index[index]
                        .clone()
                        .context("promotion context lacks a component")?,
                )
                .or_default()
                .push(index);
        }
        let mut components = by_component.into_iter().collect::<Vec<_>>();
        components.sort_by(|(left_key, left), (right_key, right)| {
            let left_real = left
                .iter()
                .filter(|index| candidates[**index].origin == DatasetOriginV2::Real)
                .count();
            let right_real = right
                .iter()
                .filter(|index| candidates[**index].origin == DatasetOriginV2::Real)
                .count();
            (right_real * left.len())
                .cmp(&(left_real * right.len()))
                .then_with(|| right.len().cmp(&left.len()))
                .then_with(|| left_key.cmp(right_key))
        });
        ensure!(
            components.len() >= panel_count
                && components
                    .iter()
                    .all(|(_, indexes)| indexes.len() <= PROMOTION_MAX_CONTEXTS),
            "{task} promotion components cannot form bounded independent panels"
        );
        let mut panel_sizes = vec![0_usize; panel_count];
        for (_, indexes) in components {
            let panel = (0..panel_count)
                .filter(|panel| panel_sizes[*panel] + indexes.len() <= PROMOTION_MAX_CONTEXTS)
                .min_by_key(|panel| (panel_sizes[*panel], *panel))
                .context("promotion component does not fit any panel")?;
            panel_sizes[panel] += indexes.len();
            let panel_name = format!("{}-promotion-{panel:03}", task.as_str());
            for index in indexes {
                panel_by_index.insert(index, panel_name.clone());
            }
        }
        ensure!(
            panel_sizes
                .iter()
                .all(|size| *size >= PROMOTION_MIN_CONTEXTS),
            "{task} component-preserving promotion panels cannot all reach 128 contexts"
        );
    }
    let entries = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| SealedPanelEntryV2 {
            task: candidate.task,
            context_identity: candidate.context_identity,
            context_sha256: candidate.context_sha256,
            source_family_sha256: candidate.source_family_sha256,
            near_duplicate_cluster_sha256: candidate.near_duplicate_cluster_sha256,
            origin: candidate.origin,
            pool: assignment[index],
            promotion_panel: panel_by_index.remove(&index),
        })
        .collect();
    let pool = SealedPanelPoolV2 {
        schema: SEALED_PANEL_POOL_V2_SCHEMA.to_owned(),
        entries,
        created_at_ms,
    };
    pool.validate()?;
    for task in [Task::Binary, Task::Regression] {
        let panels = pool
            .entries
            .iter()
            .filter(|entry| entry.task == task && entry.pool == PanelPoolKindV2::Promotion)
            .filter_map(|entry| entry.promotion_panel.as_deref())
            .collect::<BTreeSet<_>>();
        ensure!(
            panels.len() >= 3
                && panels
                    .iter()
                    .all(|panel| pool.promotion_panel(task, panel).len() >= PROMOTION_MIN_CONTEXTS),
            "{task} sealed pool lacks three usable promotion panels"
        );
    }
    Ok(pool)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SealedPanelContextFactsV1 {
    pub schema: String,
    pub task: Task,
    pub context_identity: String,
    pub context_sha256: Sha256Digest,
    pub catalog_sha256: Sha256Digest,
    pub train_sha256: Sha256Digest,
    pub test_sha256: Sha256Digest,
    pub source_family_sha256: Sha256Digest,
    pub near_duplicate_cluster_sha256: Sha256Digest,
    pub origin: DatasetOriginV2,
    pub rows: usize,
    pub columns: usize,
    pub missing_fraction: String,
    pub size_slice: String,
    pub missingness_slice: String,
    pub imbalance_slice: String,
    pub target_shape_slice: String,
    pub created_at_ms: u128,
}

impl SealedPanelContextFactsV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SEALED_PANEL_CONTEXT_FACTS_V1_SCHEMA,
            "sealed panel context facts schema is not v1"
        );
        ensure!(
            !self.context_identity.trim().is_empty()
                && self.rows > 0
                && self.columns >= 2
                && [
                    self.missing_fraction.as_str(),
                    self.size_slice.as_str(),
                    self.missingness_slice.as_str(),
                    self.imbalance_slice.as_str(),
                    self.target_shape_slice.as_str(),
                ]
                .into_iter()
                .all(|value| !value.trim().is_empty()),
            "sealed panel context facts are incomplete"
        );
        let missing_fraction = self
            .missing_fraction
            .parse::<f64>()
            .context("sealed panel missing fraction is not numeric")?;
        ensure!(
            missing_fraction.is_finite() && (0.0..=1.0).contains(&missing_fraction),
            "sealed panel missing fraction is outside 0..1"
        );
        Ok(())
    }

    pub fn slices(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            (
                "origin".to_owned(),
                format!("{:?}", self.origin).to_ascii_lowercase(),
            ),
            ("size".to_owned(), self.size_slice.clone()),
            ("missingness".to_owned(), self.missingness_slice.clone()),
            ("imbalance".to_owned(), self.imbalance_slice.clone()),
            ("target_shape".to_owned(), self.target_shape_slice.clone()),
        ])
    }
}
