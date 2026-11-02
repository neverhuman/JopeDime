use super::*;

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
