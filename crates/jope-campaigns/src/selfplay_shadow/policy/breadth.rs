use super::*;

fn breadth_order(task: Task, wave_index: u64, dataset_id: &str) -> [u8; 32] {
    Sha256::digest(format!("breadth-v2|{task}|{wave_index}|{dataset_id}").as_bytes()).into()
}

pub fn plan_breadth_cohort(
    task: Task,
    wave_index: u64,
    datasets: &[BreadthDataset],
) -> Result<BreadthCohort> {
    let registry = datasets
        .iter()
        .filter(|row| row.task == task && row.eligible && !row.sealed)
        .collect::<Vec<_>>();
    ensure!(
        registry.len() >= DATASETS_PER_COHORT,
        "fewer than 64 eligible non-sealed registry datasets"
    );
    let observed = registry
        .iter()
        .filter(|row| row.observation_count > 0)
        .count();
    let coverage = observed as f64 / registry.len() as f64;
    let (frontier_target, hard_target, anchor_target) = if coverage < 0.80 {
        (48, 8, 8)
    } else {
        (32, 16, 16)
    };
    let mut frontier = registry
        .iter()
        .copied()
        .filter(|row| row.encoded)
        .collect::<Vec<_>>();
    frontier.sort_by(|left, right| {
        left.frontier_visits
            .cmp(&right.frontier_visits)
            .then_with(|| left.observation_count.cmp(&right.observation_count))
            .then_with(|| {
                breadth_order(task, wave_index, &left.dataset_id).cmp(&breadth_order(
                    task,
                    wave_index,
                    &right.dataset_id,
                ))
            })
    });
    ensure!(
        frontier.len() >= frontier_target,
        "insufficient encoded frontier contexts"
    );
    let mut used = HashSet::<String>::new();
    let mut members = Vec::new();
    for row in frontier.into_iter().take(frontier_target) {
        used.insert(row.dataset_id.clone());
        members.push(CohortMember {
            dataset_id: row.dataset_id.clone(),
            dataset_content_sha256: row.dataset_content_sha256.clone(),
            role: CohortRole::Frontier,
        });
    }

    let mut hard = registry
        .iter()
        .copied()
        .filter(|row| {
            row.encoded
                && !used.contains(&row.dataset_id)
                && row
                    .last_hard_wave
                    .is_none_or(|last| wave_index.saturating_sub(last) >= 4)
                && row.hard_score.is_finite()
                && row.uncertainty.is_finite()
        })
        .collect::<Vec<_>>();
    hard.sort_by(|left, right| {
        (right.hard_score + right.uncertainty)
            .total_cmp(&(left.hard_score + left.uncertainty))
            .then_with(|| left.dataset_id.cmp(&right.dataset_id))
    });
    ensure!(hard.len() >= hard_target, "insufficient hard contexts");
    for row in hard.into_iter().take(hard_target) {
        used.insert(row.dataset_id.clone());
        members.push(CohortMember {
            dataset_id: row.dataset_id.clone(),
            dataset_content_sha256: row.dataset_content_sha256.clone(),
            role: CohortRole::Hard,
        });
    }

    let mut anchors = registry
        .iter()
        .copied()
        .filter(|row| row.encoded && row.stable_anchor && !used.contains(&row.dataset_id))
        .collect::<Vec<_>>();
    anchors.sort_by(|left, right| {
        breadth_order(task, 0, &left.dataset_id).cmp(&breadth_order(task, 0, &right.dataset_id))
    });
    ensure!(
        anchors.len() >= anchor_target,
        "insufficient stable anchor contexts"
    );
    for row in anchors.into_iter().take(anchor_target) {
        members.push(CohortMember {
            dataset_id: row.dataset_id.clone(),
            dataset_content_sha256: row.dataset_content_sha256.clone(),
            role: CohortRole::Anchor,
        });
    }
    ensure!(
        members.len() == DATASETS_PER_COHORT,
        "cohort does not contain 64 contexts"
    );
    Ok(BreadthCohort {
        task,
        wave_index,
        catalog_coverage: coverage,
        frontier_target,
        hard_target,
        anchor_target,
        members,
    })
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EfficiencyEvidence {
    pub complete_waves: usize,
    pub direct_evaluations: usize,
    pub lower_90_efficiency_ratio: f64,
    pub upper_90_efficiency_ratio: f64,
    pub validity: f64,
    pub failure_guard_triggered: bool,
}
