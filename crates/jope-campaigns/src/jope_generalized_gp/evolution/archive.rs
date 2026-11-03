use super::ranking::compare_ranked;
use super::*;

pub fn build_provisional_archive<F>(
    arm: CampaignArmV1,
    dataset_sha256: &str,
    split_plan_sha256: &str,
    ranked_generations: &[Vec<RankedCandidateV1>],
    existing_evaluations: &[CandidateSplitEvaluationV1],
    mut backfill: F,
) -> Result<ArchiveCheckpointV1, String>
where
    F: FnMut(&GenomeV1, u32) -> Result<CandidateSplitEvaluationV1, String>,
{
    crate::search_v3::validate_sha256("archive dataset", dataset_sha256)?;
    crate::search_v3::validate_sha256("archive split plan", split_plan_sha256)?;
    if ranked_generations.len() != 5 || ranked_generations.iter().any(Vec::is_empty) {
        return Err("archive construction requires five nonempty generation rankings".into());
    }
    fn add_candidate(
        candidates: &mut BTreeMap<String, GenomeV1>,
        selected: &mut Vec<String>,
        candidate: &GenomeV1,
    ) {
        candidates
            .entry(candidate.semantic_sha256.clone())
            .or_insert_with(|| candidate.clone());
        if !selected.contains(&candidate.semantic_sha256) {
            selected.push(candidate.semantic_sha256.clone());
        }
    }
    let mut candidates = BTreeMap::<String, GenomeV1>::new();
    let mut selected = Vec::<String>::new();
    // Preserve every generation leader.
    for generation in ranked_generations {
        add_candidate(&mut candidates, &mut selected, &generation[0].candidate);
    }
    // Preserve one leader from every observed head family before score fill.
    let mut family_leaders = BTreeMap::<String, &RankedCandidateV1>::new();
    for candidate in ranked_generations.iter().flatten() {
        family_leaders
            .entry(candidate.candidate.head_family.clone())
            .and_modify(|current| {
                if compare_ranked(candidate, current) == Ordering::Less {
                    *current = candidate;
                }
            })
            .or_insert(candidate);
    }
    for leader in family_leaders.values() {
        add_candidate(&mut candidates, &mut selected, &leader.candidate);
    }
    let all_ranked = rank_population(arm, existing_evaluations)?;
    for leader in &all_ranked {
        add_candidate(&mut candidates, &mut selected, &leader.candidate);
        if selected.len() == 20 {
            break;
        }
    }
    if selected.len() < 20 {
        for generation in ranked_generations {
            for leader in generation {
                add_candidate(&mut candidates, &mut selected, &leader.candidate);
                if selected.len() == 20 {
                    break;
                }
            }
            if selected.len() == 20 {
                break;
            }
        }
    }
    selected.truncate(20);
    if selected.len() != 20 {
        return Err("fewer than 20 unique genomes are available for the archive".into());
    }

    let mut confirmed = Vec::with_capacity(20);
    for semantic in selected {
        let candidate = candidates
            .get(&semantic)
            .cloned()
            .or_else(|| {
                ranked_generations
                    .iter()
                    .flatten()
                    .find(|ranked| ranked.candidate.semantic_sha256 == semantic)
                    .map(|ranked| ranked.candidate.clone())
            })
            .unwrap();
        let mut split_evaluations = Vec::with_capacity(5);
        for split_index in 0..5u32 {
            let evaluation = existing_evaluations
                .iter()
                .find(|evaluation| {
                    evaluation.arm == arm
                        && evaluation.candidate.semantic_sha256 == semantic
                        && evaluation.split_index == split_index
                })
                .cloned()
                .map(Ok)
                .unwrap_or_else(|| backfill(&candidate, split_index))?;
            evaluation.validate()?;
            if evaluation.arm != arm
                || evaluation.dataset_sha256 != dataset_sha256
                || evaluation.split_index != split_index
                || evaluation.candidate.semantic_sha256 != semantic
            {
                return Err("archive backfill returned stale or mismatched evidence".into());
            }
            split_evaluations.push(evaluation);
        }
        confirmed.push((candidate, split_evaluations));
    }
    let all_confirmed: Vec<_> = confirmed
        .iter()
        .flat_map(|(_, evaluations)| evaluations.iter().cloned())
        .collect();
    let confirmed_ranking = rank_population(arm, &all_confirmed)?;
    let rank: BTreeMap<_, _> = confirmed_ranking
        .iter()
        .enumerate()
        .map(|(rank, candidate)| (candidate.candidate.semantic_sha256.as_str(), rank))
        .collect();
    confirmed.sort_by_key(|(candidate, _)| rank[&candidate.semantic_sha256.as_str()]);
    let archive = ArchiveCheckpointV1 {
        schema: ArchiveCheckpointV1::SCHEMA.into(),
        arm,
        dataset_sha256: dataset_sha256.into(),
        split_plan_sha256: split_plan_sha256.into(),
        members: confirmed
            .into_iter()
            .enumerate()
            .map(|(rank, (candidate, split_evaluations))| ArchiveMemberV1 {
                candidate,
                split_evaluations,
                confirmed_rank: rank as u32,
            })
            .collect(),
    };
    archive.validate()?;
    Ok(archive)
}

fn compare_cv7(left: &Cv7CandidateEvidenceV1, right: &Cv7CandidateEvidenceV1) -> Ordering {
    right
        .valid
        .cmp(&left.valid)
        .then_with(|| right.robust_objective().total_cmp(&left.robust_objective()))
        .then_with(|| left.secondary_metric.total_cmp(&right.secondary_metric))
        .then_with(|| left.runtime_seconds.total_cmp(&right.runtime_seconds))
        .then_with(|| left.candidate.complexity.cmp(&right.candidate.complexity))
        .then_with(|| {
            left.candidate
                .semantic_sha256
                .cmp(&right.candidate.semantic_sha256)
        })
}

pub fn select_cv7_winner(
    arm: CampaignArmV1,
    dataset_sha256: &str,
    archive_sha256: &str,
    cv7_plan_sha256: &str,
    mut finalists: Vec<Cv7CandidateEvidenceV1>,
) -> Result<Cv7SelectionFreezeV1, String> {
    if finalists.len() != 10 {
        return Err("CV7 selection requires exactly ten candidates".into());
    }
    for finalist in &finalists {
        finalist.validate()?;
    }
    finalists.sort_by(compare_cv7);
    if !finalists[0].valid {
        return Err("CV7 cannot freeze a winner when every finalist is invalid".into());
    }
    let selection = Cv7SelectionFreezeV1 {
        schema: Cv7SelectionFreezeV1::SCHEMA.into(),
        arm,
        dataset_sha256: dataset_sha256.into(),
        archive_sha256: archive_sha256.into(),
        cv7_plan_sha256: cv7_plan_sha256.into(),
        winner_semantic_sha256: finalists[0].candidate.semantic_sha256.clone(),
        finalists,
    };
    selection.validate()?;
    Ok(selection)
}
