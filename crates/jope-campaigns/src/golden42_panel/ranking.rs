use super::*;

pub(super) fn v3_seed_sources() -> Vec<String> {
    [
        NEAR_WINNER,
        COMPATIBLE_WINNER,
        GOLDEN42_FORMULA_SEED,
        NEAREST_PROFILE_SEED,
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

pub(super) fn novelty(candidate: &V3Candidate, archive: &V3Archive) -> f64 {
    if archive.entries.is_empty() {
        return 1.0;
    }
    let prefix = &candidate.semantic_hash[..16];
    let nearest = archive
        .entries
        .iter()
        .map(|entry| {
            prefix
                .bytes()
                .zip(entry.candidate.semantic_hash[..16].bytes())
                .filter(|(left, right)| left != right)
                .count()
        })
        .min()
        .unwrap_or(0);
    nearest as f64 / 16.0
}

pub(super) fn scored_order(
    left: &V3ScoredCandidate,
    right: &V3ScoredCandidate,
) -> std::cmp::Ordering {
    left.rmse
        .total_cmp(&right.rmse)
        .then_with(|| {
            left.candidate
                .complexity()
                .cmp(&right.candidate.complexity())
        })
        .then_with(|| {
            left.candidate
                .semantic_hash
                .cmp(&right.candidate.semantic_hash)
        })
}
