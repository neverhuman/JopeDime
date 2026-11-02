use super::*;

// This is a structural helper only; no public admission token can be made here.
pub(crate) fn check_candidates(
    plan: &RequestPlan,
    candidates: &[CandidateClaim],
    neural_candidates: usize,
    bank: &GenericBank,
    canonicalization: &Canonicalization,
) -> Result<()> {
    ensure!(
        candidates.len() == 80 && neural_candidates <= 64,
        "experimental initial slot count changed"
    );
    let mut seen = BTreeSet::new();
    let mut previous: Option<&CandidateClaim> = None;
    for (slot, candidate) in candidates.iter().enumerate() {
        ensure!(
            candidate.slot == slot && seen.insert(candidate.recipe.executable_sha256.clone()),
            "reordered or duplicate experimental initial recipe"
        );
        validate_recipe(&candidate.recipe, plan)?;
        let validated = canonicalization
            .results
            .get(candidate.program_index)
            .context("candidate canonicalization slot absent")?;
        ensure!(
            validated.index == candidate.program_index
                && validated.error.is_none()
                && validated.original_bytes_sha256 == candidate.recipe.original_bytes_sha256
                && validated
                    .recipe
                    .as_ref()
                    .is_some_and(|r| same(r, &candidate.recipe).unwrap_or(false)),
            "candidate recipe differs from exact native batch output"
        );
        if slot < neural_candidates {
            ensure!(
                candidate.origin == "neural_policy"
                    && candidate.bank_index.is_none()
                    && candidate.candidate_seed.is_none()
                    && candidate
                        .sample_index
                        .is_some_and(|i| i < plan.sampling.count)
                    && candidate
                        .tokens
                        .as_ref()
                        .is_some_and(|t| !t.is_empty() && t.len() <= plan.sampling.max_len + 4)
                    && candidate.reward_mean.is_some_and(f64::is_finite)
                    && candidate
                        .reward_std
                        .is_some_and(|v| v.is_finite() && v >= 0.)
                    && candidate.rank_log_weight.is_some_and(f64::is_finite),
                "neural slot lacks actual sampled/ranked evidence"
            );
            if let Some(old) = previous {
                ensure!(
                    old.reward_mean.unwrap() > candidate.reward_mean.unwrap()
                        || (old.reward_mean == candidate.reward_mean
                            && old.recipe.executable_sha256 <= candidate.recipe.executable_sha256),
                    "neural rank ordering changed"
                );
            }
            previous = Some(candidate);
        } else {
            ensure!(
                candidate.origin
                    == if slot < 64 {
                        "neural_rejection_fallback"
                    } else {
                        "random_exploration"
                    }
                    && candidate.tokens.is_none()
                    && candidate.sample_index.is_none()
                    && candidate.reward_mean.is_none()
                    && candidate.reward_std.is_none()
                    && candidate.rank_log_weight.is_none(),
                "generic slot invented neural provenance"
            );
        }
    }
    if neural_candidates > 0 {
        let neural = &candidates[..neural_candidates];
        let max = neural[0].reward_mean.unwrap();
        let normalizer = max
            + neural
                .iter()
                .map(|c| (c.reward_mean.unwrap() - max).exp())
                .sum::<f64>()
                .ln();
        ensure!(
            neural.iter().all(|c| (c.rank_log_weight.unwrap()
                - (c.reward_mean.unwrap() - normalizer))
                .abs()
                <= 1e-12),
            "neural rank weights changed"
        );
    }
    // Recreate fallback/exploration ordering from every original draw. Invalid or
    // duplicate entries alone may be skipped; output cannot cherry-pick the bank.
    let mut used = candidates[..neural_candidates]
        .iter()
        .map(|c| c.recipe.executable_sha256.clone())
        .collect::<BTreeSet<_>>();
    let mut cursor = neural_candidates;
    for (index, draw) in bank.candidates.iter().enumerate() {
        let Ok(recipe) = SynthesizedRecipe::new(draw.genome.clone()) else {
            continue;
        };
        if validate_recipe(&recipe, plan).is_err() || !used.insert(recipe.executable_sha256.clone())
        {
            continue;
        }
        let candidate = &candidates[cursor];
        ensure!(
            candidate.bank_index == Some(index)
                && candidate.candidate_seed == Some(draw.seed)
                && same(&candidate.recipe, &recipe)?,
            "fallback/exploration differs from original typed draw order"
        );
        cursor += 1;
        if cursor == 80 {
            break;
        }
    }
    ensure!(
        cursor == 80,
        "generic bank cannot supply complete population"
    );
    Ok(())
}
