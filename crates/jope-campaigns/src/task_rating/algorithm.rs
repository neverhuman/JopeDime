use super::*;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug)]
struct Comparison {
    left: usize,
    right: usize,
    outcome: f64,
}

fn tie(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12 * left.abs().max(right.abs()).max(1.0)
}

fn fit_bradley_terry(
    player_count: usize,
    comparisons: &[Comparison],
) -> Result<(Vec<f64>, RatingConvergenceV1)> {
    ensure!(player_count > 1, "rating needs at least two players");
    ensure!(
        !comparisons.is_empty(),
        "rating has no pairwise comparisons"
    );
    let mut ratings = vec![INITIAL_RATING; player_count];
    let mut last_update = f64::INFINITY;
    for iteration in 1..=MAX_ITERATIONS {
        let mut gradient = vec![0.0; player_count];
        let mut curvature = vec![L2_REGULARIZATION; player_count];
        for comparison in comparisons {
            let difference = ((ratings[comparison.left] - ratings[comparison.right])
                * LOGISTIC_SCALE)
                .clamp(-700.0, 700.0);
            let probability = 1.0 / (1.0 + (-difference).exp());
            let residual = comparison.outcome - probability;
            let hessian = LOGISTIC_SCALE * LOGISTIC_SCALE * probability * (1.0 - probability);
            gradient[comparison.left] += LOGISTIC_SCALE * residual;
            gradient[comparison.right] -= LOGISTIC_SCALE * residual;
            curvature[comparison.left] += hessian;
            curvature[comparison.right] += hessian;
        }
        for index in 0..player_count {
            gradient[index] -= L2_REGULARIZATION * (ratings[index] - INITIAL_RATING);
        }
        let updates = gradient
            .iter()
            .zip(&curvature)
            .map(|(gradient, curvature)| (gradient / curvature).clamp(-UPDATE_CLAMP, UPDATE_CLAMP))
            .collect::<Vec<_>>();
        last_update = updates.iter().copied().map(f64::abs).fold(0.0, f64::max);
        for (rating, update) in ratings.iter_mut().zip(updates) {
            *rating += update;
        }
        if last_update < CONVERGENCE_TOLERANCE {
            return Ok((
                ratings,
                RatingConvergenceV1 {
                    converged: true,
                    iterations: iteration,
                    tolerance: CONVERGENCE_TOLERANCE,
                    maximum_update: last_update,
                    l2_regularization: L2_REGULARIZATION,
                    update_clamp: UPDATE_CLAMP,
                },
            ));
        }
    }
    anyhow::bail!(
        "Bradley-Terry fit did not converge after {MAX_ITERATIONS} iterations (update={last_update})"
    )
}

fn task_metric(task: Task) -> (&'static str, &'static str) {
    let semantics = task.semantics();
    (semantics.primary_metric.as_str(), task.reward_expression())
}

fn deterministic_index(
    seed: u64,
    task: Task,
    epoch: &str,
    resample: usize,
    draw: usize,
    upper: usize,
) -> usize {
    let digest = Sha256::digest(
        format!(
            "JopeTaskRatingSnapshot.v1\0{seed}\0{}\0{epoch}\0{resample}\0{draw}",
            task.as_str()
        )
        .as_bytes(),
    );
    let value = u64::from_le_bytes(digest[..8].try_into().expect("SHA prefix"));
    value as usize % upper
}

fn percentile(values: &mut [f64], numerator: usize, denominator: usize) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) * numerator + denominator / 2) / denominator;
    values[index]
}

fn p95_latency(mut values: Vec<u64>) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    Some(values[((values.len() * 95).div_ceil(100)).saturating_sub(1)])
}

pub fn build_task_rating_snapshot(
    input: TaskRatingBuildInputV1,
) -> Result<JopeTaskRatingSnapshotV1> {
    ensure!(
        input.schema == TASK_RATING_BUILD_INPUT_SCHEMA,
        "task rating input schema is not v1"
    );
    ensure!(
        input.bootstrap_resamples == RATING_BOOTSTRAP_RESAMPLES,
        "rating requires exactly 10000 family bootstrap resamples"
    );
    ensure!(
        input.players.len() >= 2 && !input.contexts.is_empty(),
        "rating input is empty"
    );
    let mut players = input.players;
    players.sort_by(|left, right| left.player.cmp(&right.player));
    ensure!(
        players
            .iter()
            .all(|player| !player.player.trim().is_empty())
            && players
                .iter()
                .map(|player| player.player.as_str())
                .collect::<BTreeSet<_>>()
                .len()
                == players.len(),
        "rating players are empty or duplicated"
    );
    let player_index = players
        .iter()
        .enumerate()
        .map(|(index, player)| (player.player.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let anchor_index = *player_index
        .get(&input.fixed_anchor_player)
        .context("fixed rating anchor is absent")?;
    ensure!(
        player_index.contains_key(&input.display_player),
        "display rating player is absent"
    );
    let mut contexts = input.contexts;
    contexts.sort_by(|left, right| left.context_sha256.cmp(&right.context_sha256));
    ensure!(
        contexts
            .iter()
            .all(|context| !context.context.trim().is_empty())
            && contexts
                .iter()
                .map(|context| &context.context_sha256)
                .collect::<BTreeSet<_>>()
                .len()
                == contexts.len(),
        "rating contexts are empty or duplicated"
    );
    let context_by_digest = contexts
        .iter()
        .map(|context| (&context.context_sha256, context))
        .collect::<BTreeMap<_, _>>();

    let mut observed = BTreeMap::new();
    for row in input.observations {
        ensure!(
            player_index.contains_key(&row.player)
                && context_by_digest.contains_key(&row.context_sha256)
                && row.slot < RATING_SLOTS
                && row.score.is_none_or(f64::is_finite),
            "rating slot references an unknown identity or invalid score"
        );
        ensure!(
            observed
                .insert(
                    (row.player.clone(), row.context_sha256.clone(), row.slot),
                    row
                )
                .is_none(),
            "rating input repeats a player/context slot"
        );
    }

    let mut policy_contexts = Vec::with_capacity(players.len() * contexts.len());
    let mut utility_by_context = BTreeMap::<Sha256Digest, Vec<f64>>::new();
    let mut valid_by_player = vec![0_u64; players.len()];
    let mut failed_by_player = vec![0_u64; players.len()];
    let mut latency_by_player = vec![Vec::new(); players.len()];
    for context in &contexts {
        let mut utilities = vec![0.0; players.len()];
        for (index, player) in players.iter().enumerate() {
            let mut slots = Vec::with_capacity(RATING_SLOTS);
            for slot in 0..RATING_SLOTS {
                let observation =
                    observed.get(&(player.player.clone(), context.context_sha256.clone(), slot));
                let valid =
                    observation.is_some_and(|row| row.valid && !row.failed && row.score.is_some());
                let utility = if valid {
                    bounded_task_utility(
                        input.task,
                        observation.and_then(|row| row.score).expect("valid score"),
                    )?
                } else {
                    -1.0
                };
                valid_by_player[index] += u64::from(valid);
                failed_by_player[index] +=
                    u64::from(observation.is_none_or(|row| row.failed || !row.valid));
                if let Some(latency) = observation.and_then(|row| row.latency_ms) {
                    latency_by_player[index].push(latency);
                }
                slots.push(RatedSlotV1 {
                    slot,
                    score: observation.and_then(|row| row.score),
                    utility,
                    valid,
                    failed: observation.is_none_or(|row| row.failed),
                    missing: observation.is_none(),
                    latency_ms: observation.and_then(|row| row.latency_ms),
                });
            }
            let policy_utility =
                slots.iter().map(|slot| slot.utility).sum::<f64>() / RATING_SLOTS as f64;
            let best_of_eight_utility = slots
                .iter()
                .map(|slot| slot.utility)
                .max_by(f64::total_cmp)
                .expect("eight slots");
            utilities[index] = policy_utility;
            policy_contexts.push(RatedPolicyContextV1 {
                player: player.player.clone(),
                context_sha256: context.context_sha256.clone(),
                source_family_sha256: context.source_family_sha256.clone(),
                slots,
                policy_utility,
                best_of_eight_utility,
            });
        }
        utility_by_context.insert(context.context_sha256.clone(), utilities);
    }

    let mut comparisons = Vec::new();
    let mut comparisons_by_family = BTreeMap::<Sha256Digest, Vec<Comparison>>::new();
    let mut wins = vec![0_u64; players.len()];
    let mut losses = vec![0_u64; players.len()];
    let mut ties = vec![0_u64; players.len()];
    for context in &contexts {
        let utility = &utility_by_context[&context.context_sha256];
        for left in 0..players.len() {
            for right in (left + 1)..players.len() {
                let outcome = if tie(utility[left], utility[right]) {
                    ties[left] += 1;
                    ties[right] += 1;
                    0.5
                } else if utility[left] > utility[right] {
                    wins[left] += 1;
                    losses[right] += 1;
                    1.0
                } else {
                    losses[left] += 1;
                    wins[right] += 1;
                    0.0
                };
                let comparison = Comparison {
                    left,
                    right,
                    outcome,
                };
                comparisons.push(comparison);
                comparisons_by_family
                    .entry(context.source_family_sha256.clone())
                    .or_default()
                    .push(comparison);
            }
        }
    }
    let (raw_ratings, convergence) = fit_bradley_terry(players.len(), &comparisons)?;
    let anchor_shift = INITIAL_RATING - raw_ratings[anchor_index];
    let ratings = raw_ratings
        .into_iter()
        .map(|rating| rating + anchor_shift)
        .collect::<Vec<_>>();

    let families = comparisons_by_family.keys().cloned().collect::<Vec<_>>();
    ensure!(!families.is_empty(), "rating has no source families");
    let mut bootstrap = vec![Vec::with_capacity(RATING_BOOTSTRAP_RESAMPLES); players.len()];
    for resample in 0..RATING_BOOTSTRAP_RESAMPLES {
        let mut sampled = Vec::new();
        for draw in 0..families.len() {
            let index = deterministic_index(
                input.bootstrap_seed,
                input.task,
                &input.rating_epoch,
                resample,
                draw,
                families.len(),
            );
            sampled.extend_from_slice(&comparisons_by_family[&families[index]]);
        }
        let (fitted, _) = fit_bradley_terry(players.len(), &sampled)?;
        let shift = INITIAL_RATING - fitted[anchor_index];
        for (index, rating) in fitted.into_iter().enumerate() {
            bootstrap[index].push(rating + shift);
        }
    }
    let denominator = (contexts.len() * RATING_SLOTS) as f64;
    let rated_players = players
        .into_iter()
        .enumerate()
        .map(|(index, player)| {
            let mut lower_values = bootstrap[index].clone();
            let mut upper_values = bootstrap[index].clone();
            RatedPlayerV1 {
                player: player.player,
                checkpoint_sha256: player.checkpoint_sha256,
                ancestry_sha256: player.ancestry_sha256,
                rating: ratings[index],
                rating_interval_95: [
                    percentile(&mut lower_values, 25, 1000),
                    percentile(&mut upper_values, 975, 1000),
                ],
                wins: wins[index],
                losses: losses[index],
                ties: ties[index],
                validity: valid_by_player[index] as f64 / denominator,
                failure_rate: failed_by_player[index] as f64 / denominator,
                p95_latency_ms: p95_latency(std::mem::take(&mut latency_by_player[index])),
                family_count: families.len(),
            }
        })
        .collect();
    let (metric, transform) = task_metric(input.task);
    let snapshot = JopeTaskRatingSnapshotV1 {
        schema: TASK_RATING_SNAPSHOT_SCHEMA.to_owned(),
        task: input.task,
        rating_kind: input.rating_kind,
        rating_epoch: input.rating_epoch,
        task_metric: metric.to_owned(),
        utility_transform: transform.to_owned(),
        fixed_anchor_player: input.fixed_anchor_player,
        display_player: input.display_player,
        proposal_selection_policy_sha256: input.proposal_selection_policy_sha256,
        proposal_slots_per_player_context: RATING_SLOTS,
        invalid_slot_utility: -1.0,
        players: rated_players,
        contexts,
        policy_contexts,
        convergence,
        bootstrap_resamples: RATING_BOOTSTRAP_RESAMPLES,
        promotion_eligible: false,
        promotion_ineligible_reason: match input.rating_kind {
            RatingKindV1::AdaptiveSearch => {
                "adaptive/test-used search evidence is not promotion evidence"
            }
            RatingKindV1::DiagnosticAnchor => {
                "reusable diagnostic anchors are permanently promotion-ineligible"
            }
        }
        .to_owned(),
        generated_at_ms: input.generated_at_ms,
    };
    snapshot.validate()?;
    Ok(snapshot)
}
