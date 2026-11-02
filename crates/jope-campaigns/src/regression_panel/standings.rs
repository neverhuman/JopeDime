use super::*;

pub(super) fn pairwise_results(scored: &[ScoredDataset]) -> Value {
    let players = players();
    let mut rows = Vec::new();
    for left_index in 0..players.len() {
        for right_index in left_index + 1..players.len() {
            let left = players[left_index];
            let right = players[right_index];
            let mut left_wins = 0usize;
            let mut right_wins = 0usize;
            let mut ties = 0usize;
            for dataset in scored {
                match dataset.scores[left].total_cmp(&dataset.scores[right]) {
                    Ordering::Less => left_wins += 1,
                    Ordering::Equal => ties += 1,
                    Ordering::Greater => right_wins += 1,
                }
            }
            rows.push(json!({
                "left": left,
                "right": right,
                "left_wins": left_wins,
                "right_wins": right_wins,
                "ties": ties,
            }));
        }
    }
    Value::Array(rows)
}

pub(super) fn players() -> Vec<&'static str> {
    let mut values = BASELINES.to_vec();
    values.push(MODEL);
    values
}

pub(super) fn elo_standings(scored: &[ScoredDataset]) -> Value {
    const K: f64 = 16.0;
    let players = players();
    let mut ratings = players
        .iter()
        .map(|player| ((*player).to_string(), 1500.0))
        .collect::<BTreeMap<_, _>>();
    for dataset in scored {
        let snapshot = ratings.clone();
        let mut deltas = players
            .iter()
            .map(|player| ((*player).to_string(), 0.0))
            .collect::<BTreeMap<_, _>>();
        for left_index in 0..players.len() {
            for right_index in left_index + 1..players.len() {
                let left = players[left_index];
                let right = players[right_index];
                let left_rating = snapshot[left];
                let right_rating = snapshot[right];
                let expected_left = 1.0 / (1.0 + 10_f64.powf((right_rating - left_rating) / 400.0));
                let actual_left = match dataset.scores[left].total_cmp(&dataset.scores[right]) {
                    Ordering::Less => 1.0,
                    Ordering::Equal => 0.5,
                    Ordering::Greater => 0.0,
                };
                let delta = K * (actual_left - expected_left);
                *deltas.get_mut(left).unwrap() += delta;
                *deltas.get_mut(right).unwrap() -= delta;
            }
        }
        for player in &players {
            *ratings.get_mut(*player).unwrap() += deltas[*player];
        }
    }
    let mut ordered = ratings.into_iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    Value::Array(
        ordered
            .into_iter()
            .enumerate()
            .map(|(index, (algorithm, rating))| {
                json!({"rank": index + 1, "algorithm": algorithm, "rating": rating})
            })
            .collect(),
    )
}
