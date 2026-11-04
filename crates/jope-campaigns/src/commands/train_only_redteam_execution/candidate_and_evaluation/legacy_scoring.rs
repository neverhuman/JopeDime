use super::*;

pub(crate) fn legacy_metrics(truth: &[f64], predictions: &[f64]) -> Result<(f64, f64)> {
    let error = squared_error(truth, predictions)?;
    let mean = truth.iter().sum::<f64>() / truth.len() as f64;
    let total = truth
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    if total <= 0.0 {
        bail!("legacy split target variance is zero");
    }
    Ok(((error / truth.len() as f64).sqrt(), 1.0 - error / total))
}

pub(crate) fn legacy_robust_fitness(scores: &[f64]) -> Result<f64> {
    if scores.is_empty() || scores.iter().any(|value| !value.is_finite()) {
        bail!("legacy robust fitness needs finite scores");
    }
    let clipped = scores
        .iter()
        .map(|value| value.max(-1.0))
        .collect::<Vec<_>>();
    let mean = clipped.iter().sum::<f64>() / clipped.len() as f64;
    let sd = (clipped
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / clipped.len() as f64)
        .sqrt();
    Ok(mean - 0.25 * sd.min(1.0))
}

pub(crate) fn legacy_split(
    dataset_hash: &str,
    generation: usize,
    targets: &[f64],
) -> Result<Vec<usize>> {
    let mut sorted = (0..targets.len()).collect::<Vec<_>>();
    sorted.sort_by(|left, right| {
        targets[*left]
            .total_cmp(&targets[*right])
            .then_with(|| left.cmp(right))
    });
    let mut deciles = vec![Vec::new(); 10];
    let rows = sorted.len();
    for (rank, row) in sorted.into_iter().enumerate() {
        deciles[(rank * 10 / rows).min(9)].push(row);
    }
    let mut seed_hash = Sha256::new();
    seed_hash.update(b"checkpoint-assisted-split-seed-v2\0");
    seed_hash.update(20_260_729_u64.to_le_bytes());
    seed_hash.update(dataset_hash.as_bytes());
    seed_hash.update((generation as u64).to_le_bytes());
    let seed = u64::from_le_bytes(seed_hash.finalize()[..8].try_into().unwrap());
    let validation_total = targets.len().div_ceil(5);
    let mut ordered = Vec::new();
    let mut base_total = 0;
    for rows in deciles {
        let mut rows = rows;
        rows.sort_by_key(|row| {
            let mut hash = Sha256::new();
            hash.update(b"checkpoint-assisted-split-order-v2\0");
            hash.update(seed.to_le_bytes());
            hash.update((*row as u64).to_le_bytes());
            hash.finalize().to_vec()
        });
        let base = rows.len() / 5;
        let remainder = rows.len() % 5;
        base_total += base;
        ordered.push((rows, base, remainder));
    }
    let mut quotas = ordered.iter().map(|entry| entry.1).collect::<Vec<_>>();
    let mut extras = validation_total - base_total;
    let mut remainder_order = (0..10).collect::<Vec<_>>();
    remainder_order.sort_by(|left, right| {
        ordered[*right]
            .2
            .cmp(&ordered[*left].2)
            .then_with(|| left.cmp(right))
    });
    for decile in remainder_order {
        if extras == 0 {
            break;
        }
        if quotas[decile] < ordered[decile].0.len() {
            quotas[decile] += 1;
            extras -= 1;
        }
    }
    if extras != 0 {
        bail!("legacy validation quota allocation failed");
    }
    let mut validation = Vec::with_capacity(validation_total);
    for (decile, (rows, _, _)) in ordered.iter().enumerate() {
        validation.extend_from_slice(&rows[..quotas[decile]]);
    }
    validation.sort_unstable();
    Ok(validation)
}

pub(crate) fn score_legacy_population(
    data: &RegressionData,
    dataset_hash: &str,
    generation: usize,
    population: &[Candidate],
    workers: usize,
) -> Result<Vec<LegacyScoredCandidate>> {
    let validations = (0..=generation)
        .map(|split| legacy_split(dataset_hash, split, &data.y))
        .collect::<Result<Vec<_>>>()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    let mut scored = pool.install(|| {
        population
            .par_iter()
            .map(|candidate| {
                let evaluated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || -> Result<Vec<f64>> {
                        let parsed = GenomeIr::parse(&candidate.canonical_genome)?;
                        let mut scores = Vec::with_capacity(validations.len());
                        for validation in &validations {
                            let training = complement(data.y.len(), validation)?;
                            let (train_x, train_y) = slice_rows(data, &training);
                            let (validation_x, validation_y) = slice_rows(data, validation);
                            let model = fit(&parsed, &train_x, &train_y)?;
                            let predictions = model.predict(&validation_x)?;
                            scores.push(legacy_metrics(&validation_y, &predictions)?.1);
                        }
                        Ok(scores)
                    },
                ));
                Ok(match evaluated {
                    Ok(Ok(split_r2)) => LegacyScoredCandidate {
                        robust_fitness: Some(legacy_robust_fitness(&split_r2)?),
                        worst_split_r2: split_r2.iter().copied().min_by(f64::total_cmp),
                        candidate: candidate.clone(),
                        split_r2,
                        failure_reason: None,
                    },
                    Ok(Err(error)) => LegacyScoredCandidate {
                        candidate: candidate.clone(),
                        split_r2: Vec::new(),
                        robust_fitness: None,
                        worst_split_r2: None,
                        failure_reason: Some(format!("{error:#}")),
                    },
                    Err(_) => LegacyScoredCandidate {
                        candidate: candidate.clone(),
                        split_r2: Vec::new(),
                        robust_fitness: None,
                        worst_split_r2: None,
                        failure_reason: Some("panic during legacy evaluation".to_string()),
                    },
                })
            })
            .collect::<Result<Vec<_>>>()
    })?;
    scored.sort_by(
        |left, right| match (left.robust_fitness, right.robust_fitness) {
            (Some(left_fitness), Some(right_fitness)) => right_fitness
                .total_cmp(&left_fitness)
                .then_with(|| {
                    right
                        .worst_split_r2
                        .unwrap()
                        .total_cmp(&left.worst_split_r2.unwrap())
                })
                .then_with(|| left.candidate.complexity.cmp(&right.candidate.complexity))
                .then_with(|| {
                    left.candidate
                        .semantic_sha256
                        .cmp(&right.candidate.semantic_sha256)
                }),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => left
                .candidate
                .semantic_sha256
                .cmp(&right.candidate.semantic_sha256),
        },
    );
    Ok(scored)
}
