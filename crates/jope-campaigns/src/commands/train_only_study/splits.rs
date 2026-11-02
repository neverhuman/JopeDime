use super::*;

pub(super) fn build_splits(
    run_id: &str,
    dataset_hash: &str,
    train_sha256: &str,
    targets: &[f64],
    generations: usize,
) -> Result<TrainOnlySplitManifest> {
    if targets.len() < 100 || targets.iter().any(|value| !value.is_finite()) {
        bail!("split construction needs at least 100 finite targets");
    }
    let mut sorted = (0..targets.len()).collect::<Vec<_>>();
    sorted.sort_by(|left, right| {
        targets[*left]
            .total_cmp(&targets[*right])
            .then_with(|| left.cmp(right))
    });
    let mut deciles = vec![Vec::new(); 10];
    let sorted_len = sorted.len();
    for (rank, row) in sorted.into_iter().enumerate() {
        let decile = (rank * 10 / sorted_len).min(9);
        deciles[decile].push(row);
    }
    let validation_total = targets.len().div_ceil(5);
    let mut receipts = Vec::with_capacity(generations);
    let mut seen_hashes = HashSet::new();
    for generation in 0..generations {
        let split_seed = {
            let mut hash = Sha256::new();
            hash.update(b"checkpoint-assisted-split-seed-v2\0");
            hash.update(STUDY_SEED.to_le_bytes());
            hash.update(dataset_hash.as_bytes());
            hash.update((generation as u64).to_le_bytes());
            u64::from_le_bytes(hash.finalize()[..8].try_into().unwrap())
        };
        let mut ordered = Vec::with_capacity(10);
        let mut base_total = 0usize;
        for (decile, rows) in deciles.iter().enumerate() {
            let mut rows = rows.clone();
            rows.sort_by_key(|row| {
                let mut hash = Sha256::new();
                hash.update(b"checkpoint-assisted-split-order-v2\0");
                hash.update(split_seed.to_le_bytes());
                hash.update((*row as u64).to_le_bytes());
                hex::encode(hash.finalize())
            });
            let base = rows.len() / 5;
            let remainder = rows.len() % 5;
            base_total += base;
            ordered.push((decile, rows, base, remainder));
        }
        let mut extra = validation_total - base_total;
        let mut remainder_order = (0..10).collect::<Vec<_>>();
        remainder_order.sort_by(|left, right| {
            ordered[*right]
                .3
                .cmp(&ordered[*left].3)
                .then_with(|| left.cmp(right))
        });
        let mut quotas = ordered.iter().map(|entry| entry.2).collect::<Vec<_>>();
        for decile in remainder_order {
            if extra == 0 {
                break;
            }
            if quotas[decile] < ordered[decile].1.len() {
                quotas[decile] += 1;
                extra -= 1;
            }
        }
        if extra != 0 {
            bail!("could not allocate exact 20 percent validation split");
        }
        let mut validation_indices = Vec::with_capacity(validation_total);
        for (decile, rows, _, _) in &ordered {
            validation_indices.extend_from_slice(&rows[..quotas[*decile]]);
        }
        validation_indices.sort_unstable();
        let validation_set = validation_indices.iter().copied().collect::<HashSet<_>>();
        let train_indices = (0..targets.len())
            .filter(|row| !validation_set.contains(row))
            .collect::<Vec<_>>();
        if validation_indices.len() != validation_total
            || train_indices.len() + validation_indices.len() != targets.len()
            || train_indices.iter().any(|row| validation_set.contains(row))
        {
            bail!("split allocation overlap or size error");
        }
        let validation_targets = validation_indices
            .iter()
            .map(|row| targets[*row])
            .collect::<Vec<_>>();
        let mean = validation_targets.iter().sum::<f64>() / validation_targets.len() as f64;
        if validation_targets
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            <= 1e-24
        {
            bail!("generation {generation} validation target variance is zero");
        }
        let validation_target_sha256 = sha256_f64(&validation_targets);
        let split_value = json!({
            "dataset_hash": dataset_hash,
            "generation": generation,
            "seed": split_seed,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "validation_target_sha256": validation_target_sha256,
        });
        let split_sha256 = sha256_bytes(&canonical_json_bytes(&split_value)?);
        if !seen_hashes.insert(split_sha256.clone()) {
            bail!("validation split hashes are not distinct");
        }
        receipts.push(SplitReceipt {
            generation,
            seed: split_seed,
            train_indices: split_value["train_indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect(),
            validation_indices: split_value["validation_indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect(),
            validation_target_sha256,
            split_sha256,
        });
    }
    Ok(TrainOnlySplitManifest {
        schema: "TrainOnlySplitManifest.v2".to_string(),
        run_id: run_id.to_string(),
        dataset_hash: dataset_hash.to_string(),
        train_sha256: train_sha256.to_string(),
        seed: STUDY_SEED,
        rows: targets.len(),
        splits: receipts,
    })
}

pub(super) fn build_final_cv_splits(
    run_id: &str,
    dataset_hash: &str,
    train_sha256: &str,
    targets: &[f64],
) -> Result<TrainOnlySplitManifest> {
    if targets.len() < 100 || targets.iter().any(|value| !value.is_finite()) {
        bail!("final CV construction needs at least 100 finite targets");
    }
    let cv_seed = {
        let mut hash = Sha256::new();
        hash.update(b"checkpoint-assisted-final-top10-cv-seed-v2\0");
        hash.update(STUDY_SEED.to_le_bytes());
        hash.update(dataset_hash.as_bytes());
        u64::from_le_bytes(hash.finalize()[..8].try_into().unwrap())
    };
    let mut sorted = (0..targets.len()).collect::<Vec<_>>();
    sorted.sort_by(|left, right| {
        targets[*left]
            .total_cmp(&targets[*right])
            .then_with(|| left.cmp(right))
    });
    let mut deciles = vec![Vec::new(); 10];
    let sorted_len = sorted.len();
    for (rank, row) in sorted.into_iter().enumerate() {
        deciles[(rank * 10 / sorted_len).min(9)].push(row);
    }
    let mut validation_by_fold = vec![Vec::new(); FINAL_CV_FOLDS];
    for (decile, rows) in deciles.iter_mut().enumerate() {
        rows.sort_by_key(|row| {
            let mut hash = Sha256::new();
            hash.update(b"checkpoint-assisted-final-top10-cv-order-v2\0");
            hash.update(cv_seed.to_le_bytes());
            hash.update((*row as u64).to_le_bytes());
            hex::encode(hash.finalize())
        });
        for (position, row) in rows.iter().enumerate() {
            validation_by_fold[(position + decile) % FINAL_CV_FOLDS].push(*row);
        }
    }
    let mut receipts = Vec::with_capacity(FINAL_CV_FOLDS);
    let mut seen_rows = HashSet::new();
    let mut seen_hashes = HashSet::new();
    for (fold, validation_indices) in validation_by_fold.iter_mut().enumerate() {
        validation_indices.sort_unstable();
        if validation_indices.is_empty()
            || validation_indices.iter().any(|row| !seen_rows.insert(*row))
        {
            bail!("final CV validation folds overlap or are empty");
        }
        let validation_set = validation_indices.iter().copied().collect::<HashSet<_>>();
        let train_indices = (0..targets.len())
            .filter(|row| !validation_set.contains(row))
            .collect::<Vec<_>>();
        let validation_targets = validation_indices
            .iter()
            .map(|row| targets[*row])
            .collect::<Vec<_>>();
        let mean = validation_targets.iter().sum::<f64>() / validation_targets.len() as f64;
        if validation_targets
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            <= 1e-24
        {
            bail!("final CV fold {fold} target variance is zero");
        }
        let split_seed = {
            let mut hash = Sha256::new();
            hash.update(b"checkpoint-assisted-final-top10-cv-fold-v2\0");
            hash.update(cv_seed.to_le_bytes());
            hash.update((fold as u64).to_le_bytes());
            u64::from_le_bytes(hash.finalize()[..8].try_into().unwrap())
        };
        let validation_target_sha256 = sha256_f64(&validation_targets);
        let split_value = json!({
            "purpose": "final_top10_cv_rerank",
            "dataset_hash": dataset_hash,
            "fold": fold,
            "seed": split_seed,
            "train_indices": &train_indices,
            "validation_indices": &validation_indices,
            "validation_target_sha256": validation_target_sha256,
        });
        let split_sha256 = sha256_bytes(&canonical_json_bytes(&split_value)?);
        if !seen_hashes.insert(split_sha256.clone()) {
            bail!("final CV split hashes are not distinct");
        }
        receipts.push(SplitReceipt {
            generation: fold,
            seed: split_seed,
            train_indices,
            validation_indices: validation_indices.clone(),
            validation_target_sha256,
            split_sha256,
        });
    }
    if seen_rows.len() != targets.len() {
        bail!("final CV folds do not cover every training row exactly once");
    }
    Ok(TrainOnlySplitManifest {
        schema: "TrainOnlyFinalCvSplitManifest.v2".to_string(),
        run_id: run_id.to_string(),
        dataset_hash: dataset_hash.to_string(),
        train_sha256: train_sha256.to_string(),
        seed: cv_seed,
        rows: targets.len(),
        splits: receipts,
    })
}

pub(super) fn rank_final_cv(left: &FinalCvCandidate, right: &FinalCvCandidate) -> Ordering {
    match (left.robust_fitness, right.robust_fitness) {
        (Some(left_fitness), Some(right_fitness)) => right_fitness
            .total_cmp(&left_fitness)
            .then_with(|| {
                right
                    .worst_fold_r2
                    .unwrap()
                    .total_cmp(&left.worst_fold_r2.unwrap())
            })
            .then_with(|| left.complexity.cmp(&right.complexity))
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256)),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => left
            .complexity
            .cmp(&right.complexity)
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256)),
    }
}
