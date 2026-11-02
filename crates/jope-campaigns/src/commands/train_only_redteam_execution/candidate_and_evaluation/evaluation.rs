use super::*;

pub(super) fn complement(rows: usize, validation: &[usize]) -> Result<Vec<usize>> {
    let mut present = vec![false; rows];
    for row in validation {
        if *row >= rows || present[*row] {
            bail!("fold validation indices repeat or exceed row count");
        }
        present[*row] = true;
    }
    Ok(present
        .into_iter()
        .enumerate()
        .filter_map(|(row, validation)| (!validation).then_some(row))
        .collect())
}

pub(super) fn slice_rows(data: &RegressionData, indices: &[usize]) -> (Vec<Vec<f64>>, Vec<f64>) {
    (
        indices.iter().map(|row| data.x[*row].clone()).collect(),
        indices.iter().map(|row| data.y[*row]).collect(),
    )
}

pub(super) fn squared_error(truth: &[f64], predictions: &[f64]) -> Result<f64> {
    if truth.len() != predictions.len()
        || truth.is_empty()
        || truth
            .iter()
            .chain(predictions)
            .any(|value| !value.is_finite())
    {
        bail!("squared error inputs are invalid");
    }
    Ok(truth
        .iter()
        .zip(predictions)
        .map(|(truth, prediction)| (truth - prediction).powi(2))
        .sum())
}

pub(super) fn evaluation_path(
    root: &Path,
    dataset_hash: &str,
    bank: &str,
    candidate: &Candidate,
    repeats: &[FoldRepeat],
) -> Result<PathBuf> {
    let key = receipt_digest(&json!({
        "bank": bank,
        "candidate": candidate.semantic_sha256,
        "repeat_sha256": repeats.iter().map(|repeat| &repeat.repeat_sha256).collect::<Vec<_>>(),
        "metric": "pooled-oof-nrmse-v1",
    }))?;
    Ok(root
        .join("evaluations")
        .join(dataset_hash)
        .join(bank)
        .join(format!("{key}.v1.json")))
}

pub(super) fn evaluate_candidate(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    data: &RegressionData,
    bank: &str,
    repeats: &[FoldRepeat],
    candidate: &Candidate,
) -> Result<(CandidateEvaluation, String, PathBuf)> {
    let path = evaluation_path(root, &dataset.dataset_hash, bank, candidate, repeats)?;
    if path.exists() {
        let (evaluation, digest): (CandidateEvaluation, _) = read_receipt(&path)?;
        if evaluation.schema != "TrainOnlyRedteamCandidateEvaluation.v1"
            || evaluation.run_id != manifest.run_id
            || evaluation.dataset_hash != dataset.dataset_hash
            || evaluation.train_sha256 != dataset.train_sha256
            || evaluation.bank != bank
            || evaluation.candidate.semantic_sha256 != candidate.semantic_sha256
            || !evaluation.development_only
            || evaluation.confirmatory_valid
            || evaluation.repeat_sha256
                != repeats
                    .iter()
                    .map(|repeat| repeat.repeat_sha256.clone())
                    .collect::<Vec<_>>()
        {
            bail!("candidate evaluation cache identity drift");
        }
        return Ok((evaluation, digest, path));
    }
    let parsed = GenomeIr::parse(&candidate.canonical_genome)?;
    let sd = target_sd(&data.y)?;
    let mut repeat_squared_error = Vec::with_capacity(repeats.len());
    let mut repeat_rows = Vec::with_capacity(repeats.len());
    let mut repeat_predictions = Vec::with_capacity(repeats.len());
    let mut fold_nrmse = Vec::with_capacity(repeats.len());
    for repeat in repeats {
        let mut oof = vec![f64::NAN; data.y.len()];
        let mut fold_scores = Vec::with_capacity(FOLDS);
        for validation in &repeat.validation_indices {
            let training = complement(data.y.len(), validation)?;
            let (train_x, train_y) = slice_rows(data, &training);
            let (validation_x, validation_y) = slice_rows(data, validation);
            let model = fit(&parsed, &train_x, &train_y)?;
            let predictions = model.predict(&validation_x)?;
            let fold_error = squared_error(&validation_y, &predictions)?;
            fold_scores.push((fold_error / validation.len() as f64).sqrt() / sd);
            for (row, prediction) in validation.iter().zip(predictions) {
                oof[*row] = prediction;
            }
        }
        let error = squared_error(&data.y, &oof)?;
        repeat_squared_error.push(error);
        repeat_rows.push(data.y.len());
        repeat_predictions.push(oof);
        fold_nrmse.push(fold_scores);
    }
    let score = robust_cv_score(&repeat_squared_error, &repeat_rows, fold_nrmse, sd)?;
    let evaluation = CandidateEvaluation {
        schema: "TrainOnlyRedteamCandidateEvaluation.v1".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        train_sha256: dataset.train_sha256.clone(),
        bank: bank.to_string(),
        repeat_sha256: repeats
            .iter()
            .map(|repeat| repeat.repeat_sha256.clone())
            .collect(),
        candidate: candidate.clone(),
        target_sd: sd,
        repeat_squared_error,
        repeat_rows,
        repeat_predictions,
        score,
        development_only: true,
        confirmatory_valid: false,
    };
    let digest = write_receipt(&path, &evaluation)?;
    Ok((evaluation, digest, path))
}

// Keep study identity, candidate pool, input folds and execution budget explicit.
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_candidates(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    data: &RegressionData,
    bank: &str,
    repeats: &[FoldRepeat],
    candidates: &[Candidate],
    workers: usize,
) -> Result<Vec<RankedCandidate>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    let mut ranked = pool.install(|| {
        candidates
            .par_iter()
            .map(|candidate| {
                let (evaluation, digest, path) =
                    evaluate_candidate(root, manifest, dataset, data, bank, repeats, candidate)?;
                Ok(RankedCandidate {
                    candidate: candidate.clone(),
                    evaluation_path: absolute(&path)?,
                    evaluation_sha256: digest,
                    robust_loss: evaluation.score.robust_loss,
                    worst_fold_nrmse: evaluation.score.worst_fold_nrmse,
                })
            })
            .collect::<Result<Vec<_>>>()
    })?;
    ranked.sort_by(rank_loss);
    Ok(ranked)
}

pub(super) fn rank_loss(left: &RankedCandidate, right: &RankedCandidate) -> Ordering {
    left.robust_loss
        .total_cmp(&right.robust_loss)
        .then_with(|| left.worst_fold_nrmse.total_cmp(&right.worst_fold_nrmse))
        .then_with(|| left.candidate.complexity.cmp(&right.candidate.complexity))
        .then_with(|| {
            left.candidate
                .semantic_sha256
                .cmp(&right.candidate.semantic_sha256)
        })
}
