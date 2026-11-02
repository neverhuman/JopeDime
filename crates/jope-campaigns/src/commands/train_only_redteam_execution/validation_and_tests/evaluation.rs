#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct DatasetValidationReplay {
    pub dataset_hash: String,
    pub g20_complete_sha256: String,
    pub smart_complete_sha256: String,
    pub selector_sha256: String,
    pub evaluation_sha256: BTreeMap<String, String>,
    pub deterministic_generation_replay: bool,
    pub uncached_metric_parity: bool,
    #[serde(default)]
    pub scq_execution: Option<ScqExecutionProvenance>,
    pub development_only: bool,
    pub confirmatory_valid: bool,
}

fn uncached_evaluation(
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    data: &RegressionData,
    bank_name: &str,
    repeats: &[FoldRepeat],
    candidate: &Candidate,
) -> Result<CandidateEvaluation> {
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
            fold_scores.push(
                (squared_error(&validation_y, &predictions)? / validation.len() as f64).sqrt() / sd,
            );
            for (row, prediction) in validation.iter().zip(predictions) {
                oof[*row] = prediction;
            }
        }
        repeat_squared_error.push(squared_error(&data.y, &oof)?);
        repeat_rows.push(data.y.len());
        repeat_predictions.push(oof);
        fold_nrmse.push(fold_scores);
    }
    let score = robust_cv_score(&repeat_squared_error, &repeat_rows, fold_nrmse, sd)?;
    Ok(CandidateEvaluation {
        schema: "TrainOnlyRedteamCandidateEvaluation.v1".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        train_sha256: dataset.train_sha256.clone(),
        bank: bank_name.to_string(),
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
    })
}

fn evaluations_match(left: &CandidateEvaluation, right: &CandidateEvaluation) -> bool {
    left.schema == right.schema
        && left.run_id == right.run_id
        && left.dataset_hash == right.dataset_hash
        && left.train_sha256 == right.train_sha256
        && left.bank == right.bank
        && left.repeat_sha256 == right.repeat_sha256
        && left.candidate == right.candidate
        && left.target_sd == right.target_sd
        && left.repeat_squared_error == right.repeat_squared_error
        && left.repeat_rows == right.repeat_rows
        && left.repeat_predictions == right.repeat_predictions
        && left.score.repeat_nrmse == right.score.repeat_nrmse
        && left.score.fold_nrmse == right.score.fold_nrmse
        && left.score.robust_loss == right.score.robust_loss
        && left.score.worst_fold_nrmse == right.score.worst_fold_nrmse
        && left.development_only == right.development_only
        && left.confirmatory_valid == right.confirmatory_valid
}

// Keep ranked receipt identity and the independent validation inputs explicit.
#[allow(clippy::too_many_arguments)]
fn validate_ranked_rows(
    rows: &[RankedCandidate],
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    data: &RegressionData,
    bank_name: &str,
    repeats: &[FoldRepeat],
    uncached: bool,
    hashes: &mut BTreeMap<String, String>,
) -> Result<()> {
    let mut semantics = HashSet::new();
    for row in rows {
        if !semantics.insert(row.candidate.semantic_sha256.clone()) {
            bail!("ranked candidate set contains a duplicate");
        }
        let cached = load_ranked_evaluation(row)?;
        if cached.bank != bank_name {
            bail!("ranked candidate uses the wrong fold bank");
        }
        if uncached {
            let replay =
                uncached_evaluation(manifest, dataset, data, bank_name, repeats, &row.candidate)?;
            if !evaluations_match(&cached, &replay) {
                bail!("cached/uncached evaluation parity failure");
            }
        }
        hashes.insert(
            format!("{bank_name}:{}", row.candidate.semantic_sha256),
            row.evaluation_sha256.clone(),
        );
    }
    Ok(())
}
