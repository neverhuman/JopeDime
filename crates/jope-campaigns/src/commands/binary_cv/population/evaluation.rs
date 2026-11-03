use super::*;

pub(crate) fn fit_test_adaptive_binary(
    canonical_genome: &str,
    train: &RegressionData,
    test: &RegressionData,
) -> Result<(GenomeIr, BoundModel, f64)> {
    if train.features != test.features {
        bail!("binary test-adaptive train/test feature widths differ");
    }
    let genome = GenomeIr::parse(canonical_genome).context("parse binary adaptive genome")?;
    if genome.canonical() != canonical_genome {
        bail!("binary test-adaptive genome is not canonical");
    }
    let task = binary_task();
    let schema = binary_schema(train.features);
    validate_v3_genome(&genome, &task, &schema)?;
    let train_rows = train.x.len().min(train.y.len());
    if train_rows != train.x.len() || train_rows == 0 || test.y.len() != test.x.len() {
        bail!("binary test-adaptive data has inconsistent row counts");
    }
    let train_table = raw_table(train.features, &train.x)?;
    let fit_started = Instant::now();
    let model = fit_table(
        &genome,
        &task,
        &schema,
        &train_table,
        &train.y[..train_rows],
    )?;
    let fit_seconds = fit_started.elapsed().as_secs_f64();
    Ok((genome, model, fit_seconds))
}

pub(crate) fn finish_test_adaptive_binary(
    complexity: usize,
    predictions: Vec<f64>,
    test: &RegressionData,
    fit_seconds: f64,
    predict_seconds: f64,
) -> Result<BinaryAdaptiveEvaluation> {
    validate_probabilities(&predictions, test.y.len())?;
    let auc = auc(&test.y, &predictions);
    let logloss = logloss(&test.y, &predictions);
    if !auc.is_finite() || !logloss.is_finite() {
        bail!("binary test-adaptive metrics are non-finite");
    }
    Ok(BinaryAdaptiveEvaluation {
        complexity,
        auc,
        logloss,
        predictions,
        fit_seconds,
        predict_seconds,
    })
}

// ---------------------------------------------------------------------------
// campaign
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct BinaryEvaluation {
    pub(in crate::commands::binary_cv) complexity: usize,
    pub(in crate::commands::binary_cv) result: Option<BinaryCvResult>,
    pub(in crate::commands::binary_cv) rejection_reason: Option<String>,
}

pub(crate) fn evaluate_binary_unique(
    member: &BinaryMember,
    fold_cache: &BinaryFoldCache<'_>,
    task: &TaskSpec,
    schema: &DataSchema,
) -> BinaryEvaluation {
    let rejected = |complexity, reason: String| BinaryEvaluation {
        complexity,
        result: None,
        rejection_reason: Some(reason),
    };
    let genome = match GenomeIr::parse(&member.canonical_genome) {
        Ok(genome) if matches!(genome, GenomeIr::Pipeline(_)) => genome,
        Ok(_) => return rejected(0, "not a pipeline genome".to_string()),
        Err(error) => return rejected(0, format!("parse: {error:#}")),
    };
    let canonical = genome.canonical();
    let complexity = genome_complexity(&genome);
    if canonical != member.canonical_genome
        || sha256_bytes(canonical.as_bytes()) != member.semantic_sha256
    {
        return rejected(
            complexity,
            "canonical or semantic identity mismatch".to_string(),
        );
    }
    if let Err(error) = validate_v3_genome(&genome, task, schema) {
        return rejected(complexity, format!("admission: {error:#}"));
    }
    match binary_cv_on_slices(
        &genome,
        fold_cache,
        CV5_FOLDS,
        &[QUICK_HOLDOUT_FOLD],
        task,
        schema,
    ) {
        Ok(result) => BinaryEvaluation {
            complexity,
            result: Some(result),
            rejection_reason: None,
        },
        Err(error) => rejected(complexity, format!("cv: {error:#}")),
    }
}
