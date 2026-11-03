use super::*;

// ---------------------------------------------------------------------------
// fold slices
// ---------------------------------------------------------------------------

struct BinaryFoldSlices {
    train_table: RawTable,
    train_y: Vec<f64>,
    valid_table: RawTable,
    valid_y: Vec<f64>,
    validation_rows: Vec<usize>,
    valid_positives: usize,
    valid_negatives: usize,
}

/// Campaign-lifetime cache of stratified fold slices keyed by
/// (fold-plan size, fold); string tables are built once per fold instead of
/// once per genome evaluation.
pub(super) struct BinaryFoldCache<'a> {
    train: &'a RegressionData,
    dataset: &'a str,
    slices: Mutex<HashMap<(usize, usize), Arc<BinaryFoldSlices>>>,
}

impl<'a> BinaryFoldCache<'a> {
    pub(super) fn new(train: &'a RegressionData, dataset: &'a str) -> Self {
        Self {
            train,
            dataset,
            slices: Mutex::new(HashMap::new()),
        }
    }

    pub(super) fn fold(
        &self,
        fold_plan_folds: usize,
        fold: usize,
    ) -> Result<Arc<BinaryFoldSlices>> {
        if let Some(hit) = self
            .slices
            .lock()
            .expect("binary fold-slice cache lock")
            .get(&(fold_plan_folds, fold))
        {
            return Ok(Arc::clone(hit));
        }
        let n = self.train.x.len().min(self.train.y.len());
        let assignments =
            stratified_assignments(self.dataset, &self.train.y[..n], fold_plan_folds)?;
        let mut train_x: Vec<Vec<f64>> = Vec::new();
        let mut train_y = Vec::new();
        let mut valid_x: Vec<Vec<f64>> = Vec::new();
        let mut valid_y = Vec::new();
        let mut validation_rows = Vec::new();
        for (row, &assigned_fold) in assignments[..n].iter().enumerate() {
            if assigned_fold == fold {
                valid_x.push(self.train.x[row].clone());
                valid_y.push(self.train.y[row]);
                validation_rows.push(row);
            } else {
                train_x.push(self.train.x[row].clone());
                train_y.push(self.train.y[row]);
            }
        }
        let valid_positives = valid_y.iter().filter(|v| **v == 1.0).count();
        let valid_negatives = valid_y.len() - valid_positives;
        let built = Arc::new(BinaryFoldSlices {
            train_table: raw_table(self.train.features, &train_x)?,
            train_y,
            valid_table: raw_table(self.train.features, &valid_x)?,
            valid_y,
            validation_rows,
            valid_positives,
            valid_negatives,
        });
        let mut slices = self.slices.lock().expect("binary fold-slice cache lock");
        Ok(Arc::clone(
            slices.entry((fold_plan_folds, fold)).or_insert(built),
        ))
    }
}

// ---------------------------------------------------------------------------
// CV evaluation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct BinaryCvResult {
    /// Per evaluated fold; NaN when the fold's validation split is single-class.
    pub(crate) fold_auc: Vec<f64>,
    pub(crate) fold_defined: Vec<bool>,
    pub(crate) pooled_oof_auc: f64,
    pub(crate) pooled_oof_logloss: f64,
    pub(crate) pooled_oof_brier: f64,
    pub(crate) robust_fitness: f64,
    /// Wall-clock diagnostics; intentionally absent from every receipt so
    /// binary receipts stay byte-deterministic.
    #[allow(dead_code)]
    pub(crate) fit_seconds: Vec<f64>,
    #[allow(dead_code)]
    pub(crate) predict_seconds: Vec<f64>,
    /// Pooled OOF evidence for threshold freezing.
    pub(crate) oof_truth: Vec<f64>,
    pub(crate) oof_probabilities: Vec<f64>,
}

/// Public-shape binary CV entry (tests + future audit lanes); campaigns use
/// the shared-slice body directly.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn exact_cv_on_folds_binary(
    genome: &GenomeIr,
    train: &RegressionData,
    fold_plan_folds: usize,
    evaluated_folds: &[usize],
    dataset: &str,
) -> Result<BinaryCvResult> {
    binary_cv_on_slices(
        genome,
        &BinaryFoldCache::new(train, dataset),
        fold_plan_folds,
        evaluated_folds,
        &binary_task(),
        &binary_schema(train.features),
    )
}

pub(super) fn binary_cv_on_slices(
    genome: &GenomeIr,
    fold_cache: &BinaryFoldCache<'_>,
    fold_plan_folds: usize,
    evaluated_folds: &[usize],
    task: &TaskSpec,
    schema: &DataSchema,
) -> Result<BinaryCvResult> {
    let train = fold_cache.train;
    let n = train.x.len().min(train.y.len());
    let mut oof = vec![f64::NAN; n];
    let mut fold_auc = Vec::with_capacity(evaluated_folds.len());
    let mut fold_defined = Vec::with_capacity(evaluated_folds.len());
    let mut fit_seconds = Vec::with_capacity(evaluated_folds.len());
    let mut predict_seconds = Vec::with_capacity(evaluated_folds.len());
    if evaluated_folds.is_empty() || evaluated_folds.iter().any(|fold| *fold >= fold_plan_folds) {
        bail!("evaluated folds are empty or outside the fold plan");
    }
    for &fold in evaluated_folds {
        let slices = fold_cache.fold(fold_plan_folds, fold)?;
        if slices.valid_table.rows.is_empty() || slices.train_table.rows.len() < 8 {
            bail!("fold {fold} is empty or has fewer than eight training rows");
        }
        let fit_started = Instant::now();
        let model = fit_table(genome, task, schema, &slices.train_table, &slices.train_y)
            .with_context(|| format!("fit fold {fold}"))?;
        fit_seconds.push(fit_started.elapsed().as_secs_f64());
        let predict_started = Instant::now();
        let predictions = model
            .predict_table(&slices.valid_table)
            .with_context(|| format!("predict fold {fold}"))?;
        predict_seconds.push(predict_started.elapsed().as_secs_f64());
        validate_probabilities(&predictions, slices.valid_y.len())
            .with_context(|| format!("fold {fold}"))?;
        // Single-class validation split: AUC is undefined for the fold, its
        // rows still pool into the OOF evidence.
        if slices.valid_positives == 0 || slices.valid_negatives == 0 {
            fold_auc.push(f64::NAN);
            fold_defined.push(false);
        } else {
            let score = auc(&slices.valid_y, &predictions);
            if !score.is_finite() {
                bail!("fold {fold} AUC is non-finite despite both classes present");
            }
            fold_auc.push(score);
            fold_defined.push(true);
        }
        for (index, prediction) in slices.validation_rows.iter().copied().zip(predictions) {
            oof[index] = prediction;
        }
    }
    let mut oof_truth = Vec::new();
    let mut oof_probabilities = Vec::new();
    for (row, prediction) in oof.into_iter().enumerate() {
        if prediction.is_finite() {
            oof_truth.push(train.y[row]);
            oof_probabilities.push(prediction);
        }
    }
    if oof_truth.is_empty() {
        bail!("CV produced no held-out probabilities");
    }
    let pooled_positives = oof_truth.iter().filter(|v| **v == 1.0).count();
    if pooled_positives == 0 || pooled_positives == oof_truth.len() {
        bail!("pooled OOF evidence is single-class; AUC authority is undefined");
    }
    let pooled_oof_auc = auc(&oof_truth, &oof_probabilities);
    let pooled_oof_logloss = logloss(&oof_truth, &oof_probabilities);
    let pooled_oof_brier = brier(&oof_truth, &oof_probabilities);
    let robust_fitness = robust_fitness_binary(pooled_oof_auc, &fold_auc);
    if !pooled_oof_auc.is_finite() || !robust_fitness.is_finite() || !pooled_oof_logloss.is_finite()
    {
        bail!("pooled OOF metrics or robust fitness are non-finite");
    }
    Ok(BinaryCvResult {
        fold_auc,
        fold_defined,
        pooled_oof_auc,
        pooled_oof_logloss,
        pooled_oof_brier,
        robust_fitness,
        fit_seconds,
        predict_seconds,
        oof_truth,
        oof_probabilities,
    })
}
