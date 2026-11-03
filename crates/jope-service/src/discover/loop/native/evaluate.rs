use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct NativeFolds {
    pub(super) version: u32,
    pub(super) matrix_sha256: String,
    pub(super) target: String,
    pub(super) features: Vec<String>,
    pub(super) folds: Vec<Indices>,
    pub(super) refit_output: Option<PathBuf>,
    pub(super) refit_max_bytes: Option<u64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Indices {
    repeat: u8,
    train: Vec<usize>,
    validation: Vec<usize>,
}

/// Existing WRK-01 callback. It reads only its supplied FIT mapping and returns
/// three finite values: M1 RMSE surrogate, minimization objective, user value.
/// This heuristic surrogate is explicit; it is not qualified as bundle R0/M3.
pub fn evaluate_native<B: AsRef<[u8]> + Sync>(
    request: &Request,
    matrix: &Matrix<B>,
) -> std::result::Result<worker::Scores, WorkerFailure> {
    // A dedicated numerical pool overrides ambient Rayon settings. The binary's
    // fixed one-thread fits do not change when live worker capacity changes.
    let result = jope_runtime::budget::Budget::new(None, 1)
        .install(|| native_scores(request, matrix))
        .map_err(|_| WorkerFailure::Unavailable)?;
    result.map_err(|error| {
        if error.is::<UndefinedFitness>() {
            WorkerFailure::NonFinite
        } else {
            WorkerFailure::InvalidRequest
        }
    })
}

#[derive(Debug)]
pub(super) struct UndefinedFitness;
impl std::fmt::Display for UndefinedFitness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("candidate has undefined fitness or numerical predictions")
    }
}
impl std::error::Error for UndefinedFitness {}

pub(super) fn native_scores<B: AsRef<[u8]>>(
    request: &Request,
    matrix: &Matrix<B>,
) -> Result<worker::Scores> {
    let spec: NativeFolds = serde_json::from_value(request.fold_spec.clone())?;
    ensure!(
        spec.version == NATIVE_FOLDS_VERSION && spec.folds.len() >= 2,
        "invalid native folds"
    );
    ensure!(
        digest(matrix.bytes()) == spec.matrix_sha256,
        "mapped FIT matrix identity mismatch"
    );
    let objective = RegressionObjective::parse(
        request
            .objective
            .as_str()
            .context("objective must be a specification string")?,
    )?;
    feat_math::GenomeIr::parse(&request.dsl)?;
    match (&spec.refit_output, spec.refit_max_bytes) {
        (Some(path), Some(maximum)) => {
            return super::model::native_refit(request, matrix, &spec, &objective, path, maximum)
        }
        (None, None) => {}
        _ => bail!("incomplete private refit handoff"),
    }
    let mut targets = vec![];
    let mut predictions = vec![];
    let mut validation_rows = BTreeSet::new();
    let mut losses = vec![];
    let mut values = vec![];
    let mut surrogate = vec![];
    let mut fold_scores = vec![];
    for fold in spec.folds {
        let train: BTreeSet<_> = fold.train.iter().copied().collect();
        ensure!(
            !train.is_empty() && train.len() == fold.train.len() && !fold.validation.is_empty(),
            "empty or duplicated fold"
        );
        ensure!(
            fold.validation
                .iter()
                .all(|r| !train.contains(r) && validation_rows.insert((fold.repeat, *r))),
            "overlapping fold rows"
        );
        let (x, y) = numeric_rows(matrix, &spec.target, &spec.features, &fold.train)?;
        let fitted =
            NativeModel::fit(&spec.features, &request.dsl, &x, &y).map_err(|_| UndefinedFitness)?;
        let (vx, vy) = numeric_rows(matrix, &spec.target, &spec.features, &fold.validation)?;
        let prediction = fitted.predict(&vx).map_err(|_| UndefinedFitness)?;
        let score = objective
            .evaluate(&vy, &prediction)
            .map_err(|_| UndefinedFitness)?;
        let rmse = RegressionObjective::parse("rmse")?
            .evaluate(&vy, &prediction)
            .map_err(|_| UndefinedFitness)?;
        losses.push(score.loss);
        values.push(score.value);
        surrogate.push(rmse.loss);
        fold_scores.push(rmse.loss);
        targets.extend(vy);
        predictions.extend(prediction);
    }
    let mean = |v: &[f64]| v.iter().map(|x| x / v.len() as f64).sum::<f64>();
    let scores = vec![mean(&surrogate), mean(&losses), mean(&values)];
    ensure!(scores.iter().all(|x| x.is_finite()), "non-finite CV score");
    let mut bytes = Vec::with_capacity(predictions.len() * 16);
    for (y, p) in targets.into_iter().zip(predictions) {
        ensure!(p.is_finite(), "non-finite prediction");
        bytes.extend(y.to_le_bytes());
        bytes.extend(p.to_le_bytes());
    }
    Ok(worker::Scores {
        values: scores,
        fold_scores,
        predictions_digest: digest(&bytes),
    })
}

pub(in crate::discover::r#loop) fn numeric_rows<B: AsRef<[u8]>>(
    matrix: &Matrix<B>,
    target: &str,
    features: &[String],
    rows: &[usize],
) -> Result<(feat_math::Rows, Vec<f64>)> {
    ensure!(
        !features.is_empty()
            && features.iter().collect::<BTreeSet<_>>().len() == features.len()
            && !features.iter().any(|f| f == target),
        "invalid native features"
    );
    let find = |name: &str| {
        matrix
            .columns()
            .iter()
            .position(|c| c.name == name)
            .context("unknown native column")
    };
    let target = find(target)?;
    let columns: Vec<_> = features.iter().map(|s| find(s)).collect::<Result<_>>()?;
    let mut x = vec![];
    let mut y = vec![];
    for &row in rows {
        let JmatCell::Numeric(label) =
            matrix.cell(row, target).context("row outside FIT matrix")?
        else {
            bail!("invalid numeric target");
        };
        ensure!(label.is_finite(), "non-finite target");
        y.push(label);
        x.push(
            columns
                .iter()
                .map(|&col| -> Result<f64> {
                    match matrix.cell(row, col) {
                        Some(JmatCell::Numeric(v)) if v.is_finite() => Ok(v),
                        Some(JmatCell::Missing) => Ok(f64::NAN),
                        _ => bail!("invalid numeric feature"),
                    }
                })
                .collect::<Result<_>>()?,
        );
    }
    Ok((x, y))
}

/// Internal CAS fit, deliberately separate from the frozen exported bundle
/// format. Every preprocessing parameter is fitted inside the training fold.
#[derive(Serialize, Deserialize)]
pub(crate) struct NativeModel {
    pub(super) version: u32,
    pub(super) features: Vec<String>,
    pub(super) dsl: String,
    pub(super) min: Vec<f64>,
    pub(super) max: Vec<f64>,
    pub(super) fill: Vec<f64>,
    pub(super) target_min: f64,
    pub(super) target_scale: f64,
    pub(super) model: feat_math::Model,
}
