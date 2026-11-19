use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::discover::r#loop) struct NativeFolds {
    pub(super) version: u32,
    pub(super) matrix_sha256: String,
    pub(super) target: String,
    pub(super) features: Vec<String>,
    pub(super) folds: Vec<Indices>,
    pub(super) refit_output: Option<PathBuf>,
    pub(super) refit_max_bytes: Option<u64>,
    #[serde(default)]
    pub(super) calibration_mode: Option<String>,
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
        if error.is::<EvalWallClockTimeout>() {
            WorkerFailure::Timeout
        } else if error.is::<UndefinedFitness>() {
            WorkerFailure::NonFinite
        } else {
            WorkerFailure::InvalidRequest
        }
    })
}

#[derive(Debug)]
struct EvalWallClockTimeout;
impl std::fmt::Display for EvalWallClockTimeout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("candidate evaluation exceeded configured wall-clock budget")
    }
}
impl std::error::Error for EvalWallClockTimeout {}

#[derive(Default)]
struct SerializedByteCounter(u64);

impl std::io::Write for SerializedByteCounter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 = self.0.saturating_add(buf.len() as u64);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn serialized_json_byte_len<T: Serialize>(value: &T) -> u64 {
    let mut counter = SerializedByteCounter::default();
    if serde_json::to_writer(&mut counter, value).is_ok() {
        counter.0
    } else {
        0
    }
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
        matrix.sha256_hex() == spec.matrix_sha256,
        "mapped FIT matrix identity mismatch"
    );
    let objective_raw = request
        .objective
        .as_str()
        .context("objective must be a specification string")?;
    if let Ok(binary_objective) =
        crate::objective::binary::BinaryObjective::parse_json(objective_raw)
    {
        return binary_native_scores(request, matrix, spec, &binary_objective);
    }
    let objective = RegressionObjective::parse(objective_raw)?;
    let calibration_family =
        resolve_calibration_family(spec.calibration_mode.as_deref(), objective_raw)?;
    let aux_names = objective.columns();
    let parsed_genome = feat_math::GenomeIr::parse(&request.dsl)?;
    let complexity_stages = match &parsed_genome {
        feat_math::GenomeIr::Pipeline(expr) => expr.stages.len() + 1,
        feat_math::GenomeIr::Symbolic { .. } => 1,
        feat_math::GenomeIr::Portfolio(p) => {
            p.members.iter().map(|m| m.stages.len() + 1).sum::<usize>() + 1
        }
    };
    match (&spec.refit_output, spec.refit_max_bytes) {
        (Some(path), Some(maximum)) => {
            return super::model::native_refit(
                request,
                matrix,
                &spec,
                &objective,
                calibration_family,
                path,
                maximum,
            )
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
    let eval_started = std::time::Instant::now();
    for (fold_idx, fold) in spec.folds.into_iter().enumerate() {
        if fold_idx > 0
            && request.limits.wall_ms > 0
            && eval_started.elapsed().as_millis() >= u128::from(request.limits.wall_ms)
        {
            return Err(EvalWallClockTimeout.into());
        }
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
        let train_aux = aux_columns_for_rows(matrix, &aux_names, &fold.train)?;
        let train_t0 = std::time::Instant::now();
        let fitted = NativeModel::fit_with_calibration(
            &spec.features,
            &request.dsl,
            &x,
            &y,
            &objective,
            calibration_family,
            &train_aux,
        )
        .map_err(|_| UndefinedFitness)?;
        let train_secs = train_t0.elapsed().as_secs_f64();
        let (vx, vy) = numeric_rows(matrix, &spec.target, &spec.features, &fold.validation)?;
        let val_aux = aux_columns_for_rows(matrix, &aux_names, &fold.validation)?;
        let infer_t0 = std::time::Instant::now();
        let prediction = fitted.predict(&vx).map_err(|_| UndefinedFitness)?;
        let infer_secs = infer_t0.elapsed().as_secs_f64();
        let rmse = RegressionObjective::parse("rmse")?
            .evaluate(&vy, &prediction)
            .map_err(|_| UndefinedFitness)?;
        surrogate.push(rmse.loss);
        fold_scores.push(rmse.loss);
        let model_bytes = serialized_json_byte_len(&fitted);
        let feat_w = spec.features.len().max(1);
        let fold_mean = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        let fold_std = (fold_scores
            .iter()
            .map(|s| {
                let d = *s - fold_mean;
                d * d
            })
            .sum::<f64>()
            / fold_scores.len() as f64)
            .sqrt();
        let telemetry = jope_runtime::fitness::ExecutionTelemetry {
            train_time_seconds: train_secs,
            inference_time_seconds: infer_secs,
            train_memory_bytes: model_bytes + (fold.train.len() * feat_w * 16) as u64,
            inference_memory_bytes: model_bytes + (fold.validation.len() * feat_w * 16) as u64,
            model_bytes,
            complexity_stages,
            complexity_parameters: (model_bytes as usize / 8).max(feat_w),
            feature_count: spec.features.len(),
            fold_std,
            train_rows: fold.train.len(),
            inference_rows: fold.validation.len(),
        };
        let score = objective
            .evaluate_with_telemetry(&vy, &prediction, &val_aux, &telemetry)
            .map_err(|_| UndefinedFitness)?;
        losses.push(score.loss);
        values.push(score.value);
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

pub(in crate::discover::r#loop) fn aux_columns_for_rows<B: AsRef<[u8]>>(
    matrix: &Matrix<B>,
    col_names: &[String],
    rows: &[usize],
) -> Result<jope_runtime::fitness::Columns> {
    let mut out = jope_runtime::fitness::Columns::new();
    for name in col_names {
        let col_idx = matrix
            .columns()
            .iter()
            .position(|c| &c.name == name)
            .with_context(|| format!("unknown auxiliary objective column '{name}'"))?;
        let mut vals = Vec::with_capacity(rows.len());
        for &row in rows {
            match matrix.cell(row, col_idx) {
                Some(JmatCell::Numeric(v)) if v.is_finite() => vals.push(v),
                _ => {
                    bail!("non-finite or missing auxiliary objective column '{name}' in row {row}")
                }
            }
        }
        out.insert(name.clone(), vals);
    }
    Ok(out)
}

pub(super) fn resolve_calibration_family(
    mode: Option<&str>,
    objective_spec: &str,
) -> Result<jope_runtime::calibration::Calibration> {
    use jope_runtime::calibration::Calibration;
    match mode {
        None | Some("none") => Ok(Calibration::None),
        Some("shift") => Ok(Calibration::Shift),
        Some("affine") => Ok(Calibration::Affine),
        Some("auto") => {
            let prep_spec = jope_prep::objective::ObjectiveSpec::parse(objective_spec)
                .map_err(|e| anyhow::anyhow!("invalid objective for calibration check: {e}"))?;
            if prep_spec.benefits_from_post_hoc_calibration() {
                Ok(Calibration::Affine)
            } else {
                Ok(Calibration::None)
            }
        }
        Some(other) => {
            bail!("unsupported calibration_mode '{other}'; expected none, auto, shift, or affine")
        }
    }
}

/// Internal CAS fit, deliberately separate from the frozen exported bundle
/// format. Every preprocessing parameter is fitted inside the training fold.
#[derive(Serialize, Deserialize)]
pub(crate) struct NativeModel {
    pub(crate) version: u32,
    pub(crate) features: Vec<String>,
    pub(crate) dsl: String,
    pub(crate) min: Vec<f64>,
    pub(crate) max: Vec<f64>,
    pub(crate) fill: Vec<f64>,
    pub(crate) target_min: f64,
    pub(crate) target_scale: f64,
    pub(crate) model: feat_math::Model,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) calibration: Option<jope_runtime::calibration::OutputTransform>,
}

pub(in crate::discover::r#loop) struct BinaryMatrixData {
    pub(super) schema: feat_math::advanced_v3::DataSchema,
    pub(super) table: feat_math::advanced_v3::RawTable,
    pub(super) targets: Vec<f64>,
    pub(super) labels: Vec<String>,
    pub(super) classes: Vec<String>,
}

pub(in crate::discover::r#loop) fn resolve_binary_method(
    dsl: &str,
) -> Result<jope_runtime::methods::registry::RustMethod> {
    let canonical = feat_math::GenomeIr::parse(dsl)?.canonical();
    jope_runtime::methods::registry::rust_methods(jope_runtime::task::TaskKind::Binary)?
        .into_iter()
        .find(|m| m.dsl == canonical)
        .context("unsupported binary candidate pipeline")
}

pub(in crate::discover::r#loop) fn subset_raw_table(
    table: &feat_math::advanced_v3::RawTable,
    rows: &[usize],
) -> Result<feat_math::advanced_v3::RawTable> {
    feat_math::advanced_v3::RawTable::new(
        table.headers.clone(),
        rows.iter()
            .map(|&r| table.rows.get(r).cloned().context("row outside FIT matrix"))
            .collect::<Result<Vec<_>>>()?,
    )
}

pub(in crate::discover::r#loop) fn binary_matrix_data<B: AsRef<[u8]>>(
    matrix: &Matrix<B>,
    target: &str,
    features: &[String],
    class_order: [&str; 2],
) -> Result<BinaryMatrixData> {
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
    let target_col = find(target)?;
    let feature_cols: Vec<usize> = features.iter().map(|s| find(s)).collect::<Result<_>>()?;
    let schema_cols: Vec<feat_math::advanced_v3::ColumnSpec> = features
        .iter()
        .zip(&feature_cols)
        .map(|(name, &col_idx)| match matrix.columns()[col_idx].kind {
            crate::dataset::jmat::ColumnKind::Numeric => {
                feat_math::advanced_v3::ColumnSpec::numeric(name)
            }
            crate::dataset::jmat::ColumnKind::Categorical => {
                feat_math::advanced_v3::ColumnSpec::categorical(name)
            }
        })
        .collect();
    let schema = jope_runtime::table::iid_schema(schema_cols)?;
    let mut values = Vec::with_capacity(matrix.rows());
    let mut targets = Vec::with_capacity(matrix.rows());
    let mut labels = Vec::with_capacity(matrix.rows());
    for row in 0..matrix.rows() {
        let cls =
            crate::discover::r#loop::binary_cell_class(matrix.cell(row, target_col), class_order)?;
        targets.push(cls as f64);
        labels.push(class_order[cls].to_owned());
        let feature_row = feature_cols
            .iter()
            .map(|&col| match matrix.cell(row, col) {
                Some(JmatCell::Missing) => Ok(None),
                Some(JmatCell::Numeric(v)) if v.is_finite() => Ok(Some(v.to_string())),
                Some(JmatCell::Categorical(s)) => Ok(Some(s.to_owned())),
                _ => bail!("invalid feature cell in binary FIT matrix"),
            })
            .collect::<Result<Vec<_>>>()?;
        values.push(feature_row);
    }
    let table = jope_runtime::table::raw_table(&schema, &values)?;
    Ok(BinaryMatrixData {
        schema,
        table,
        targets,
        labels,
        classes: vec![class_order[0].to_owned(), class_order[1].to_owned()],
    })
}

fn binary_native_scores<B: AsRef<[u8]>>(
    request: &Request,
    matrix: &Matrix<B>,
    spec: NativeFolds,
    objective: &crate::objective::binary::BinaryObjective,
) -> Result<worker::Scores> {
    use jope_runtime::calibration::Calibration;
    use jope_runtime::methods::native::{fit_native, predict_bound, Predictions};
    use jope_runtime::task::TaskKind;

    let method = resolve_binary_method(&request.dsl)?;
    let cal_family = match spec.calibration_mode.as_deref() {
        Some("none") => Calibration::None,
        Some("platt") => Calibration::Platt,
        None | Some("auto") => {
            let probability_sensitive = !matches!(
                objective.fitness().metric,
                jope_runtime::fitness_classification::Metric::RocAuc
                    | jope_runtime::fitness_classification::Metric::PrAuc
            );
            Calibration::resolve_binary(Calibration::Auto, probability_sensitive)
        }
        Some(other) => {
            bail!("unsupported binary calibration_mode '{other}'; expected none, auto, or platt")
        }
    };
    match (&spec.refit_output, spec.refit_max_bytes) {
        (Some(path), Some(maximum)) => {
            return super::model::binary_native_refit(
                request, matrix, &spec, &method, objective, cal_family, path, maximum,
            )
        }
        (None, None) => {}
        _ => bail!("incomplete private refit handoff"),
    }

    let data = binary_matrix_data(
        matrix,
        &spec.target,
        &spec.features,
        objective.class_order(),
    )?;
    let mut validation_rows = BTreeSet::new();
    let mut surrogate = Vec::with_capacity(spec.folds.len());
    let mut fold_scores = Vec::with_capacity(spec.folds.len());

    struct RepeatOof<'a> {
        labels: Vec<&'a str>,
        targets: Vec<f64>,
        raw_p1: Vec<f64>,
        fold_ids: Vec<usize>,
    }
    let mut by_repeat: BTreeMap<u8, RepeatOof<'_>> = BTreeMap::new();
    let eval_started = std::time::Instant::now();

    for (fold_idx, fold) in spec.folds.into_iter().enumerate() {
        if fold_idx > 0
            && request.limits.wall_ms > 0
            && eval_started.elapsed().as_millis() >= u128::from(request.limits.wall_ms)
        {
            return Err(EvalWallClockTimeout.into());
        }
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
        let train_table = subset_raw_table(&data.table, &fold.train)?;
        let train_targets: Vec<f64> = fold.train.iter().map(|&r| data.targets[r]).collect();
        let bound = fit_native(
            &method,
            TaskKind::Binary,
            &data.schema,
            &data.classes,
            &train_table,
            &train_targets,
        )
        .map_err(|_| UndefinedFitness)?;
        let val_table = subset_raw_table(&data.table, &fold.validation)?;
        let Predictions::Probabilities { values, .. } =
            predict_bound(&bound, &val_table, &data.classes).map_err(|_| UndefinedFitness)?
        else {
            return Err(UndefinedFitness.into());
        };
        let mut brier_sum = 0.0;
        let oof = by_repeat.entry(fold.repeat).or_insert_with(|| RepeatOof {
            labels: Vec::new(),
            targets: Vec::new(),
            raw_p1: Vec::new(),
            fold_ids: Vec::new(),
        });
        for (&r, probs) in fold.validation.iter().zip(values) {
            let p1 = probs[1];
            ensure!(
                p1.is_finite() && (0.0..=1.0).contains(&p1),
                "non-finite binary probability"
            );
            brier_sum += (p1 - data.targets[r]).powi(2);
            oof.labels.push(data.labels[r].as_str());
            oof.targets.push(data.targets[r]);
            oof.raw_p1.push(p1);
            oof.fold_ids.push(fold_idx);
        }
        let brier = brier_sum / fold.validation.len() as f64;
        surrogate.push(brier);
        fold_scores.push(brier);
    }

    let mut raw_losses = Vec::with_capacity(by_repeat.len());
    let mut raw_values = Vec::with_capacity(by_repeat.len());
    let mut raw_ok = true;
    let mut cal_losses = Vec::with_capacity(by_repeat.len());
    let mut cal_values = Vec::with_capacity(by_repeat.len());
    let mut cal_probs_by_repeat: Vec<Vec<[f64; 2]>> = Vec::with_capacity(by_repeat.len());
    let mut cal_ok = cal_family == Calibration::Platt;

    for oof in by_repeat.values() {
        let raw_pairs: Vec<[f64; 2]> = oof.raw_p1.iter().map(|&p| [1.0 - p, p]).collect();
        match objective.evaluate(&oof.labels, &raw_pairs) {
            Ok(score) => {
                raw_losses.push(score.loss);
                raw_values.push(score.value);
            }
            Err(_) => raw_ok = false,
        }
        if cal_ok {
            match jope_runtime::calibration_platt::cross_fit(
                &oof.raw_p1,
                &oof.labels,
                objective.class_order(),
                &oof.fold_ids,
            )
            .and_then(|cf| {
                let score = objective.evaluate(&oof.labels, &cf.probabilities)?;
                Ok((score, cf.probabilities))
            }) {
                Ok((score, probs)) => {
                    cal_losses.push(score.loss);
                    cal_values.push(score.value);
                    cal_probs_by_repeat.push(probs);
                }
                Err(_) => cal_ok = false,
            }
        }
    }

    let mean = |v: &[f64]| v.iter().map(|x| x / v.len() as f64).sum::<f64>();
    let use_cal = cal_ok && (!raw_ok || mean(&cal_losses) < mean(&raw_losses));
    let (chosen_loss, chosen_value) = if use_cal {
        (mean(&cal_losses), mean(&cal_values))
    } else if raw_ok {
        (mean(&raw_losses), mean(&raw_values))
    } else {
        return Err(UndefinedFitness.into());
    };
    let scores = vec![mean(&surrogate), chosen_loss, chosen_value];
    ensure!(scores.iter().all(|x| x.is_finite()), "non-finite CV score");

    let mut bytes = Vec::new();
    for (rep_idx, oof) in by_repeat.values().enumerate() {
        for (i, &y) in oof.targets.iter().enumerate() {
            let p1 = if use_cal {
                cal_probs_by_repeat[rep_idx][i][1]
            } else {
                oof.raw_p1[i]
            };
            ensure!(p1.is_finite(), "non-finite binary prediction");
            bytes.extend(y.to_le_bytes());
            bytes.extend(p1.to_le_bytes());
        }
    }
    Ok(worker::Scores {
        values: scores,
        fold_scores,
        predictions_digest: digest(&bytes),
    })
}
