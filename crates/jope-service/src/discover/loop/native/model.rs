use super::evaluate::{
    aux_columns_for_rows, binary_matrix_data, multiclass_matrix_data, numeric_rows,
    subset_raw_table, NativeFolds, UndefinedFitness,
};
use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BinaryRefitState {
    pub(crate) version: u32,
    pub(crate) fitted_bytes: Vec<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) platt: Option<jope_runtime::calibration_platt::Calibration>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MulticlassRefitState {
    pub(crate) version: u32,
    pub(crate) fitted_bytes: Vec<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<jope_runtime::calibration_temperature::Calibration>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RefitEnvelope {
    attempt: Attempt,
    model_json: String,
    model_digest: String,
    predictions_digest: String,
}

impl NativeModel {
    pub(crate) fn fit(
        features: &[String],
        dsl: &str,
        x: &feat_math::Rows,
        y: &[f64],
    ) -> Result<Self> {
        ensure!(
            !x.is_empty() && x.len() == y.len() && y.iter().all(|y| y.is_finite()),
            "invalid training rows"
        );
        ensure!(
            x.iter().all(|row| row.len() == features.len()),
            "ragged features"
        );
        let mut min = vec![];
        let mut max = vec![];
        let mut fill = vec![];
        for col in 0..features.len() {
            let present: Vec<_> = x.iter().map(|r| r[col]).filter(|v| v.is_finite()).collect();
            min.push(present.iter().copied().reduce(f64::min).unwrap_or(0.0));
            max.push(present.iter().copied().reduce(f64::max).unwrap_or(0.0));
            fill.push(
                present
                    .iter()
                    .map(|v| v / present.len() as f64)
                    .sum::<f64>(),
            );
        }
        let target_min = y.iter().copied().reduce(f64::min).context("no target")?;
        let target_max = y.iter().copied().reduce(f64::max).context("no target")?;
        let target_scale = (target_max - target_min).max(1e-12);
        ensure!(
            target_scale.is_finite()
                && min.iter().chain(&max).chain(&fill).all(|v| v.is_finite())
                && min.iter().zip(&max).all(|(lo, hi)| (hi - lo).is_finite()),
            "preprocessing overflow"
        );
        let normalize = |rows: &feat_math::Rows| -> feat_math::Rows {
            rows.iter()
                .map(|r| {
                    r.iter()
                        .enumerate()
                        .map(|(j, &v)| {
                            let value = if v.is_finite() { v } else { fill[j] };
                            if max[j] == min[j] {
                                0.0
                            } else {
                                ((value - min[j]) / (max[j] - min[j])).clamp(0.0, 1.0)
                            }
                        })
                        .collect()
                })
                .collect()
        };
        let tx = normalize(x);
        let ty = y
            .iter()
            .map(|v| (v - target_min) / target_scale)
            .collect::<Vec<_>>();
        let model = feat_math::fit(&feat_math::GenomeIr::parse(dsl)?, &tx, &ty)?;
        Ok(Self {
            version: NATIVE_ARTIFACT_VERSION,
            features: features.to_vec(),
            dsl: dsl.into(),
            min,
            max,
            fill,
            target_min,
            target_scale,
            model,
            calibration: None,
        })
    }

    pub(crate) fn fit_with_calibration(
        features: &[String],
        dsl: &str,
        x: &feat_math::Rows,
        y: &[f64],
        objective: &RegressionObjective,
        calibration_family: jope_runtime::calibration::Calibration,
        train_aux_cols: &jope_runtime::fitness::Columns,
    ) -> Result<Self> {
        let mut fitted = Self::fit(features, dsl, x, y)?;
        if calibration_family != jope_runtime::calibration::Calibration::None {
            let raw_preds = fitted.predict(x)?;
            let transform = jope_runtime::calibration::fit(
                objective.fitness(),
                calibration_family,
                y,
                &raw_preds,
                train_aux_cols,
            );
            transform.validate()?;
            if !transform.is_identity() {
                fitted.calibration = Some(transform);
            }
        }
        Ok(fitted)
    }

    pub(crate) fn predict(&self, x: &feat_math::Rows) -> Result<Vec<f64>> {
        ensure!(
            x.iter().all(|r| r.len() == self.features.len()),
            "prediction width mismatch"
        );
        let tx = x
            .iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, &v)| {
                        let value = if v.is_finite() { v } else { self.fill[j] };
                        if self.max[j] == self.min[j] {
                            0.0
                        } else {
                            ((value - self.min[j]) / (self.max[j] - self.min[j])).clamp(0.0, 1.0)
                        }
                    })
                    .collect()
            })
            .collect();
        let values: Vec<_> = self
            .model
            .predict(&tx)?
            .into_iter()
            .map(|p| {
                let uncalibrated = p * self.target_scale + self.target_min;
                match &self.calibration {
                    Some(cal) => cal.apply(uncalibrated),
                    None => uncalibrated,
                }
            })
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "non-finite native prediction"
        );
        Ok(values)
    }
}

// This function is reached only inside the supervised numerical worker.
// Private scratch output is never a CAS or redb write.
pub(super) fn native_refit<B: AsRef<[u8]>>(
    request: &Request,
    matrix: &Matrix<B>,
    spec: &NativeFolds,
    objective: &RegressionObjective,
    calibration_family: jope_runtime::calibration::Calibration,
    output: &Path,
    maximum: u64,
) -> Result<worker::Scores> {
    ensure!(
        maximum > 0 && maximum <= REFIT_HANDOFF_BYTES,
        "invalid private fit bound"
    );
    ensure!(
        output.is_absolute() && output.file_name().is_some_and(|n| n == "model.json"),
        "invalid private fit path"
    );
    let parent = output.parent().context("private fit directory missing")?;
    ensure!(
        fs::symlink_metadata(parent)?.is_dir() && fs::canonicalize(parent)? == parent,
        "private fit directory must be canonical"
    );
    let all_rows: Vec<usize> = (0..matrix.rows()).collect();
    let (x, y) = numeric_rows(matrix, &spec.target, &spec.features, &all_rows)?;
    let aux_cols = aux_columns_for_rows(matrix, &objective.columns(), &all_rows)?;
    let train_t0 = std::time::Instant::now();
    let fitted = NativeModel::fit_with_calibration(
        &spec.features,
        &request.dsl,
        &x,
        &y,
        objective,
        calibration_family,
        &aux_cols,
    )
    .map_err(|_| UndefinedFitness)?;
    let train_secs = train_t0.elapsed().as_secs_f64();
    let infer_t0 = std::time::Instant::now();
    let predictions = fitted.predict(&x).map_err(|_| UndefinedFitness)?;
    let infer_secs = infer_t0.elapsed().as_secs_f64();
    let model_json = serde_json::to_string(&fitted)?;
    let model_bytes = model_json.len() as u64;
    let feat_w = spec.features.len().max(1);
    let telemetry = jope_runtime::fitness::ExecutionTelemetry {
        train_time_seconds: train_secs,
        inference_time_seconds: infer_secs,
        train_memory_bytes: model_bytes + (x.len() * feat_w * 16) as u64,
        inference_memory_bytes: model_bytes + (x.len() * feat_w * 16) as u64,
        model_bytes,
        complexity_stages: 1,
        complexity_parameters: (model_bytes as usize / 8).max(feat_w),
        feature_count: spec.features.len(),
        fold_std: 0.0,
        train_rows: x.len(),
        inference_rows: x.len(),
    };
    let score = objective
        .evaluate_with_telemetry(&y, &predictions, &aux_cols, &telemetry)
        .map_err(|_| UndefinedFitness)?;
    let surrogate = RegressionObjective::parse("rmse")?
        .evaluate(&y, &predictions)
        .map_err(|_| UndefinedFitness)?;
    let mut measured = Vec::with_capacity(y.len() * 16);
    for (label, prediction) in y.iter().zip(&predictions) {
        measured.extend(label.to_le_bytes());
        measured.extend(prediction.to_le_bytes());
    }
    let predictions_digest = digest(&measured);
    let _: NativeModel = serde_json::from_str(&model_json).map_err(|_| UndefinedFitness)?;
    let envelope = RefitEnvelope {
        attempt: request.attempt.clone(),
        model_digest: digest(model_json.as_bytes()),
        model_json,
        predictions_digest: predictions_digest.clone(),
    };
    let bytes = serde_json::to_vec(&envelope)?;
    ensure!(
        bytes.len() as u64 <= maximum,
        "promoted model exceeds the M1 private handoff bound"
    );
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(output)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(worker::Scores {
        values: vec![surrogate.loss, score.loss, score.value],
        fold_scores: vec![],
        predictions_digest,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn binary_native_refit<B: AsRef<[u8]>>(
    request: &Request,
    matrix: &Matrix<B>,
    spec: &NativeFolds,
    method: &jope_runtime::methods::registry::RustMethod,
    objective: &crate::objective::binary::BinaryObjective,
    calibration_family: jope_runtime::calibration::Calibration,
    output: &Path,
    maximum: u64,
) -> Result<worker::Scores> {
    use jope_runtime::budget::{BudgetRecord, ENFORCEMENT};
    use jope_runtime::calibration::Calibration;
    use jope_runtime::fitted::{CreatedBy, FitRecord, FittedModel};
    use jope_runtime::methods::native::{fit_native, predict_bound, Predictions};
    use jope_runtime::model_object::ModelObject;
    use jope_runtime::task::TaskKind;

    ensure!(
        maximum > 0 && maximum <= REFIT_HANDOFF_BYTES,
        "invalid private fit bound"
    );
    ensure!(
        output.is_absolute() && output.file_name().is_some_and(|n| n == "model.json"),
        "invalid private fit path"
    );
    let parent = output.parent().context("private fit directory missing")?;
    ensure!(
        fs::symlink_metadata(parent)?.is_dir() && fs::canonicalize(parent)? == parent,
        "private fit directory must be canonical"
    );

    let data = binary_matrix_data(
        matrix,
        &spec.target,
        &spec.features,
        objective.class_order(),
    )?;
    let mut platt = None;
    if calibration_family == Calibration::Platt {
        let int_labels: Vec<usize> = data.targets.iter().map(|&t| t as usize).collect();
        let c0 = int_labels.iter().filter(|&&c| c == 0).count();
        let min_class = c0.min(int_labels.len() - c0);
        if min_class >= 2 {
            let k = 5usize.min(min_class).max(2);
            let val_parts = crate::discover::splits::stratified_k_fold(
                &int_labels,
                k,
                request.seed,
                "binary-refit/platt-v1",
            )?;
            let mut oof_p1 = vec![0.0; data.targets.len()];
            let mut fold_ids = vec![0usize; data.targets.len()];
            let mut oof_ok = true;
            for (fold_id, val_rows) in val_parts.into_iter().enumerate() {
                let val_set: BTreeSet<usize> = val_rows.iter().copied().collect();
                let train_rows: Vec<usize> = (0..data.targets.len())
                    .filter(|r| !val_set.contains(r))
                    .collect();
                let train_table = subset_raw_table(&data.table, &train_rows)?;
                let train_targets: Vec<f64> = train_rows.iter().map(|&r| data.targets[r]).collect();
                let val_table = subset_raw_table(&data.table, &val_rows)?;
                match fit_native(
                    method,
                    TaskKind::Binary,
                    &data.schema,
                    &data.classes,
                    &train_table,
                    &train_targets,
                )
                .and_then(|bound| predict_bound(&bound, &val_table, &data.classes))
                {
                    Ok(Predictions::Probabilities { values, .. }) => {
                        for (&r, probs) in val_rows.iter().zip(values) {
                            oof_p1[r] = probs[1];
                            fold_ids[r] = fold_id;
                        }
                    }
                    _ => {
                        oof_ok = false;
                        break;
                    }
                }
            }
            if oof_ok {
                let label_refs: Vec<&str> = data.labels.iter().map(String::as_str).collect();
                let raw_pairs: Vec<[f64; 2]> = oof_p1.iter().map(|&p| [1.0 - p, p]).collect();
                let raw_res = objective.evaluate(&label_refs, &raw_pairs);
                if let Ok(cf) = jope_runtime::calibration_platt::cross_fit(
                    &oof_p1,
                    &label_refs,
                    objective.class_order(),
                    &fold_ids,
                ) {
                    if let Ok(cal_score) = objective.evaluate(&label_refs, &cf.probabilities) {
                        let improves = match raw_res {
                            Ok(raw_score) => cal_score.loss < raw_score.loss,
                            Err(_) => true,
                        };
                        if improves {
                            platt = Some(cf.calibration);
                        }
                    }
                }
            }
        }
    }

    let bound = fit_native(
        method,
        TaskKind::Binary,
        &data.schema,
        &data.classes,
        &data.table,
        &data.targets,
    )
    .map_err(|_| UndefinedFitness)?;
    let budget = BudgetRecord {
        time_limit_seconds: (request.limits.wall_ms > 0)
            .then(|| request.limits.wall_ms as f64 / 1000.0),
        consumed_seconds: 0.0,
        threads: 1,
        enforcement: ENFORCEMENT,
    };
    let fitted = FittedModel::new(FitRecord {
        method,
        kind: TaskKind::Binary,
        classes: &data.classes,
        target: &spec.target,
        schema: &data.schema,
        model: bound,
        rows: data.targets.len(),
        fitting_input_sha256: jope_runtime::table::fitting_input_sha256(
            &data.table,
            &data.targets,
        )?,
        training_file_sha256: matrix.sha256_hex().to_owned(),
        selection: None,
        budget: &budget,
        created_by: CreatedBy::default(),
    })
    .map_err(|_| UndefinedFitness)?;
    let fitted_bytes = fitted.to_bytes().map_err(|_| UndefinedFitness)?;
    let mut object =
        ModelObject::from_fitted_bytes(fitted_bytes.clone()).map_err(|_| UndefinedFitness)?;
    if let Some(cal) = &platt {
        object = object
            .with_platt_calibration(cal.clone())
            .map_err(|_| UndefinedFitness)?;
    }
    let Predictions::Probabilities { values, .. } =
        object.predict(&data.table).map_err(|_| UndefinedFitness)?
    else {
        return Err(UndefinedFitness.into());
    };
    let pairs: Vec<[f64; 2]> = values.iter().map(|row| [row[0], row[1]]).collect();
    let label_refs: Vec<&str> = data.labels.iter().map(String::as_str).collect();
    let score = objective
        .evaluate(&label_refs, &pairs)
        .map_err(|_| UndefinedFitness)?;
    let brier = pairs
        .iter()
        .zip(&data.targets)
        .map(|(p, &y)| (p[1] - y).powi(2))
        .sum::<f64>()
        / pairs.len() as f64;
    let mut measured = Vec::with_capacity(data.targets.len() * 16);
    for (&label, pair) in data.targets.iter().zip(&pairs) {
        measured.extend(label.to_le_bytes());
        measured.extend(pair[1].to_le_bytes());
    }
    let predictions_digest = digest(&measured);
    let state = BinaryRefitState {
        version: NATIVE_ARTIFACT_VERSION,
        fitted_bytes,
        platt,
    };
    let model_json = serde_json::to_string(&state)?;
    let _: BinaryRefitState = serde_json::from_str(&model_json).map_err(|_| UndefinedFitness)?;
    let envelope = RefitEnvelope {
        attempt: request.attempt.clone(),
        model_digest: digest(model_json.as_bytes()),
        model_json,
        predictions_digest: predictions_digest.clone(),
    };
    let bytes = serde_json::to_vec(&envelope)?;
    ensure!(
        bytes.len() as u64 <= maximum,
        "promoted model exceeds the M1 private handoff bound"
    );
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(output)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(worker::Scores {
        values: vec![brier, score.loss, score.value],
        fold_scores: vec![],
        predictions_digest,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn multiclass_native_refit<B: AsRef<[u8]>>(
    request: &Request,
    matrix: &Matrix<B>,
    spec: &NativeFolds,
    method: &jope_runtime::methods::registry::RustMethod,
    objective: &crate::objective::multiclass::MulticlassObjective,
    calibration_family: jope_runtime::calibration::Calibration,
    output: &Path,
    maximum: u64,
) -> Result<worker::Scores> {
    use jope_runtime::budget::{BudgetRecord, ENFORCEMENT};
    use jope_runtime::calibration::Calibration;
    use jope_runtime::fitted::{CreatedBy, FitRecord, FittedModel};
    use jope_runtime::methods::native::{fit_native, predict_bound, Predictions};
    use jope_runtime::model_object::ModelObject;
    use jope_runtime::task::TaskKind;

    ensure!(
        maximum > 0 && maximum <= REFIT_HANDOFF_BYTES,
        "invalid private fit bound"
    );
    ensure!(
        output.is_absolute() && output.file_name().is_some_and(|n| n == "model.json"),
        "invalid private fit path"
    );
    let parent = output.parent().context("private fit directory missing")?;
    ensure!(
        fs::symlink_metadata(parent)?.is_dir() && fs::canonicalize(parent)? == parent,
        "private fit directory must be canonical"
    );

    let data = multiclass_matrix_data(
        matrix,
        &spec.target,
        &spec.features,
        objective.class_order(),
    )?;
    let num_classes = data.classes.len();
    let class_refs: Vec<&str> = data.classes.iter().map(String::as_str).collect();

    let mut temperature = None;
    if calibration_family == Calibration::Temperature {
        let int_labels: Vec<usize> = data.targets.iter().map(|&t| t as usize).collect();
        let mut counts = vec![0usize; num_classes];
        for &c in &int_labels {
            counts[c] += 1;
        }
        let min_class = counts.into_iter().min().unwrap_or(0);
        if min_class >= 2 {
            let k = 5usize.min(min_class).max(2);
            let val_parts = crate::discover::splits::stratified_k_fold(
                &int_labels,
                k,
                request.seed,
                "multiclass-refit/temperature-v1",
            )?;
            let mut oof_probs = vec![Vec::new(); data.targets.len()];
            let mut fold_ids = vec![0usize; data.targets.len()];
            let mut oof_ok = true;
            for (fold_id, val_rows) in val_parts.into_iter().enumerate() {
                let val_set: BTreeSet<usize> = val_rows.iter().copied().collect();
                let train_rows: Vec<usize> = (0..data.targets.len())
                    .filter(|r| !val_set.contains(r))
                    .collect();
                let train_table = subset_raw_table(&data.table, &train_rows)?;
                let train_targets: Vec<f64> = train_rows.iter().map(|&r| data.targets[r]).collect();
                let val_table = subset_raw_table(&data.table, &val_rows)?;
                match fit_native(
                    method,
                    TaskKind::Multiclass,
                    &data.schema,
                    &data.classes,
                    &train_table,
                    &train_targets,
                )
                .and_then(|bound| predict_bound(&bound, &val_table, &data.classes))
                {
                    Ok(Predictions::Probabilities { values, .. }) => {
                        for (&r, probs) in val_rows.iter().zip(values) {
                            oof_probs[r] = probs;
                            fold_ids[r] = fold_id;
                        }
                    }
                    _ => {
                        oof_ok = false;
                        break;
                    }
                }
            }
            if oof_ok {
                let label_refs: Vec<&str> = data.labels.iter().map(String::as_str).collect();
                let raw_res = objective.evaluate(&label_refs, &oof_probs);
                if let Ok(cf) = jope_runtime::calibration_temperature::cross_fit_with_fitness(
                    &oof_probs,
                    &label_refs,
                    &class_refs,
                    &fold_ids,
                    objective.fitness(),
                ) {
                    if let Ok(cal_score) = objective.evaluate(&label_refs, &cf.probabilities) {
                        let improves = match raw_res {
                            Ok(raw_score) => cal_score.loss < raw_score.loss,
                            Err(_) => true,
                        };
                        if improves {
                            temperature = Some(cf.calibration);
                        }
                    }
                }
            }
        }
    }

    let bound = fit_native(
        method,
        TaskKind::Multiclass,
        &data.schema,
        &data.classes,
        &data.table,
        &data.targets,
    )
    .map_err(|_| UndefinedFitness)?;
    let budget = BudgetRecord {
        time_limit_seconds: (request.limits.wall_ms > 0)
            .then(|| request.limits.wall_ms as f64 / 1000.0),
        consumed_seconds: 0.0,
        threads: 1,
        enforcement: ENFORCEMENT,
    };
    let fitted = FittedModel::new(FitRecord {
        method,
        kind: TaskKind::Multiclass,
        classes: &data.classes,
        target: &spec.target,
        schema: &data.schema,
        model: bound,
        rows: data.targets.len(),
        fitting_input_sha256: jope_runtime::table::fitting_input_sha256(
            &data.table,
            &data.targets,
        )?,
        training_file_sha256: matrix.sha256_hex().to_owned(),
        selection: None,
        budget: &budget,
        created_by: CreatedBy::default(),
    })
    .map_err(|_| UndefinedFitness)?;
    let fitted_bytes = fitted.to_bytes().map_err(|_| UndefinedFitness)?;
    let mut object =
        ModelObject::from_fitted_bytes(fitted_bytes.clone()).map_err(|_| UndefinedFitness)?;
    if let Some(cal) = &temperature {
        object = object
            .with_temperature_calibration(cal.clone())
            .map_err(|_| UndefinedFitness)?;
    }
    let Predictions::Probabilities { values, .. } =
        object.predict(&data.table).map_err(|_| UndefinedFitness)?
    else {
        return Err(UndefinedFitness.into());
    };
    let label_refs: Vec<&str> = data.labels.iter().map(String::as_str).collect();
    let score = objective
        .evaluate(&label_refs, &values)
        .map_err(|_| UndefinedFitness)?;
    let brier = values
        .iter()
        .zip(&data.targets)
        .map(|(probs, &y)| {
            let y_idx = y as usize;
            probs
                .iter()
                .enumerate()
                .map(|(c, &p)| {
                    let ind = if c == y_idx { 1.0 } else { 0.0 };
                    (p - ind).powi(2)
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        / values.len() as f64;
    let mut measured = Vec::with_capacity(data.targets.len() * (num_classes + 1) * 8);
    for (&label, probs) in data.targets.iter().zip(&values) {
        measured.extend(label.to_le_bytes());
        for &p in probs {
            measured.extend(p.to_le_bytes());
        }
    }
    let predictions_digest = digest(&measured);
    let state = MulticlassRefitState {
        version: NATIVE_ARTIFACT_VERSION,
        fitted_bytes,
        temperature,
    };
    let model_json = serde_json::to_string(&state)?;
    let _: MulticlassRefitState =
        serde_json::from_str(&model_json).map_err(|_| UndefinedFitness)?;
    let envelope = RefitEnvelope {
        attempt: request.attempt.clone(),
        model_digest: digest(model_json.as_bytes()),
        model_json,
        predictions_digest: predictions_digest.clone(),
    };
    let bytes = serde_json::to_vec(&envelope)?;
    ensure!(
        bytes.len() as u64 <= maximum,
        "promoted model exceeds the M1 private handoff bound"
    );
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(output)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(worker::Scores {
        values: vec![brier, score.loss, score.value],
        fold_scores: vec![],
        predictions_digest,
    })
}
