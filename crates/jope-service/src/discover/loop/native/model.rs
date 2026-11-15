use super::evaluate::{aux_columns_for_rows, numeric_rows, NativeFolds, UndefinedFitness};
use super::*;

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
