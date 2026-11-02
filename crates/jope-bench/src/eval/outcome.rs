use super::*;

/// Metric id → value.
pub type Metrics = BTreeMap<String, f64>;

/// One program scored on one cut.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Outcome {
    /// `ok` or `rejected`.
    pub status: String,
    /// Why it was rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The task's primary score on the held-out rows ([`Task::score_metric`]; higher is
    /// better). `None` on a rejection, and for a binary cut whose held-out rows are one class
    /// (AUC is undefined there; `note` says so and the other metrics are still recorded).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Every metric on the held-out rows ([`Task::metrics`]; `auc` is absent when undefined).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metrics: Metrics,
    /// Something worth knowing about an `ok` outcome.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// Wall seconds in `fit`.
    pub fit_seconds: f64,
    /// Wall seconds in `predict` (for the table API this includes rendering the table).
    pub predict_seconds: f64,
    /// Held-out rows scored.
    pub hold_rows: usize,
    /// Held-out predictions, in `cut.hold` order (not serialised: reports carry scores, not rows).
    #[serde(skip)]
    pub predictions: Vec<f64>,
}

impl Outcome {
    pub(crate) fn rejected(
        reason: String,
        fit_seconds: f64,
        predict_seconds: f64,
        hold_rows: usize,
    ) -> Self {
        Self {
            status: "rejected".to_owned(),
            reason: Some(reason),
            score: None,
            metrics: Metrics::new(),
            note: None,
            fit_seconds,
            predict_seconds,
            hold_rows,
            predictions: Vec::new(),
        }
    }

    /// Whether the program fitted and predicted (its primary score may still be undefined).
    pub fn is_ok(&self) -> bool {
        self.status == "ok"
    }
}

/// Rows and targets at `indices`.
pub fn take(x: &Rows, y: &[f64], indices: &[usize]) -> (Rows, Vec<f64>) {
    let xs = indices.iter().map(|&i| x[i].clone()).collect();
    let ys = indices.iter().map(|&i| y[i]).collect();
    (xs, ys)
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_else(|| "non-string panic payload".to_owned())
}

/// Synthetic all-numeric schema `c0..c{features-1}`, IID rows.
pub fn numeric_schema(features: usize) -> DataSchema {
    DataSchema {
        columns: (0..features)
            .map(|j| ColumnSpec::numeric(format!("c{j}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    }
}

/// Shortest-round-trip string table: parses back to the exact f64 cells.
pub fn numeric_table(x: &Rows) -> Result<RawTable> {
    let features = x.first().map(Vec::len).unwrap_or(0);
    let headers = (0..features).map(|j| format!("c{j}")).collect::<Vec<_>>();
    let cells = x
        .iter()
        .map(|row| row.iter().map(|v| format!("{v:?}")).collect())
        .collect();
    RawTable::new(headers, cells)
}

/// A fitted program: predicts a row block.
type Predictor = Box<dyn Fn(&Rows) -> Result<Vec<f64>>>;

fn fit(task: Task, genome: &GenomeIr, x: &Rows, y: &[f64]) -> Result<Predictor> {
    match task.v3_spec() {
        None => {
            let model = feat_math::fit(genome, x, y)?;
            Ok(Box::new(move |rows: &Rows| Ok(model.predict(rows)?)))
        }
        Some(spec) => {
            let features = x.first().map(Vec::len).unwrap_or(0);
            let schema = numeric_schema(features);
            let table = numeric_table(x)?;
            let model = fit_table(genome, &spec, &schema, &table, y)?;
            Ok(Box::new(move |rows: &Rows| {
                model.predict_table(&numeric_table(rows)?)
            }))
        }
    }
}

/// Score held-out predictions for `task`: the primary score (`None` when undefined), every
/// metric, and a note. `Err` names a prediction the task does not allow.
pub fn score(
    task: Task,
    y: &[f64],
    p: &[f64],
) -> std::result::Result<(Option<f64>, Metrics, Option<String>), String> {
    let mut metrics = Metrics::new();
    match task {
        Task::Regression => {
            let (rmse_value, r2_value) = (rmse(y, p), r2(y, p));
            if !rmse_value.is_finite() || !r2_value.is_finite() {
                return Err("non-finite score".to_owned());
            }
            metrics.insert("rmse".to_owned(), rmse_value);
            metrics.insert("r2".to_owned(), r2_value);
            Ok((Some(r2_value), metrics, None))
        }
        Task::Binary => {
            if p.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)) {
                return Err("probability outside [0, 1]".to_owned());
            }
            let (ll, br) = (logloss(y, p), brier(y, p));
            if !ll.is_finite() || !br.is_finite() {
                return Err("non-finite score".to_owned());
            }
            metrics.insert("logloss".to_owned(), ll);
            metrics.insert("brier".to_owned(), br);
            let positives = y.iter().filter(|v| **v == 1.0).count();
            if positives == 0 || positives == y.len() {
                return Ok((
                    None,
                    metrics,
                    Some("held-out rows are one class; AUC undefined".to_owned()),
                ));
            }
            let auc_value = auc(y, p);
            if !auc_value.is_finite() {
                return Err("non-finite score".to_owned());
            }
            metrics.insert("auc".to_owned(), auc_value);
            Ok((Some(auc_value), metrics, None))
        }
        Task::Multiclass => {
            if p.iter().any(|v| !v.is_finite() || v.fract() != 0.0) {
                return Err("non-integer class label".to_owned());
            }
            let (f1, acc) = (macro_f1(y, p), multiclass_accuracy(y, p));
            if !f1.is_finite() || !acc.is_finite() {
                return Err("non-finite score".to_owned());
            }
            metrics.insert("macro_f1".to_owned(), f1);
            metrics.insert("accuracy".to_owned(), acc);
            Ok((Some(f1), metrics, None))
        }
    }
}

/// Fit `dsl` on `cut.train`, predict `cut.hold`, score for `task`.
pub fn evaluate(dsl: &str, x: &Rows, y: &[f64], cut: &Cut, task: Task) -> Outcome {
    let hold_rows = cut.hold.len();
    let genome = match catch_unwind(AssertUnwindSafe(|| GenomeIr::parse(dsl))) {
        Ok(Ok(genome)) => genome,
        Ok(Err(error)) => return Outcome::rejected(format!("parse: {error}"), 0.0, 0.0, hold_rows),
        Err(payload) => {
            return Outcome::rejected(
                format!("parse panicked: {}", panic_text(payload)),
                0.0,
                0.0,
                hold_rows,
            )
        }
    };
    let (x_train, y_train) = take(x, y, &cut.train);
    let (x_hold, y_hold) = take(x, y, &cut.hold);

    let started = Instant::now();
    let predictor = match catch_unwind(AssertUnwindSafe(|| fit(task, &genome, &x_train, &y_train)))
    {
        Ok(Ok(predictor)) => predictor,
        Ok(Err(error)) => {
            return Outcome::rejected(
                format!("fit: {error:#}"),
                started.elapsed().as_secs_f64(),
                0.0,
                hold_rows,
            )
        }
        Err(payload) => {
            return Outcome::rejected(
                format!("fit panicked: {}", panic_text(payload)),
                started.elapsed().as_secs_f64(),
                0.0,
                hold_rows,
            )
        }
    };
    let fit_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let predictions = match catch_unwind(AssertUnwindSafe(|| predictor(&x_hold))) {
        Ok(Ok(predictions)) => predictions,
        Ok(Err(error)) => {
            return Outcome::rejected(
                format!("predict: {error:#}"),
                fit_seconds,
                started.elapsed().as_secs_f64(),
                hold_rows,
            )
        }
        Err(payload) => {
            return Outcome::rejected(
                format!("predict panicked: {}", panic_text(payload)),
                fit_seconds,
                started.elapsed().as_secs_f64(),
                hold_rows,
            )
        }
    };
    let predict_seconds = started.elapsed().as_secs_f64();
    if predictions.len() != hold_rows {
        return Outcome::rejected(
            format!(
                "predict returned {} values for {hold_rows} rows",
                predictions.len()
            ),
            fit_seconds,
            predict_seconds,
            hold_rows,
        );
    }
    if predictions.iter().any(|p| !p.is_finite()) {
        return Outcome::rejected(
            "non-finite prediction".to_owned(),
            fit_seconds,
            predict_seconds,
            hold_rows,
        );
    }
    match score(task, &y_hold, &predictions) {
        Ok((score, metrics, note)) => Outcome {
            status: "ok".to_owned(),
            reason: None,
            score,
            metrics,
            note,
            fit_seconds,
            predict_seconds,
            hold_rows,
            predictions,
        },
        Err(reason) => Outcome::rejected(reason, fit_seconds, predict_seconds, hold_rows),
    }
}
