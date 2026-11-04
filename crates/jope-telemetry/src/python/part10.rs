// ---- validation ----------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (label=None))]
fn check_partition_label(label: Option<&str>) -> &'static str {
    validation::check_partition_label(label).as_str()
}

#[pyclass(name = "ValidationRecorder", frozen)]
pub struct PyValidationRecorder {
    inner: Mutex<ValidationRecorder>,
    sink: Option<RecordHandle>,
}

#[pymethods]
impl PyValidationRecorder {
    #[new]
    #[pyo3(signature = (sink=None, family_roster_sha256=None, fold_map_sha256=None, max_datasets=validation::MAX_DATASETS))]
    fn new(
        sink: Option<&Bound<'_, PyRecordSink>>,
        family_roster_sha256: Option<String>,
        fold_map_sha256: Option<String>,
        max_datasets: usize,
    ) -> Self {
        Self {
            inner: Mutex::new(ValidationRecorder::new(
                family_roster_sha256,
                fold_map_sha256,
                max_datasets,
            )),
            sink: sink_of(sink),
        }
    }

    #[pyo3(signature = (*, dataset, task, partition, family=None, fold=None, prediction_mean=None, prediction_std=None, selected_program_sha256=None, selected_in_bank=None, **metrics))]
    #[allow(clippy::too_many_arguments)]
    fn add(
        &self,
        dataset: String,
        task: String,
        partition: Option<String>,
        family: Option<String>,
        fold: Option<i64>,
        prediction_mean: Option<f64>,
        prediction_std: Option<f64>,
        selected_program_sha256: Option<String>,
        selected_in_bank: Option<bool>,
        metrics: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let mut values = Vec::new();
        if let Some(m) = metrics {
            for (k, v) in m.iter() {
                values.push((k.extract::<String>()?, v.extract::<f64>().ok()));
            }
        }
        let r = DatasetResult {
            dataset,
            task,
            partition,
            family,
            fold,
            prediction_mean,
            prediction_std,
            selected_program_sha256,
            selected_in_bank,
            metrics: values,
        };
        lock(&self.inner).add(r).map_err(|e| match e {
            ValidationError::Sealed(m) => SealedPartitionError::new_err(m),
            ValidationError::Invalid(m) => PyValueError::new_err(m),
        })
    }

    /// The validation record (emitted to the sink and returned); the recorder resets.
    fn finish<'py>(&self, py: Python<'py>, at: i64) -> PyResult<Bound<'py, PyAny>> {
        let rec = lock(&self.inner).finish(at);
        emit_record(&self.sink, validation::RECORD_KIND, &rec);
        to_py(py, &rec)
    }
}
