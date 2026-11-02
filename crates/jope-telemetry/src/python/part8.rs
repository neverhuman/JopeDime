/// `resource.getrusage(RUSAGE_CHILDREN)` user + system seconds (µs resolution).
fn children_cpu_s(py: Python<'_>) -> PyResult<f64> {
    let resource = py.import("resource")?;
    let ru = resource.call_method1("getrusage", (resource.getattr("RUSAGE_CHILDREN")?,))?;
    Ok(ru.getattr("ru_utime")?.extract::<f64>()? + ru.getattr("ru_stime")?.extract::<f64>()?)
}

#[pyclass(name = "EvalBatch", frozen)]
pub struct PyEvalBatch {
    inner: EvalBatch,
    sink: Option<RecordHandle>,
}

#[pymethods]
impl PyEvalBatch {
    #[new]
    #[pyo3(signature = (sink=None))]
    fn new(sink: Option<&Bound<'_, PyRecordSink>>) -> Self {
        Self {
            inner: EvalBatch::new(),
            sink: sink_of(sink),
        }
    }

    fn batch_begin(
        &self,
        py: Python<'_>,
        kind: &str,
        n: usize,
        workers_requested: i64,
        workers_effective: i64,
    ) -> PyResult<()> {
        let cpu = children_cpu_s(py)?;
        self.inner
            .batch_begin(kind, n, workers_requested, workers_effective, cpu);
        Ok(())
    }

    fn worker_start(&self, i: usize) {
        self.inner.worker_start(i);
    }

    fn guard_wait(&self, i: usize, ns: i64) {
        self.inner.guard_wait(i, ns);
    }

    fn fit_done(&self, i: usize, ns: i64, rc: i64) {
        self.inner.fit_done(i, ns, rc);
    }

    fn outer_done(&self, i: usize, ns: i64) {
        self.inner.outer_done(i, ns);
    }

    fn worker_end(&self, i: usize) {
        self.inner.worker_end(i);
    }

    /// One record per batch, emitted to the sink and returned.
    fn batch_end<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rec = self.inner.batch_end(children_cpu_s(py)?);
        emit_record(&self.sink, eval::RECORD_KIND, &rec);
        to_py(py, &rec)
    }

    fn rc_of(&self, i: usize) -> Option<i64> {
        self.inner.rc_of(i)
    }

    #[getter]
    fn batches(&self) -> i64 {
        self.inner
            .batches
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[getter]
    fn slots_dropped(&self) -> i64 {
        self.inner
            .slots_dropped
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}
