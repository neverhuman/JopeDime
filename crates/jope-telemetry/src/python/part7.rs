// ---- startup reads -------------------------------------------------------------------------

#[pyclass(name = "LocalReads", frozen)]
pub struct PyLocalReads {
    inner: Mutex<ReadStats>,
}

fn file_size(path: &PathBuf) -> PyResult<u64> {
    std::fs::metadata(path)
        .map(|m| m.len())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyOSError, _>(e.to_string()))
}

#[pymethods]
impl PyLocalReads {
    #[new]
    fn new() -> Self {
        Self {
            inner: Mutex::new(ReadStats::default()),
        }
    }

    /// Call `operation()` and record the logical size of `path` and the call's wall time.
    fn read<'py>(
        &self,
        path: PathBuf,
        operation: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let size = file_size(&path)?;
        let started = Instant::now();
        let value = operation.call0()?;
        lock(&self.inner).record(size, started, Instant::now());
        Ok(value)
    }

    /// A context manager around `opener()`'s own context manager (`np.load(path)`): the read
    /// is recorded when the inner context exits, also on error.
    fn open(slf: &Bound<'_, Self>, path: PathBuf, opener: Py<PyAny>) -> PyReadSpan {
        PyReadSpan {
            reads: slf.clone().unbind(),
            path,
            opener,
            inner: None,
            size: 0,
            started: None,
        }
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &lock(&self.inner).snapshot())
    }

    fn take_snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &lock(&self.inner).take_snapshot())
    }
}

#[pyclass(name = "ReadSpan", unsendable)]
pub struct PyReadSpan {
    reads: Py<PyLocalReads>,
    path: PathBuf,
    opener: Py<PyAny>,
    inner: Option<Py<PyAny>>,
    size: u64,
    started: Option<Instant>,
}

impl PyReadSpan {
    fn record(&mut self, py: Python<'_>) {
        if let Some(started) = self.started.take() {
            lock(&self.reads.bind(py).get().inner).record(self.size, started, Instant::now());
        }
    }
}

#[pymethods]
impl PyReadSpan {
    fn __enter__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.size = file_size(&self.path)?;
        self.started = Some(Instant::now());
        let entered = self.opener.bind(py).call0().and_then(|cm| {
            let value = cm.call_method0("__enter__")?;
            Ok((cm.unbind(), value))
        });
        match entered {
            Ok((cm, value)) => {
                self.inner = Some(cm);
                Ok(value)
            }
            Err(error) => {
                self.record(py);
                Err(error)
            }
        }
    }

    #[pyo3(signature = (exc_type=None, exc=None, tb=None))]
    fn __exit__(
        &mut self,
        py: Python<'_>,
        exc_type: Option<Py<PyAny>>,
        exc: Option<Py<PyAny>>,
        tb: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        let result = match self.inner.take() {
            Some(cm) => cm
                .bind(py)
                .call_method1("__exit__", (exc_type, exc, tb))
                .and_then(|r| r.is_truthy()),
            None => Ok(false),
        };
        self.record(py);
        result
    }
}

// ---- evaluator batches ---------------------------------------------------------------------
