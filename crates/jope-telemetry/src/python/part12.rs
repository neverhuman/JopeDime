// ---- DIME score trainers ---------------------------------------------------------------------

/// GPU 0 through NVML, read on a Rust thread; `snapshot()` is the `gpu_*` dict the DIME
/// trainers merge into their progress and epoch records.
#[pyclass(name = "HardwareMonitor", frozen)]
struct PyHardwareMonitor {
    inner: Mutex<HardwareMonitor>,
    minimum_headroom_mib: f64,
}

#[pymethods]
impl PyHardwareMonitor {
    #[new]
    fn new(py: Python<'_>, hz: f64, minimum_headroom_mib: f64) -> PyResult<Self> {
        let inner = py
            .detach(|| HardwareMonitor::new(hz))
            .map_err(value_error)?;
        Ok(Self {
            inner: Mutex::new(inner),
            minimum_headroom_mib,
        })
    }

    fn start(&self) -> PyResult<()> {
        lock(&self.inner).start().map_err(runtime_error)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let reading = lock(&self.inner).reading();
        to_py(
            py,
            &hardware::snapshot(reading.as_ref(), self.minimum_headroom_mib),
        )
    }

    fn stop(&self, py: Python<'_>) {
        py.detach(|| lock(&self.inner).stop());
    }
}

/// One NVML reading of GPU 0, as `HardwareMonitor.snapshot()` reports it.
#[pyfunction]
fn gpu_snapshot(py: Python<'_>, minimum_headroom_mib: f64) -> PyResult<Bound<'_, PyAny>> {
    let reading = py.detach(|| GpuReader::new().read());
    to_py(
        py,
        &hardware::snapshot(reading.as_ref(), minimum_headroom_mib),
    )
}

/// `live-metrics.jsonl` windows (`jope-prime.dime-live-metrics.v1`).
#[pyclass(name = "LiveMetricsWriter", unsendable)]
struct PyLiveMetricsWriter {
    inner: LiveMetricsWriter,
}

#[pymethods]
impl PyLiveMetricsWriter {
    #[new]
    #[pyo3(signature = (path, hz, rotate_bytes=live::ROTATE_BYTES))]
    fn new(path: PathBuf, hz: f64, rotate_bytes: u64) -> PyResult<Self> {
        let inner = LiveMetricsWriter::new(path, hz, rotate_bytes).map_err(value_error)?;
        Ok(Self { inner })
    }

    /// One step's row; the window is written once it is `1 / hz` seconds old.
    fn add(&mut self, row: &Bound<'_, PyDict>) -> PyResult<()> {
        let row = live_row(row)?;
        Ok(self.inner.add(row)?)
    }

    fn flush(&mut self) -> PyResult<()> {
        Ok(self.inner.flush()?)
    }

    fn close(&mut self) -> PyResult<()> {
        Ok(self.inner.close()?)
    }
}

fn live_row(row: &Bound<'_, PyDict>) -> PyResult<LiveRow> {
    let py = row.py();
    let item = |key: &str| {
        row.get_item(key)?
            .ok_or_else(|| PyKeyError::new_err(key.to_owned()))
    };
    let int = |value: Bound<'_, PyAny>| -> PyResult<i64> {
        py.get_type::<PyInt>().call1((value,))?.extract()
    };
    let mut means = [0.0; live::MEAN_FIELDS.len()];
    for (slot, key) in means.iter_mut().zip(live::MEAN_FIELDS) {
        *slot = item(key)?.extract::<f64>()?;
    }
    let queue_depth = match row.get_item("queue_depth")? {
        Some(depth) if !depth.is_none() => Some(int(depth)?),
        _ => None,
    };
    let mut latest: [String; live::LATEST_FIELDS.len()] = Default::default();
    for (slot, key) in latest.iter_mut().zip(live::LATEST_FIELDS) {
        match row.get_item(key)? {
            Some(value) => write_py_json(&value, slot, 0)?,
            None => slot.push_str("null"),
        }
    }
    Ok(LiveRow {
        batch_samples: int(item("batch_samples")?)?,
        starved: item("starved")?.is_truthy()?,
        means,
        queue_depth,
        latest,
    })
}

/// The broad trainer's `live.v2.json` (`RatchetLiveTelemetry.v2`), from a `HardwareMonitor`.
#[pyclass(name = "RatchetLive", unsendable)]
struct PyRatchetLive {
    inner: RatchetLive,
}

#[pymethods]
impl PyRatchetLive {
    #[new]
    fn new(path: PathBuf, hardware: &Bound<'_, PyHardwareMonitor>, owner: &str) -> Self {
        let reading = lock(&hardware.get().inner).shared();
        Self {
            inner: RatchetLive::new(path, reading, owner),
        }
    }

    /// Rewrite the file unless it was written under 100 ms ago; `True` when it wrote.
    fn tick(&mut self, best_known: bool) -> PyResult<bool> {
        Ok(self.inner.tick(best_known)?)
    }
}

#[pyclass(name = "FullFlywheelReporter", unsendable)]
struct PyFullFlywheel {
    inner: crate::hooks::flywheel::FullFlywheel,
}
#[pymethods]
impl PyFullFlywheel {
    #[new]
    fn new(packet: PathBuf, output: PathBuf, history: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: crate::hooks::flywheel::FullFlywheel::new(&packet, output, history)
                .map_err(runtime_error)?,
        })
    }
    fn training_order(&self, seed: u64, limit: usize, gp: bool) -> PyResult<Vec<String>> {
        self.inner
            .training_order(seed, limit, gp)
            .map_err(runtime_error)
    }
    fn utility<'py>(
        &self,
        py: Python<'py>,
        cids: Vec<String>,
        fitness: Vec<f64>,
        gates: Vec<bool>,
        programs: Vec<String>,
        role: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        to_py(
            py,
            &self
                .inner
                .utility(&cids, &fitness, &gates, &programs, role)
                .map_err(runtime_error)?,
        )
    }
    fn begin(&mut self) {
        self.inner.begin();
    }
    fn begin_pack(&mut self) {
        self.inner.begin_pack();
    }
    fn packed(&mut self, cids: Vec<String>, payload_bytes: Vec<u64>) -> PyResult<()> {
        self.inner
            .packed(&cids, &payload_bytes)
            .map_err(runtime_error)
    }
    #[allow(clippy::too_many_arguments)]
    fn scores(
        &self,
        cids: Vec<String>,
        values: Vec<f64>,
        metric: &str,
        higher: bool,
        model: &str,
        model_sha: &str,
        round: u64,
    ) -> PyResult<()> {
        self.inner
            .scores(&cids, &values, metric, higher, model, model_sha, round)
            .map_err(runtime_error)
    }
    fn finish<'py>(
        &mut self,
        py: Python<'py>,
        row: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let row: Value = serde_json::from_str(&to_json(row)?).map_err(runtime_error)?;
        to_py(py, &self.inner.finish(row).map_err(runtime_error)?)
    }
}
