// ---- DOPE guard ----------------------------------------------------------------------------

#[pyclass(name = "DopeGuard", frozen)]
pub struct PyDopeGuard {
    inner: Mutex<DopeGuard>,
    sink: Option<RecordHandle>,
}

/// Hash `(name, tensor)` pairs one at a time: `to_bytes(tensor)` returns `(dtype tag, shape,
/// uint8 buffer)` in little-endian C order, and only one host copy is alive at a time.
fn digests(
    py: Python<'_>,
    pairs: &Bound<'_, PyAny>,
    to_bytes: &Bound<'_, PyAny>,
    sort: bool,
) -> PyResult<Vec<TensorDigest>> {
    let mut items: Vec<(String, Bound<'_, PyAny>)> = Vec::new();
    for item in pairs.try_iter()? {
        let (name, tensor): (String, Bound<'_, PyAny>) = item?.extract()?;
        items.push((name, tensor));
    }
    if sort {
        items.sort_by(|a, b| a.0.cmp(&b.0));
    }
    let mut out = Vec::with_capacity(items.len());
    for (name, tensor) in items {
        let (tag, shape, data): (String, Vec<usize>, Bound<'_, PyAny>) =
            to_bytes.call1((tensor,))?.extract()?;
        // One host tensor copy at a time; the guard runs at epoch granularity, never per step.
        let bytes = PyBuffer::<u8>::get(&data)?.to_vec(py)?;
        let mut hasher = dope::tensor_hasher(&tag, &shape);
        sha2::Digest::update(&mut hasher, &bytes);
        out.push(TensorDigest::finish(name, hasher, &shape));
    }
    Ok(out)
}

impl PyDopeGuard {
    fn emit_and_enforce<'py>(&self, py: Python<'py>, mut rec: J) -> PyResult<Bound<'py, PyAny>> {
        let reasons = dope::enforce(&mut rec);
        emit_record(&self.sink, dope::RECORD_KIND, &rec);
        if !reasons.is_empty() {
            return Err(DopeFreezeViolationError::new_err(reasons.join("; ")));
        }
        to_py(py, &rec)
    }
}

#[pymethods]
impl PyDopeGuard {
    #[new]
    #[pyo3(signature = (sink=None, sample_every=100))]
    fn new(sink: Option<&Bound<'_, PyRecordSink>>, sample_every: u64) -> Self {
        Self {
            inner: Mutex::new(DopeGuard::new(sample_every)),
            sink: sink_of(sink),
        }
    }

    /// One boundary observation (run start, epoch, checkpoint). `params`, `buffers` and
    /// `normalizers` are `(name, tensor)` pairs; `named` is `(name, numel, requires_grad)` for
    /// every parameter of the fused model; `totals` is `(trainable, frozen)` over
    /// `model.parameters()`. Raises `DopeFreezeViolationError` after recording.
    #[pyo3(signature = (boundary, at, compare, *, params, buffers, normalizers, to_bytes, named, totals, in_optimizer, grads_present, training, reset=false))]
    #[allow(clippy::too_many_arguments)]
    fn observe<'py>(
        &self,
        py: Python<'py>,
        boundary: &str,
        at: i64,
        compare: bool,
        params: &Bound<'py, PyAny>,
        buffers: &Bound<'py, PyAny>,
        normalizers: &Bound<'py, PyAny>,
        to_bytes: &Bound<'py, PyAny>,
        named: Vec<(String, u64, bool)>,
        totals: (u64, u64),
        in_optimizer: u64,
        grads_present: u64,
        training: Vec<String>,
        reset: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let started = Instant::now();
        let snapshot = dope::snapshot(
            digests(py, params, to_bytes, true)?,
            digests(py, buffers, to_bytes, true)?,
            digests(py, normalizers, to_bytes, false)?,
        )
        .map_err(DopeFreezeViolationError::new_err)?;
        let named: Vec<NamedParam> = named
            .into_iter()
            .map(|(name, numel, requires_grad)| NamedParam {
                name,
                numel,
                requires_grad,
            })
            .collect();
        let observation = Observation {
            snapshot,
            counts: dope::param_counts(&named, totals.0, totals.1),
            dope_params_in_optimizer: in_optimizer,
            dope_grads_present: grads_present,
            training_submodules: training,
        };
        let rec = {
            let mut guard = lock(&self.inner);
            if reset {
                guard.reference = None;
            }
            let ms = started.elapsed().as_secs_f64() * 1e3;
            guard.observe(boundary, at, compare, observation, ms)
        };
        self.emit_and_enforce(py, rec)
    }

    /// Row D4 between epoch hashes: every `sample_every` steps, `probe()` returns
    /// `(dope_grads_present, dope_training_submodules)`.
    fn step(&self, py: Python<'_>, step: u64, probe: &Bound<'_, PyAny>) -> PyResult<()> {
        if !lock(&self.inner).should_sample(step) {
            return Ok(());
        }
        let (grads, training): (u64, Vec<String>) = probe.call0()?.extract()?;
        if let Some(rec) = DopeGuard::step_record(step as i64, grads, training) {
            self.emit_and_enforce(py, rec)?;
        }
        Ok(())
    }

    #[getter]
    fn reference_composite_sha256(&self) -> Option<String> {
        lock(&self.inner)
            .reference
            .as_ref()
            .map(|r| r.composite_sha256.clone())
    }
}
