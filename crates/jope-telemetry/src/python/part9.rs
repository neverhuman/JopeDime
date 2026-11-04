// ---- GP ------------------------------------------------------------------------------------

#[pyclass(name = "GpSession", frozen)]
pub struct PyGpSession {
    inner: Mutex<GpSession>,
}

#[pymethods]
impl PyGpSession {
    #[new]
    #[pyo3(signature = (op_vocab_size=None, seed=None, probe_panel=false))]
    fn new(op_vocab_size: Option<usize>, seed: Option<u64>, probe_panel: bool) -> Self {
        Self {
            inner: Mutex::new(GpSession::new(op_vocab_size, seed, probe_panel)),
        }
    }

    fn summarize_json(&self, generation_json: &str) -> PyResult<String> {
        lock(&self.inner)
            .summarize_json(generation_json)
            .map_err(runtime_error)
    }

    fn dictionaries_json(&self) -> PyResult<String> {
        lock(&self.inner).dictionaries_json().map_err(runtime_error)
    }
}

#[pyclass(name = "GenerationRecorder", frozen)]
pub struct PyGenerationRecorder {
    session: Py<PyGpSession>,
    inner: Mutex<GenerationRecorder>,
    sink: Option<RecordHandle>,
}

/// Any iterable of str (a list, set or frozenset of program hashes).
fn strings(items: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    match items {
        Some(items) if !items.is_none() => {
            items.try_iter()?.map(|s| s?.extract::<String>()).collect()
        }
        _ => Ok(Vec::new()),
    }
}

fn opt_f64(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
    Ok(match d.get_item(key)? {
        Some(v) if !v.is_none() => v.extract::<f64>().ok(),
        _ => None,
    })
}

#[pymethods]
impl PyGenerationRecorder {
    #[new]
    #[pyo3(signature = (session, context, round_idx, sink=None, bank_shas=None, corpus_shas=None))]
    fn new(
        session: Py<PyGpSession>,
        context: String,
        round_idx: u32,
        sink: Option<&Bound<'_, PyRecordSink>>,
        bank_shas: Option<&Bound<'_, PyAny>>,
        corpus_shas: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            session,
            inner: Mutex::new(GenerationRecorder::new(
                context,
                round_idx,
                strings(bank_shas)?,
                strings(corpus_shas)?,
            )),
            sink: sink_of(sink),
        })
    }

    fn proposals(&self, n: u64, seen_rejects: u64) {
        lock(&self.inner).proposals(n, seen_rejects);
    }

    fn provenance(&self, program_sha256: &str) -> &'static str {
        lock(&self.inner).provenance(program_sha256)
    }

    /// Add one evaluator result dict. `rc` overrides the result's own `rc`.
    #[pyo3(signature = (result, parent_dsl=None, improved=None, hwm_winner=false, behavior=None, rc=None))]
    fn add(
        &self,
        result: &Bound<'_, PyDict>,
        parent_dsl: Option<String>,
        improved: Option<bool>,
        hwm_winner: bool,
        behavior: Option<Vec<Option<f64>>>,
        rc: Option<i64>,
    ) -> PyResult<()> {
        let dsl: String = result
            .get_item("dsl")?
            .ok_or_else(|| PyValueError::new_err("result has no dsl"))?
            .extract()?;
        let get_str = |key: &str| -> PyResult<Option<String>> {
            Ok(match result.get_item(key)? {
                Some(v) if !v.is_none() => Some(v.str()?.to_cow()?.into_owned()),
                _ => None,
            })
        };
        let proj16 = match result.get_item("proj16")? {
            Some(v) if !v.is_none() => Some(
                v.try_iter()?
                    .map(|x| x?.extract::<f64>())
                    .collect::<PyResult<Vec<f64>>>()?,
            ),
            _ => None,
        };
        let complexity = match result.get_item("complexity_open_parens")? {
            Some(v) if !v.is_none() => Some(v.extract::<u32>()?),
            _ => None,
        };
        let rc = match rc {
            Some(rc) => Some(rc),
            None => match result.get_item("rc")? {
                Some(v) if !v.is_none() => Some(v.extract::<i64>()?),
                _ => None,
            },
        };
        let r = EvalResult {
            dsl,
            program_sha256: get_str("program_sha256")?,
            operator: get_str("operator")?.unwrap_or_default(),
            status: get_str("status")?.unwrap_or_else(|| "failed".to_owned()),
            rc,
            cv_gate_passed: match result.get_item("cv_gate_passed")? {
                Some(v) => v.is_truthy()?,
                None => false,
            },
            fitness: opt_f64(result, "fitness")?.unwrap_or(100.0),
            complexity,
            proj16,
            cv_mean_log_ratio: opt_f64(result, "cv_mean_log_ratio")?,
            cv_tail_log_ratio: opt_f64(result, "cv_tail_log_ratio")?,
            cv_log_ratio_sd: opt_f64(result, "cv_log_ratio_sd")?,
            test_rmse_ratio: opt_f64(result, "test_rmse_ratio")?,
        };
        lock(&self.inner).add(r, parent_dsl, improved, hwm_winner, behavior);
        Ok(())
    }

    /// Summarize the generation (emitted to the sink and returned) and start the next one.
    fn finish<'py>(&self, py: Python<'py>, generation: u32) -> PyResult<Bound<'py, PyAny>> {
        let summary = {
            let session = self.session.bind(py).get();
            let mut session = lock(&session.inner);
            lock(&self.inner)
                .finish(generation, &mut session)
                .map_err(runtime_error)?
        };
        emit_record(&self.sink, generation::GEN_KIND, &summary);
        to_py(py, &summary)
    }
}
