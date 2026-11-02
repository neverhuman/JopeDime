// ---- record sink ---------------------------------------------------------------------------

/// A `Send + Sync` handle on a run's record queue (`Recorder.record_sink()`).
#[pyclass(name = "RecordSink", frozen)]
pub struct PyRecordSink {
    inner: RecordHandle,
}

#[pymethods]
impl PyRecordSink {
    /// Queue `record` (a JSON-compatible dict) as a `kind` record. False when it was dropped.
    fn emit(&self, kind: &str, record: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(emit_json_line(&self.inner, kind, &to_json(record)?))
    }
}

fn sink_of(sink: Option<&Bound<'_, PyRecordSink>>) -> Option<RecordHandle> {
    sink.map(|s| s.get().inner.clone())
}

// ---- recorder ------------------------------------------------------------------------------

struct SubmitBuffers {
    steps: Vec<u64>,
    values: Vec<f64>,
    width: usize,
}

impl SubmitBuffers {
    fn new() -> Self {
        Self {
            steps: vec![0; 32],
            values: Vec::new(),
            width: 0,
        }
    }

    fn empty() -> Self {
        Self {
            steps: Vec::new(),
            values: Vec::new(),
            width: 0,
        }
    }
}

struct StepState {
    batch: StepBatch,
    timer: StepTimer,
}

#[pyclass(name = "Recorder", unsendable)]
pub struct PyRecorder {
    recorder: Option<Recorder>,
    producers: Vec<Option<ProducerHandle>>,
    submit_buffers: Vec<SubmitBuffers>,
    host_collector: Option<SystemCollector>,
    metric_groups: MetricGroups,
    steps: Vec<Option<StepState>>,
    h2d: Vec<Option<H2dSampler>>,
    io: Option<BatchIo>,
    round: Option<u32>,
}

fn metadata_map(metadata: Option<&Bound<'_, PyAny>>) -> PyResult<BTreeMap<String, Value>> {
    let text = match metadata {
        None => "{}".to_owned(),
        Some(m) if m.is_none() => "{}".to_owned(),
        Some(m) => match m.cast::<PyString>() {
            Ok(s) => s.to_cow()?.into_owned(),
            Err(_) => to_json(m)?,
        },
    };
    if text.len() > 1 << 20 {
        return Err(value_error("metadata exceeds 1 MiB registry cap"));
    }
    serde_json::from_str(&text).map_err(|e| value_error(format!("metadata: {e}")))
}
