#[pymethods]
impl PyRecorder {
    #[new]
    #[pyo3(signature = (run_dir, producer_names, metadata=None))]
    fn new(
        run_dir: PathBuf,
        producer_names: Vec<String>,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Self::start(
            run_dir,
            &producer_names,
            metadata_map(metadata)?,
            MetricGroups::new(),
        )
    }

    fn set_metric_layout(&mut self, source: usize, columns: Vec<(u16, u16)>) -> PyResult<()> {
        let width = columns.len();
        let producer = self.producer_mut(source)?;
        producer.set_metric_layout(columns).map_err(value_error)?;
        let buffers = self
            .submit_buffers
            .get_mut(source)
            .ok_or_else(|| PyValueError::new_err("unknown producer index"))?;
        buffers.width = width;
        buffers.values.resize(width.saturating_mul(32), 0.0);
        Ok(())
    }

    #[pyo3(signature = (source, metric_id, group_id, step, value, sampled=true))]
    fn emit_scalar(
        &mut self,
        source: usize,
        metric_id: u16,
        group_id: u16,
        step: u64,
        value: f64,
        sampled: bool,
    ) -> PyResult<bool> {
        Ok(self
            .producer_mut(source)?
            .emit_scalar(metric_id, group_id, step, value, sampled))
    }

    #[pyo3(signature = (source, metric_id, group_id, step, reason_id, sampled=true))]
    fn emit_unavailable(
        &mut self,
        source: usize,
        metric_id: u16,
        group_id: u16,
        step: u64,
        reason_id: u16,
        sampled: bool,
    ) -> PyResult<bool> {
        Ok(self
            .producer_mut(source)?
            .emit_unavailable(metric_id, group_id, step, reason_id, sampled))
    }

    /// One step's throughput from raw counters, divided and emitted here. tok/s is the labeled
    /// pair `throughput.tokens_per_s.rust` (over `rust_seconds` when the Rust output was timed
    /// alone, else `seconds`) and `throughput.tokens_per_s.all_languages`, so `rust_tokens` and
    /// `all_tokens` come together. Also `throughput.samples_per_s`, `io.batch_bytes_per_s` with
    /// `io.batch_gb_s`, and `io.h2d_gb_s` over `h2d_seconds`. Returns the events queued.
    #[pyo3(signature = (
        source, step, seconds, *, rust_tokens=None, all_tokens=None, rust_seconds=None,
        samples=None, batch_bytes=None, h2d_bytes=None, h2d_seconds=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn emit_rates(
        &mut self,
        source: usize,
        step: u64,
        seconds: f64,
        rust_tokens: Option<f64>,
        all_tokens: Option<f64>,
        rust_seconds: Option<f64>,
        samples: Option<f64>,
        batch_bytes: Option<f64>,
        h2d_bytes: Option<f64>,
        h2d_seconds: Option<f64>,
    ) -> PyResult<usize> {
        let counters = io::RateCounters {
            rust_tokens,
            all_tokens,
            rust_seconds,
            samples,
            batch_bytes,
            h2d_bytes,
            h2d_seconds,
        };
        let rates = io::rates(seconds, &counters).map_err(value_error)?;
        let producer = self.producer_mut(source)?;
        Ok(rates
            .into_iter()
            .filter(|&(metric, value)| {
                let sampled = crate::train::metric(metric)
                    .is_some_and(|m| matches!(m.cadence, Cadence::Sampled));
                producer.emit_scalar(metric, 0, step, value, sampled)
            })
            .count())
    }

    /// The run's reasoning tok/s, a third tok/s metric apart from the Rust-only and
    /// all-languages pair: `reasoning_tokens` (the run's total so far, counted by `tokenizer`)
    /// over `wall_seconds`, or over the seconds since this recorder started when absent. Emit it
    /// at the end of the run, or periodically with the running total; the last one is the run's
    /// value. Writes the three `throughput.reasoning_*` events and a `reasoning-tok-s` record
    /// with the tokenizer. Returns the rate.
    #[pyo3(signature = (source, step, *, reasoning_tokens, tokenizer, wall_seconds=None))]
    fn emit_reasoning_tok_per_s(
        &mut self,
        source: usize,
        step: u64,
        reasoning_tokens: f64,
        tokenizer: &str,
        wall_seconds: Option<f64>,
    ) -> PyResult<f64> {
        let elapsed = self.producer_mut(source)?.run_elapsed_s();
        let run = io::reasoning_rate(reasoning_tokens, tokenizer, wall_seconds, elapsed)
            .map_err(value_error)?;
        let producer = self.producer_mut(source)?;
        for (metric, value) in run.events() {
            producer.emit_scalar(metric, 0, step, value, false);
        }
        crate::record::record(self.recorder_ref()?, io::REASONING_KIND, &run);
        Ok(run.reasoning_tok_per_s)
    }

    /// Raw block submit of `rows` x layout-width float64 values (`array('d')` buffers).
    fn submit(
        &mut self,
        py: Python<'_>,
        source: usize,
        steps: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
        rows: usize,
    ) -> PyResult<u64> {
        if rows > 32 {
            return Err(value_error("rows exceeds the frozen batch cap of 32"));
        }
        let step_buffer = PyBuffer::<u64>::get(steps)?;
        let value_buffer = PyBuffer::<f64>::get(values)?;
        if step_buffer.item_count() > 32 || value_buffer.item_count() > 32 * 4096 {
            return Err(value_error(
                "Python telemetry batch exceeds frozen buffer bounds",
            ));
        }
        let buffers = self
            .submit_buffers
            .get_mut(source)
            .ok_or_else(|| PyValueError::new_err("unknown producer index"))?;
        if buffers.width == 0
            || step_buffer.item_count() < rows
            || value_buffer.item_count() != buffers.values.len()
        {
            return Err(value_error(
                "Python telemetry buffers do not match the registered layout",
            ));
        }
        step_buffer.copy_to_slice(py, &mut buffers.steps[..step_buffer.item_count()])?;
        value_buffer.copy_to_slice(py, &mut buffers.values)?;
        let slot = self
            .producers
            .get_mut(source)
            .ok_or_else(|| PyValueError::new_err("unknown producer index"))?;
        let producer = slot
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("producer is already submitting"))?;
        let buffers = std::mem::replace(buffers, SubmitBuffers::empty());
        let width = buffers.width;
        let (producer, buffers, result) = py.detach(move || {
            let mut producer = producer;
            let result = producer.submit_values(
                &buffers.steps[..rows],
                rows,
                &buffers.values[..rows.saturating_mul(width)],
            );
            (producer, buffers, result)
        });
        *slot = Some(producer);
        self.submit_buffers[source] = buffers;
        result.map_err(runtime_error)
    }

    fn record_json(&self, kind: &str, body_json: &str) -> PyResult<bool> {
        Ok(emit_json_line(self.recorder_ref()?, kind, body_json))
    }

    /// Queue `record` (a JSON-compatible dict) as a `kind` record. False when it was dropped.
    fn emit(&self, kind: &str, record: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(emit_json_line(
            self.recorder_ref()?,
            kind,
            &to_json(record)?,
        ))
    }

    /// A thread-safe handle on this run's record queue, for evaluator workers and the
    /// evolution queue.
    fn record_sink(&self) -> PyResult<PyRecordSink> {
        Ok(PyRecordSink {
            inner: self.recorder_ref()?.record_handle(),
        })
    }

    fn run_dir(&self) -> PyResult<String> {
        Ok(self.recorder_ref()?.run_dir().display().to_string())
    }

    /// Fix the step row of `source` to `columns` (`metric` or `metric/<group label>`).
    #[pyo3(signature = (source, columns, rows=32))]
    fn step_layout(&mut self, source: usize, columns: Vec<String>, rows: usize) -> PyResult<()> {
        self.set_step_layout(source, columns, rows)
    }

    #[getter]
    fn step_columns(&self) -> Vec<String> {
        self.steps
            .get(STEP_SOURCE)
            .and_then(Option::as_ref)
            .map(|s| s.batch.columns().to_vec())
            .unwrap_or_default()
    }

    #[getter]
    fn submitted_rows(&self) -> u64 {
        self.steps
            .iter()
            .flatten()
            .map(|s| s.batch.submitted_rows)
            .sum()
    }

    #[getter]
    fn submit_errors(&self) -> u64 {
        self.steps
            .iter()
            .flatten()
            .map(|s| s.batch.submit_errors)
            .sum()
    }

    /// Start the host wall-clock spans of one step.
    fn begin(&mut self) -> PyResult<()> {
        self.step_state()?.timer.begin();
        Ok(())
    }

    /// Close the span since the previous mark into `column`; returns it in ms.
    fn mark(&mut self, column: &str) -> PyResult<f64> {
        let col = self.step_col(column)?;
        let state = self.step_state()?;
        let ms = state.timer.mark();
        state.batch.put(col, ms).map_err(value_error)?;
        Ok(ms)
    }

    /// Store one already-materialized host value in the current step row.
    fn put(&mut self, column: &str, value: f64) -> PyResult<()> {
        let col = self.step_col(column)?;
        self.step_state()?
            .batch
            .put(col, value)
            .map_err(value_error)
    }

    /// Write `step.wall` and `step.unattributed` (when in the layout) for the current row.
    fn end(&mut self) -> PyResult<()> {
        let state = self.step_state()?;
        let (wall, unattributed) = state.timer.end();
        if let Some(c) = state.batch.col("step.wall") {
            state.batch.put(c, wall).map_err(value_error)?;
        }
        if let Some(c) = state.batch.col("step.unattributed") {
            state.batch.put(c, unattributed).map_err(value_error)?;
        }
        Ok(())
    }

    /// Close the current row as optimizer step `step`; a full block is submitted.
    fn commit(&mut self, step: u64) -> PyResult<()> {
        let (state, producer) = (
            self.steps.get_mut(STEP_SOURCE).and_then(Option::as_mut),
            self.producers.get_mut(STEP_SOURCE).and_then(Option::as_mut),
        );
        match (state, producer) {
            (Some(state), Some(producer)) => {
                state.batch.commit(step, producer);
                Ok(())
            }
            _ => Err(PyRuntimeError::new_err("no step layout or producer")),
        }
    }

    /// Submit every partial block (epoch end, shutdown). Never called per step.
    fn flush(&mut self) {
        self.flush_all();
    }

    /// Data-supply counters for one popped flywheel batch dict; returns its array bytes.
    fn record_io(
        &mut self,
        batch: &Bound<'_, PyDict>,
        prep_ms: f64,
        queue: &Bound<'_, PyAny>,
        data_wait_ms: f64,
    ) -> PyResult<u64> {
        let mut bytes = 0u64;
        for key in io::ARRAY_KEYS {
            let array = batch
                .get_item(key)?
                .ok_or_else(|| PyValueError::new_err(format!("batch has no {key} array")))?;
            bytes += array.getattr("nbytes")?.extract::<u64>()?;
        }
        let samples = batch
            .get_item("ctx_b")?
            .ok_or_else(|| PyValueError::new_err("batch has no ctx_b array"))?
            .getattr("shape")?
            .get_item(0)?
            .extract::<u64>()?;
        let queue = if queue.is_none() {
            None
        } else {
            Some(QueueSample {
                depth: queue.getattr("last_queue_depth")?.extract()?,
                wait_ms: queue.getattr("last_queue_wait_ms")?.extract()?,
                fallback: queue.getattr("last_pop_fallback")?.extract()?,
            })
        };
        let sample = IoSample {
            prep_ms,
            queue,
            data_wait_ms,
            samples,
            bytes,
        };
        let (Some(io), Some(state)) = (
            self.io.as_mut(),
            self.steps.get_mut(STEP_SOURCE).and_then(Option::as_mut),
        ) else {
            return Err(PyRuntimeError::new_err("recorder has no IO step columns"));
        };
        io.record(&mut state.batch, &sample).map_err(value_error)?;
        Ok(bytes)
    }

    // H2D sampler slots; the CUDA events live in `jope.telemetry.cuda_events.CudaH2D`.

    fn h2d_enabled(&self, source: usize) -> bool {
        self.h2d
            .get(source)
            .and_then(Option::as_ref)
            .is_some_and(|s| s.enabled)
    }

    fn h2d_will_sample(&self, source: usize, step: u64) -> bool {
        self.h2d
            .get(source)
            .and_then(Option::as_ref)
            .is_some_and(|s| s.will_sample(step))
    }

    #[pyo3(signature = (source, step, size_bytes, data_wait_ms=None))]
    fn h2d_begin(
        &mut self,
        source: usize,
        step: u64,
        size_bytes: u64,
        data_wait_ms: Option<f64>,
    ) -> PyResult<i64> {
        Ok(self
            .h2d_mut(source)?
            .begin(step, size_bytes, data_wait_ms)
            .map_or(-1, |s| s as i64))
    }

    fn h2d_recorded(&mut self, source: usize, slot: usize) -> PyResult<()> {
        self.h2d_mut(source)?.recorded(slot);
        Ok(())
    }

    fn h2d_pending(&self, source: usize) -> Vec<usize> {
        self.h2d
            .get(source)
            .and_then(Option::as_ref)
            .map(H2dSampler::pending_slots)
            .unwrap_or_default()
    }

    fn h2d_complete(&mut self, source: usize, slot: usize, elapsed_ms: f64) -> PyResult<()> {
        let Some(row) = self.h2d_mut(source)?.complete(slot, elapsed_ms) else {
            return Ok(());
        };
        let (Some(state), Some(producer)) = (
            self.steps.get_mut(source).and_then(Option::as_mut),
            self.producers.get_mut(source).and_then(Option::as_mut),
        ) else {
            return Err(PyRuntimeError::new_err("H2D source has no step layout"));
        };
        for (col, value) in row.values.iter().enumerate() {
            state.batch.put(col, *value).map_err(value_error)?;
        }
        state.batch.commit(row.step, producer);
        Ok(())
    }

    #[pyo3(signature = (source, slot=None))]
    fn h2d_fail(&mut self, source: usize, slot: Option<usize>) -> PyResult<()> {
        self.h2d_mut(source)?.fail(slot);
        Ok(())
    }

    /// The shutdown stream synchronization failed: count the pending samples as dropped.
    fn h2d_abandon(&mut self, source: usize) -> PyResult<()> {
        self.h2d_mut(source)?.abandon_pending();
        Ok(())
    }

    /// CUDA events could not be created for `source`: record its columns as unavailable.
    fn h2d_unavailable(&mut self, source: usize) -> PyResult<()> {
        self.h2d_mut(source)?.enabled = false;
        let evolution = source == trainers::W_SOURCE_EVO_H2D;
        trainers::emit_h2d_unavailable(self.producer_mut(source)?, evolution, false);
        Ok(())
    }

    #[getter]
    fn h2d_dropped(&self) -> u64 {
        self.h2d_dropped_total()
    }

    /// End a flywheel round: flush, emit the `evo-queue` round record, close. Sets
    /// `queue_stats["h2d_event_sample_drops"]` and `queue_stats["telemetry"]` (the manifest).
    fn finish_round<'py>(
        &mut self,
        py: Python<'py>,
        queue_stats: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.flush_all();
        queue_stats.set_item("h2d_event_sample_drops", self.h2d_dropped_total())?;
        let body = self.round_record(queue_stats, &[])?;
        emit_json_line(self.recorder_ref()?, "jope.telemetry.evo-queue/v1", &body);
        let manifest = py
            .import("json")?
            .call_method1("loads", (self.finalize(true)?,))?;
        queue_stats.set_item("telemetry", &manifest)?;
        Ok(manifest)
    }

    /// Finalize an interrupted round: flush, emit the interrupted round record, and finish the
    /// run as interrupted. Never raises; returns notes for the original error.
    fn interrupt_round(
        &mut self,
        optimizer_steps: u64,
        queue_stats: &Bound<'_, PyDict>,
    ) -> Vec<String> {
        let mut notes = Vec::new();
        self.flush_all();
        let extra = [
            ("optimizer_steps", J::from(optimizer_steps)),
            ("h2d_event_sample_drops", J::from(self.h2d_dropped_total())),
            ("interrupted", J::from(true)),
        ];
        match self.round_record(queue_stats, &extra) {
            Ok(body) => match self.recorder_ref() {
                Ok(r) => {
                    emit_json_line(r, "jope.telemetry.evo-queue/v1", &body);
                }
                Err(e) => notes.push(format!("telemetry interrupted round record failed ({e})")),
            },
            Err(e) => notes.push(format!("telemetry interrupted round record failed ({e})")),
        }
        if let Err(e) = self.finalize(false) {
            notes.push(format!(
                "telemetry interrupted recorder finalization failed ({e})"
            ));
        }
        notes
    }

    fn close<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let text = self.finalize(true)?;
        py.import("json")?.call_method1("loads", (text,))
    }

    /// Finalize an incomplete run as interrupted after flushing its producers.
    fn interrupt<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let text = self.finalize(false)?;
        py.import("json")?.call_method1("loads", (text,))
    }
}
