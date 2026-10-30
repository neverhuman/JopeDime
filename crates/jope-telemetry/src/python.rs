//! PyO3 bridge (`jope._telemetry`). Every telemetry computation is Rust; the torch loops call
//! these classes with values they already have. Python buffers are copied into bounded
//! Rust-owned blocks while the GIL is held, then the GIL is released before queue submission.
//!
//! * `Recorder`: one run. Besides the raw producer API it owns the trainer step row (`begin`,
//!   `mark`, `put`, `end`, `commit`), the flywheel IO counters (`record_io`), the H2D sampler
//!   slots (`h2d_*`, driven by `jope.telemetry.cuda_events`) and the round records.
//! * `RecordSink`: a `Send + Sync` handle on a run's record queue for worker threads.
//! * `EvalBatch`, `GenerationRecorder`, `ValidationRecorder`, `DopeGuard`, `LocalReads`.
//! * `HardwareMonitor`, `gpu_snapshot`, `LiveMetricsWriter`, `RatchetLive`: the DIME score
//!   trainers' NVML monitor, `live-metrics.jsonl` windows and `live.v2.json` snapshot.

use crate::api::GpSession;
use crate::collectors::SystemCollector;
use crate::core::{ProducerHandle, RecordHandle, Recorder, RecorderBuilder};
use crate::hooks::dope::{self, DopeGuard, NamedParam, Observation, TensorDigest};
use crate::hooks::eval::{self, EvalBatch};
use crate::hooks::generation::{self, EvalResult, GenerationRecorder};
use crate::hooks::hardware::{self, GpuReader, HardwareMonitor, RatchetLive};
use crate::hooks::io::{self, BatchIo, H2dSampler, IoSample, QueueSample, ReadStats};
use crate::hooks::json::J;
use crate::hooks::live::{self, LiveMetricsWriter, LiveRow};
use crate::hooks::step::{StepBatch, StepTimer};
use crate::hooks::trainers::{self, MetricGroups};
use crate::hooks::validation::{self, DatasetResult, ValidationError, ValidationRecorder};
use crate::manifest::RunMetadata;
use crate::record::RecordSink;
use crate::train::Cadence;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

pyo3::create_exception!(
    _telemetry,
    DopeFreezeViolationError,
    PyRuntimeError,
    "A DOPE weight, buffer, grad flag, optimizer membership or eval-mode invariant broke."
);
pyo3::create_exception!(
    _telemetry,
    SealedPartitionError,
    PyRuntimeError,
    "An evaluated result was labeled as the sealed partition (row V7, alarm A8)."
);

const STEP_SOURCE: usize = 0;
const MAX_JSON_DEPTH: u32 = 128;

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ---- JSON bridging -------------------------------------------------------------------------

/// `json.dumps(obj, allow_nan=False, separators=(",", ":"))` in Rust: dicts keep their order,
/// non-finite floats and unsupported types raise like `json` does.
fn write_json(obj: &Bound<'_, PyAny>, out: &mut String, depth: u32) -> PyResult<()> {
    if depth > MAX_JSON_DEPTH {
        return Err(value_error("Circular reference detected"));
    }
    if obj.is_none() {
        out.push_str("null");
    } else if let Ok(b) = obj.cast::<PyBool>() {
        out.push_str(if b.is_true() { "true" } else { "false" });
    } else if obj.cast::<PyInt>().is_ok() {
        out.push_str(&obj.str()?.to_cow()?);
    } else if let Ok(f) = obj.cast::<PyFloat>() {
        let v = f.value();
        if !v.is_finite() {
            return Err(value_error(format!(
                "Out of range float values are not JSON compliant: {v}"
            )));
        }
        out.push_str(&serde_json::to_string(&v).map_err(value_error)?);
    } else if let Ok(s) = obj.cast::<PyString>() {
        out.push_str(&serde_json::to_string(&*s.to_cow()?).map_err(value_error)?);
    } else if let Ok(d) = obj.cast::<PyDict>() {
        out.push('{');
        for (i, (k, v)) in d.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let key = if let Ok(s) = k.cast::<PyString>() {
                s.to_cow()?.into_owned()
            } else if k.is_none() {
                "null".to_owned()
            } else if let Ok(b) = k.cast::<PyBool>() {
                (if b.is_true() { "true" } else { "false" }).to_owned()
            } else if k.cast::<PyInt>().is_ok() || k.cast::<PyFloat>().is_ok() {
                k.str()?.to_cow()?.into_owned()
            } else {
                return Err(PyTypeError::new_err(format!(
                    "keys must be str, int, float, bool or None, not {}",
                    k.get_type().name()?
                )));
            };
            out.push_str(&serde_json::to_string(&key).map_err(value_error)?);
            out.push(':');
            write_json(&v, out, depth + 1)?;
        }
        out.push('}');
    } else if obj.cast::<PyList>().is_ok() || obj.cast::<PyTuple>().is_ok() {
        out.push('[');
        for (i, item) in obj.try_iter()?.enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_json(&item?, out, depth + 1)?;
        }
        out.push(']');
    } else {
        return Err(PyTypeError::new_err(format!(
            "Object of type {} is not JSON serializable",
            obj.get_type().name()?
        )));
    }
    Ok(())
}

fn to_json(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut out = String::new();
    write_json(obj, &mut out, 0)?;
    Ok(out)
}

/// `json.dumps(obj, sort_keys=True)` with the default separators, `repr` floats and `NaN`: the
/// spelling `live-metrics.jsonl` had, for the fields a live row passes through.
fn write_py_json(obj: &Bound<'_, PyAny>, out: &mut String, depth: u32) -> PyResult<()> {
    if depth > MAX_JSON_DEPTH {
        return Err(value_error("Circular reference detected"));
    }
    if obj.is_none() {
        out.push_str("null");
    } else if let Ok(b) = obj.cast::<PyBool>() {
        out.push_str(if b.is_true() { "true" } else { "false" });
    } else if obj.cast::<PyInt>().is_ok() {
        out.push_str(&obj.str()?.to_cow()?);
    } else if let Ok(f) = obj.cast::<PyFloat>() {
        out.push_str(&live::py_float(f.value()));
    } else if let Ok(s) = obj.cast::<PyString>() {
        out.push_str(&live::py_str(&s.to_cow()?));
    } else if let Ok(d) = obj.cast::<PyDict>() {
        let mut items = Vec::with_capacity(d.len());
        for (k, v) in d.iter() {
            let key = k
                .cast::<PyString>()
                .map_err(|_| PyTypeError::new_err("live metrics dict keys must be str"))?
                .to_cow()?
                .into_owned();
            items.push((key, v));
        }
        items.sort_by(|a, b| a.0.cmp(&b.0));
        out.push('{');
        for (i, (k, v)) in items.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&live::py_str(k));
            out.push_str(": ");
            write_py_json(v, out, depth + 1)?;
        }
        out.push('}');
    } else if obj.cast::<PyList>().is_ok() || obj.cast::<PyTuple>().is_ok() {
        out.push('[');
        for (i, item) in obj.try_iter()?.enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            write_py_json(&item?, out, depth + 1)?;
        }
        out.push(']');
    } else {
        return Err(PyTypeError::new_err(format!(
            "Object of type {} is not JSON serializable",
            obj.get_type().name()?
        )));
    }
    Ok(())
}

/// Serialize in Rust, then `json.loads`, so the dict keeps the serialized key order.
fn to_py<'py, T: Serialize + ?Sized>(py: Python<'py>, value: &T) -> PyResult<Bound<'py, PyAny>> {
    let text = serde_json::to_string(value).map_err(runtime_error)?;
    py.import("json")?.call_method1("loads", (text,))
}

fn emit_record(sink: &Option<RecordHandle>, kind: &str, body: &impl Serialize) -> bool {
    sink.as_ref()
        .is_some_and(|s| crate::record::record(s, kind, body))
}

fn emit_json_line(sink: &dyn RecordSink, kind: &str, body_json: &str) -> bool {
    match crate::record::encode_record_json_line(kind, body_json) {
        Ok(line) => sink.record_line(kind, line),
        Err(error) => {
            sink.record_encoding_error(error.contains("exceeds size cap"));
            false
        }
    }
}

// ---- registry ------------------------------------------------------------------------------

#[pyfunction]
fn metric_registry() -> Vec<(u16, &'static str, &'static str, &'static str, bool)> {
    crate::train::METRICS
        .iter()
        .map(|m| (m.id, m.name, m.unit, m.row, m.grouped))
        .collect()
}

#[pyfunction]
fn metric_id(name: &str) -> PyResult<u16> {
    crate::train::metric_by_name(name)
        .map(|m| m.id)
        .ok_or_else(|| PyValueError::new_err(format!("unknown telemetry metric {name}")))
}

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

impl PyRecorder {
    fn start(
        run_dir: PathBuf,
        producer_names: &[String],
        mut metadata: BTreeMap<String, Value>,
        mut metric_groups: MetricGroups,
    ) -> PyResult<Self> {
        if let Some(groups) = metadata.remove("metric_groups") {
            metric_groups.extend(parse_metric_groups(groups)?);
        }
        if producer_names.iter().any(|name| name == "host-probe") {
            if let Ok(nvml) = nvml_wrapper::Nvml::init() {
                if let Ok(count) = nvml.device_count() {
                    let labels: BTreeMap<u16, String> = (0..count.min(4096))
                        .map(|index| (index as u16, format!("nvml_device_{index}")))
                        .collect();
                    metric_groups.insert(crate::train::GPU_UTIL, labels.clone());
                    metric_groups.insert(crate::train::GPU_MEMORY_BYTES, labels);
                }
            }
        }
        let mut builder = RecorderBuilder::new(
            run_dir,
            RunMetadata {
                metadata,
                metric_groups: metric_groups.clone(),
                ..RunMetadata::default()
            },
        );
        for name in producer_names {
            builder.add_producer(name.clone()).map_err(value_error)?;
        }
        let (recorder, producers) = builder.start().map_err(runtime_error)?;
        let mut producers: Vec<Option<ProducerHandle>> = producers.into_iter().map(Some).collect();
        let host_collector =
            if let Some(index) = producer_names.iter().position(|name| name == "host-probe") {
                let producer = producers
                    .get_mut(index)
                    .and_then(Option::take)
                    .ok_or_else(|| PyRuntimeError::new_err("host-probe producer is unavailable"))?;
                Some(
                    SystemCollector::start(producer, std::time::Duration::from_secs(1)).map_err(
                        |error| PyRuntimeError::new_err(format!("start host collector: {error}")),
                    )?,
                )
            } else {
                None
            };
        let n = producers.len();
        Ok(Self {
            recorder: Some(recorder),
            submit_buffers: (0..n).map(|_| SubmitBuffers::new()).collect(),
            producers,
            host_collector,
            metric_groups,
            steps: (0..n).map(|_| None).collect(),
            h2d: (0..n).map(|_| None).collect(),
            io: None,
            round: None,
        })
    }

    fn producer_mut(&mut self, source: usize) -> PyResult<&mut ProducerHandle> {
        self.producers
            .get_mut(source)
            .ok_or_else(|| PyValueError::new_err("unknown producer index"))?
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("producer is unavailable"))
    }

    fn recorder_ref(&self) -> PyResult<&Recorder> {
        self.recorder
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("recorder is closed"))
    }

    fn set_step_layout(
        &mut self,
        source: usize,
        columns: Vec<String>,
        rows: usize,
    ) -> PyResult<()> {
        let layout =
            trainers::resolve_layout(&columns, &self.metric_groups).map_err(value_error)?;
        let batch = StepBatch::new(columns, rows).map_err(value_error)?;
        self.producer_mut(source)?
            .set_metric_layout(layout)
            .map_err(value_error)?;
        *self
            .steps
            .get_mut(source)
            .ok_or_else(|| PyValueError::new_err("unknown producer index"))? = Some(StepState {
            batch,
            timer: StepTimer::default(),
        });
        Ok(())
    }

    fn step_state(&mut self) -> PyResult<&mut StepState> {
        self.steps
            .get_mut(STEP_SOURCE)
            .and_then(Option::as_mut)
            .ok_or_else(|| PyRuntimeError::new_err("no step layout: call step_layout first"))
    }

    fn step_col(&mut self, column: &str) -> PyResult<usize> {
        let state = self.step_state()?;
        state
            .batch
            .col(column)
            .ok_or_else(|| PyValueError::new_err(format!("step layout has no {column} column")))
    }

    fn flush_all(&mut self) {
        for (state, producer) in self.steps.iter_mut().zip(self.producers.iter_mut()) {
            if let (Some(state), Some(producer)) = (state.as_mut(), producer.as_mut()) {
                state.batch.flush(producer);
            }
        }
    }

    fn h2d_mut(&mut self, source: usize) -> PyResult<&mut H2dSampler> {
        self.h2d
            .get_mut(source)
            .and_then(Option::as_mut)
            .ok_or_else(|| PyValueError::new_err(format!("source {source} has no H2D sampler")))
    }

    fn h2d_dropped_total(&self) -> u64 {
        self.h2d.iter().flatten().map(|s| s.dropped_samples).sum()
    }

    fn stop_collector(&mut self) {
        if let Some(collector) = &mut self.host_collector {
            collector.stop();
        }
        self.host_collector.take();
    }

    fn finalize(&mut self, complete: bool) -> PyResult<String> {
        self.flush_all();
        self.stop_collector();
        self.producers.clear();
        let recorder = self
            .recorder
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("recorder is already closed"))?;
        let manifest = if complete {
            recorder.close()
        } else {
            recorder.interrupt()
        }
        .map_err(runtime_error)?;
        serde_json::to_string(&manifest).map_err(runtime_error)
    }

    fn round_record(
        &self,
        queue_stats: &Bound<'_, PyDict>,
        extra: &[(&str, J)],
    ) -> PyResult<String> {
        let mut body = String::from("{\"round\":");
        body.push_str(&self.round.unwrap_or(0).to_string());
        for (k, v) in queue_stats.iter() {
            let key = k
                .cast::<PyString>()
                .map_err(|_| PyTypeError::new_err("queue_stats keys must be str"))?
                .to_cow()?
                .into_owned();
            if key == "round" || extra.iter().any(|(e, _)| *e == key) {
                continue;
            }
            body.push(',');
            body.push_str(&serde_json::to_string(&key).map_err(value_error)?);
            body.push(':');
            write_json(&v, &mut body, 1)?;
        }
        for (k, v) in extra {
            body.push(',');
            body.push_str(&serde_json::to_string(k).map_err(value_error)?);
            body.push(':');
            body.push_str(&serde_json::to_string(v).map_err(value_error)?);
        }
        body.push('}');
        Ok(body)
    }
}

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

fn parse_metric_groups(value: Value) -> PyResult<MetricGroups> {
    let outer = value
        .as_object()
        .ok_or_else(|| PyValueError::new_err("metric_groups must be an object"))?;
    let mut result = MetricGroups::new();
    let mut total_labels = 0_usize;
    for (metric, groups) in outer {
        let metric_id = metric
            .parse::<u16>()
            .map_err(|e| PyValueError::new_err(format!("metric_groups key {metric:?}: {e}")))?;
        let groups = groups
            .as_object()
            .ok_or_else(|| PyValueError::new_err("metric group labels must be objects"))?;
        let mut labels = BTreeMap::new();
        for (group, label) in groups {
            let group_id = group
                .parse::<u16>()
                .map_err(|e| PyValueError::new_err(format!("group id {group:?}: {e}")))?;
            let label = label
                .as_str()
                .ok_or_else(|| PyValueError::new_err("metric group label must be a string"))?;
            if label.is_empty() || label.len() > 128 {
                return Err(PyValueError::new_err(
                    "metric group label must contain 1..=128 bytes",
                ));
            }
            total_labels += label.len();
            if total_labels > 1 << 20
                || result.values().map(BTreeMap::len).sum::<usize>() + labels.len() >= 4096
            {
                return Err(PyValueError::new_err(
                    "metric group registry exceeds frozen bounds",
                ));
            }
            labels.insert(group_id, label.to_owned());
        }
        result.insert(metric_id, labels);
    }
    Ok(result)
}

/// P (`jope.train`, active-v2): one `python-trainer` producer with the loss components, LR and
/// host spans as its step row. The directory is the run directory itself.
#[pyfunction]
#[pyo3(signature = (run_dir, components, metadata=None))]
fn open_python_trainer(
    run_dir: PathBuf,
    components: Vec<String>,
    metadata: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyRecorder> {
    let (columns, groups) = trainers::python_trainer_layout(&components);
    let mut rec = PyRecorder::start(
        run_dir,
        &["python-trainer".to_owned()],
        metadata_map(metadata)?,
        groups,
    )?;
    rec.set_step_layout(STEP_SOURCE, columns, 32)?;
    Ok(rec)
}

/// W (`research/flywheel_selfplay`, one recorder per training round) under
/// `<run_root>/w-step-r<round>-s<seed>-<32 hex>/`: the step row with IO columns, two H2D
/// sources (main and evolution batch) and the 1 s host probe. `startup_reads` is the
/// `LocalReads` that observed the startup loads; its totals are taken (and reset) here.
#[pyfunction]
#[pyo3(signature = (run_root, round_idx, seed, batch_size, device, cuda_available, startup_reads=None))]
fn open_flywheel_step(
    run_root: PathBuf,
    round_idx: u32,
    seed: u64,
    batch_size: u64,
    device: &str,
    cuda_available: bool,
    startup_reads: Option<&Bound<'_, PyLocalReads>>,
) -> PyResult<PyRecorder> {
    let startup = startup_reads.map(|r| lock(&r.get().inner).take_snapshot());
    let cuda = device.starts_with("cuda") && cuda_available;
    let run_dir = run_root.join(format!(
        "w-step-r{round_idx}-s{seed}-{:032x}",
        ulid::Ulid::new().0
    ));
    let startup_json = match &startup {
        Some(s) => serde_json::to_value(s).map_err(runtime_error)?,
        None => serde_json::json!({"enabled": false}),
    };
    let metadata: BTreeMap<String, Value> = [
        ("trainer", Value::from("research.flywheel_selfplay")),
        ("round", round_idx.into()),
        ("seed", seed.into()),
        ("batch_size", batch_size.into()),
        ("device", device.into()),
        ("evo_queue", "enabled_or_disabled_per_run".into()),
        (
            "data_identity",
            "unavailable_without_extra_input_read".into(),
        ),
        ("not_fused_dope_training", true.into()),
        ("host", trainers::hostname().into()),
        ("io_startup_reads", startup_json),
        ("s3_get_capability", false.into()),
        ("object_cache_capability", false.into()),
        ("pin_memory_capability", false.into()),
        ("h2d_event_capability", cuda.into()),
        ("h2d_sample_stride", io::H2D_SAMPLE_STRIDE.into()),
        ("startup_io_capability", startup.is_some().into()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect();
    let producers: Vec<String> = trainers::W_PRODUCERS
        .iter()
        .map(|p| (*p).to_owned())
        .collect();
    let mut rec = PyRecorder::start(
        run_dir,
        &producers,
        metadata,
        trainers::flywheel_metric_groups(trainers::cpu_count()),
    )?;
    rec.round = Some(round_idx);
    rec.set_step_layout(STEP_SOURCE, trainers::flywheel_step_columns(), 32)?;
    let io = BatchIo::new(&rec.step_state()?.batch).map_err(runtime_error)?;
    rec.io = Some(io);
    for (source, evolution) in [
        (trainers::W_SOURCE_MAIN_H2D, false),
        (trainers::W_SOURCE_EVO_H2D, true),
    ] {
        rec.set_step_layout(source, trainers::flywheel_h2d_columns(evolution), 32)?;
        rec.h2d[source] = Some(H2dSampler::new(
            cuda,
            !evolution,
            io::H2D_SAMPLE_STRIDE,
            io::H2D_SLOTS,
        ));
    }
    trainers::emit_flywheel_startup(rec.producer_mut(STEP_SOURCE)?, startup.as_ref());
    if !cuda {
        for source in [trainers::W_SOURCE_MAIN_H2D, trainers::W_SOURCE_EVO_H2D] {
            let evolution = source == trainers::W_SOURCE_EVO_H2D;
            trainers::emit_h2d_unavailable(rec.producer_mut(source)?, evolution, true);
        }
    }
    Ok(rec)
}

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

#[pymodule]
fn _telemetry(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<PyRecorder>()?;
    module.add_class::<PyRecordSink>()?;
    module.add_class::<PyGpSession>()?;
    module.add_class::<PyGenerationRecorder>()?;
    module.add_class::<PyEvalBatch>()?;
    module.add_class::<PyValidationRecorder>()?;
    module.add_class::<PyDopeGuard>()?;
    module.add_class::<PyLocalReads>()?;
    module.add_class::<PyReadSpan>()?;
    module.add_class::<PyHardwareMonitor>()?;
    module.add_class::<PyLiveMetricsWriter>()?;
    module.add_class::<PyRatchetLive>()?;
    module.add_function(wrap_pyfunction!(metric_registry, module)?)?;
    module.add_function(wrap_pyfunction!(metric_id, module)?)?;
    module.add_function(wrap_pyfunction!(open_python_trainer, module)?)?;
    module.add_function(wrap_pyfunction!(open_flywheel_step, module)?)?;
    module.add_function(wrap_pyfunction!(check_partition_label, module)?)?;
    module.add_function(wrap_pyfunction!(gpu_snapshot, module)?)?;
    module.add(
        "DopeFreezeViolationError",
        py.get_type::<DopeFreezeViolationError>(),
    )?;
    module.add(
        "SealedPartitionError",
        py.get_type::<SealedPartitionError>(),
    )?;
    module.add("DOPE_SUBMODULE", dope::DOPE_SUBMODULE)?;
    module.add(
        "DOPE_NORMALIZER_BUFFERS",
        dope::DOPE_NORMALIZER_BUFFERS.to_vec(),
    )?;
    Ok(())
}
