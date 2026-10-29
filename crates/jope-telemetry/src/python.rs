//! Optional PyO3 bridge. Python buffers are copied into bounded Rust-owned blocks while the GIL
//! is held, then the GIL is released before queue submission.

use crate::collectors::SystemCollector;
use crate::core::{ProducerHandle, Recorder, RecorderBuilder};
use crate::manifest::RunMetadata;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[pyclass(name = "Recorder", unsendable)]
pub struct PyRecorder {
    recorder: Option<Recorder>,
    producers: Vec<Option<ProducerHandle>>,
    submit_buffers: Vec<SubmitBuffers>,
    host_collector: Option<SystemCollector>,
}

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

#[pyclass(name = "GpSession", unsendable)]
pub struct PyGpSession {
    inner: crate::api::GpSession,
}

#[pymethods]
impl PyGpSession {
    #[new]
    #[pyo3(signature = (op_vocab_size=None, seed=None, probe_panel=false))]
    fn new(op_vocab_size: Option<usize>, seed: Option<u64>, probe_panel: bool) -> Self {
        Self {
            inner: crate::api::GpSession::new(op_vocab_size, seed, probe_panel),
        }
    }

    fn summarize_json(&mut self, generation_json: &str) -> PyResult<String> {
        self.inner
            .summarize_json(generation_json)
            .map_err(runtime_error)
    }
}

#[pyfunction]
fn metric_registry() -> Vec<(u16, &'static str, &'static str, &'static str, bool)> {
    crate::train::METRICS
        .iter()
        .map(|m| (m.id, m.name, m.unit, m.row, m.grouped))
        .collect()
}

#[pymethods]
impl PyRecorder {
    #[new]
    #[pyo3(signature = (run_dir, producer_names, metadata_json="{}"))]
    fn new(run_dir: String, producer_names: Vec<String>, metadata_json: &str) -> PyResult<Self> {
        if metadata_json.len() > 1 << 20 {
            return Err(PyValueError::new_err(
                "metadata_json exceeds 1 MiB registry cap",
            ));
        }
        let mut metadata: BTreeMap<String, Value> = serde_json::from_str(metadata_json)
            .map_err(|e| PyValueError::new_err(format!("metadata_json: {e}")))?;
        let mut metric_groups = metadata
            .remove("metric_groups")
            .map(parse_metric_groups)
            .transpose()?
            .unwrap_or_default();
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
            PathBuf::from(run_dir),
            RunMetadata {
                metadata,
                metric_groups,
                ..RunMetadata::default()
            },
        );
        for name in &producer_names {
            builder
                .add_producer(name.clone())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
        Ok(Self {
            recorder: Some(recorder),
            submit_buffers: (0..producer_names.len())
                .map(|_| SubmitBuffers::new())
                .collect(),
            producers,
            host_collector,
        })
    }

    fn set_metric_layout(&mut self, source: usize, columns: Vec<(u16, u16)>) -> PyResult<()> {
        let width = columns.len();
        let producer = self.producer_mut(source)?;
        producer
            .set_metric_layout(columns)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let buffers = self
            .submit_buffers
            .get_mut(source)
            .ok_or_else(|| PyValueError::new_err("unknown producer index"))?;
        buffers.width = width;
        buffers.values.resize(width.saturating_mul(32), 0.0);
        Ok(())
    }

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

    fn submit(
        &mut self,
        py: Python<'_>,
        source: usize,
        steps: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
        rows: usize,
    ) -> PyResult<u64> {
        if rows > 32 {
            return Err(PyValueError::new_err(
                "rows exceeds the frozen batch cap of 32",
            ));
        }
        let step_buffer = PyBuffer::<u64>::get(steps)?;
        let value_buffer = PyBuffer::<f64>::get(values)?;
        if step_buffer.item_count() > 32 || value_buffer.item_count() > 32 * 4096 {
            return Err(PyValueError::new_err(
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
            return Err(PyValueError::new_err(
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
        let recorder = self
            .recorder
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("recorder is closed"))?;
        let line = match crate::record::encode_record_json_line(kind, body_json) {
            Ok(line) => line,
            Err(error) => {
                crate::record::RecordSink::record_encoding_error(
                    recorder,
                    error.contains("exceeds size cap"),
                );
                return Ok(false);
            }
        };
        Ok(crate::record::RecordSink::record_line(recorder, kind, line))
    }

    fn run_dir(&self) -> PyResult<String> {
        self.recorder
            .as_ref()
            .map(|r| r.run_dir().display().to_string())
            .ok_or_else(|| PyRuntimeError::new_err("recorder is closed"))
    }

    fn close(&mut self) -> PyResult<String> {
        if let Some(collector) = &mut self.host_collector {
            collector.stop();
        }
        self.host_collector.take();
        self.producers.clear();
        let recorder = self
            .recorder
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("recorder is already closed"))?;
        let manifest = recorder.close().map_err(runtime_error)?;
        serde_json::to_string(&manifest).map_err(runtime_error)
    }
}

fn parse_metric_groups(value: Value) -> PyResult<BTreeMap<u16, BTreeMap<u16, String>>> {
    let outer = value
        .as_object()
        .ok_or_else(|| PyValueError::new_err("metric_groups must be an object"))?;
    let mut result = BTreeMap::<u16, BTreeMap<u16, String>>::new();
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

impl PyRecorder {
    fn producer_mut(&mut self, source: usize) -> PyResult<&mut ProducerHandle> {
        self.producers
            .get_mut(source)
            .ok_or_else(|| PyValueError::new_err("unknown producer index"))?
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("producer is unavailable"))
    }
}

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[pymodule]
fn _telemetry(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRecorder>()?;
    module.add_class::<PyGpSession>()?;
    module.add_function(wrap_pyfunction!(metric_registry, module)?)?;
    Ok(())
}
