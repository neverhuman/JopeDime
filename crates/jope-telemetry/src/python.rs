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
use jope_datasets::catalog::Task;
use jope_datasets::family_contracts::{
    build_binary_target_summaries, build_split_view_manifest, derive_binary_source_family,
    freeze_source_family_split as rust_freeze_source_family_split, OuterSplit,
};
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

// Bounded source fragments keep the bridge in this module scope.
include!("python/part1.rs");
include!("python/part2.rs");
include!("python/part3.rs");
include!("python/part4.rs");
include!("python/part5.rs");
include!("python/part6.rs");
include!("python/part7.rs");
include!("python/part8.rs");
include!("python/part9.rs");
include!("python/part10.rs");
include!("python/part11.rs");
include!("python/part12.rs");
include!("python/part13.rs");
