//! Resource variables: what a candidate *cost* to train and to serve, as fitness inputs.
//!
//! A business objective often trades accuracy against training time (a model retrained every
//! market session must finish before the gate closes), inference latency (a pricing call has a
//! p99 budget), throughput or memory (an edge device has a fixed heap). These variables let a
//! specification read those costs the same way it reads `y`, `pred` or an auxiliary column.
//!
//! Two kinds exist, and the difference matters for reproducibility:
//!
//! - **measured** variables (wall/CPU seconds, latency, peak heap) are readings of this machine
//!   under its current load. Two runs never agree on them, so a search ranking by one records
//!   every reading it ranked by and re-derives selection from those records, never by measuring
//!   again;
//! - **structural** variables (model bytes, program nodes, stage count, row counts) are functions
//!   of the program and the data alone, bit-identical across runs and hosts.
//!
//! Prefer a structural proxy when one expresses the business cost; it is the reproducible one.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Every resource a specification may read. Names are the snake_case JSON spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceVar {
    /// Wall seconds of one fit on the training rows.
    FitSeconds,
    /// CPU seconds of the whole process during one fit (every thread the fit used).
    FitCpuSeconds,
    /// Wall seconds to predict the held-out batch (median of three passes).
    PredictSeconds,
    /// Batch prediction latency per row, in microseconds.
    PredictUsPerRow,
    /// Batch prediction throughput in rows per second (maximize).
    PredictRowsPerSecond,
    /// Median single-row (online) prediction latency, in microseconds.
    RowLatencyP50Us,
    /// 99th-percentile single-row (online) prediction latency, in microseconds.
    RowLatencyP99Us,
    /// Peak process heap growth above its starting level during one fit, in bytes.
    FitPeakHeapBytes,
    /// Peak process heap growth above its starting level while predicting the held-out batch.
    PredictPeakHeapBytes,
    /// Bytes of the fitted model in its canonical serialized form (structural).
    ModelBytes,
    /// Program cost: stages plus expression nodes (structural).
    ProgramNodes,
    /// Pipeline stages, the head included (structural).
    StageCount,
    /// Training rows the fit saw (structural; for per-row normalisation).
    TrainRows,
    /// Held-out rows scored (structural; for per-row normalisation).
    HoldRows,
}

/// Readings for one candidate, keyed by variable.
pub type Resources = BTreeMap<ResourceVar, f64>;

/// One hard limit of a [`super::Fitness::Constrained`] specification. At least one bound is set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceLimit {
    pub resource: ResourceVar,
    /// The largest feasible value (inclusive).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// The smallest feasible value (inclusive).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
}

impl ResourceVar {
    pub const ALL: [ResourceVar; 14] = [
        Self::FitSeconds,
        Self::FitCpuSeconds,
        Self::PredictSeconds,
        Self::PredictUsPerRow,
        Self::PredictRowsPerSecond,
        Self::RowLatencyP50Us,
        Self::RowLatencyP99Us,
        Self::FitPeakHeapBytes,
        Self::PredictPeakHeapBytes,
        Self::ModelBytes,
        Self::ProgramNodes,
        Self::StageCount,
        Self::TrainRows,
        Self::HoldRows,
    ];

    /// The JSON spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FitSeconds => "fit_seconds",
            Self::FitCpuSeconds => "fit_cpu_seconds",
            Self::PredictSeconds => "predict_seconds",
            Self::PredictUsPerRow => "predict_us_per_row",
            Self::PredictRowsPerSecond => "predict_rows_per_second",
            Self::RowLatencyP50Us => "row_latency_p50_us",
            Self::RowLatencyP99Us => "row_latency_p99_us",
            Self::FitPeakHeapBytes => "fit_peak_heap_bytes",
            Self::PredictPeakHeapBytes => "predict_peak_heap_bytes",
            Self::ModelBytes => "model_bytes",
            Self::ProgramNodes => "program_nodes",
            Self::StageCount => "stage_count",
            Self::TrainRows => "train_rows",
            Self::HoldRows => "hold_rows",
        }
    }

    /// Whether the value is a reading of the machine (non-reproducible) rather than a function of
    /// the program and the data.
    pub fn is_measured(self) -> bool {
        !matches!(
            self,
            Self::ModelBytes
                | Self::ProgramNodes
                | Self::StageCount
                | Self::TrainRows
                | Self::HoldRows
        )
    }

    /// Whether more is better. Only throughput is.
    pub fn maximized(self) -> bool {
        self == Self::PredictRowsPerSecond
    }

    /// Whether producing the value needs repeated single-row predictions.
    pub fn needs_row_latency(self) -> bool {
        matches!(self, Self::RowLatencyP50Us | Self::RowLatencyP99Us)
    }

    /// Whether producing the value needs the process heap meter.
    pub fn needs_heap_meter(self) -> bool {
        matches!(self, Self::FitPeakHeapBytes | Self::PredictPeakHeapBytes)
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::FitSeconds => "Wall seconds of one fit on the training rows (measured).",
            Self::FitCpuSeconds => {
                "CPU seconds of the process during one fit, all threads (measured)."
            }
            Self::PredictSeconds => {
                "Wall seconds to predict the held-out batch, median of 3 passes (measured)."
            }
            Self::PredictUsPerRow => "Batch prediction microseconds per row (measured).",
            Self::PredictRowsPerSecond => {
                "Batch prediction rows per second; more is better (measured)."
            }
            Self::RowLatencyP50Us => {
                "Median one-row prediction latency in microseconds, up to 256 calls (measured)."
            }
            Self::RowLatencyP99Us => {
                "99th-percentile one-row prediction latency in microseconds (measured)."
            }
            Self::FitPeakHeapBytes => {
                "Peak process heap growth during one fit, in bytes (measured; needs the meter)."
            }
            Self::PredictPeakHeapBytes => {
                "Peak process heap growth predicting the held-out batch (measured; needs the meter)."
            }
            Self::ModelBytes => "Bytes of the fitted model, canonical serialized form (structural).",
            Self::ProgramNodes => "Program cost: stages plus expression nodes (structural).",
            Self::StageCount => "Pipeline stages including the head (structural).",
            Self::TrainRows => "Training rows the fit saw (structural).",
            Self::HoldRows => "Held-out rows scored (structural).",
        }
    }
}

impl std::fmt::Display for ResourceVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A candidate that breaks a [`ResourceLimit`]: refused, never ranked. Callers that record a
/// failure status can find it with `error.downcast_ref::<Infeasible>()`.
#[derive(Clone, Debug, PartialEq)]
pub struct Infeasible {
    pub resource: ResourceVar,
    pub value: f64,
    pub limit: f64,
    pub bound: &'static str,
}

impl std::fmt::Display for Infeasible {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let op = if self.bound == "max" { ">" } else { "<" };
        write!(
            f,
            "resource_infeasible: {} = {} {op} {} {}",
            self.resource, self.value, self.bound, self.limit
        )
    }
}

impl std::error::Error for Infeasible {}
