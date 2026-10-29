//! Training-side metric registry and helpers: TELEMETRY_SPEC rows T1–T16, D1–D5 and V1–V7.
//!
//! Hot-path values travel as fixed numeric events keyed by the `u16` ids below; the names,
//! units and producer coverage go into `manifest.json` once. Python producers fill a device
//! stats ring whose column layout is a list of these ids (see [`RingLayout`]).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cadence {
    Step,
    Sampled,
    Epoch,
    Validation,
    Checkpoint,
    Run,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct MetricDef {
    pub id: u16,
    pub name: &'static str,
    pub unit: &'static str,
    pub row: &'static str,
    pub cadence: Cadence,
    /// Values are per parameter group, with the group id in the event's `group` field. For grouped
    /// validation metrics, the run-local `group_id` indexes `manifest.metric_groups[metric_id]`; its
    /// dictionary label identifies family/task/fold without putting strings in hot-path events.
    pub grouped: bool,
}

macro_rules! metrics {
    ($( $konst:ident = $id:literal, $name:literal, $unit:literal, $row:literal, $cad:ident, $grouped:literal; )*) => {
        $( pub const $konst: u16 = $id; )*
        /// Every TRAIN-side metric. Ids are stable: append, never renumber or reuse.
        pub const METRICS: &[MetricDef] = &[
            $( MetricDef { id: $id, name: $name, unit: $unit, row: $row, cadence: Cadence::$cad, grouped: $grouped }, )*
        ];
    };
}

metrics! {
    TRAIN_LOSS = 0x0101, "train.loss", "loss", "T1", Step, false;
    TRAIN_LOSS_COMPONENT = 0x0102, "train.loss.component", "loss", "T2", Step, true;
    TRAIN_NONFINITE = 0x0103, "train.nonfinite", "count", "T3", Step, false;
    GRAD_NORM_GLOBAL = 0x0104, "optim.grad_norm.global", "l2", "T4", Step, false;
    GRAD_CLIPPED = 0x0105, "optim.clipped", "bool", "T4", Step, false;
    GRAD_CLIP_COEF = 0x0106, "optim.clip_coef", "ratio", "T4", Step, false;
    GRAD_NORM_GROUP = 0x0107, "optim.grad_norm.group", "l2", "T5", Sampled, true;
    WEIGHT_NORM_GROUP = 0x0108, "optim.weight_norm.group", "l2", "T6", Sampled, true;
    UPDATE_NORM_GROUP = 0x0109, "optim.update_norm.group", "l2", "T7", Sampled, true;
    UPDATE_RATIO_GROUP = 0x010a, "optim.update_ratio.group", "ratio", "T7", Sampled, true;
    LR_GROUP = 0x010b, "optim.lr.group", "lr", "T8", Step, true;
    SCHED_STEP = 0x010c, "sched.step", "step", "T8", Step, false;
    EPOCH_FRAC = 0x010d, "epoch_frac", "epoch", "T8", Step, false;
    ADAM_EXP_AVG_SQ_MEAN = 0x010e, "optim.adam.exp_avg_sq.mean", "scalar", "T9", Epoch, true;
    AMP_LOSS_SCALE = 0x010f, "amp.loss_scale", "scale", "T10", Step, false;
    AMP_SKIPPED = 0x0110, "amp.skipped_steps", "count", "T10", Step, false;
    SPAN_FWD_MS = 0x0111, "step.span.fwd", "ms", "T11", Step, false;
    SPAN_BWD_MS = 0x0112, "step.span.bwd", "ms", "T11", Step, false;
    SPAN_OPT_MS = 0x0113, "step.span.opt", "ms", "T11", Step, false;
    SPAN_VAL_MS = 0x0114, "step.span.val", "ms", "T11", Validation, false;
    SPAN_CKPT_MS = 0x0115, "step.span.ckpt", "ms", "T11", Checkpoint, false;
    SPAN_GUARD_MS = 0x0116, "step.span.guard", "ms", "T11", Epoch, false;
    SAMPLES_PER_S = 0x0117, "throughput.samples_per_s", "1/s", "T12", Step, false;
    DATASETS_PER_S = 0x0118, "throughput.datasets_per_s", "1/s", "T12", Step, false;
    TOKENS_PER_S = 0x0119, "throughput.tokens_per_s", "1/s", "T12", Step, false;
    EXISTING_TELEMETRY_S = 0x011a, "train.existing_telemetry_s", "s", "T13", Step, false;
    COORDINATION_S = 0x011b, "train.coordination_s", "s", "T13", Step, false;
    COMMUNICATION_S = 0x011c, "train.communication_s", "s", "T13", Step, false;
    CHECKPOINT_S = 0x011d, "train.checkpoint_s", "s", "T13", Checkpoint, false;
    EXPOSURE_FAMILIES = 0x011e, "train.exposure.families_seen", "count", "T15", Epoch, false;
    EXPOSURE_DATASETS = 0x011f, "train.exposure.datasets_seen", "count", "T15", Epoch, false;
    PLATEAU_EPOCHS = 0x0120, "train.plateau.epochs_since_best", "count", "T16", Epoch, false;
    STEP_WALL_MS = 0x0121, "step.wall", "ms", "T11", Step, false;
    STEP_UNATTRIBUTED_MS = 0x0122, "step.unattributed", "ms", "T11", Step, false;
    GPU_SPAN_FWD_MS = 0x0123, "gpu.span.fwd", "ms", "T11", Sampled, false;
    GPU_SPAN_BWD_MS = 0x0124, "gpu.span.bwd", "ms", "T11", Sampled, false;
    GPU_SPAN_OPT_MS = 0x0125, "gpu.span.opt", "ms", "T11", Sampled, false;
    RING_SLOT_BUSY_DROPS = 0x0126, "ring.slot_busy_drops", "count", "T1", Step, false;
    SPAN_DATA_MS = 0x0127, "step.span.data", "ms", "T11", Step, false;
    SPAN_H2D_MS = 0x0128, "step.span.h2d", "ms", "T11", Step, false;
    DOPE_HASH_EQUAL = 0x0201, "dope.freeze.equal", "bool", "D1", Epoch, false;
    PARAMS_TRAINABLE = 0x0202, "params.count.trainable", "count", "D2", Epoch, true;
    PARAMS_FROZEN = 0x0203, "params.count.frozen", "count", "D2", Epoch, true;
    DOPE_IN_OPTIMIZER = 0x0204, "dope.in_optimizer", "count", "D3", Epoch, false;
    DOPE_GRAD_PRESENT = 0x0205, "dope.grad_present", "count", "D4", Sampled, false;
    DOPE_TRAINING_MODE = 0x0206, "dope.training_mode", "bool", "D4", Sampled, false;
    DOPE_GUARD_MS = 0x0207, "dope.freeze.guard_ms", "ms", "D5", Epoch, false;
    VAL_LOSS = 0x0301, "val.loss", "loss", "V1", Validation, false;
    VAL_LOSS_COMPONENT = 0x0302, "val.loss.component", "loss", "V1", Validation, true;
    VAL_SEALED_TOUCH = 0x0307, "val.sealed_touch", "count", "V7", Validation, false;
    VAL_SEALED_LABEL_MISSING = 0x0308, "val.sealed_label_missing", "count", "V7", Validation, false;
    VAL_LOSS_FAMILY_TASK = 0x0310, "val.loss.family_task", "loss", "V1", Validation, true;
    VAL_R2_FAMILY_TASK = 0x0311, "val.r2.family_task", "r2", "V2", Validation, true;
    VAL_AUC_FAMILY_TASK = 0x0312, "val.auc.family_task", "auc", "V3", Validation, true;
    VAL_LOGLOSS_FAMILY_TASK = 0x0313, "val.logloss.family_task", "loss", "V3", Validation, true;
    VAL_BRIER_FAMILY_TASK = 0x0314, "val.brier.family_task", "loss", "V3", Validation, true;
    VAL_ACC_FAMILY_TASK = 0x0315, "val.acc.family_task", "fraction", "V3", Validation, true;
    VAL_ECE_FAMILY_TASK = 0x0316, "val.ece.family_task", "fraction", "V3", Validation, true;
    VAL_LOSS_FAMILY_TASK = 0x0310, "val.loss.family_task", "loss", "V1", Validation, true;
    VAL_R2_FAMILY_TASK = 0x0311, "val.r2.family_task", "r2", "V2", Validation, true;
    VAL_AUC_FAMILY_TASK = 0x0312, "val.auc.family_task", "auc", "V3", Validation, true;
    VAL_LOGLOSS_FAMILY_TASK = 0x0313, "val.logloss.family_task", "loss", "V3", Validation, true;
    VAL_BRIER_FAMILY_TASK = 0x0314, "val.brier.family_task", "loss", "V3", Validation, true;
    VAL_ACC_FAMILY_TASK = 0x0315, "val.acc.family_task", "fraction", "V3", Validation, true;
    VAL_ECE_FAMILY_TASK = 0x0316, "val.ece.family_task", "fraction", "V3", Validation, true;
    IO_LOCAL_BYTES = 0x0401, "io.local.bytes", "bytes", "I1", Run, false;
    IO_LOCAL_READ_MS = 0x0402, "io.local.read", "ms", "I1", Sampled, false;
    IO_LOCAL_READ_MIB_S = 0x0403, "io.local.read_mib_s", "MiB/s", "I1", Sampled, false;
    IO_BATCH_PREP_MS = 0x0404, "io.batch_prep", "ms", "I2", Step, true;
    IO_QUEUE_DEPTH = 0x0405, "io.queue.depth", "batches", "I3", Step, false;
    IO_QUEUE_WAIT_MS = 0x0406, "io.queue.wait", "ms", "I3", Step, false;
    IO_QUEUE_FALLBACK = 0x0407, "io.queue.fallback", "count", "I3", Step, false;
    IO_H2D_BYTES = 0x0408, "io.h2d.bytes", "bytes", "I4", Sampled, true;
    IO_H2D_MS = 0x0409, "io.h2d", "ms", "I4", Sampled, true;
    IO_H2D_GB_S = 0x040a, "io.h2d_gb_s", "GB/s", "I4", Sampled, true;
    IO_S3_GET_MS = 0x040b, "io.s3.get", "ms", "I5", Step, false;
    IO_S3_GET_BYTES = 0x040c, "io.s3.bytes", "bytes", "I5", Step, false;
    IO_S3_RETRIES = 0x040d, "io.s3.retries", "count", "I5", Step, false;
    IO_CACHE_HIT = 0x040e, "io.cache.hit", "count", "I6", Step, false;
    IO_CACHE_MISS = 0x040f, "io.cache.miss", "count", "I6", Step, false;
    IO_DATA_WAIT_MS = 0x0410, "io.data_wait", "ms", "I9", Step, false;
    GPU_UTIL = 0x0411, "host.gpu.util", "percent", "I7", Sampled, true;
    GPU_MEMORY_BYTES = 0x0412, "host.gpu.memory", "bytes", "I7", Sampled, true;
    CPU_PROCESS_UTIL = 0x0413, "host.cpu.process_util", "percent", "I8", Sampled, false;
    CPU_NODE_UTIL = 0x0414, "host.cpu.node_util", "percent", "I8", Sampled, false;
    CPU_CORE_UTIL = 0x0415, "host.cpu.core_util", "percent", "I8", Sampled, true;
    CPU_PROCESS_RSS_BYTES = 0x0416, "host.cpu.process_rss", "bytes", "I8", Sampled, false;
    IO_BATCHES_PER_S = 0x0417, "io.batches_per_s", "1/s", "I2", Step, false;
    IO_SAMPLES_PER_S = 0x0418, "io.samples_per_s", "1/s", "I2", Step, false;
    IO_BATCH_BYTES = 0x0419, "io.batch.bytes", "bytes", "I2", Step, false;
    IO_BATCH_BYTES_PER_S = 0x041a, "io.batch_bytes_per_s", "bytes/s", "I2", Step, false;
    IO_BATCH_GB_S = 0x041b, "io.batch_gb_s", "GB/s", "I2", Step, false;
}

pub fn metric(id: u16) -> Option<&'static MetricDef> {
    METRICS.iter().find(|m| m.id == id)
}

/// Column layout of one device stats ring slot: `[rows, columns.len()]` float32, row-major.
/// Each column is a metric id plus a group id for grouped metrics (for example, the loss
/// component index). Step numbers are not in the block: float32 is exact only up to 2^24, so
/// the trainer keeps them in a host-side `u64` array per slot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RingLayout {
    pub columns: Vec<(u16, u16)>,
}

/// One decoded ring cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepValue {
    pub step: u64,
    pub metric: u16,
    pub group: u16,
    pub value: f64,
}

impl RingLayout {
    pub fn new(columns: Vec<(u16, u16)>) -> Result<Self, String> {
        for &(id, _) in &columns {
            if metric(id).is_none() {
                return Err(format!("unknown metric id {id:#06x}"));
            }
        }
        Ok(Self { columns })
    }

    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// Decode a row-major `[rows, width]` block and return how many cells were non-finite.
    /// Those cells are kept as they are: T3 counts them, and the alarm engine sees the raw value.
    pub fn decode(
        &self,
        steps: &[u64],
        block: &[f32],
        rows: usize,
        out: &mut Vec<StepValue>,
    ) -> Result<u64, String> {
        let w = self.width();
        if steps.len() < rows || block.len() < rows * w {
            return Err(format!(
                "slot has {} steps and {} cells, need {rows} and {}",
                steps.len(),
                block.len(),
                rows * w
            ));
        }
        let mut nonfinite = 0;
        for (r, &step) in steps[..rows].iter().enumerate() {
            for (&(id, group), &v) in self.columns.iter().zip(&block[r * w..(r + 1) * w]) {
                nonfinite += !v.is_finite() as u64;
                out.push(StepValue {
                    step,
                    metric: id,
                    group,
                    value: f64::from(v),
                });
            }
        }
        Ok(nonfinite)
    }
}

/// The schedule a config declares, used by alarm A6 to check what the optimizer reports.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Schedule {
    Constant {
        lr: f64,
    },
    /// `torch.optim.lr_scheduler.CosineAnnealingLR(T_max, eta_min)`, closed form, stepped once
    /// per optimizer step as `train_flywheel_model_for_round` does. The recursive PyTorch
    /// update differs from the closed form only by rounding, far inside the A6 tolerance.
    Cosine {
        base_lr: f64,
        eta_min: f64,
        t_max: u64,
    },
}

impl Schedule {
    /// LR in effect for optimizer step `t` (0-based, before `scheduler.step()` for that step).
    pub fn lr_at(&self, t: u64) -> f64 {
        match *self {
            Self::Constant { lr } => lr,
            Self::Cosine {
                base_lr,
                eta_min,
                t_max,
            } => {
                let t = t.min(t_max) as f64;
                eta_min
                    + (base_lr - eta_min)
                        * (1.0 + (std::f64::consts::PI * t / t_max.max(1) as f64).cos())
                        / 2.0
            }
        }
    }
}

/// Top-level submodule of a parameter name, used for D2 counts: `dope_model.encoder.w` →
/// `dope_model`.
pub fn submodule(param_name: &str) -> &str {
    param_name.split('.').next().unwrap_or(param_name)
}

/// The submodules whose parameters are DOPE. Anything else is the trainable JopeDime portion.
pub const DOPE_SUBMODULES: &[&str] = &["dope_model"];

pub fn is_dope_param(param_name: &str) -> bool {
    DOPE_SUBMODULES.contains(&submodule(param_name))
}

/// What one evaluation result says about its partition (row V7). Only a label already present
/// in an authorized result or the frozen fold manifest is consulted; the sealed roster and
/// sealed data are never opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartitionCheck {
    NotSealed,
    Sealed,
    LabelMissing,
}

/// Labels that mark the sealed partition in result and manifest fields.
pub const SEALED_LABELS: &[&str] = &["sealed", "meta_test", "meta-test", "sealed_meta_test"];

pub fn check_partition_label(label: Option<&str>) -> PartitionCheck {
    match label.map(str::trim) {
        None | Some("") => PartitionCheck::LabelMissing,
        Some(l) if SEALED_LABELS.iter().any(|s| s.eq_ignore_ascii_case(l)) => {
            PartitionCheck::Sealed
        }
        Some(_) => PartitionCheck::NotSealed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ids_and_names_are_unique() {
        let ids: HashSet<u16> = METRICS.iter().map(|m| m.id).collect();
        let names: HashSet<&str> = METRICS.iter().map(|m| m.name).collect();
        assert_eq!(ids.len(), METRICS.len());
        assert_eq!(names.len(), METRICS.len());
    }

    #[test]
    fn ring_decode_counts_nonfinite_and_keeps_exact_steps() {
        let l = RingLayout::new(vec![(TRAIN_LOSS, 0), (GRAD_NORM_GLOBAL, 0)]).unwrap();
        let big = (1u64 << 24) + 1; // not representable in f32
        let block = [1.5f32, 0.2, f32::NAN, 0.3];
        let mut out = Vec::new();
        assert_eq!(l.decode(&[big, big + 1], &block, 2, &mut out).unwrap(), 1);
        assert_eq!(out.len(), 4);
        assert_eq!((out[2].step, out[2].metric), (big + 1, TRAIN_LOSS));
        assert!(out[2].value.is_nan());
        assert!(l.decode(&[0], &block, 2, &mut out).is_err());
        assert!(RingLayout::new(vec![(0xffff, 0)]).is_err());
    }

    #[test]
    fn cosine_schedule_endpoints() {
        let s = Schedule::Cosine {
            base_lr: 1.2e-3,
            eta_min: 1e-4,
            t_max: 100,
        };
        assert!((s.lr_at(0) - 1.2e-3).abs() < 1e-15);
        assert!((s.lr_at(100) - 1e-4).abs() < 1e-15);
        assert!((s.lr_at(50) - (1e-4 + 1.1e-3 / 2.0)).abs() < 1e-15);
        assert_eq!(s.lr_at(500), s.lr_at(100));
    }

    #[test]
    fn partition_labels() {
        assert_eq!(
            check_partition_label(Some("train")),
            PartitionCheck::NotSealed
        );
        assert_eq!(
            check_partition_label(Some("Meta_Test")),
            PartitionCheck::Sealed
        );
        assert_eq!(
            check_partition_label(Some(" ")),
            PartitionCheck::LabelMissing
        );
        assert_eq!(check_partition_label(None), PartitionCheck::LabelMissing);
    }

    #[test]
    fn dope_param_classification() {
        assert!(is_dope_param("dope_model.layers.0.weight"));
        assert!(!is_dope_param("jope_head.weight"));
        assert!(!is_dope_param("dope_modelx.w"));
    }
}
