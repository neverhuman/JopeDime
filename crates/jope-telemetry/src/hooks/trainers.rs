//! Column layouts, metric-group labels and availability records for the instrumented trainers:
//! P (`python/src/jope/train.py`, active-v2) and W (`research/flywheel_selfplay`, one recorder
//! per training round). These were Python in `train.py` and `selfplay_loop.py`; the trainers
//! now only pass their identity and the host values they already have.

use crate::core::ProducerHandle;
use crate::hooks::io::{ReadSnapshot, END_TO_END_COLUMN, H2D_COLUMNS, IO_COLUMNS};
use crate::manifest::{
    REASON_CUDA_EVENT_UNAVAILABLE, REASON_NO_NVML_OR_GPU, REASON_NO_TEXT_TOKEN_PATH,
    REASON_PROCFS_UNAVAILABLE, REASON_STARTUP_READS_NOT_OBSERVED,
};
use crate::train::{self, metric_by_name};
use anyhow::{anyhow, bail, Result};
use std::collections::BTreeMap;

pub type MetricGroups = BTreeMap<u16, BTreeMap<u16, String>>;

/// W's step row: host scalars, host spans, then the data-supply counters.
pub const W_STEP_COLUMNS: [&str; 12] = [
    "train.loss",
    "train.loss.component/dir_margin",
    "train.loss.component/evo_val",
    "optim.lr.group",
    "step.span.data",
    "step.span.h2d",
    "step.span.fwd",
    "step.span.readback",
    "step.span.bwd",
    "step.span.opt",
    "step.wall",
    "step.unattributed",
];
pub const W_COMPONENTS: [&str; 2] = ["dir_margin", "evo_val"];
/// Producer order of a W step recorder.
pub const W_PRODUCERS: [&str; 4] = [
    "flywheel-trainer",
    "h2d-events",
    "h2d-evo-events",
    "host-probe",
];
pub const W_SOURCE_MAIN_H2D: usize = 1;
pub const W_SOURCE_EVO_H2D: usize = 2;

/// Resolve `metric` or `metric/<group label>` column names to `(metric_id, group_id)`, with
/// group labels looked up in the run's `metric_groups`.
pub fn resolve_layout(columns: &[String], groups: &MetricGroups) -> Result<Vec<(u16, u16)>> {
    columns
        .iter()
        .map(|column| {
            let (name, label) = match column.split_once('/') {
                Some((n, l)) => (n, Some(l)),
                None => (column.as_str(), None),
            };
            let def =
                metric_by_name(name).ok_or_else(|| anyhow!("unknown telemetry metric {name}"))?;
            let group = match label {
                None => 0,
                Some(label) => {
                    if !def.grouped {
                        bail!("telemetry metric {name} is not declared grouped");
                    }
                    *groups
                        .get(&def.id)
                        .and_then(|g| g.iter().find(|(_, l)| l.as_str() == label))
                        .ok_or_else(|| anyhow!("{name} has no group labelled {label:?}"))?
                        .0
                }
            };
            Ok((def.id, group))
        })
        .collect()
}

fn labels<S: AsRef<str>>(names: impl IntoIterator<Item = S>) -> BTreeMap<u16, String> {
    names
        .into_iter()
        .enumerate()
        .map(|(i, n)| (i as u16, n.as_ref().to_owned()))
        .collect()
}

/// P's columns for its loss components, in sorted component order.
pub fn python_trainer_layout(components: &[String]) -> (Vec<String>, MetricGroups) {
    let mut components = components.to_vec();
    components.sort();
    let mut columns = vec!["train.loss".to_owned()];
    columns.extend(
        components
            .iter()
            .map(|c| format!("train.loss.component/{c}")),
    );
    columns.extend(
        [
            "optim.lr.group",
            "step.span.data",
            "step.span.fwd",
            "step.span.bwd",
            "step.span.opt",
            "step.span.readback",
            "step.wall",
            "step.unattributed",
        ]
        .map(str::to_owned),
    );
    let groups = MetricGroups::from([
        (train::TRAIN_LOSS_COMPONENT, labels(&components)),
        (train::LR_GROUP, labels(["optimizer_group_0"])),
    ]);
    (columns, groups)
}

/// W's step row. Ungrouped names (`optim.lr.group`, `io.batch_prep`) write group 0, which is
/// labelled `optimizer_group_0` / `batch_supply` in the run's metric groups.
pub fn flywheel_step_columns() -> Vec<String> {
    W_STEP_COLUMNS
        .iter()
        .chain(IO_COLUMNS.iter())
        .map(|c| (*c).to_owned())
        .collect()
}

/// The H2D source layouts: the main batch writes group 0 and the end-to-end column; the
/// evolution batch writes group 1 and leaves the end-to-end column missing.
pub fn flywheel_h2d_columns(evolution: bool) -> Vec<String> {
    let label = if evolution {
        "evolution_batch"
    } else {
        "main_batch"
    };
    let mut columns: Vec<String> = H2D_COLUMNS.iter().map(|c| format!("{c}/{label}")).collect();
    columns.push(END_TO_END_COLUMN.to_owned());
    columns
}

pub fn flywheel_metric_groups(cpu_count: usize) -> MetricGroups {
    let mut groups = MetricGroups::from([
        (train::TRAIN_LOSS_COMPONENT, labels(W_COMPONENTS)),
        (train::LR_GROUP, labels(["optimizer_group_0"])),
        (train::IO_BATCH_PREP_MS, labels(["batch_supply"])),
        (
            train::CPU_CORE_UTIL,
            labels((0..cpu_count.max(1)).map(|i| format!("cpu_{i}"))),
        ),
    ]);
    for column in H2D_COLUMNS {
        let id = metric_by_name(column).map_or(0, |m| m.id);
        groups.insert(id, labels(["main_batch", "evolution_batch"]));
    }
    groups
}

fn id(name: &str) -> u16 {
    metric_by_name(name).map_or(0, |m| m.id)
}

/// Startup-read scalars (or their unavailability) and the always-unavailable S3, cache and
/// token metrics on W's main producer, at step 0.
pub fn emit_flywheel_startup(main: &mut ProducerHandle, startup: Option<&ReadSnapshot>) {
    match startup.filter(|s| s.enabled) {
        Some(s) => {
            main.emit_scalar(id("io.local.bytes"), 0, 0, s.logical_bytes as f64, true);
            main.emit_scalar(id("io.local.read"), 0, 0, s.read_ms, true);
            if s.read_ms > 0.0 {
                let mib_s = s.logical_bytes as f64 / (1024.0 * 1024.0) / (s.read_ms / 1000.0);
                main.emit_scalar(id("io.local.read_mib_s"), 0, 0, mib_s, true);
            } else {
                main.emit_unavailable(
                    id("io.local.read_mib_s"),
                    0,
                    0,
                    REASON_PROCFS_UNAVAILABLE,
                    true,
                );
            }
        }
        None => {
            for name in ["io.local.bytes", "io.local.read", "io.local.read_mib_s"] {
                main.emit_unavailable(id(name), 0, 0, REASON_STARTUP_READS_NOT_OBSERVED, true);
            }
        }
    }
    for name in ["io.s3.get", "io.s3.bytes", "io.s3.retries"] {
        main.emit_unavailable(id(name), 0, 0, crate::manifest::REASON_NO_S3_PATH, true);
    }
    for name in ["io.cache.hit", "io.cache.miss"] {
        main.emit_unavailable(
            id(name),
            0,
            0,
            crate::manifest::REASON_NO_OBJECT_CACHE,
            true,
        );
    }
    for name in [
        "throughput.tokens_per_s",
        "throughput.tokens_per_s.rust",
        "throughput.tokens_per_s.all_languages",
    ] {
        main.emit_unavailable(id(name), 0, 0, REASON_NO_TEXT_TOKEN_PATH, true);
    }
}

/// H2D columns of one source are unavailable: no CUDA device (`no_gpu`) or no CUDA events.
pub fn emit_h2d_unavailable(producer: &mut ProducerHandle, evolution: bool, no_gpu: bool) {
    let reason = if no_gpu {
        REASON_NO_NVML_OR_GPU
    } else {
        REASON_CUDA_EVENT_UNAVAILABLE
    };
    let group = u16::from(evolution);
    for name in H2D_COLUMNS {
        producer.emit_unavailable(id(name), group, 0, reason, true);
    }
    if !evolution {
        producer.emit_unavailable(id(END_TO_END_COLUMN), 0, 0, reason, true);
    }
}

/// `os.uname().nodename` on Linux.
pub fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|h| h.trim().to_owned())
        .ok()
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "unknown".to_owned())
}

/// Online CPUs: the `cpuN` lines of `/proc/stat`, the same cores the host probe reports as
/// `host.cpu.core_util` groups.
pub fn cpu_count() -> usize {
    std::fs::read_to_string("/proc/stat")
        .ok()
        .map(|stat| {
            stat.lines()
                .filter(|l| {
                    l.strip_prefix("cpu")
                        .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
                })
                .count()
        })
        .filter(|n| *n > 0)
        .or_else(|| std::thread::available_parallelism().ok().map(usize::from))
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_resolve_against_their_groups() {
        let (cols, groups) = python_trainer_layout(&["rank".into(), "huber".into()]);
        assert_eq!(cols[1], "train.loss.component/huber");
        let layout = resolve_layout(&cols, &groups).unwrap();
        assert_eq!(layout[1], (train::TRAIN_LOSS_COMPONENT, 0));
        assert_eq!(layout[2], (train::TRAIN_LOSS_COMPONENT, 1));
        assert_eq!(layout[3], (train::LR_GROUP, 0));

        let groups = flywheel_metric_groups(4);
        let w = resolve_layout(&flywheel_step_columns(), &groups).unwrap();
        assert_eq!(w.len(), 21);
        assert_eq!(w[2], (train::TRAIN_LOSS_COMPONENT, 1));
        let evo = resolve_layout(&flywheel_h2d_columns(true), &groups).unwrap();
        assert_eq!(evo[0].1, 1);
        assert_eq!(evo[3], (train::IO_BATCH_GB_S, 0));
        assert_eq!(groups[&train::CPU_CORE_UTIL].len(), 4);
    }

    #[test]
    fn layout_errors_are_explicit() {
        let groups = MetricGroups::new();
        assert!(resolve_layout(&["no.such.metric".into()], &groups).is_err());
        assert!(resolve_layout(&["train.loss/x".into()], &groups).is_err());
        assert!(resolve_layout(&["train.loss.component/x".into()], &groups).is_err());
        assert!(cpu_count() >= 1 && !hostname().is_empty());
    }
}
