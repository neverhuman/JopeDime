//! Task-type plumbing for the campaign driver and the selfplay plane.
//!
//! The task kind is OUT-OF-BAND: it comes from a pinned manifest and strict
//! data validation, never from auto-detecting the target values — any of the
//! 42 frozen regression datasets could happen to have y in {0,1}, and silent
//! rerouting would invalidate pinned receipts. Genomes stay task-agnostic;
//! probability semantics are carried by probability-capable heads.
//!
//! This module is the library's task-vocabulary authority (R2): the wire type
//! `selfplay_shadow::Task` carries receipt-pinned serde names and converts
//! to/from `TaskKind`, while the task→string, task→corpus-directory, and
//! task→primary-metric maps live here.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(dead_code)] // consumed by the binary-classification campaign lane
pub enum TaskKind {
    Regression,
    Binary,
    Timeseries,
    Multiclass,
}

impl TaskKind {
    /// Receipt-facing name, stable across schema versions.
    #[allow(dead_code)]
    pub fn as_str(self) -> &'static str {
        match self {
            TaskKind::Regression => "regression",
            TaskKind::Binary => "binary",
            TaskKind::Timeseries => "timeseries",
            TaskKind::Multiclass => "multiclass",
        }
    }

    /// On-disk corpus directory under the authoritative quant root (D1).
    ///
    /// The multiclass corpus keeps its legacy `multi_label` directory name
    /// (the on-disk compatibility name; the problem type is multiclass);
    /// every other task's directory matches its wire name.
    #[allow(dead_code)]
    pub fn corpus_dir_name(self) -> &'static str {
        match self {
            TaskKind::Multiclass => "multi_label",
            other => other.as_str(),
        }
    }

    /// The higher-is-better authority metric for this task (R1/D2). Every
    /// duplicated per-file metric map must agree with this one.
    #[allow(dead_code)]
    pub fn primary_metric(self) -> &'static str {
        match self {
            TaskKind::Regression => "r2",
            TaskKind::Binary => "auc",
            TaskKind::Timeseries => "negative_mase",
            TaskKind::Multiclass => "macro_f1",
        }
    }

    #[allow(dead_code)]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "regression" => Some(TaskKind::Regression),
            "binary" => Some(TaskKind::Binary),
            "timeseries" => Some(TaskKind::Timeseries),
            "multiclass" => Some(TaskKind::Multiclass),
            _ => None,
        }
    }
}

impl std::fmt::Display for TaskKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_kind_parses_and_displays_timeseries() {
        assert_eq!(TaskKind::parse("timeseries"), Some(TaskKind::Timeseries));
        assert_eq!(TaskKind::Timeseries.as_str(), "timeseries");
        assert_eq!(TaskKind::Timeseries.to_string(), "timeseries");
        assert_eq!(TaskKind::parse("multiclass"), Some(TaskKind::Multiclass));
    }

    #[test]
    fn task_kind_maps_corpus_dirs_and_metrics() {
        assert_eq!(TaskKind::Regression.corpus_dir_name(), "regression");
        assert_eq!(TaskKind::Binary.corpus_dir_name(), "binary");
        assert_eq!(TaskKind::Timeseries.corpus_dir_name(), "timeseries");
        assert_eq!(TaskKind::Multiclass.corpus_dir_name(), "multi_label");
        assert_eq!(TaskKind::Regression.primary_metric(), "r2");
        assert_eq!(TaskKind::Binary.primary_metric(), "auc");
        assert_eq!(TaskKind::Timeseries.primary_metric(), "negative_mase");
        assert_eq!(TaskKind::Multiclass.primary_metric(), "macro_f1");
        // The on-disk compatibility directory name is NOT a task string.
        assert_eq!(TaskKind::parse("multi_label"), None);
    }
}
