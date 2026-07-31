//! Task-type plumbing for the campaign driver.
//!
//! The task kind is OUT-OF-BAND: it comes from a pinned manifest and strict
//! data validation, never from auto-detecting the target values — any of the
//! 42 frozen regression datasets could happen to have y in {0,1}, and silent
//! rerouting would invalidate pinned receipts. Genomes stay task-agnostic;
//! probability semantics are carried by probability-capable heads.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(dead_code)] // consumed by the binary-classification campaign lane
pub(crate) enum TaskKind {
    Regression,
    Binary,
    Timeseries,
}

impl TaskKind {
    /// Receipt-facing name, stable across schema versions.
    #[allow(dead_code)]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            TaskKind::Regression => "regression",
            TaskKind::Binary => "binary",
            TaskKind::Timeseries => "timeseries",
        }
    }

    #[allow(dead_code)]
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "regression" => Some(TaskKind::Regression),
            "binary" => Some(TaskKind::Binary),
            "timeseries" => Some(TaskKind::Timeseries),
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
        assert_eq!(TaskKind::parse("multiclass"), None);
    }
}
