//! The runtime task contract: `feat_math`'s [`TaskKind`], which every fitted model already
//! carries, rather than the dataset catalog's enum (SPECIFICATION.md §4.2 item 4). The
//! benchmark adapters map `jope_datasets::catalog::Task` onto it.

pub use feat_math::advanced_v3::TaskKind;

/// Stable lower-case name.
pub fn name(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::Regression => "regression",
        TaskKind::Binary => "binary",
        TaskKind::Multiclass => "multiclass",
        TaskKind::Timeseries => "timeseries",
    }
}

/// The tasks the appliance fits.
pub const SUPPORTED: [TaskKind; 3] = [TaskKind::Regression, TaskKind::Binary, TaskKind::Multiclass];

/// The metric automatic selection minimizes, matching the frozen panel's primary metric per task.
pub fn metric(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::Regression => "rmse",
        TaskKind::Binary => "auroc",
        TaskKind::Multiclass => "log_loss",
        TaskKind::Timeseries => "unsupported",
    }
}

/// `--task`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskChoice {
    Auto,
    Fixed(TaskKind),
}

impl TaskChoice {
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "auto" => TaskChoice::Auto,
            "regression" => TaskChoice::Fixed(TaskKind::Regression),
            "binary" => TaskChoice::Fixed(TaskKind::Binary),
            "multiclass" => TaskChoice::Fixed(TaskKind::Multiclass),
            _ => return None,
        })
    }
}
