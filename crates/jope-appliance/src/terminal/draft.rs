//! Terminal-owned RunSpec draft state. The client validates complete drafts with a dry run.

use jope_api::{
    decimal::DecimalU64, ids::DatasetId, Budget, DatasetProfile, Effort, Guidance, LandmarkMode,
    Language, ObjectiveSpec, RunSpec, SplitKind, SplitSpec, Task,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct Draft {
    dataset: Option<DatasetId>,
    target: Option<String>,
    task: Task,
    suggested_tasks: BTreeMap<String, Task>,
    ignore_columns: Vec<String>,
    split: SplitSpec,
    /// `None` until `\objective` sets one: the run then gets [`default_objective`] for its task.
    objective: Option<ObjectiveSpec>,
    effort: Effort,
    budget: Budget,
    landmarks: LandmarkMode,
    guidance: Guidance,
    languages: Vec<Language>,
    require_all_languages: bool,
    device: String,
    seed: DecimalU64,
    pub columns: Vec<String>,
}

impl Default for Draft {
    fn default() -> Self {
        Self {
            dataset: None,
            target: None,
            task: Task::Auto,
            suggested_tasks: BTreeMap::new(),
            ignore_columns: Vec::new(),
            split: SplitSpec {
                kind: SplitKind::Random,
                holdout_fraction: 0.2,
                group_column: None,
                time_column: None,
            },
            objective: None,
            effort: Effort::Fast,
            budget: Budget {
                evaluations: None,
                wall_seconds: None,
            },
            landmarks: LandmarkMode::Full,
            guidance: Guidance::Auto,
            languages: vec![Language::Rust, Language::Python],
            require_all_languages: false,
            device: "auto".into(),
            seed: DecimalU64(0),
            columns: Vec::new(),
        }
    }
}

/// Apply the runtime's existing Auto policy to the registered public column profile.
pub(crate) fn inferred_tasks(profile: &DatasetProfile) -> BTreeMap<String, Task> {
    let mut tasks: BTreeMap<_, _> = profile
        .columns
        .iter()
        .filter_map(|column| {
            if column.distinct.0 == 0 {
                return None;
            }
            let task = if column.distinct.0 == 2 {
                Task::Binary
            } else if matches!(column.data_type.as_str(), "numeric" | "number") {
                Task::Regression
            } else {
                Task::Multiclass
            };
            Some((column.name.clone(), task))
        })
        .collect();
    for suggestion in &profile.target_suggestions {
        if suggestion.task != Task::Auto {
            tasks.insert(suggestion.column.clone(), suggestion.task);
        }
    }
    tasks
}

/// The objective a run gets when none is named: the runtime's primary metric for its task (#2132),
/// so a classification target is not scored by RMSE. An unresolved `auto` task keeps `rmse`.
pub(crate) fn default_objective(task: Task) -> &'static str {
    use jope_runtime::task::{metric, TaskKind};
    metric(match task {
        Task::Binary => TaskKind::Binary,
        Task::Multiclass => TaskKind::Multiclass,
        Task::Regression | Task::Auto => TaskKind::Regression,
    })
}

impl Draft {
    pub fn set_seed(&mut self, seed: DecimalU64) {
        self.seed = seed;
    }

    pub fn set_dataset(&mut self, profile: DatasetProfile) {
        self.suggested_tasks = inferred_tasks(&profile);
        self.dataset = Some(profile.dataset_id);
        self.columns = profile
            .columns
            .into_iter()
            .map(|column| column.name)
            .collect();
        // A target kept from a previously opened table that this one lacks is replaced by
        // this table's suggestion.
        if self
            .target
            .as_ref()
            .is_some_and(|target| !self.columns.contains(target))
        {
            self.target = None;
        }
        if self.target.is_none() {
            self.target = profile
                .target_suggestions
                .first()
                .map(|suggestion| suggestion.column.clone());
        }
    }

    pub fn languages(&self) -> Vec<Language> {
        self.languages.clone()
    }

    pub fn apply(&mut self, input: &str) -> Result<(), String> {
        let parsed = crate::terminal::commands::parse(input).map_err(|error| error.to_string())?;
        let args = parsed.arguments;
        match parsed.command.name {
            "\\target" => {
                let value = args.first().ok_or("usage: \\target COLUMN")?.clone();
                self.target = Some(value);
                let mut i = 1;
                while i < args.len() {
                    match args[i].as_str() {
                        "--task" => {
                            i += 1;
                            self.task = parse_task(args.get(i).ok_or("--task needs a value")?)?;
                        }
                        "--ignore" => {
                            i += 1;
                            self.ignore_columns = args
                                .get(i)
                                .ok_or("--ignore needs a value")?
                                .split(',')
                                .filter(|column| !column.is_empty())
                                .map(str::to_owned)
                                .collect();
                        }
                        "--group" | "--time" => {
                            return Err(
                                "this engine supports random splits only. Grouped and time-ordered splits are not available"
                                    .into(),
                            );
                        }
                        flag => return Err(format!("unknown target option {flag}")),
                    }
                    i += 1;
                }
            }
            "\\objective" => {
                let value = args.first().ok_or("usage: \\objective NAME|JSON|@FILE")?;
                self.objective = Some(parse_objective(value)?);
            }
            "\\effort" => {
                self.effort = match args.first().map(String::as_str) {
                    Some("fast") => Effort::Fast,
                    Some("medium") => Effort::Medium,
                    Some("high") => Effort::High,
                    Some("ultra") => Effort::Ultra,
                    _ => return Err("usage: \\effort fast|medium|high|ultra".into()),
                };
                let mut i = 1;
                while i < args.len() {
                    let option = args[i].as_str();
                    i += 1;
                    let value = args.get(i).ok_or("effort option needs a value")?;
                    match option {
                        "--evals" => {
                            let parsed = value
                                .parse::<u64>()
                                .map_err(|_| "evaluation limit must be a nonnegative integer")?;
                            self.budget.evaluations = Some(DecimalU64(parsed));
                        }
                        "--time" => self.budget.wall_seconds = Some(parse_duration(value)?),
                        flag => return Err(format!("unknown effort option {flag}")),
                    }
                    i += 1;
                }
            }
            "\\landmarks" => {
                self.landmarks = match args.first().map(String::as_str) {
                    Some("full") => LandmarkMode::Full,
                    Some("none") => LandmarkMode::None,
                    _ => return Err("usage: \\landmarks full|none".into()),
                };
            }
            "\\lang" | "\\language" => {
                let value = args.first().ok_or("usage: \\lang rust,python,javascript")?;
                let mut languages = Vec::new();
                for language in value.split(',') {
                    languages.push(match language {
                        "rust" => Language::Rust,
                        "python" => Language::Python,
                        "javascript" => Language::Javascript,
                        "typescript" => Language::Typescript,
                        _ => return Err(format!("unsupported language {language}")),
                    });
                }
                self.languages = languages;
                self.require_all_languages = !args.iter().any(|arg| arg == "--best-effort");
            }
            command => return Err(format!("\\{command} does not edit the run draft")),
        }
        Ok(())
    }

    pub fn run_spec(&self, dry_run: bool) -> Option<RunSpec> {
        let task = if self.task == Task::Auto {
            self.suggested_tasks
                .get(self.target.as_ref()?)
                .copied()
                .unwrap_or(Task::Auto)
        } else {
            self.task
        };
        Some(RunSpec {
            dataset: self.dataset.clone()?,
            target: self.target.clone()?,
            task,
            ignore_columns: self.ignore_columns.clone(),
            test_dataset: None,
            split: self.split.clone(),
            objective: self
                .objective
                .clone()
                .unwrap_or_else(|| ObjectiveSpec::Name(default_objective(task).into())),
            effort: self.effort,
            budget: self.budget.clone(),
            landmarks: self.landmarks,
            guidance: self.guidance,
            languages: self.languages.clone(),
            require_all_languages: self.require_all_languages,
            device: self.device.clone(),
            seed: self.seed,
            dry_run,
            plan_digest: None,
        })
    }
}

fn parse_task(value: &str) -> Result<Task, String> {
    match value {
        "auto" => Ok(Task::Auto),
        "regression" => Ok(Task::Regression),
        "binary" => Ok(Task::Binary),
        "multiclass" => Ok(Task::Multiclass),
        _ => Err(format!("unsupported task {value}")),
    }
}

fn parse_duration(value: &str) -> Result<DecimalU64, String> {
    let (number, scale) = if let Some(value) = value.strip_suffix('s') {
        (value, 1)
    } else if let Some(value) = value.strip_suffix('m') {
        (value, 60)
    } else {
        (value, 1)
    };
    number
        .parse::<u64>()
        .ok()
        .and_then(|seconds| seconds.checked_mul(scale))
        .map(DecimalU64)
        .ok_or_else(|| "time limit must be a nonnegative duration in seconds or minutes".into())
}

fn parse_objective(value: &str) -> Result<ObjectiveSpec, String> {
    let json = if let Some(path) = value.strip_prefix('@') {
        Some(std::fs::read_to_string(path).map_err(|error| error.to_string())?)
    } else if value.starts_with('{') {
        Some(value.to_owned())
    } else {
        None
    };
    let Some(json) = json else {
        return Ok(ObjectiveSpec::Name(value.to_owned()));
    };
    let object: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(&json).map_err(|error| error.to_string())?;
    Ok(ObjectiveSpec::Specification(object))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_commands_update_a_runspec_without_inventing_a_dataset() {
        let mut draft = Draft::default();
        draft
            .apply(r"\target value --task regression --ignore id,group")
            .unwrap();
        draft.apply(r"\effort medium --evals 64 --time 2m").unwrap();
        draft.apply(r"\lang rust,python --best-effort").unwrap();
        assert!(draft.run_spec(true).is_none());
        assert_eq!(draft.target.as_deref(), Some("value"));
        assert_eq!(draft.task, Task::Regression);
        assert_eq!(draft.ignore_columns, ["id", "group"]);
        assert_eq!(draft.split.kind, SplitKind::Random);
        assert!(draft.split.group_column.is_none());
        assert_eq!(draft.budget.evaluations, Some(DecimalU64(64)));
        assert_eq!(draft.budget.wall_seconds, Some(DecimalU64(120)));
        assert!(!draft.require_all_languages);
    }

    #[test]
    fn malformed_draft_updates_are_rejected() {
        let mut draft = Draft::default();
        assert!(draft.apply(r"\effort very-fast").is_err());
        assert!(draft.apply(r"\lang rust,brainfuck").is_err());
        assert!(draft.apply(r"\objective {invalid}").is_err());
    }

    #[test]
    fn dataset_suggestions_and_draft_options_project_into_dry_run_spec() {
        let profile: DatasetProfile = serde_json::from_value(serde_json::json!({
            "dataset_id": "ds_0123456789abcdef01234567",
            "name": "measurements.csv",
            "bytes": "128",
            "rows": "12",
            "columns": [
                {"name": "target", "data_type": "number", "missing": "0", "distinct": "8"},
                {"name": "group", "data_type": "string", "missing": "0", "distinct": "3"}
            ],
            "target_suggestions": [
                {"column": "target", "task": "auto", "reason": "non-binding target hint"}
            ],
            "warnings": []
        }))
        .unwrap();
        let mut draft = Draft::default();
        draft.set_dataset(profile);
        assert_eq!(draft.run_spec(true).unwrap().task, Task::Regression);
        draft
            .apply(r"\target target --task regression --ignore group")
            .unwrap();
        draft.apply(r"\effort medium --evals 64 --time 2m").unwrap();
        draft.apply(r"\landmarks none").unwrap();
        draft.apply(r"\lang rust,python --best-effort").unwrap();
        draft.set_seed(DecimalU64(7));

        let spec = draft.run_spec(true).unwrap();
        assert_eq!(spec.dataset.to_string(), "ds_0123456789abcdef01234567");
        assert_eq!(spec.target, "target");
        assert_eq!(spec.task, Task::Regression);
        assert_eq!(spec.ignore_columns, ["group"]);
        assert_eq!(spec.split.kind, SplitKind::Random);
        assert!(spec.split.group_column.is_none());
        assert_eq!(spec.effort, Effort::Medium);
        assert_eq!(spec.budget.evaluations, Some(DecimalU64(64)));
        assert_eq!(spec.budget.wall_seconds, Some(DecimalU64(120)));
        assert_eq!(spec.landmarks, LandmarkMode::None);
        assert_eq!(spec.languages, [Language::Rust, Language::Python]);
        assert!(!spec.require_all_languages);
        assert_eq!(spec.seed, DecimalU64(7));
        assert!(spec.dry_run);
        assert_eq!(draft.columns, ["target", "group"]);
    }

    #[test]
    fn objective_defaults_to_the_inferred_task_metric_until_set() {
        let profile: DatasetProfile = serde_json::from_value(serde_json::json!({
            "dataset_id": "ds_0123456789abcdef01234567",
            "name": "churn.csv",
            "bytes": "128",
            "rows": "12",
            "columns": [
                {"name": "churned", "data_type": "number", "missing": "0", "distinct": "2"},
                {"name": "spend", "data_type": "number", "missing": "0", "distinct": "9"}
            ],
            "target_suggestions": [
                {"column": "churned", "task": "auto", "reason": "non-binding target hint"}
            ],
            "warnings": []
        }))
        .unwrap();
        let mut draft = Draft::default();
        draft.set_dataset(profile);
        let spec = draft.run_spec(true).unwrap();
        assert_eq!(spec.task, Task::Binary);
        assert_eq!(spec.objective, ObjectiveSpec::Name("auroc".into()));

        draft.apply(r"\target spend").unwrap();
        let spec = draft.run_spec(true).unwrap();
        assert_eq!(spec.task, Task::Regression);
        assert_eq!(spec.objective, ObjectiveSpec::Name("rmse".into()));

        draft.apply(r"\target churned").unwrap();
        draft.apply(r"\objective mae").unwrap();
        assert_eq!(
            draft.run_spec(true).unwrap().objective,
            ObjectiveSpec::Name("mae".into())
        );
    }

    #[test]
    fn default_objective_follows_each_task() {
        assert_eq!(default_objective(Task::Regression), "rmse");
        assert_eq!(default_objective(Task::Auto), "rmse");
        assert_eq!(default_objective(Task::Binary), "auroc");
        assert_eq!(default_objective(Task::Multiclass), "log_loss");
    }

    #[test]
    fn task_duration_and_effort_options_reject_invalid_boundaries() {
        assert_eq!(parse_task("binary").unwrap(), Task::Binary);
        assert!(parse_task("unsupported").is_err());
        assert_eq!(parse_duration("9").unwrap(), DecimalU64(9));
        assert_eq!(parse_duration("3s").unwrap(), DecimalU64(3));
        assert_eq!(parse_duration("2m").unwrap(), DecimalU64(120));
        assert!(parse_duration("18446744073709551615m").is_err());
        assert!(parse_duration("-1s").is_err());

        let mut draft = Draft::default();
        assert!(draft.apply(r"\effort fast --evals -1").is_err());
        assert!(draft.apply(r"\effort fast --time nope").is_err());
    }

    #[test]
    fn grouped_and_time_splits_are_refused() {
        let mut draft = Draft::default();
        let group = draft.apply(r"\target value --group subject").unwrap_err();
        assert!(group.contains("random splits only"), "{group}");
        let time = draft.apply(r"\target value --time date").unwrap_err();
        assert!(time.contains("random splits only"), "{time}");
        assert_eq!(draft.split.kind, SplitKind::Random);
        assert!(draft.split.group_column.is_none());
        assert!(draft.split.time_column.is_none());
    }
}
