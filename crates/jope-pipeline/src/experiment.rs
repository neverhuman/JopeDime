//! Private experiment identities. These contracts compose the existing dataset,
//! typed-program and fitted-model implementations; they define no new engine.
use anyhow::{ensure, Result};
use jope_datasets::catalog::Task;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const EXPERIMENT_SCHEMA: &str = "jope.experiment/v1";
pub const PANEL_SCHEMA: &str = "jope.benchmark-panel/v1";
pub const REQUEST_SCHEMA: &str = "jope.evaluation-request/v1";
pub const RESULT_SCHEMA: &str = "jope.evaluation-result/v1";

pub fn sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    Rmse,
    Auroc,
    LogLoss,
}
impl Metric {
    pub fn for_task(task: Task) -> Result<Self> {
        Ok(match task {
            Task::Regression => Self::Rmse,
            Task::Binary => Self::Auroc,
            Task::Multiclass => Self::LogLoss,
            other => anyhow::bail!("task {} is outside this panel", other.as_str()),
        })
    }
    pub fn validate_score(self, value: f64) -> Result<()> {
        ensure!(
            value.is_finite() && value >= 0.,
            "nonfinite or negative primary score"
        );
        ensure!(self != Self::Auroc || value <= 1., "AUROC exceeds one");
        Ok(())
    }
    /// All primary metrics become a smaller-is-better selection loss.
    pub fn loss(self, value: f64) -> Result<f64> {
        self.validate_score(value)?;
        Ok(if self == Self::Auroc {
            1. - value
        } else {
            value
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Resources {
    pub workers: u16,
    pub cpus: u16,
    pub memory_bytes: u64,
    pub swap_bytes: u64,
    pub gpu_tasks: u16,
    pub threads_per_worker: u16,
    pub total_seconds: u64,
    pub reporting_seconds: u64,
    pub per_fit_seconds: u64,
}
impl Resources {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.workers > 0 && self.workers <= self.cpus && self.cpus <= 20,
            "CPU/worker allowance exceeds 20 or is zero"
        );
        ensure!(
            self.memory_bytes > 0 && self.memory_bytes <= 64 * 1024 * 1024 * 1024,
            "memory allowance exceeds 64 GiB or is zero"
        );
        ensure!(
            self.swap_bytes == 0 && self.gpu_tasks <= 1 && self.threads_per_worker == 1,
            "requires no swap, at most one GPU task and one numerical thread per worker"
        );
        ensure!(
            self.total_seconds > 0 && self.total_seconds <= 7200,
            "panel allowance exceeds two hours or is zero"
        );
        ensure!(
            self.reporting_seconds > 0 && self.reporting_seconds < self.total_seconds,
            "reporting window must be reserved inside the deadline"
        );
        ensure!(
            self.per_fit_seconds > 0
                && self.per_fit_seconds <= self.total_seconds - self.reporting_seconds,
            "fit allowance exceeds execution window"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Experiment {
    pub schema: String,
    pub panel_sha256: String,
    pub methods_sha256: String,
    pub source_sha: String,
    pub checkpoints: BTreeMap<String, String>,
    pub seed: u64,
    pub resources: Resources,
    pub expected_artifacts: Vec<String>,
    pub protocol: String,
    pub numerical_policy: String,
}
impl Experiment {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == EXPERIMENT_SCHEMA,
            "unknown experiment schema"
        );
        ensure!(
            sha256(&self.panel_sha256) && sha256(&self.methods_sha256),
            "invalid panel/method identity"
        );
        ensure!(
            self.source_sha.len() == 40 && self.source_sha.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid full source SHA"
        );
        ensure!(
            self.checkpoints.values().all(|v| sha256(v)),
            "invalid checkpoint identity"
        );
        ensure!(
            !self.protocol.is_empty() && !self.numerical_policy.is_empty(),
            "missing protocol or numerical policy"
        );
        let artifacts: BTreeSet<_> = self.expected_artifacts.iter().collect();
        ensure!(
            !artifacts.is_empty()
                && artifacts.len() == self.expected_artifacts.len()
                && artifacts.iter().all(|a| !a.is_empty()),
            "missing or duplicate expected artifacts"
        );
        self.resources.validate()
    }
    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        Ok(crate::sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Written once at execution admission, after dependencies and inputs verify.
/// Resume validates this record; it never grants a fresh time allowance.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExecutionWindow {
    pub experiment_sha256: String,
    pub started_unix_ms: u64,
    pub deadline_unix_ms: u64,
}
impl ExecutionWindow {
    pub fn validate(&self, experiment: &Experiment) -> Result<()> {
        ensure!(
            self.experiment_sha256 == experiment.identity()?,
            "stale experiment window"
        );
        let expected = self
            .started_unix_ms
            .checked_add(experiment.resources.total_seconds * 1000);
        ensure!(
            expected == Some(self.deadline_unix_ms),
            "deadline changed or overflowed"
        );
        Ok(())
    }
    pub fn may_admit(&self, experiment: &Experiment, now_ms: u64) -> Result<bool> {
        self.validate(experiment)?;
        ensure!(
            now_ms >= self.started_unix_ms,
            "clock precedes original start"
        );
        Ok(now_ms < self.deadline_unix_ms - experiment.resources.reporting_seconds * 1000)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ColumnKind {
    Numeric,
    Categorical,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Column {
    pub name: String,
    pub kind: ColumnKind,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub url: String,
    pub sha256: String,
    pub bytes: u64,
    pub metadata_url: String,
    pub metadata_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Split {
    pub protocol: String,
    pub sha256: String,
    pub development_sha256: String,
    pub sealed_sha256: String,
    pub development_rows: usize,
    pub sealed_rows: usize,
    pub fold_rows: [usize; 5],
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct License {
    pub original: String,
    pub evidence_url: String,
    pub repository: String,
    pub note: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PanelDataset {
    pub id: String,
    pub task: Task,
    pub metric: Metric,
    pub source: Source,
    pub rows: usize,
    pub features: Vec<Column>,
    pub classes: Vec<String>,
    pub group_columns: Vec<String>,
    pub split: Split,
    pub license: License,
    pub checkpoint_exposure: String,
    pub description: String,
    pub group_limitations: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Panel {
    pub schema: String,
    pub panel_id: String,
    pub repository: String,
    pub revision: String,
    pub seed: u64,
    pub selection: String,
    pub folds: usize,
    pub datasets: Vec<PanelDataset>,
}
impl Panel {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == PANEL_SCHEMA && self.folds == 5 && self.datasets.len() == 30,
            "requires frozen 30-dataset, five-fold panel"
        );
        ensure!(
            !self.selection.is_empty() && !self.panel_id.is_empty(),
            "missing selection provenance"
        );
        let mut ids = BTreeSet::new();
        let mut inputs = BTreeSet::new();
        let mut counts = BTreeMap::new();
        ensure!(
            self.revision.len() == 40 && self.revision.bytes().all(|b| b.is_ascii_hexdigit()),
            "source revision must be a full commit"
        );
        for d in &self.datasets {
            ensure!(
                inputs.insert(&d.source.sha256),
                "duplicate dataset bytes under different names"
            );
            ensure!(
                !d.id.is_empty()
                    && d.id
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
                    && ids.insert(&d.id),
                "unsafe or duplicate dataset id"
            );
            *counts.entry(d.task).or_insert(0) += 1;
            ensure!(
                Metric::for_task(d.task)? == d.metric,
                "task/metric mismatch for {}",
                d.id
            );
            for hash in [
                &d.source.sha256,
                &d.source.metadata_sha256,
                &d.split.sha256,
                &d.split.development_sha256,
                &d.split.sealed_sha256,
            ] {
                ensure!(sha256(hash), "invalid dataset/split digest");
            }
            ensure!(
                d.source.bytes > 0 && d.rows > 0 && !d.features.is_empty(),
                "empty input"
            );
            ensure!(
                d.split.development_rows.checked_add(d.split.sealed_rows) == Some(d.rows)
                    && d.split.sealed_rows > 0,
                "partition row count mismatch"
            );
            ensure!(
                d.split.fold_rows.iter().all(|n| *n > 0)
                    && d.split
                        .fold_rows
                        .iter()
                        .try_fold(0usize, |a, n| a.checked_add(*n))
                        == Some(d.split.development_rows),
                "empty fold or development row count mismatch"
            );
            ensure!(
                d.split.development_sha256 != d.split.sealed_sha256,
                "sealed input aliases development"
            );
            let features: BTreeSet<_> = d.features.iter().map(|c| &c.name).collect();
            ensure!(
                features.len() == d.features.len()
                    && features.iter().all(|n| !n.is_empty())
                    && d.group_columns.iter().all(|g| features.contains(g)),
                "invalid feature/group columns"
            );
            let classes: BTreeSet<_> = d.classes.iter().collect();
            ensure!(
                classes.len() == d.classes.len() && classes.iter().all(|c| !c.is_empty()),
                "duplicate or empty class label"
            );
            ensure!(
                match d.task {
                    Task::Regression => classes.is_empty(),
                    Task::Binary => classes.len() == 2,
                    Task::Multiclass => classes.len() >= 3,
                    _ => false,
                },
                "class ordering incompatible with task"
            );
            ensure!(
                !d.license.original.is_empty() && !d.checkpoint_exposure.is_empty(),
                "license/exposure must be explicit, including unknown"
            );
        }
        ensure!(
            counts.len() == 3 && counts.values().all(|n| *n == 10),
            "requires ten datasets per task"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationStage {
    Screen,
    Confirm,
    TrainOof,
    Diagnostic,
}
/// Every field participates in identity, including fitting and scoring row order.
/// No sealed-final request is accepted by the development worker contract.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EvaluationRequest {
    pub schema: String,
    pub experiment_sha256: String,
    pub dataset_sha256: String,
    pub split_sha256: String,
    pub fitting_rows_sha256: String,
    pub scoring_rows_sha256: String,
    pub fitting_rows: usize,
    pub scoring_rows: usize,
    pub recipe_sha256: String,
    pub evaluator_sha256: String,
    pub checkpoint_sha256: Option<String>,
    pub task: Task,
    pub metric: Metric,
    pub classes: Vec<String>,
    pub fold: u8,
    pub stage: EvaluationStage,
    pub seed: u64,
    pub numerical_policy: String,
}
impl EvaluationRequest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == REQUEST_SCHEMA && self.fold < 5,
            "unknown request schema or fold"
        );
        ensure!(
            self.metric == Metric::for_task(self.task)?,
            "request task/metric mismatch"
        );
        for h in [
            &self.experiment_sha256,
            &self.dataset_sha256,
            &self.split_sha256,
            &self.fitting_rows_sha256,
            &self.scoring_rows_sha256,
            &self.recipe_sha256,
            &self.evaluator_sha256,
        ] {
            ensure!(sha256(h), "invalid evaluation identity");
        }
        ensure!(
            self.checkpoint_sha256.as_ref().is_none_or(|h| sha256(h)),
            "invalid checkpoint identity"
        );
        ensure!(
            self.fitting_rows_sha256 != self.scoring_rows_sha256,
            "fitting and scoring partitions alias"
        );
        ensure!(
            self.fitting_rows > 0 && self.scoring_rows > 0,
            "empty evaluation partition"
        );
        ensure!(
            !self.numerical_policy.is_empty(),
            "missing numerical policy"
        );
        let n = self.classes.iter().collect::<BTreeSet<_>>().len();
        ensure!(
            n == self.classes.len()
                && self.classes.iter().all(|c| !c.is_empty())
                && match self.task {
                    Task::Regression => n == 0,
                    Task::Binary => n == 2,
                    Task::Multiclass => n >= 3,
                    _ => false,
                },
            "invalid class order"
        );
        ensure!(
            self.stage != EvaluationStage::Screen || self.fold == 0,
            "screen must use fold zero"
        );
        Ok(())
    }
    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        Ok(crate::sha256_hex(&serde_json::to_vec(self)?))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationStatus {
    Completed,
    Failed,
    TimedOut,
    Unavailable,
    Cancelled,
    BudgetExhausted,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Consumption {
    pub wall_seconds: f64,
    pub cpu_seconds: f64,
    pub gpu_seconds: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EvaluationResult {
    pub schema: String,
    pub request_sha256: String,
    pub status: EvaluationStatus,
    pub reason: Option<String>,
    pub primary: Option<f64>,
    pub secondary: BTreeMap<String, f64>,
    pub scored_rows: usize,
    pub predictions_sha256: Option<String>,
    pub fitted_state_sha256: Option<String>,
    pub consumption: Consumption,
}
impl EvaluationResult {
    pub fn validate(&self, request: &EvaluationRequest) -> Result<()> {
        ensure!(
            self.schema == RESULT_SCHEMA && self.request_sha256 == request.identity()?,
            "stale or mismatched result identity"
        );
        for t in [
            Some(self.consumption.wall_seconds),
            Some(self.consumption.cpu_seconds),
            self.consumption.gpu_seconds,
        ]
        .into_iter()
        .flatten()
        {
            ensure!(t.is_finite() && t >= 0., "invalid resource measurement");
        }
        ensure!(
            self.secondary.values().all(|v| v.is_finite()),
            "nonfinite secondary score"
        );
        if self.status == EvaluationStatus::Completed {
            ensure!(
                self.scored_rows == request.scoring_rows,
                "zero or incomplete result"
            );
            request.metric.validate_score(
                self.primary
                    .ok_or_else(|| anyhow::anyhow!("completed result lacks primary score"))?,
            )?;
            ensure!(
                self.predictions_sha256.as_ref().is_some_and(|h| sha256(h))
                    && self.fitted_state_sha256.as_ref().is_some_and(|h| sha256(h)),
                "completed result lacks prediction/fitted evidence"
            );
        } else {
            ensure!(
                self.primary.is_none() && self.secondary.is_empty() && self.scored_rows == 0,
                "incomplete result cannot publish scores"
            );
            ensure!(
                self.reason.as_ref().is_some_and(|r| !r.is_empty()),
                "incomplete result requires reason"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn experiment() -> Experiment {
        Experiment {
            schema: EXPERIMENT_SCHEMA.into(),
            panel_sha256: "a".repeat(64),
            methods_sha256: "b".repeat(64),
            source_sha: "c".repeat(40),
            checkpoints: BTreeMap::new(),
            seed: 42,
            resources: Resources {
                workers: 20,
                cpus: 20,
                memory_bytes: 64 << 30,
                swap_bytes: 0,
                gpu_tasks: 1,
                threads_per_worker: 1,
                total_seconds: 7200,
                reporting_seconds: 120,
                per_fit_seconds: 10,
            },
            expected_artifacts: vec!["report.json".into()],
            protocol: "development-v1".into(),
            numerical_policy: "strict-v1".into(),
        }
    }
    fn request() -> EvaluationRequest {
        EvaluationRequest {
            schema: REQUEST_SCHEMA.into(),
            experiment_sha256: "a".repeat(64),
            dataset_sha256: "b".repeat(64),
            split_sha256: "c".repeat(64),
            fitting_rows_sha256: "d".repeat(64),
            scoring_rows_sha256: "e".repeat(64),
            fitting_rows: 80,
            scoring_rows: 20,
            recipe_sha256: "f".repeat(64),
            evaluator_sha256: "0".repeat(64),
            checkpoint_sha256: None,
            task: Task::Binary,
            metric: Metric::Auroc,
            classes: vec!["no".into(), "yes".into()],
            fold: 0,
            stage: EvaluationStage::Screen,
            seed: 42,
            numerical_policy: "strict-v1".into(),
        }
    }
    fn result(r: &EvaluationRequest) -> EvaluationResult {
        EvaluationResult {
            schema: RESULT_SCHEMA.into(),
            request_sha256: r.identity().unwrap(),
            status: EvaluationStatus::Completed,
            reason: None,
            primary: Some(0.6),
            secondary: BTreeMap::new(),
            scored_rows: 20,
            predictions_sha256: Some("1".repeat(64)),
            fitted_state_sha256: Some("2".repeat(64)),
            consumption: Consumption {
                wall_seconds: 2.,
                cpu_seconds: 1.,
                gpu_seconds: None,
                peak_rss_bytes: Some(1000),
            },
        }
    }
    #[test]
    fn budgets_fail_closed_and_resume_keeps_original_deadline() {
        let e = experiment();
        e.validate().unwrap();
        let window = ExecutionWindow {
            experiment_sha256: e.identity().unwrap(),
            started_unix_ms: 1000,
            deadline_unix_ms: 7_201_000,
        };
        assert!(window.may_admit(&e, 7_080_999).unwrap());
        assert!(!window.may_admit(&e, 7_081_000).unwrap());
        assert!(!window.may_admit(&e, 9_000_000).unwrap());
        assert!(window.may_admit(&e, 999).is_err());
        let mut extended = window.clone();
        extended.deadline_unix_ms += 1;
        assert!(extended.validate(&e).is_err());
        for field in [
            "workers",
            "cpus",
            "memory_bytes",
            "swap_bytes",
            "gpu_tasks",
            "threads_per_worker",
            "total_seconds",
            "reporting_seconds",
            "per_fit_seconds",
        ] {
            let mut value = serde_json::to_value(&e).unwrap();
            value["resources"][field] = match field {
                "memory_bytes" => (65u64 << 30).into(),
                "swap_bytes" => 1.into(),
                "gpu_tasks" | "threads_per_worker" => 2.into(),
                "total_seconds" => 7201.into(),
                "reporting_seconds" | "per_fit_seconds" => 7200.into(),
                _ => 21.into(),
            };
            assert!(
                serde_json::from_value::<Experiment>(value)
                    .unwrap()
                    .validate()
                    .is_err(),
                "{field}"
            );
        }
    }
    #[test]
    fn identity_binds_fold_classes_checkpoint_and_all_evaluation_inputs() {
        let r = request();
        let measured = result(&r);
        measured.validate(&r).unwrap();
        for field in [
            "experiment_sha256",
            "dataset_sha256",
            "split_sha256",
            "fitting_rows_sha256",
            "scoring_rows_sha256",
            "recipe_sha256",
            "evaluator_sha256",
            "checkpoint_sha256",
        ] {
            let mut changed = serde_json::to_value(&r).unwrap();
            changed[field] = "9".repeat(64).into();
            let changed: EvaluationRequest = serde_json::from_value(changed).unwrap();
            assert!(measured.validate(&changed).is_err(), "{field}");
        }
        let mut changed = r.clone();
        changed.classes.reverse();
        assert!(measured.validate(&changed).is_err());
        changed = r.clone();
        changed.stage = EvaluationStage::Confirm;
        assert!(measured.validate(&changed).is_err());
        changed.fold = 1;
        assert!(measured.validate(&changed).is_err());
        changed = r.clone();
        changed.seed += 1;
        assert!(measured.validate(&changed).is_err());
        changed = r.clone();
        changed.numerical_policy = "alternate".into();
        assert!(measured.validate(&changed).is_err());
        changed = r;
        changed.scoring_rows_sha256 = changed.fitting_rows_sha256.clone();
        assert!(changed.validate().is_err());
    }
    #[test]
    fn zero_nonfinite_incomplete_and_wrong_task_results_cannot_pass() {
        let r = request();
        let mut m = result(&r);
        m.scored_rows = 0;
        assert!(m.validate(&r).is_err());
        for value in [f64::NAN, f64::INFINITY, -1., 1.1] {
            m = result(&r);
            m.primary = Some(value);
            assert!(m.validate(&r).is_err());
        }
        m = result(&r);
        m.status = EvaluationStatus::TimedOut;
        assert!(m.validate(&r).is_err());
        m.primary = None;
        m.scored_rows = 0;
        m.reason = Some("fit deadline exceeded".into());
        m.validate(&r).unwrap();
        let mut bad = r;
        bad.metric = Metric::Rmse;
        assert!(bad.validate().is_err());
        bad.task = Task::Multiclass;
        bad.metric = Metric::LogLoss;
        assert!(bad.validate().is_err());
        assert_eq!(Metric::Auroc.loss(0.8).unwrap(), 1. - 0.8);
        assert_eq!(Metric::Rmse.loss(0.8).unwrap(), 0.8);
    }
    #[test]
    fn frozen_panel_has_ten_of_each_original_task_and_valid_identities() {
        let panel: Panel =
            serde_json::from_str(include_str!("../../../benchmarks/panel.json")).unwrap();
        panel.validate().unwrap();
        let mut changed = panel.clone();
        changed.datasets[0].metric = Metric::Auroc;
        assert!(changed.validate().is_err());
        changed = panel.clone();
        changed.datasets[0].split.fold_rows[0] = 0;
        assert!(changed.validate().is_err());
        changed = panel.clone();
        changed.datasets[0].split.sealed_sha256 =
            changed.datasets[0].split.development_sha256.clone();
        assert!(changed.validate().is_err());
        changed = panel;
        changed.datasets[0].id = changed.datasets[1].id.clone();
        assert!(changed.validate().is_err());
    }
}
