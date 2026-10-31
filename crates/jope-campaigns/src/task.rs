//! Exhaustive task semantics for Epoch 6 and Search V3.
//!
//! `Task` is the only task vocabulary in this crate. `TaskKind` remains a
//! type alias so frozen V1/V2 callers keep compiling, but new code should use
//! `Task` and the registry returned by [`crate::task::Task::semantics`].

use anyhow::{ensure, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::str::FromStr;

pub const TASK_SEMANTICS_SCHEMA: &str = "JopeTaskSemantics.v1";
pub const TASK_SEMANTICS_JSON: &str = include_str!("../manifests/epoch6/JopeTaskSemantics.v1.json");
pub const TASK_SEMANTICS_SHA256: &str =
    "d158798880758dafcce9241226a1585015c4a95c29c2c2a830fdd888fa1b5422";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Binary,
    Regression,
    Timeseries,
    Multiclass,
}

/// Compatibility name for frozen V1/V2 call sites. This is not a second enum.
pub type TaskKind = Task;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskMetric {
    R2,
    Auc,
    MacroF1,
    NegativeMase,
}

impl TaskMetric {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::R2 => "r2",
            Self::Auc => "auc",
            Self::MacroF1 => "macro_f1",
            Self::NegativeMase => "negative_mase",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeErrorMetric {
    Rmse,
    LogLoss,
    OneMinusMacroF1,
    Mase,
}

impl NativeErrorMetric {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rmse => "rmse",
            Self::LogLoss => "log_loss",
            Self::OneMinusMacroF1 => "one_minus_macro_f1",
            Self::Mase => "mase",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BaselineMeaning {
    TrainingMean,
    TrainingPrevalence,
    TrainingMajorityClass,
    SeasonalNaive,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardTransform {
    ClampUnit,
    NegativeMaseParity,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetValidationClass {
    FiniteContinuous,
    BinaryZeroOne,
    ContiguousClassIndex,
    OrderedFiniteContinuous,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadCapability {
    Regression,
    BinaryProbability,
    MulticlassProbability,
    OrderedForecast,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GpSupportState {
    Qualified,
    ShadowOnly,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationReadiness {
    Active,
    ShadowReady,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingReadiness {
    Active,
    RequiresReadinessReceipt,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TaskSemantics {
    pub task: Task,
    pub wire_name: &'static str,
    pub corpus_directory: &'static str,
    pub primary_metric: TaskMetric,
    pub native_error_metric: NativeErrorMetric,
    pub baseline_meaning: BaselineMeaning,
    pub reward_transform: RewardTransform,
    pub target_validation: TargetValidationClass,
    pub required_head_capabilities: &'static [HeadCapability],
    pub gp_support: GpSupportState,
    pub evaluation_readiness: EvaluationReadiness,
    pub training_readiness: TrainingReadiness,
    pub minimum_distinct_v2_identities: u32,
    pub minimum_rolling_origins: u8,
}

const REGRESSION_HEADS: &[HeadCapability] = &[HeadCapability::Regression];
const BINARY_HEADS: &[HeadCapability] = &[HeadCapability::BinaryProbability];
const MULTICLASS_HEADS: &[HeadCapability] = &[HeadCapability::MulticlassProbability];
const TIMESERIES_HEADS: &[HeadCapability] = &[HeadCapability::OrderedForecast];

pub const TASK_SEMANTICS: [TaskSemantics; 4] = [
    TaskSemantics {
        task: Task::Regression,
        wire_name: "regression",
        corpus_directory: "regression",
        primary_metric: TaskMetric::R2,
        native_error_metric: NativeErrorMetric::Rmse,
        baseline_meaning: BaselineMeaning::TrainingMean,
        reward_transform: RewardTransform::ClampUnit,
        target_validation: TargetValidationClass::FiniteContinuous,
        required_head_capabilities: REGRESSION_HEADS,
        gp_support: GpSupportState::Qualified,
        evaluation_readiness: EvaluationReadiness::Active,
        training_readiness: TrainingReadiness::Active,
        minimum_distinct_v2_identities: 0,
        minimum_rolling_origins: 0,
    },
    TaskSemantics {
        task: Task::Binary,
        wire_name: "binary",
        corpus_directory: "binary",
        primary_metric: TaskMetric::Auc,
        native_error_metric: NativeErrorMetric::LogLoss,
        baseline_meaning: BaselineMeaning::TrainingPrevalence,
        reward_transform: RewardTransform::ClampUnit,
        target_validation: TargetValidationClass::BinaryZeroOne,
        required_head_capabilities: BINARY_HEADS,
        gp_support: GpSupportState::Qualified,
        evaluation_readiness: EvaluationReadiness::Active,
        training_readiness: TrainingReadiness::Active,
        minimum_distinct_v2_identities: 0,
        minimum_rolling_origins: 0,
    },
    TaskSemantics {
        task: Task::Multiclass,
        wire_name: "multiclass",
        corpus_directory: "multi_label",
        primary_metric: TaskMetric::MacroF1,
        native_error_metric: NativeErrorMetric::OneMinusMacroF1,
        baseline_meaning: BaselineMeaning::TrainingMajorityClass,
        reward_transform: RewardTransform::ClampUnit,
        target_validation: TargetValidationClass::ContiguousClassIndex,
        required_head_capabilities: MULTICLASS_HEADS,
        gp_support: GpSupportState::ShadowOnly,
        evaluation_readiness: EvaluationReadiness::ShadowReady,
        training_readiness: TrainingReadiness::RequiresReadinessReceipt,
        minimum_distinct_v2_identities: 500,
        minimum_rolling_origins: 0,
    },
    TaskSemantics {
        task: Task::Timeseries,
        wire_name: "timeseries",
        corpus_directory: "timeseries_reduced_v2",
        primary_metric: TaskMetric::NegativeMase,
        native_error_metric: NativeErrorMetric::Mase,
        baseline_meaning: BaselineMeaning::SeasonalNaive,
        reward_transform: RewardTransform::NegativeMaseParity,
        target_validation: TargetValidationClass::OrderedFiniteContinuous,
        required_head_capabilities: TIMESERIES_HEADS,
        gp_support: GpSupportState::ShadowOnly,
        evaluation_readiness: EvaluationReadiness::ShadowReady,
        training_readiness: TrainingReadiness::RequiresReadinessReceipt,
        minimum_distinct_v2_identities: 42,
        minimum_rolling_origins: 3,
    },
];

impl Task {
    /// Historical operational iteration order. It is deliberately frozen so
    /// introducing the canonical display order cannot alter V1/V2 replay.
    pub const ALL: [Task; 4] = [
        Task::Regression,
        Task::Binary,
        Task::Timeseries,
        Task::Multiclass,
    ];

    /// Canonical Epoch 6 serialization and display order.
    pub const CANONICAL_ORDER: [Task; 4] = [
        Task::Regression,
        Task::Binary,
        Task::Multiclass,
        Task::Timeseries,
    ];

    /// Frozen V1/V2 rotation. Replay V2 binds this separately from canonical
    /// display order.
    pub const ROTATION_CYCLE: [Task; 8] = [
        Task::Regression,
        Task::Binary,
        Task::Regression,
        Task::Binary,
        Task::Timeseries,
        Task::Regression,
        Task::Binary,
        Task::Multiclass,
    ];

    pub const fn semantics(self) -> &'static TaskSemantics {
        match self {
            Self::Regression => &TASK_SEMANTICS[0],
            Self::Binary => &TASK_SEMANTICS[1],
            Self::Multiclass => &TASK_SEMANTICS[2],
            Self::Timeseries => &TASK_SEMANTICS[3],
        }
    }

    pub const fn as_str(self) -> &'static str {
        self.semantics().wire_name
    }

    pub const fn corpus_dir_name(self) -> &'static str {
        self.semantics().corpus_directory
    }

    pub const fn primary_metric(self) -> &'static str {
        self.semantics().primary_metric.as_str()
    }

    pub const fn native_error_metric(self) -> &'static str {
        self.semantics().native_error_metric.as_str()
    }

    /// Frozen receipt-facing expression. The metric name is part of existing
    /// V1/V2 scientific identity, so this remains task-specific even when
    /// several tasks share the same numerical transform.
    pub const fn reward_expression(self) -> &'static str {
        match self {
            Self::Regression => "clamp(r2,-1,1)",
            Self::Binary => "clamp(auc,-1,1)",
            Self::Multiclass => "clamp(macro_f1,-1,1)",
            Self::Timeseries => "clamp(negative_mase+1,-1,1)",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "regression" => Some(Self::Regression),
            "binary" => Some(Self::Binary),
            "multiclass" => Some(Self::Multiclass),
            "timeseries" => Some(Self::Timeseries),
            _ => None,
        }
    }

    pub fn rotation_successor(position: usize, eligible: impl Fn(Task) -> bool) -> (Task, usize) {
        let length = Self::ROTATION_CYCLE.len();
        for offset in 0..length {
            let slot = (position + offset) % length;
            let task = Self::ROTATION_CYCLE[slot];
            if eligible(task) {
                return (task, (slot + 1) % length);
            }
        }
        let slot = position % length;
        (Self::ROTATION_CYCLE[slot], (slot + 1) % length)
    }
}

impl Display for Task {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for Task {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        Self::parse(value).ok_or_else(|| {
            anyhow::anyhow!("task must be regression, binary, multiclass, or timeseries")
        })
    }
}

pub trait MetricMarker: Copy + std::fmt::Debug {
    const METRIC: TaskMetric;
    fn validate(value: f64) -> bool;
}

macro_rules! metric_marker {
    ($name:ident, $metric:ident, $valid:expr) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub struct $name;
        impl MetricMarker for $name {
            const METRIC: TaskMetric = TaskMetric::$metric;
            fn validate(value: f64) -> bool {
                ($valid)(value)
            }
        }
    };
}

metric_marker!(R2Metric, R2, |value: f64| value.is_finite());
metric_marker!(AucMetric, Auc, |value: f64| value.is_finite()
    && (0.0..=1.0).contains(&value));
metric_marker!(MacroF1Metric, MacroF1, |value: f64| value.is_finite()
    && (0.0..=1.0).contains(&value));
metric_marker!(NegativeMaseMetric, NegativeMase, |value: f64| value
    .is_finite()
    && value <= 0.0);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativeScore<M: MetricMarker> {
    value: f64,
    marker: PhantomData<M>,
}

impl<M: MetricMarker> NativeScore<M> {
    pub fn new(value: f64) -> Result<Self> {
        ensure!(M::validate(value), "invalid {} score", M::METRIC.as_str());
        Ok(Self {
            value,
            marker: PhantomData,
        })
    }

    pub const fn get(self) -> f64 {
        self.value
    }
}

impl<M: MetricMarker> Serialize for NativeScore<M> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.value.serialize(serializer)
    }
}

impl<'de, M: MetricMarker> Deserialize<'de> for NativeScore<M> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = f64::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct NormalizedReward(f64);

impl NormalizedReward {
    pub fn new(value: f64) -> Result<Self> {
        ensure!(
            value.is_finite() && (-1.0..=1.0).contains(&value),
            "normalized reward is non-finite or outside [-1, 1]"
        );
        Ok(Self(value))
    }

    pub const fn get(self) -> f64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for NormalizedReward {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = f64::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

pub fn normalize_reward(task: Task, native_score: f64) -> Result<NormalizedReward> {
    ensure!(native_score.is_finite(), "task score is not finite");
    let value = match task {
        Task::Regression => native_score.clamp(-1.0, 1.0),
        Task::Binary => NativeScore::<AucMetric>::new(native_score)?.get(),
        Task::Multiclass => NativeScore::<MacroF1Metric>::new(native_score)?.get(),
        Task::Timeseries => {
            (NativeScore::<NegativeMaseMetric>::new(native_score)?.get() + 1.0).clamp(-1.0, 1.0)
        }
    };
    NormalizedReward::new(value)
}

pub fn operational_loss(task: Task, native_score: f64) -> Result<f64> {
    ensure!(native_score.is_finite(), "task score is not finite");
    Ok(match task {
        Task::Regression => 1.0 - native_score.clamp(-1.0, 1.0),
        Task::Binary => 1.0 - NativeScore::<AucMetric>::new(native_score)?.get(),
        Task::Multiclass => 1.0 - NativeScore::<MacroF1Metric>::new(native_score)?.get(),
        Task::Timeseries => -NativeScore::<NegativeMaseMetric>::new(native_score)?.get(),
    })
}

pub fn embedded_task_semantics_sha256() -> String {
    hex::encode(Sha256::digest(TASK_SEMANTICS_JSON.as_bytes()))
}

pub fn verify_embedded_task_semantics() -> Result<()> {
    ensure!(
        embedded_task_semantics_sha256() == TASK_SEMANTICS_SHA256,
        "embedded JopeTaskSemantics.v1 digest drift"
    );
    let value: serde_json::Value = serde_json::from_str(TASK_SEMANTICS_JSON)?;
    ensure!(
        value.get("schema").and_then(serde_json::Value::as_str) == Some(TASK_SEMANTICS_SCHEMA),
        "embedded task-semantics schema drift"
    );
    let tasks = value
        .get("tasks")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("embedded task-semantics tasks are absent"))?;
    ensure!(
        tasks.len() == TASK_SEMANTICS.len(),
        "embedded task count drift"
    );
    for (row, semantics) in tasks.iter().zip(TASK_SEMANTICS) {
        ensure!(
            row.get("task").and_then(serde_json::Value::as_str) == Some(semantics.wire_name)
                && row
                    .get("corpus_directory")
                    .and_then(serde_json::Value::as_str)
                    == Some(semantics.corpus_directory)
                && row
                    .get("primary_metric")
                    .and_then(serde_json::Value::as_str)
                    == Some(semantics.primary_metric.as_str())
                && row
                    .get("native_error_metric")
                    .and_then(serde_json::Value::as_str)
                    == Some(semantics.native_error_metric.as_str()),
            "embedded semantics drift for {}",
            semantics.wire_name
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn one_registry_owns_all_task_names_directories_and_metrics() {
        assert_eq!(
            Task::CANONICAL_ORDER.map(Task::as_str),
            ["regression", "binary", "multiclass", "timeseries",]
        );
        assert_eq!(Task::Timeseries.corpus_dir_name(), "timeseries_reduced_v2");
        assert_eq!(Task::Multiclass.corpus_dir_name(), "multi_label");
        assert_eq!(Task::Timeseries.primary_metric(), "negative_mase");
        assert_eq!(Task::Multiclass.native_error_metric(), "one_minus_macro_f1");
        assert_eq!(Task::Regression.reward_expression(), "clamp(r2,-1,1)");
        assert_eq!(Task::Binary.reward_expression(), "clamp(auc,-1,1)");
        assert_eq!(Task::parse("multi_label"), None);
    }

    #[test]
    fn legacy_derived_order_is_frozen_separately_from_canonical_order() {
        let ordered = [
            Task::Multiclass,
            Task::Timeseries,
            Task::Regression,
            Task::Binary,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
        assert_eq!(
            ordered,
            [
                Task::Binary,
                Task::Regression,
                Task::Timeseries,
                Task::Multiclass,
            ]
        );
        assert_eq!(
            Task::CANONICAL_ORDER,
            [
                Task::Regression,
                Task::Binary,
                Task::Multiclass,
                Task::Timeseries,
            ]
        );
    }

    #[test]
    fn typed_scores_reject_wrong_domains_and_timeseries_parity_is_zero() {
        assert!(NativeScore::<AucMetric>::new(1.01).is_err());
        assert!(NativeScore::<MacroF1Metric>::new(-0.01).is_err());
        assert!(NativeScore::<NegativeMaseMetric>::new(0.01).is_err());
        assert_eq!(normalize_reward(Task::Timeseries, -1.0).unwrap().get(), 0.0);
    }

    #[test]
    fn embedded_and_rust_task_semantics_match() {
        verify_embedded_task_semantics().unwrap();
    }
}
