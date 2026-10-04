//! Opt-in ordered-axis admission for versioned stages.
//!
//! Legacy lag/roll recipe programs shift feature positions within one row.
//! This contract does not change their codec or the existing advanced-v3
//! ordered-row implementation. A future SampleTime stage must carry this
//! metadata and pass admission before it can use a fold's ordered input.

use super::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub(crate) mod codec;

pub const ORDERED_AXIS_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    FeatureWithinRow,
    SampleTime,
    #[serde(rename = "iid_row")]
    IIDRow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Causality {
    Causal,
    Offline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitScope {
    TrainFold,
    Stateless,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskMode {
    IID,
    Ordered,
    Forecasting,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageAxis {
    pub version: u16,
    pub axis: Axis,
    pub causality: Causality,
    pub fit_scope: FitScope,
}

impl StageAxis {
    pub const fn new(axis: Axis, causality: Causality, fit_scope: FitScope) -> Self {
        Self {
            version: ORDERED_AXIS_VERSION,
            axis,
            causality,
            fit_scope,
        }
    }

    pub const fn legacy_within_row() -> Self {
        Self::new(
            Axis::FeatureWithinRow,
            Causality::Causal,
            FitScope::Stateless,
        )
    }

    pub fn admit(self, task: TaskMode, input: Option<&OrderedInputIdentity>) -> Result<()> {
        if self.version != ORDERED_AXIS_VERSION {
            return Err(Error::Unsupported(format!(
                "ordered-axis metadata version {} is unsupported",
                self.version
            )));
        }
        if let Some(input) = input {
            input.validate()?;
        }
        if self.axis == Axis::SampleTime && task == TaskMode::IID {
            return Err(Error::Unsupported(
                "SampleTime stage cannot enter an IID task".into(),
            ));
        }
        if self.axis == Axis::SampleTime && input.is_none() {
            return Err(Error::Shape(
                "SampleTime stage requires an ordered input identity".into(),
            ));
        }
        if task == TaskMode::Forecasting && self.causality == Causality::Offline {
            return Err(Error::Unsupported(
                "offline stage cannot enter forecasting".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TiePolicy {
    StableInputOrder,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrderedInputIdentity {
    pub version: u16,
    pub group_key_column: String,
    pub timestamp_column: String,
    pub tie_policy: TiePolicy,
}

impl OrderedInputIdentity {
    pub fn new(group_key_column: impl Into<String>, timestamp_column: impl Into<String>) -> Self {
        Self {
            version: ORDERED_AXIS_VERSION,
            group_key_column: group_key_column.into(),
            timestamp_column: timestamp_column.into(),
            tie_policy: TiePolicy::StableInputOrder,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != ORDERED_AXIS_VERSION {
            return Err(Error::Unsupported(format!(
                "ordered input version {} is unsupported",
                self.version
            )));
        }
        if self.group_key_column.trim().is_empty() || self.timestamp_column.trim().is_empty() {
            return Err(Error::Shape(
                "ordered input requires group and timestamp columns".into(),
            ));
        }
        if self.group_key_column == self.timestamp_column {
            return Err(Error::Shape(
                "group and timestamp columns must differ".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GroupKey {
    NumericBits(u64),
    Categorical(String),
}

impl GroupKey {
    pub fn numeric(value: f64) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::Domain("group key must be finite".into()));
        }
        Ok(Self::NumericBits(if value == 0.0 {
            0.0f64.to_bits()
        } else {
            value.to_bits()
        }))
    }

    pub fn categorical(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(Error::Domain("group key must not be blank".into()));
        }
        Ok(Self::Categorical(value))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoldBoundary {
    pub cut_id: String,
    pub train_end: usize,
    pub total_rows: usize,
}

impl FoldBoundary {
    pub fn new(cut_id: impl Into<String>, train_end: usize, total_rows: usize) -> Result<Self> {
        let fold = Self {
            cut_id: cut_id.into(),
            train_end,
            total_rows,
        };
        fold.validate()?;
        Ok(fold)
    }

    pub fn validate(&self) -> Result<()> {
        if self.cut_id.trim().is_empty() || self.train_end == 0 || self.train_end > self.total_rows
        {
            return Err(Error::Bounds(
                "fold requires a named, nonempty train prefix within the input".into(),
            ));
        }
        Ok(())
    }

    pub fn fit_rows(&self) -> std::ops::Range<usize> {
        0..self.train_end
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldPhase {
    Train,
    HeldOut,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowAdmission {
    pub ordinal: usize,
    pub group: GroupKey,
    pub group_reset: bool,
    pub phase: FoldPhase,
    pub may_fit: bool,
    pub prior_in_group: usize,
    pub prior_train_in_group: usize,
}

#[derive(Clone, Default)]
struct GroupProgress {
    last_timestamp: Option<f64>,
    prior_rows: usize,
    prior_train_rows: usize,
}

/// One cursor per cut. A stage emits a row before updating its group-local
/// state with that row; held-out observations may then carry forward causally,
/// but only `may_fit` rows may update fitted parameters. A new group begins
/// with empty state and no state crosses from a different cut.
#[derive(Clone)]
pub struct CausalCursor {
    stage: StageAxis,
    input: OrderedInputIdentity,
    fold: FoldBoundary,
    next_ordinal: usize,
    groups: BTreeMap<GroupKey, GroupProgress>,
}

/// Admission adapter for the historical recipe planner. It deliberately
/// leaves legacy parsing, encoding, fitting and emission untouched.
pub fn admit_legacy_genome(
    genome: &crate::GenomeIr,
    task: &crate::advanced_v3::TaskSpec,
    schema: &crate::advanced_v3::DataSchema,
) -> Result<()> {
    let plan =
        crate::spec::build_plan(genome).map_err(|problems| Error::Syntax(problems.join("; ")))?;
    if let crate::spec::Plan::Pipeline { stages, .. } = &plan {
        for stage in stages {
            let axis = match &stage.op {
                crate::spec::PlannedOp::Lag { .. } => {
                    crate::kernel::capabilities::legacy_recipe_axis("lag")
                }
                crate::spec::PlannedOp::Roll { .. } => {
                    crate::kernel::capabilities::legacy_recipe_axis("roll")
                }
                _ => None,
            };
            if let Some(axis) = axis {
                task.admit_ordered_stage(schema, axis)
                    .map_err(|error| Error::Unsupported(error.to_string()))?;
            }
        }
    }
    Ok(())
}

impl CausalCursor {
    pub fn new(
        stage: StageAxis,
        task: TaskMode,
        input: &OrderedInputIdentity,
        fold: FoldBoundary,
    ) -> Result<Self> {
        stage.admit(task, Some(input))?;
        fold.validate()?;
        if stage.axis != Axis::SampleTime || stage.causality != Causality::Causal {
            return Err(Error::Unsupported(
                "causal fold cursor requires a causal SampleTime stage".into(),
            ));
        }
        Ok(Self {
            stage,
            input: input.clone(),
            fold,
            next_ordinal: 0,
            groups: BTreeMap::new(),
        })
    }

    pub fn push(&mut self, group: GroupKey, timestamp: f64) -> Result<RowAdmission> {
        if self.next_ordinal >= self.fold.total_rows {
            return Err(Error::Bounds("row exceeds declared fold input".into()));
        }
        if !timestamp.is_finite() {
            return Err(Error::Domain("timestamp must be finite".into()));
        }
        let progress = self.groups.entry(group.clone()).or_default();
        if progress
            .last_timestamp
            .is_some_and(|previous| timestamp < previous)
        {
            return Err(Error::Domain(
                "timestamps must be monotone within each group".into(),
            ));
        }
        let phase = if self.next_ordinal < self.fold.train_end {
            FoldPhase::Train
        } else {
            FoldPhase::HeldOut
        };
        let admission = RowAdmission {
            ordinal: self.next_ordinal,
            group,
            group_reset: progress.prior_rows == 0,
            phase,
            may_fit: phase == FoldPhase::Train && self.stage.fit_scope == FitScope::TrainFold,
            prior_in_group: progress.prior_rows,
            prior_train_in_group: progress.prior_train_rows,
        };
        progress.last_timestamp = Some(timestamp);
        progress.prior_rows += 1;
        if phase == FoldPhase::Train {
            progress.prior_train_rows += 1;
        }
        self.next_ordinal += 1;
        Ok(admission)
    }

    pub fn finish(self) -> Result<()> {
        if self.next_ordinal != self.fold.total_rows {
            return Err(Error::Bounds(
                "fold input ended before declared row count".into(),
            ));
        }
        Ok(())
    }
}
