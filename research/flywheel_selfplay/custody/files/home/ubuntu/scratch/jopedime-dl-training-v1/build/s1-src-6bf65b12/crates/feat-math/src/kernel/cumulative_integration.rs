//! Opt-in causal cumulative integration, version 2 / wire 60.
use crate::kernel::ordered_axis::{
    Axis, CausalCursor, Causality, FitScope, FoldBoundary, GroupKey, OrderedInputIdentity,
    StageAxis, TaskMode,
};
use crate::kernel::{Error, Result};
use crate::math_expr::FaultSet;
use std::collections::BTreeMap;

mod profile;
pub use profile::{Edit, ExperimentalProfile, IntegrationSearch, PROFILE as EXPERIMENTAL_PROFILE};
mod emit;
mod model;
mod model_validation;
mod program;
mod recipe;
mod state_codec;
pub use crate::kernel::finite_difference::DifferenceLanguage as IntegrationLanguage;
pub use crate::kernel::finite_difference::OrderedRow;
pub use model::{IntegrationModel, IntegrationPrediction};
pub use program::IntegrationProgram;
pub use recipe::{IntegrationFittedCpp, IntegrationRecipe, FITTED_PROFILE, RECIPE_PROFILE};

const MAX_ABS: f64 = 1e12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Left rectangle: the previous sample applies over the next time interval.
    Rect,
    Trap,
}

#[derive(Clone, Debug)]
pub struct Config {
    mode: Mode,
    initial: Vec<f64>,
}

impl Config {
    pub fn mode(&self) -> Mode {
        self.mode
    }
    pub fn initial(&self) -> &[f64] {
        &self.initial
    }
    pub fn new(version: u8, wire: u16, mode: Mode, initial: Vec<f64>) -> Result<Self> {
        if version != 2 || wire != 60 {
            return Err(Error::Unsupported(
                "integration requires version2/wire60".into(),
            ));
        }
        if initial.is_empty() || initial.len() > 512 {
            return Err(Error::Shape(
                "integration initial width must be1..512".into(),
            ));
        }
        if initial.iter().any(|x| !x.is_finite() || x.abs() > MAX_ABS) {
            return Err(Error::Domain(
                "integration initial state must be finite and bounded".into(),
            ));
        }
        Ok(Self { mode, initial })
    }
}

#[derive(Clone)]
struct GroupState {
    timestamp: f64,
    previous: Vec<f64>,
    accumulator: Vec<f64>,
}

#[derive(Clone)]
pub struct IntegrationState {
    config: Config,
    cursor: CausalCursor,
    groups: BTreeMap<GroupKey, GroupState>,
}

#[derive(Debug)]
pub struct Output {
    pub values: Vec<f64>,
    pub faults: Vec<u32>,
}

impl IntegrationState {
    pub fn new(
        config: Config,
        task: TaskMode,
        input: &OrderedInputIdentity,
        fold: FoldBoundary,
    ) -> Result<Self> {
        let stage = StageAxis::new(Axis::SampleTime, Causality::Causal, FitScope::Stateless);
        Ok(Self {
            config,
            cursor: CausalCursor::new(stage, task, input, fold)?,
            groups: BTreeMap::new(),
        })
    }

    /// A refused time interval or row leaves the entire cursor and history intact.
    pub fn push(&mut self, group: GroupKey, timestamp: f64, row: &[f64]) -> Result<Output> {
        if row.len() != self.config.initial.len() {
            return Err(Error::Shape(
                "integration row width differs from initial state".into(),
            ));
        }
        let mut cursor = self.cursor.clone();
        let admission = cursor.push(group, timestamp)?;
        let group = admission.group;
        let mut values = self.config.initial.clone();
        let mut faults = vec![0; row.len()];
        for (flag, value) in faults.iter_mut().zip(row) {
            if !value.is_finite() {
                *flag |= FaultSet::NONFINITE_INPUT.bits();
            }
        }
        if let Some(previous) = self.groups.get(&group) {
            let dt = timestamp - previous.timestamp;
            if !dt.is_finite() || dt <= 0.0 {
                return Err(Error::Domain(
                    "integration requires finite positive time intervals".into(),
                ));
            }
            for column in 0..row.len() {
                let left = finite(previous.previous[column], &mut faults[column]);
                let right = finite(row[column], &mut faults[column]);
                let contribution = match self.config.mode {
                    Mode::Rect => left * dt,
                    Mode::Trap => {
                        let sum = left + right;
                        if sum.is_finite() && sum.abs() < f64::MIN_POSITIVE * 2.0 {
                            // Halve the interval before scaling tiny sums: the
                            // midpoint alone can round in the subnormal range while its area is normal.
                            sum * (dt * 0.5)
                        } else if sum.is_finite() {
                            (sum * 0.5) * dt
                        } else {
                            (left * 0.5 + right * 0.5) * dt
                        }
                    }
                };
                let accumulated = previous.accumulator[column] + contribution;
                if accumulated.abs() > MAX_ABS {
                    faults[column] |= FaultSet::MAGNITUDE_CLAMPED.bits();
                }
                values[column] = accumulated.clamp(-MAX_ABS, MAX_ABS);
            }
        }
        self.groups.insert(
            group,
            GroupState {
                timestamp,
                previous: row.to_vec(),
                accumulator: values.clone(),
            },
        );
        self.cursor = cursor;
        Ok(Output { values, faults })
    }

    pub fn cursor_bytes(&self) -> Result<Vec<u8>> {
        self.cursor.to_bytes()
    }
    pub fn finish(self) -> Result<()> {
        self.cursor.finish()
    }
}

fn finite(value: f64, faults: &mut u32) -> f64 {
    if value.is_finite() {
        value
    } else {
        *faults |= FaultSet::NONFINITE_INPUT.bits();
        0.0
    }
}
