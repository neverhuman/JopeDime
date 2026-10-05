//! Opt-in causal fixed-five realized path summaries, version 2 / wire 64.
use crate::kernel::ordered_axis::{
    Axis, CausalCursor, Causality, FitScope, FoldBoundary, GroupKey, OrderedInputIdentity,
    StageAxis, TaskMode,
};
use crate::kernel::{Error, Result};
use crate::math_expr::FaultSet;
use std::collections::{BTreeMap, VecDeque};

pub const BOUNDARY_HISTORY: u32 = 1 << 17;
const MAX_INCREMENT: f64 = 1e6;
// pi/2 is applied after the ordered sum of adjacent absolute-increment products.
const BV_FACTOR: f64 = f64::from_bits(0x3ff9_21fb_5444_2d18);

#[derive(Clone, Copy, Debug)]
pub struct Config {
    window: usize,
}

impl Config {
    pub fn window(self) -> usize {
        self.window
    }

    /// Window counts observations, so at most 128 increments are retained.
    pub fn new(version: u8, wire: u16, window: usize) -> Result<Self> {
        if version != 2 || wire != 64 {
            return Err(Error::Unsupported(
                "path summaries require version2/wire64".into(),
            ));
        }
        if !(2..=129).contains(&window) {
            return Err(Error::Bounds(
                "path summary window must be2..129 observations".into(),
            ));
        }
        Ok(Self { window })
    }
}

#[derive(Debug)]
pub struct Output {
    /// RV, bipower variation, max(RV-BV,0), positive SV, negative SV.
    pub values: [f64; 5],
    pub faults: [u32; 5],
}

#[derive(Clone)]
pub struct PathState {
    config: Config,
    cursor: CausalCursor,
    history: BTreeMap<GroupKey, VecDeque<f64>>,
}

impl PathState {
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
            history: BTreeMap::new(),
        })
    }

    /// One scalar path observation yields five features. Stable timestamp ties
    /// keep input order; this stage performs no division by a time interval.
    pub fn push(&mut self, group: GroupKey, timestamp: f64, row: &[f64]) -> Result<Output> {
        if row.len() != 1 {
            return Err(Error::Shape(
                "path summaries require one input value".into(),
            ));
        }
        let mut cursor = self.cursor.clone();
        let admission = cursor.push(group, timestamp)?;
        let mut history = self
            .history
            .get(&admission.group)
            .cloned()
            .unwrap_or_default();
        history.push_back(row[0]);
        if history.len() > self.config.window {
            history.pop_front();
        }
        let output = summarize(history.make_contiguous(), self.config);
        self.history.insert(admission.group, history);
        self.cursor = cursor;
        Ok(output)
    }

    /// The ordered cursor identity, separate from the full raw-history checkpoint.
    pub fn cursor_bytes(&self) -> Result<Vec<u8>> {
        self.cursor.to_bytes()
    }
    pub fn finish(self) -> Result<()> {
        self.cursor.finish()
    }
}

/// Partial windows use only the observed causal prefix and report a boundary.
/// Nonfinite observations become +0 before differencing. Finite subtraction
/// overflow clamps in its signed direction; all increments clamp before squaring.
pub fn summarize(observations: &[f64], config: Config) -> Output {
    let observations = &observations[observations.len().saturating_sub(config.window)..];
    let mut flags = if observations.len() < config.window {
        BOUNDARY_HISTORY
    } else {
        0
    };
    let mut values = [0.; 5];
    let mut previous_value = None;
    let mut previous_increment: Option<f64> = None;
    let mut products = 0.;
    for &raw in observations {
        let value = if raw.is_finite() {
            raw
        } else {
            flags |= FaultSet::NONFINITE_INPUT.bits();
            0.
        };
        if let Some(previous) = previous_value {
            let delta: f64 = value - previous;
            if delta.abs() > MAX_INCREMENT {
                flags |= FaultSet::MAGNITUDE_CLAMPED.bits();
            }
            let delta = delta.clamp(-MAX_INCREMENT, MAX_INCREMENT);
            let squared = delta * delta;
            values[0] += squared;
            if let Some(previous) = previous_increment {
                products += previous.abs() * delta.abs();
            }
            if delta > 0. {
                values[3] += squared;
            }
            if delta < 0. {
                values[4] += squared;
            }
            previous_increment = Some(delta);
        }
        previous_value = Some(value);
    }
    values[1] = products * BV_FACTOR;
    values[2] = (values[0] - values[1]).max(0.);
    Output {
        values,
        faults: [flags; 5],
    }
}

mod model;
mod model_validation;
mod program;
mod state_codec;
pub use crate::kernel::finite_difference::{DifferenceLanguage as PathStatsLanguage, OrderedRow};
pub use model::{PathStatsModel, PathStatsPrediction};
pub use program::PathStatsProgram;

mod emit;
mod recipe;
pub use recipe::{PathStatsFittedCpp, PathStatsRecipe};

mod profile;
pub use profile::{Edit, ExperimentalProfile, PathStatsSearch, PROFILE as EXPERIMENTAL_PROFILE};
