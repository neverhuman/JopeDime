//! Opt-in causal irregular-grid backward difference, version 2 / wire 75.
use crate::kernel::ordered_axis::{
    Axis, CausalCursor, Causality, FitScope, FoldBoundary, GroupKey, OrderedInputIdentity,
    StageAxis, TaskMode,
};
use crate::kernel::{Error, Result};
use crate::math_expr::FaultSet;
use std::collections::BTreeMap;

pub const BOUNDARY_HISTORY: u32 = 1 << 17;
/// Stage-local fault for refused timestamps or minimum-gap policy violations.
pub const TIME_GAP_REJECTED: u32 = 1 << 18;

#[derive(Clone, Debug)]
pub struct Config {
    width: usize,
    minimum_gap: f64,
}

impl Config {
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn minimum_gap(&self) -> f64 {
        self.minimum_gap
    }

    pub fn new(version: u8, wire: u16, width: usize, minimum_gap: f64) -> Result<Self> {
        if version != 2 || wire != 75 {
            return Err(Error::Unsupported(
                "timescale difference requires version2/wire75".into(),
            ));
        }
        if width == 0 || width > 512 {
            return Err(Error::Shape("timescale width must be1..512".into()));
        }
        if !minimum_gap.is_finite() || minimum_gap <= 0.0 {
            return Err(Error::Domain(
                "timescale minimum gap must be explicit, finite and positive".into(),
            ));
        }
        Ok(Self { width, minimum_gap })
    }
}

#[derive(Debug)]
pub struct Rejection {
    pub kind: Error,
    pub faults: u32,
}

#[derive(Debug)]
pub struct Output {
    pub values: Vec<f64>,
    pub faults: Vec<u32>,
}

#[derive(Clone)]
pub struct TimeGradientState {
    config: Config,
    cursor: CausalCursor,
    groups: BTreeMap<GroupKey, (f64, Vec<f64>)>,
}

impl TimeGradientState {
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

    pub fn push(
        &mut self,
        group: GroupKey,
        timestamp: f64,
        row: &[f64],
    ) -> std::result::Result<Output, Rejection> {
        if row.len() != self.config.width {
            return Err(Rejection {
                kind: Error::Shape("timescale row width".into()),
                faults: 0,
            });
        }
        if !timestamp.is_finite() {
            return Err(Rejection {
                kind: Error::Domain("timescale timestamp must be finite".into()),
                faults: TIME_GAP_REJECTED,
            });
        }
        let group = match group {
            GroupKey::NumericBits(bits) if f64::from_bits(bits) == 0.0 => GroupKey::NumericBits(0),
            other => other,
        };
        if let Some((time, _)) = self.groups.get(&group) {
            let gap = timestamp - time;
            if !gap.is_finite() || gap < self.config.minimum_gap {
                return Err(Rejection {
                    kind: Error::Domain(
                        "timescale gap violates declared minimum; no clipping".into(),
                    ),
                    faults: TIME_GAP_REJECTED,
                });
            }
        }
        let mut cursor = self.cursor.clone();
        let admitted = cursor
            .push(group, timestamp)
            .map_err(|kind| Rejection { kind, faults: 0 })?;
        let group = admitted.group;
        let mut values = vec![0.0; row.len()];
        let mut faults = vec![0; row.len()];
        if let Some((time, previous)) = self.groups.get(&group) {
            let gap = timestamp - time;
            if !gap.is_finite() || gap < self.config.minimum_gap {
                return Err(Rejection {
                    kind: Error::Domain(
                        "timescale gap violates declared minimum; no clipping".into(),
                    ),
                    faults: TIME_GAP_REJECTED,
                });
            }
            for column in 0..row.len() {
                let left = finite(previous[column], &mut faults[column]);
                let right = finite(row[column], &mut faults[column]);
                let difference = right - left;
                let value = if difference.is_finite() {
                    difference / gap
                } else {
                    right / gap - left / gap
                };
                if !value.is_finite() {
                    faults[column] |= FaultSet::NONFINITE_SANITIZED.bits();
                } else {
                    if value.abs() > 1e12 {
                        faults[column] |= FaultSet::MAGNITUDE_CLAMPED.bits();
                    }
                    values[column] = value.clamp(-1e12, 1e12);
                }
            }
        } else {
            for column in 0..row.len() {
                faults[column] = BOUNDARY_HISTORY;
                finite(row[column], &mut faults[column]);
            }
        }
        self.groups.insert(group, (timestamp, row.to_vec()));
        self.cursor = cursor;
        Ok(Output { values, faults })
    }

    pub fn finish(self) -> Result<()> {
        self.cursor.finish()
    }

    pub fn cursor_bytes(&self) -> Result<Vec<u8>> {
        self.cursor.to_bytes()
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

impl From<Error> for Rejection {
    fn from(kind: Error) -> Self {
        Self { kind, faults: 0 }
    }
}
impl From<Rejection> for crate::kernel::wire_v2::VersionError {
    fn from(error: Rejection) -> Self {
        Self::StageRejected {
            kind: error.kind,
            faults: error.faults,
        }
    }
}
impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}; faults={}", self.kind, self.faults)
    }
}
impl std::error::Error for Rejection {}
pub type TimeResult<T> = std::result::Result<T, Rejection>;
mod model;
mod model_validation;
mod program;
mod state_codec;
pub use crate::kernel::finite_difference::{
    DifferenceLanguage as TimeGradientLanguage, OrderedRow,
};
pub use model::{TimeGradientModel, TimeGradientPrediction};
pub use program::TimeGradientProgram;

mod emit;
mod recipe;
pub use recipe::{TimeGradientFittedCpp, TimeGradientRecipe};

mod profile;
pub use profile::{Edit, ExperimentalProfile, TimeGradientSearch, PROFILE as EXPERIMENTAL_PROFILE};
