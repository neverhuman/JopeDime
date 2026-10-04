//! Opt-in causal integer backward differences, version 2 / wire 59.
//! Standalone kernels and bounded group history do not admit complete recipes,
//! models, fitted predictors or generation. The legacy registry stays frozen.
use super::ordered_axis::{
    Axis, CausalCursor, Causality, FitScope, FoldBoundary, GroupKey, OrderedInputIdentity,
    RowAdmission, StageAxis, TaskMode,
};
use super::{Error, Result};
use crate::math_expr::FaultSet;
use std::collections::{BTreeMap, VecDeque};

mod emit;
pub use emit::{emit_helpers, DifferenceLanguage};

pub const KERNEL_ID: &str = "feat-math.causal-integer-difference.v2";
/// Stage-local bit 17: insufficient history. This is NOT a scalar FaultSet
/// class; the scalar catalog and its old decoder remain unchanged.
pub const BOUNDARY_HISTORY: u32 = 1 << 17;
const NONFINITE_INPUT: u32 = FaultSet::NONFINITE_INPUT.bits();
const NONFINITE_SANITIZED: u32 = FaultSet::NONFINITE_SANITIZED.bits();
const MAGNITUDE_CLAMPED: u32 = FaultSet::MAGNITUDE_CLAMPED.bits();

#[derive(Clone, Copy, Debug)]
pub struct DifferenceConfig {
    order: u8,
    step: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DifferenceOutput {
    pub values: Vec<f64>,
    /// Scalar numerical fault bits plus the explicitly stage-local boundary bit.
    pub faults: Vec<u32>,
}

impl DifferenceConfig {
    pub fn new(version: u8, wire: u16, order: u8, step: f64) -> Result<Self> {
        if version != 2 || wire != 59 {
            return Err(Error::Unsupported(
                "causal difference requires version 2 / wire 59".into(),
            ));
        }
        if !(1..=4).contains(&order) || !step.is_finite() || step <= 0.0 {
            return Err(Error::Domain(
                "difference order must be 1..4 and step finite and positive".into(),
            ));
        }
        Ok(Self { order, step })
    }

    /// Oldest observation first; one through m+1 equally wide rows. Short
    /// histories yield +0 with BOUNDARY_HISTORY. Nonfinite cells normalize to
    /// +0 with NONFINITE_INPUT. Integer stencils [1,-1], [1,-2,1],
    /// [1,-3,3,-1], [1,-4,6,-4,1] are evaluated as a fixed triangular
    /// sequence of backward differences, dividing by h at each level rather
    /// than forming h^m (which can overflow/underflow for admitted h).
    /// An overflowing finite subtraction retries a/h-b/h. A nonfinite final
    /// result is +0 with NONFINITE_SANITIZED; finite output projects to +/-1e12.
    /// Intermediate projections are forbidden. Valid signed zeros are kept.
    /// Before output projection, when intermediates are normal and finite,
    /// |error| <= gamma(2m) * sum_k(C(m,k)*|x_k|) / h^m, where
    /// gamma(n)=n*u/(1-n*u), u=2^-53. Cancellation can make relative error
    /// large; the bound does not cover underflow or sanitized paths. This is not a correctly
    /// rounded exact-real stencil or a cross-host libm qualification claim.
    pub fn evaluate(self, rows: &[Vec<f64>]) -> Result<DifferenceOutput> {
        if rows.is_empty() || rows.len() > usize::from(self.order) + 1 {
            return Err(Error::Shape(
                "difference history must contain 1..m+1 rows".into(),
            ));
        }
        let width = rows[0].len();
        validate_width(width)?;
        if rows.iter().any(|row| row.len() != width) {
            return Err(Error::Shape(
                "difference history rows must have identical width".into(),
            ));
        }
        let (values, faults) = difference_values(self.order, self.step, rows);
        Ok(DifferenceOutput { values, faults })
    }
}

fn validate_width(width: usize) -> Result<()> {
    if width == 0 || width > crate::MAX_WIDTH {
        Err(Error::Shape("difference width must be 1..MAX_WIDTH".into()))
    } else {
        Ok(())
    }
}

// arithmetic-begin
fn difference_values(order: u8, step: f64, rows: &[Vec<f64>]) -> (Vec<f64>, Vec<u32>) {
    let width = rows[0].len();
    let mut values = Vec::with_capacity(width);
    let mut faults = Vec::with_capacity(width);
    for column in 0..width {
        let mut work = [0.0_f64; 5];
        let mut bits = 0;
        for (i, row) in rows.iter().enumerate() {
            let x = row[column];
            if x.is_finite() {
                work[i] = x;
            } else {
                bits |= NONFINITE_INPUT;
            }
        }
        if rows.len() <= usize::from(order) {
            values.push(0.0);
            faults.push(bits | BOUNDARY_HISTORY);
            continue;
        }
        for level in 0..usize::from(order) {
            for i in 0..usize::from(order) - level {
                let a = work[i + 1];
                let b = work[i];
                let difference = a - b;
                work[i] = if !difference.is_finite() && a.is_finite() && b.is_finite() {
                    a / step - b / step
                } else {
                    difference / step
                };
            }
        }
        let value = work[0];
        if !value.is_finite() {
            values.push(0.0);
            faults.push(bits | NONFINITE_SANITIZED);
        } else {
            if value.abs() > 1e12 {
                bits |= MAGNITUDE_CLAMPED;
            }
            values.push(value.clamp(-1e12, 1e12));
            faults.push(bits);
        }
    }
    (values, faults)
}
// arithmetic-end

pub struct DifferenceObservation {
    pub admission: RowAdmission,
    pub output: DifferenceOutput,
}

/// One declared cut, with at most m previous rows per admitted group. A new
/// instance resets all groups; train-to-held-out carry never resets the cut.
/// No labels, learned parameters or caller-supplied admission records enter it.
pub struct DifferenceState {
    config: DifferenceConfig,
    width: usize,
    cursor: CausalCursor,
    history: BTreeMap<GroupKey, VecDeque<Vec<f64>>>,
}

impl DifferenceState {
    pub fn new(
        config: DifferenceConfig,
        width: usize,
        task: TaskMode,
        input: &OrderedInputIdentity,
        fold: FoldBoundary,
    ) -> Result<Self> {
        validate_width(width)?;
        let stage = StageAxis::new(Axis::SampleTime, Causality::Causal, FitScope::Stateless);
        let cursor = CausalCursor::new(stage, task, input, fold)?;
        Ok(Self {
            config,
            width,
            cursor,
            history: BTreeMap::new(),
        })
    }

    pub fn push(
        &mut self,
        group: GroupKey,
        timestamp: f64,
        row: &[f64],
    ) -> Result<DifferenceObservation> {
        if row.len() != self.width {
            return Err(Error::Shape(
                "difference row differs from declared width".into(),
            ));
        }
        // Public enum variants must not bypass the group constructor guards.
        let group = match group {
            GroupKey::NumericBits(bits) => GroupKey::numeric(f64::from_bits(bits))?,
            GroupKey::Categorical(value) => GroupKey::categorical(value)?,
        };
        let admission = self.cursor.push(group.clone(), timestamp)?;
        let history = self.history.entry(group).or_default();
        history.push_back(row.to_vec());
        // Width/count invariants are established before spending the cursor.
        let output = self.config.evaluate(history.make_contiguous())?;
        if history.len() > usize::from(self.config.order) {
            history.pop_front();
        }
        Ok(DifferenceObservation { admission, output })
    }

    pub fn finish(self) -> Result<()> {
        self.cursor.finish()
    }
}
