//! Separate B07 numerical kernels. Legacy evaluators and emitted artifacts do
//! not select these helpers; program, model and recipe admission remain held.
//! Asinh reference: <https://netlib.org/fdlibm/s_asinh.c>.
use crate::math_expr::FaultSet;

mod core;
mod emit;
pub use emit::{emit_helpers, NumericalLanguage};

pub const POLICY_ID: &str = "v2";
pub const KERNEL_ID: &str = "feat-math.fdlibm-neumaier-lassq.v2";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    Sum,
    L1,
    Mean,
    L2,
}

impl Reduction {
    fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::L1 => "l1",
            Self::Mean => "mean",
            Self::L2 => "l2",
        }
    }
}

/// The fdlibm branch form preserves tiny finite inputs, including signed zero.
/// Nonfinite input returns positive zero with NONFINITE_INPUT.
pub fn asinh_v2(input: f64) -> (f64, FaultSet) {
    let (value, bits) = core::numerical_asinh_v2(input);
    (value, FaultSet::from_bits(bits).unwrap())
}

/// Ordered Neumaier sum/l1/mean or a LASSQ scaled Euclidean norm. Nonfinite
/// cells refuse the entire reduction with NONFINITE_INPUT. Empty sum/l1/l2
/// are identities; empty mean raises EMPTY_ROW_REDUCTION. Outputs are bounded
/// at ±1e12 with MAGNITUDE_CLAMPED. An unrepresentable compensated sum uses
/// the guarded positive-zero NONFINITE_SANITIZED fallback.
pub fn reduce_v2(reduction: Reduction, row: &[f64]) -> (f64, FaultSet) {
    let (value, bits) = core::numerical_reduce_v2(reduction.name(), row).unwrap();
    (value, FaultSet::from_bits(bits).unwrap())
}
