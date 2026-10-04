//! Guarded version-two exponential helpers. This numerical prerequisite does
//! not admit scalar programs, fitted models, full recipes or shipped search.
use super::wire_v2::{VersionError, VersionResult};
use crate::math_expr::FaultSet;
use crate::portable_math;

mod emit;
pub use emit::{emit_helpers, StableExponentialLanguage};

pub const KERNEL_ID: &str = "feat-math.stable-exponential.fdlibm.v2";
pub const NEG_LN_2: f64 = f64::from_bits(0xbfe6_2e42_fefa_39ef);

/// Codec-local IDs; the frozen v1 function enum and its behavior are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum StableExponential {
    Expm1 = 21,
    Log1mexp = 22,
}

impl StableExponential {
    pub fn from_versioned_id(version: u8, id: u16) -> VersionResult<Self> {
        if version != 2 {
            return Err(VersionError::UnsupportedVersion(format!(
                "stable exponential codec version {version}"
            )));
        }
        match id {
            21 => Ok(Self::Expm1),
            22 => Ok(Self::Log1mexp),
            _ => Err(VersionError::UnsupportedVersion(format!(
                "stable exponential function {id}"
            ))),
        }
    }

    pub const fn id(self) -> u16 {
        self as u16
    }

    /// Nonfinite input returns +0 with NONFINITE_INPUT. Expm1 projects its
    /// argument to [-30,30]; log1mexp requires x<0, including negative
    /// subnormals, and returns +0 with LOG_DOMAIN otherwise. Finite outputs
    /// project to +/-1e12. Valid underflow tails retain negative zero.
    /// The pure portable primitives, branch bits and arithmetic order are fixed.
    pub fn evaluate(self, x: f64) -> (f64, FaultSet) {
        if !x.is_finite() {
            return (0.0, FaultSet::NONFINITE_INPUT);
        }
        let (value, mut faults) = match self {
            Self::Expm1 => {
                let q = x.clamp(-30.0, 30.0);
                let faults = if q == x {
                    FaultSet::EMPTY
                } else {
                    FaultSet::EXP_ARG_CLAMPED
                };
                (portable_math::expm1(q), faults)
            }
            Self::Log1mexp if x >= 0.0 => return (0.0, FaultSet::LOG_DOMAIN),
            Self::Log1mexp if x < NEG_LN_2 => (
                portable_math::log1p(-portable_math::exp(x)),
                FaultSet::EMPTY,
            ),
            Self::Log1mexp => (portable_math::ln(-portable_math::expm1(x)), FaultSet::EMPTY),
        };
        if !value.is_finite() {
            return (0.0, faults.union(FaultSet::NONFINITE_SANITIZED));
        }
        if value.abs() > 1.0e12 {
            faults = faults.union(FaultSet::MAGNITUDE_CLAMPED);
        }
        (value.clamp(-1.0e12, 1.0e12), faults)
    }
}
