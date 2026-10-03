//! B05's versioned inverse-link numerical contract. Search admission remains
//! disabled until expression/program/model and full recipe closure are proven.
//! Old Blom and its emitter helpers retain their original clamp and bytes.
use super::wire_v2::{VersionError, VersionResult};
use crate::math_expr::FaultSet;

mod emit;
pub use emit::{emit_helpers, InverseLinkLanguage};

pub const KERNEL_ID: &str = "feat-math.inverse-links.acklam.v2";
pub const MIN_P: f64 = f64::from_bits(0x3ca0_0000_0000_0000); // 2^-53
pub const MAX_P: f64 = f64::from_bits(0x3fef_ffff_ffff_ffff); // 1 - 2^-53

/// Codec-local function IDs. These are separate from the frozen v1 Func enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum InverseLink {
    Probit = 19,
    Logit = 20,
}

impl InverseLink {
    pub fn from_versioned_id(version: u8, id: u16) -> VersionResult<Self> {
        if version != 2 {
            return Err(VersionError::UnsupportedVersion(format!(
                "inverse-link codec version {version}"
            )));
        }
        match id {
            19 => Ok(Self::Probit),
            20 => Ok(Self::Logit),
            _ => Err(VersionError::UnsupportedVersion(format!(
                "inverse-link function {id}"
            ))),
        }
    }

    pub const fn id(self) -> u16 {
        self as u16
    }

    /// Nonfinite inputs produce +0 with a Domain fault. Finite probabilities
    /// outside the closed interval project to its endpoint with Saturation.
    /// The approved probit bound is relative <=1.2e-9, without erfc refinement.
    pub fn evaluate(self, p: f64) -> (f64, FaultSet) {
        if !p.is_finite() {
            return (0.0, FaultSet::NONFINITE_INPUT);
        }
        let q = p.clamp(MIN_P, MAX_P);
        let faults = if q == p {
            FaultSet::EMPTY
        } else {
            FaultSet::PROBABILITY_CLAMPED
        };
        let value = match self {
            Self::Probit => acklam(q),
            Self::Logit if (0.25..=0.75).contains(&q) => {
                let d = (q - 0.5) * 2.0;
                d.ln_1p() - (-d).ln_1p()
            }
            Self::Logit => q.ln() - (-q).ln_1p(),
        };
        (value, faults)
    }
}

// Frozen coefficient bits and Horner order from the historical Acklam kernel.
// The new guard is above; the historical 1e-12 guard is deliberately untouched.
const A: &[f64] = &[
    -3.969683028665376e1,
    2.209460984245205e2,
    -2.759285104469687e2,
    1.383_577_518_672_69e2,
    -3.066479806614716e1,
    2.506628277459239,
];
const B: &[f64] = &[
    -5.447609879822406e1,
    1.615858368580409e2,
    -1.556989798598866e2,
    6.680131188771972e1,
    -1.328068155288572e1,
    1.0,
];
const C: &[f64] = &[
    -7.784894002430293e-3,
    -3.223964580411365e-1,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
];
const D: &[f64] = &[
    7.784695709041462e-3,
    3.224671290700398e-1,
    2.445134137142996,
    3.754408661907416,
    1.0,
];

fn polynomial(coefficients: &[f64], x: f64) -> f64 {
    coefficients[1..]
        .iter()
        .fold(coefficients[0], |value, coefficient| {
            value * x + coefficient
        })
}

fn acklam(p: f64) -> f64 {
    if p < 0.02425 {
        let q = (-2.0 * p.ln()).sqrt();
        polynomial(C, q) / polynomial(D, q)
    } else if p <= 1.0 - 0.02425 {
        let q = p - 0.5;
        let r = q * q;
        polynomial(A, r) * q / polynomial(B, r)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -polynomial(C, q) / polynomial(D, q)
    }
}
