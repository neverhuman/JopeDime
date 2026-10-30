//! Feature-position Savitzky–Golay, v2 wire 58. Unit feature spacing only.
//! Shrink clips the centered window to the row and reduces degree to n-1.
//! A derivative above that degree is +0 with BOUNDARY_DEGREE. This mode
//! never reads another observation and cannot be admitted as SampleTime.
use super::ordered_axis::Axis;
use super::{Error, Result};
use crate::math_expr::FaultSet;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const FEATURE_ID: &str = "feat-math.feature-savgol.v2";
pub const BOUNDARY_DEGREE: u32 = 1 << 18;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    window: u8,
    degree: u8,
    derivative: u8,
}

impl Config {
    pub fn new(
        version: u8,
        wire: u16,
        axis: Axis,
        window: u8,
        degree: u8,
        derivative: u8,
        boundary: &str,
    ) -> Result<Self> {
        if version != 2 || wire != 58 || axis != Axis::FeatureWithinRow {
            return Err(Error::Unsupported(
                "savgol requires v2 / wire 58 / FeatureWithinRow".into(),
            ));
        }
        if !(3..=129).contains(&window)
            || window.is_multiple_of(2)
            || degree > 6
            || degree >= window
            || derivative > degree
            || boundary != "shrink"
        {
            return Err(Error::Domain(
                "savgol requires odd w=3..129, p<=min(6,w-1), r<=p, boundary=shrink".into(),
            ));
        }
        Ok(Self {
            window,
            degree,
            derivative,
        })
    }

    pub fn window(self) -> u8 {
        self.window
    }
    pub fn degree(self) -> u8 {
        self.degree
    }
    pub fn derivative(self) -> u8 {
        self.derivative
    }

    pub fn fit(self, width: usize) -> Result<Fitted> {
        Self::new(
            2,
            58,
            Axis::FeatureWithinRow,
            self.window,
            self.degree,
            self.derivative,
            "shrink",
        )?;
        if width > crate::MAX_WIDTH {
            return Err(Error::Shape("savgol width exceeds MAX_WIDTH".into()));
        }
        let half = usize::from(self.window / 2);
        let mut cache: BTreeMap<(usize, usize), Vec<u64>> = BTreeMap::new();
        let mut stencils = Vec::with_capacity(width);
        for j in 0..width {
            let left = j.min(half);
            let right = (width - 1 - j).min(half);
            let bits = if let Some(bits) = cache.get(&(left, right)) {
                bits.clone()
            } else {
                let bits = coefficients(left, right, self.degree, self.derivative)?
                    .into_iter()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>();
                cache.insert((left, right), bits.clone());
                bits
            };
            stencils.push(Stencil {
                start: j - left,
                bits,
            });
        }
        Ok(Fitted {
            config: self,
            width,
            stencils,
        })
    }
}

/// Coefficients use integer bit patterns in all serialized state. Restoring
/// validates them against the canonical setup; arbitrary FIRs are not savgol.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fitted {
    config: Config,
    width: usize,
    stencils: Vec<Stencil>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Stencil {
    pub(crate) start: usize,
    pub(crate) bits: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub values: Vec<f64>,
    pub faults: Vec<u32>,
}

impl Fitted {
    pub fn config(&self) -> Config {
        self.config
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub(crate) fn stencils(&self) -> &[Stencil] {
        &self.stencils
    }

    pub fn validate(&self) -> Result<()> {
        if *self != self.config.fit(self.width)? {
            return Err(Error::Domain("noncanonical savgol coefficients".into()));
        }
        Ok(())
    }

    pub fn apply(&self, row: &[f64]) -> Result<Output> {
        self.validate()?;
        self.apply_validated(row)
    }

    pub(crate) fn apply_validated(&self, row: &[f64]) -> Result<Output> {
        if row.len() != self.width {
            return Err(Error::Shape(
                "savgol row width differs from fitted width".into(),
            ));
        }
        let mut values = Vec::with_capacity(self.width);
        let mut faults = Vec::with_capacity(self.width);
        for stencil in &self.stencils {
            let mut value = 0.0;
            let mut fault = if usize::from(self.config.derivative) >= stencil.bits.len() {
                BOUNDARY_DEGREE
            } else {
                0
            };
            for (offset, &bits) in stencil.bits.iter().enumerate() {
                let (x, input_fault) = normalize(row[stencil.start + offset]);
                fault |= input_fault;
                let term = f64::from_bits(bits) * x;
                value += term;
            }
            let (value, final_fault) = project(value);
            values.push(value);
            faults.push(fault | final_fault);
        }
        Ok(Output { values, faults })
    }
}

pub(crate) fn normalize(x: f64) -> (f64, u32) {
    if !x.is_finite() {
        (0.0, FaultSet::NONFINITE_INPUT.bits())
    } else {
        project(x)
    }
}

pub(crate) fn project(x: f64) -> (f64, u32) {
    if !x.is_finite() {
        (0.0, FaultSet::NONFINITE_SANITIZED.bits())
    } else if x.abs() > 1e12 {
        (x.clamp(-1e12, 1e12), FaultSet::MAGNITUDE_CLAMPED.bits())
    } else {
        (x, 0)
    }
}

// Reorthogonalized modified Gram–Schmidt on scaled local monomials. Solve
// R^T z = r! e_r / scale^r and return Q z. No normal equations or dependency.
// The setup is deterministic on this host; stored bits drive all artifacts.
// Fixed row/column indices keep the two reorthogonalization passes explicit.
#[allow(clippy::needless_range_loop)]
fn coefficients(left: usize, right: usize, degree: u8, derivative: u8) -> Result<Vec<f64>> {
    let n = left + right + 1;
    let p = usize::from(degree).min(n - 1);
    let d = usize::from(derivative);
    if d > p {
        return Ok(vec![0.0; n]);
    }
    let scale = left.max(right).max(1) as f64;
    let x = (0..n)
        .map(|i| (i as f64 - left as f64) / scale)
        .collect::<Vec<_>>();
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(p + 1);
    let mut r = [[0.0; 7]; 7];
    let mut powers = vec![1.0; n];
    for k in 0..=p {
        let mut v = powers.clone();
        for _ in 0..2 {
            for i in 0..k {
                let dot = q[i].iter().zip(&v).fold(0.0, |s, (a, b)| s + a * b);
                r[i][k] += dot;
                for j in 0..n {
                    v[j] -= dot * q[i][j];
                }
            }
        }
        let norm = v.iter().fold(0.0, |s, v| s + v * v).sqrt();
        if !norm.is_finite() || norm < 1e-14 {
            return Err(Error::Domain("singular savgol setup".into()));
        }
        r[k][k] = norm;
        q.push(v.iter().map(|v| v / norm).collect());
        for j in 0..n {
            powers[j] *= x[j];
        }
    }
    let mut target = 1.0;
    for i in 1..=d {
        target *= i as f64 / scale;
    }
    let mut z = [0.0; 7];
    for i in 0..=p {
        let mut v = if i == d { target } else { 0.0 };
        for k in 0..i {
            v -= r[k][i] * z[k];
        }
        z[i] = v / r[i][i];
    }
    Ok((0..n)
        .map(|j| (0..=p).fold(0.0, |s, k| s + q[k][j] * z[k]))
        .collect())
}
