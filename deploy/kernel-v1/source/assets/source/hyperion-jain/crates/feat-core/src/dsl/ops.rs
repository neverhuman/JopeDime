//! Element-wise operator implementations for the DSL (numerically identical to the original).
use super::{Bin, Par, Un};
use crate::metrics::{standardize, EPS};

/// Abramowitz-Stegun 7.1.26 error function.
pub fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    if x < 0.0 {
        -y
    } else {
        y
    }
}

pub fn apply_un(op: Un, v: &mut [f32]) {
    match op {
        Un::Zscore => {
            let (s, _) = standardize(v);
            v.copy_from_slice(&s);
            return;
        }
        Un::Rank => {
            rank_inplace(v);
            return;
        }
        _ => {}
    }
    for x in v.iter_mut() {
        let a = *x;
        *x = match op {
            Un::Neg => -a,
            Un::Abs => a.abs(),
            Un::Square => a * a,
            Un::Cube => a * a * a,
            Un::SqrtAbs => a.abs().sqrt(),
            Un::Recip => {
                if a.abs() < EPS {
                    f32::NAN
                } else {
                    1.0 / a
                }
            }
            Un::Log1p => (1.0 + a.max(-0.999999)).ln(),
            Un::Log => a.abs().max(EPS).ln(),
            Un::Exp => a.clamp(-30.0, 30.0).exp(),
            Un::Sigmoid => 1.0 / (1.0 + (-a.clamp(-30.0, 30.0)).exp()),
            Un::Tanh => a.tanh(),
            Un::Arcsinh => (a + (a * a + 1.0).sqrt()).ln(),
            Un::Erf => erf(a),
            Un::GaussCdf => 0.5 * (1.0 + erf(a / std::f32::consts::SQRT_2)),
            Un::Sin => a.sin(),
            Un::Cos => a.cos(),
            Un::Softplus => {
                let c = a.clamp(-30.0, 30.0);
                (1.0 + c.exp()).ln()
            }
            Un::Relu => a.max(0.0),
            Un::Logit => {
                let p = a.clamp(1e-6, 1.0 - 1e-6);
                (p / (1.0 - p)).ln()
            }
            Un::Sign => {
                if a > 0.0 {
                    1.0
                } else if a < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Un::Zscore | Un::Rank => a,
        };
    }
}

pub fn rank_inplace(v: &mut [f32]) {
    let mut idx: Vec<usize> = (0..v.len()).filter(|&i| v[i].is_finite()).collect();
    let nf = idx.len();
    if nf < 2 {
        for x in v.iter_mut() {
            if x.is_finite() {
                *x = 0.5;
            }
        }
        return;
    }
    idx.sort_unstable_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let denom = (nf - 1) as f32;
    let mut ranks = vec![0f32; v.len()];
    // AVERAGE ranks for ties: tied values must share one rank, otherwise a heavily-tied (near-constant)
    // column would encode each row's POSITION in the unstable sort — and if the rows happen to be ordered
    // with any target structure, that position leaks the target (a real bug found on a target-sorted split:
    // rank(RaceDocumented), 99% constant, hit holdout AUC 0.99). Averaging ties makes rank order-invariant.
    let mut a = 0usize;
    while a < nf {
        let mut b = a;
        while b + 1 < nf && v[idx[b + 1]] == v[idx[a]] {
            b += 1;
        }
        let avg = ((a + b) as f32) / 2.0 / denom; // mean of positions a..=b, normalized to [0,1]
        for &i in &idx[a..=b] {
            ranks[i] = avg;
        }
        a = b + 1;
    }
    for i in 0..v.len() {
        if v[i].is_finite() {
            v[i] = ranks[i];
        }
    }
}

pub fn apply_par(op: Par, v: &mut [f32], p: f32, q: f32) {
    for x in v.iter_mut() {
        let a = *x;
        *x = match op {
            Par::Hill => {
                let h = q.clamp(0.5, 4.0);
                let k = p.abs().max(1e-3);
                let xh = a.abs().powf(h);
                let kh = k.powf(h);
                xh / (kh + xh + EPS)
            }
            Par::Mm => {
                let vmax = p;
                let km = q.abs().max(1e-3);
                vmax * a / (km + a + EPS)
            }
            Par::Rbf => {
                let c = p;
                let s = q.abs().max(0.1);
                (-((a - c) * (a - c)) / (2.0 * s * s + 1e-9)).exp()
            }
            Par::Clip => {
                let (lo, hi) = if p <= q { (p, q) } else { (q, p) };
                a.clamp(lo, hi)
            }
            Par::Powr => {
                let pw = p.clamp(-4.0, 4.0);
                a.abs().powf(pw)
            }
        };
    }
}

pub fn apply_bin(op: Bin, a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| match op {
            Bin::Add => x + y,
            Bin::Sub => x - y,
            Bin::Mul => x * y,
            Bin::SafeDiv => {
                if y.abs() < EPS {
                    f32::NAN
                } else {
                    x / y
                }
            }
            Bin::Min => {
                if x.is_nan() {
                    y
                } else if y.is_nan() {
                    x
                } else {
                    x.min(y)
                }
            }
            Bin::Max => {
                if x.is_nan() {
                    y
                } else if y.is_nan() {
                    x
                } else {
                    x.max(y)
                }
            }
            Bin::LogRatio => ((x.abs() + 1.0) / (y.abs() + 1.0)).ln(),
            Bin::Hinge => (x - y).max(0.0),
            Bin::Gt => {
                if x.is_finite() && y.is_finite() && x > y {
                    1.0
                } else {
                    0.0
                }
            }
            Bin::Lt => {
                if x.is_finite() && y.is_finite() && x < y {
                    1.0
                } else {
                    0.0
                }
            }
            Bin::Ge => {
                if x.is_finite() && y.is_finite() && x >= y {
                    1.0
                } else {
                    0.0
                }
            }
            Bin::Le => {
                if x.is_finite() && y.is_finite() && x <= y {
                    1.0
                } else {
                    0.0
                }
            }
            Bin::EqA => {
                if (x - y).abs() < 1e-6 {
                    1.0
                } else {
                    0.0
                }
            }
        })
        .collect()
}
