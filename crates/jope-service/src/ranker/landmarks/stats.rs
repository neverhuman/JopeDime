use super::*;

pub(super) fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.len() <= 1 {
        return sorted.first().copied().unwrap_or(0.0);
    }
    let at = q * (sorted.len() - 1) as f64;
    let lo = at.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] * (1.0 - (at - lo as f64)) + sorted[hi] * (at - lo as f64)
}

pub(super) fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

pub(super) fn stddev(values: &[f64]) -> f64 {
    let m = mean(values);
    (mean(&values.iter().map(|v| (v - m) * (v - m)).collect::<Vec<_>>())).sqrt()
}

pub(super) fn squash(value: f64) -> f64 {
    if value.is_finite() {
        value / (1.0 + value.abs())
    } else {
        0.0
    }
}

pub(super) fn gini(values: &[f64]) -> f64 {
    let mut a: Vec<f64> = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(f64::abs)
        .collect();
    if a.len() <= 1 || a.iter().sum::<f64>() <= 1e-12 {
        return 0.0;
    }
    a.sort_by(f64::total_cmp);
    let n = a.len() as f64;
    let weighted = a
        .iter()
        .enumerate()
        .map(|(i, v)| (i + 1) as f64 * v)
        .sum::<f64>();
    ((2.0 * weighted / (n * a.iter().sum::<f64>())) - (n + 1.0) / n).clamp(0.0, 1.0)
}

pub(super) fn corr(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || a.len() != b.len() {
        return 0.0;
    }
    let ma = mean(a);
    let mb = mean(b);
    let sa = stddev(a);
    let sb = stddev(b);
    if sa <= 1e-9 || sb <= 1e-9 {
        return 0.0;
    }
    let c = a
        .iter()
        .zip(b)
        .map(|(x, y)| ((x - ma) / sa) * ((y - mb) / sb))
        .sum::<f64>()
        / a.len() as f64;
    if c.is_finite() {
        c.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

pub(super) fn abs_corr_with_y(values: &[f64], y_z: &[f64]) -> f64 {
    if values.is_empty() || values.len() != y_z.len() {
        return 0.0;
    }
    let m = mean(values);
    let sd = (values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / values.len() as f64).sqrt();
    if sd <= 1e-9 || !sd.is_finite() {
        return 0.0;
    }
    let value = values
        .iter()
        .zip(y_z)
        .map(|(v, y)| ((v - m) / sd) * y)
        .sum::<f64>()
        .abs()
        / values.len() as f64;
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

// NumPy 1.26's default indirect quicksort is intentionally unstable. The
// source recipe uses np.argsort twice; matching its tie permutation is needed
// for Chatterjee xi on quantized/clipped columns.
pub(super) fn recipe_argsort(values: &[f64]) -> Vec<usize> {
    const SMALL_QUICKSORT: usize = 15;
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    if n < 2 {
        return order;
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    let mut depth = (usize::BITS - 1 - n.leading_zeros()) as i32 * 2;
    let mut stack: Vec<(usize, usize, i32)> = Vec::new();
    loop {
        if depth < 0 {
            // The pinned fixtures do not reach this introsort fallback; this
            // keeps worst-case runtime bounded for arbitrary caller inputs.
            order[lo..=hi].sort_unstable_by(|a, b| values[*a].total_cmp(&values[*b]));
        } else {
            while hi.saturating_sub(lo) > SMALL_QUICKSORT {
                let mid = lo + ((hi - lo) >> 1);
                if values[order[mid]] < values[order[lo]] {
                    order.swap(mid, lo);
                }
                if values[order[hi]] < values[order[mid]] {
                    order.swap(hi, mid);
                }
                if values[order[mid]] < values[order[lo]] {
                    order.swap(mid, lo);
                }
                let pivot = values[order[mid]];
                let mut pi = lo;
                let mut pj = hi - 1;
                order.swap(mid, pj);
                loop {
                    pi += 1;
                    while values[order[pi]] < pivot {
                        pi += 1;
                    }
                    pj -= 1;
                    while pivot < values[order[pj]] {
                        pj -= 1;
                    }
                    if pi >= pj {
                        break;
                    }
                    order.swap(pi, pj);
                }
                let pk = hi - 1;
                order.swap(pi, pk);
                if pi - lo < hi - pi {
                    stack.push((pi + 1, hi, depth - 1));
                    hi = pi - 1;
                } else {
                    stack.push((lo, pi - 1, depth - 1));
                    lo = pi + 1;
                }
                depth -= 1;
            }
            for pi in lo + 1..=hi {
                let vi = order[pi];
                let vp = values[vi];
                let mut pj = pi;
                while pj > lo && vp < values[order[pj - 1]] {
                    order[pj] = order[pj - 1];
                    pj -= 1;
                }
                order[pj] = vi;
            }
        }
        match stack.pop() {
            Some((next_lo, next_hi, next_depth)) => {
                lo = next_lo;
                hi = next_hi;
                depth = next_depth;
            }
            None => break,
        }
    }
    order
}

pub(super) fn scale_numeric_regression(
    x: &[f32],
    n: usize,
    p: usize,
    y: &[f32],
) -> (Vec<f64>, Vec<f64>) {
    let yy: Vec<f64> = y
        .iter()
        .map(|v| if v.is_finite() { f64::from(*v) } else { 0.0 })
        .collect();
    let ym = mean(&yy);
    let ys = stddev(&yy).max(1e-6);
    let sy: Vec<f64> = y
        .iter()
        .map(|v| {
            if v.is_finite() {
                8.0 * (((f64::from(*v) - ym) / ys) / 8.0).tanh()
            } else {
                0.0
            }
        })
        .collect();
    let qgrid = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];
    let norm = [
        -2.32634787,
        -1.64485363,
        -1.28155157,
        -0.67448975,
        0.0,
        0.67448975,
        1.28155157,
        1.64485363,
        2.32634787,
    ];
    let mut out = vec![0.0; n * p];
    for j in 0..p {
        let mut col: Vec<f64> = (0..n)
            .map(|i| {
                let v = f64::from(x[i * p + j]);
                if v.is_finite() {
                    v
                } else {
                    0.0
                }
            })
            .collect();
        let mut sorted = col.clone();
        sorted.sort_by(f64::total_cmp);
        let unique =
            sorted.windows(2).filter(|w| w[0] != w[1]).count() + usize::from(!sorted.is_empty());
        if unique <= 1 {
            continue;
        }
        let m = mean(&col);
        let s = stddev(&col);
        if unique == 2 {
            let den = s.max(1e-6);
            for i in 0..n {
                out[i * p + j] = 6.0 * (((col[i] - m) / den).clamp(-6.0, 6.0) / 6.0).tanh();
            }
            continue;
        }
        let q: [f64; 9] = std::array::from_fn(|k| quantile(&sorted, qgrid[k]));
        let mut skew = 0.0;
        let mut kurt = 0.0;
        if n >= 4 && s > 1e-12 {
            skew = mean(
                &col.iter()
                    .map(|v| ((v - m) / s).powi(3))
                    .collect::<Vec<_>>(),
            )
            .clamp(-50.0, 50.0);
            kurt = (mean(
                &col.iter()
                    .map(|v| ((v - m) / s).powi(4))
                    .collect::<Vec<_>>(),
            ) - 3.0)
                .clamp(-50.0, 200.0);
        }
        let med = q[4];
        let iqr = (q[5] - q[3]).max(0.0);
        let span90 = (q[7] - q[1]).max(0.0);
        let tail = if iqr > 1e-9 { span90 / iqr } else { 1.0 };
        let mut abs_dev: Vec<f64> = sorted.iter().map(|v| (v - med).abs()).collect();
        abs_dev.sort_by(f64::total_cmp);
        let robust = (iqr / 1.3489795)
            .max(quantile(&abs_dev, 0.5) * 1.4826)
            .max(s * 0.25)
            .max(1e-6);
        let use_log = skew.abs() > 1.0 || kurt > 4.0 || tail > 4.2;
        let blend = if use_log { 0.35 } else { 0.30 };
        let mut zlog = if use_log {
            col.iter()
                .map(|v| {
                    let d = v - med;
                    d.signum() * (d.abs() / robust).ln_1p()
                })
                .collect::<Vec<_>>()
        } else {
            col.iter()
                .map(|v| (v - m) / s.max(1e-6))
                .collect::<Vec<_>>()
        };
        let lm = mean(&zlog);
        let ls = stddev(&zlog).max(1e-6);
        for z in &mut zlog {
            *z = (*z - lm) / ls;
        }
        let strict = q.windows(2).all(|w| w[1] - w[0] > 1e-12);
        let iqr_safe = (q[5] - q[3]).max(1e-9);
        for i in 0..n {
            let mut z = zlog[i];
            if strict {
                let v = col[i];
                let zq = if v < q[0] {
                    norm[0] + (v - q[0]) / iqr_safe * 1.3489795
                } else if v > q[8] {
                    norm[8] + (v - q[8]) / iqr_safe * 1.3489795
                } else {
                    let seg = (0..8).find(|&k| v >= q[k] && v <= q[k + 1]).unwrap_or(0);
                    let t = (v - q[seg]) / (q[seg + 1] - q[seg]);
                    norm[seg] + t * (norm[seg + 1] - norm[seg])
                };
                z = (1.0 - blend) * z + blend * zq;
            }
            out[i * p + j] = 6.0 * (z / 6.0).tanh();
        }
        col.clear();
    }
    (out, sy)
}
