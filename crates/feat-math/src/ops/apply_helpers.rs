use super::*;

/// Append semantics + width cap + sanitize (mirrors the artifact epilogue of
/// every transform function).
fn finish(op: &FittedOp, input: &Rows, mut out: Rows) -> Rows {
    let append = op_append(op);
    let width = out.first().map_or(0, Vec::len);
    if out.len() * width.max(1) >= 65_536 && crate::is_parallel_enabled() {
        out.par_iter_mut().enumerate().for_each(|(i, row)| {
            if append {
                let keep = (input[i].len() + row.len()).min(crate::MAX_WIDTH);
                let from_input = input[i].len().min(keep);
                let mut merged = Vec::with_capacity(keep);
                merged.extend_from_slice(&input[i][..from_input]);
                merged.extend_from_slice(&row[..keep - from_input]);
                *row = merged;
            } else if row.len() > crate::MAX_WIDTH {
                row.truncate(crate::MAX_WIDTH);
                row.shrink_to_fit();
            }
            for v in row.iter_mut() {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }
            if row.is_empty() {
                row.push(0.0);
            }
        });
    } else {
        for (i, row) in out.iter_mut().enumerate() {
            if append {
                // `input ++ row`, cut to the width cap, built once at its final size: the cap would
                // otherwise truncate a longer allocation and keep all of its capacity.
                let keep = (input[i].len() + row.len()).min(crate::MAX_WIDTH);
                let from_input = input[i].len().min(keep);
                let mut merged = Vec::with_capacity(keep);
                merged.extend_from_slice(&input[i][..from_input]);
                merged.extend_from_slice(&row[..keep - from_input]);
                *row = merged;
            } else if row.len() > crate::MAX_WIDTH {
                // Expanding ops (one-hot, hinge, Chebyshev, products, splines) can generate several
                // times the cap; release the dropped tail so cached blocks hold only what they keep.
                row.truncate(crate::MAX_WIDTH);
                row.shrink_to_fit();
            }
            for v in row.iter_mut() {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }
            if row.is_empty() {
                row.push(0.0);
            }
        }
    }
    out
}

/// Ten per-row summary statistics: mean, std, min, max, median, q25, q75,
/// skewness, excess kurtosis, energy.
fn row_stats(r: &[f64]) -> Vec<f64> {
    let n = r.len() as f64;
    let mean = r.iter().sum::<f64>() / n;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    let mut energy = 0.0;
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for v in r {
        let d = v - mean;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
        energy += v * v;
        if *v < mn {
            mn = *v;
        }
        if *v > mx {
            mx = *v;
        }
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;
    energy /= n;
    let std = m2.sqrt();
    let skew = if m2 > 1e-24 { m3 / (m2 * std) } else { 0.0 };
    let kurt = if m2 > 1e-24 {
        m4 / (m2 * m2) - 3.0
    } else {
        0.0
    };
    let mut s = r.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    vec![
        mean,
        std,
        mn,
        mx,
        quantile_sorted(&s, 0.5),
        quantile_sorted(&s, 0.25),
        quantile_sorted(&s, 0.75),
        skew,
        kurt,
        energy,
    ]
}

/// Magnitudes of the plain O(d²) DFT of one row at frequencies f = 1..=k:
/// mag_f = sqrt((Σ_j x_j cos(2π f j / d))² + (Σ_j x_j sin(2π f j / d))²) / d.
/// Angle is computed left-to-right as ((2π · f) · j) / d for exact Python parity.
fn dft_mags(r: &[f64], k: usize) -> Vec<f64> {
    let d = r.len() as f64;
    let mut out = Vec::with_capacity(k);
    for f in 1..=k {
        let mut re = 0.0;
        let mut im = 0.0;
        for (j, v) in r.iter().enumerate() {
            let ang = PI2 * f as f64 * j as f64 / d;
            re += v * ang.cos();
            im += v * ang.sin();
        }
        out.push((re * re + im * im).sqrt() / d);
    }
    out
}

/// Five per-row spectral summaries over the full DFT magnitudes at f = 1..=d/2:
/// spectral entropy of the normalized spectrum (-Σ p ln p, terms with
/// p <= 1e-12 skipped), spectral moments m1 = Σ f·p and m2 = Σ f²·p, peak
/// frequency (lowest f on ties, as f64), peak magnitude. Width-1 rows (no
/// frequencies) emit all zeros.
fn row_spectral(r: &[f64]) -> Vec<f64> {
    let nf = r.len() / 2;
    if nf == 0 {
        return vec![0.0; 5];
    }
    let mags = dft_mags(r, nf);
    let mut total = 0.0;
    for m in &mags {
        total += m;
    }
    let denom = total.max(1e-12);
    let mut ent = 0.0;
    let mut m1 = 0.0;
    let mut m2 = 0.0;
    let mut peak = 0usize;
    for (i, m) in mags.iter().enumerate() {
        let p = m / denom;
        if p > 1e-12 {
            ent -= p * p.ln();
        }
        let f = (i + 1) as f64;
        m1 += f * p;
        m2 += f * f * p;
        // Strict `>` keeps the lowest frequency among tied maxima.
        if *m > mags[peak] {
            peak = i;
        }
    }
    vec![ent, m1, m2, (peak + 1) as f64, mags[peak]]
}

/// Eight per-row stats over first differences d_j = x_{j+1} − x_j:
/// mean, std (population), min, max, mean |d|, sign changes of consecutive
/// diffs / (d−1), lag-1 autocorrelation of the ORIGINAL row (Pearson of
/// x[..d−1] vs x[1..], zero variance → 0), energy = mean d². Rows shorter
/// than 2 emit all zeros.
fn row_diff_stats(r: &[f64]) -> Vec<f64> {
    let d = r.len();
    if d < 2 {
        return vec![0.0; 8];
    }
    let nd = d - 1;
    let ndf = nd as f64;
    let diffs: Vec<f64> = (0..nd).map(|j| r[j + 1] - r[j]).collect();
    let mean = diffs.iter().sum::<f64>() / ndf;
    let mut m2 = 0.0;
    let mut mean_abs = 0.0;
    let mut energy = 0.0;
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    let mut signs = 0usize;
    for (j, v) in diffs.iter().enumerate() {
        let dd = v - mean;
        m2 += dd * dd;
        mean_abs += v.abs();
        energy += v * v;
        if *v < mn {
            mn = *v;
        }
        if *v > mx {
            mx = *v;
        }
        // A sign change is a strictly negative product of consecutive diffs.
        if j > 0 && diffs[j - 1] * v < 0.0 {
            signs += 1;
        }
    }
    let std = (m2 / ndf).sqrt();
    let auto = crate::metrics::pearson(&r[..nd], &r[1..]);
    vec![
        mean,
        std,
        mn,
        mx,
        mean_abs / ndf,
        signs as f64 / ndf,
        auto,
        energy / ndf,
    ]
}

// B12: the v2 executor never changes legacy FittedOp or fit_op admission.
pub(crate) fn apply_row_vector_v2(
    stage: &crate::kernel::row_vector::FittedStage,
    row: &[f64],
) -> crate::kernel::Result<crate::kernel::savgol::Output> {
    stage.apply(row)
}
