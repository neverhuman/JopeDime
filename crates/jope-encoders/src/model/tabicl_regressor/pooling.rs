/// Row pooling for the foundation vector: for D columns over M rows, return
/// `concat[mean(D), std(D), p10(D), p90(D)]` = 4*D. Computed in f64, cast to
/// f32. Must match numpy exactly:
///  - `std` is the POPULATION std (ddof=0, divide by M, not M-1);
///  - percentiles use numpy linear interpolation on the per-column SORTED
///    values (`pos = q*(M-1)`, interpolate between floor/ceil), q10=0.10,
///    q90=0.90;
///  - block order is exactly `[mean, std, p10, p90]`.
pub(super) fn pool_mean_std_p10_p90(rows: &[Vec<f32>]) -> Vec<f32> {
    let m = rows.len();
    let d = rows.first().map_or(0, |r| r.len());
    let mm = m.max(1) as f64;

    let mut mean = vec![0f64; d];
    for row in rows {
        for (j, &v) in row.iter().enumerate() {
            mean[j] += v as f64;
        }
    }
    for v in &mut mean {
        *v /= mm;
    }

    let mut std = vec![0f64; d];
    for row in rows {
        for (j, &v) in row.iter().enumerate() {
            let dd = v as f64 - mean[j];
            std[j] += dd * dd;
        }
    }
    for v in &mut std {
        *v = (*v / mm).sqrt(); // population std (ddof=0)
    }

    let mut p10 = vec![0f64; d];
    let mut p90 = vec![0f64; d];
    if m > 0 {
        let mut col = vec![0f64; m];
        for j in 0..d {
            for (i, row) in rows.iter().enumerate() {
                col[i] = row[j] as f64;
            }
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            p10[j] = numpy_percentile_sorted(&col, 0.10);
            p90[j] = numpy_percentile_sorted(&col, 0.90);
        }
    }

    let mut out = Vec::with_capacity(4 * d);
    out.extend(mean.iter().map(|&v| v as f32));
    out.extend(std.iter().map(|&v| v as f32));
    out.extend(p10.iter().map(|&v| v as f32));
    out.extend(p90.iter().map(|&v| v as f32));
    out
}

/// numpy `percentile` with linear interpolation, on an already-sorted slice.
/// `q` is a fraction in [0,1]: position `q*(len-1)`, interpolate floor/ceil.
fn numpy_percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let m = sorted.len();
    if m == 0 {
        return 0.0;
    }
    if m == 1 {
        return sorted[0];
    }
    let pos = q * (m as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = pos - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}
