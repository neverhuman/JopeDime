/// Keep first/last and each horizontal bucket's extrema; never discard a spike
/// merely because the full recorded history exceeds the terminal width.
///
/// The input is not modified. Empty input returns an empty vector; `columns == 0`
/// still uses one bucket when reduction is needed. `bounds` controls bucket
/// placement and should describe the x range shown by the chart.
pub fn envelope(points: &[(f64, f64)], columns: usize, bounds: [f64; 2]) -> Vec<(f64, f64)> {
    if points.len() <= columns.saturating_mul(2) {
        return points.to_vec();
    }
    let mut bins = vec![None::<((f64, f64), (f64, f64))>; columns.max(1)];
    for &(x, y) in points {
        let index = (((x - bounds[0]) / (bounds[1] - bounds[0]).max(1.) * bins.len() as f64)
            as usize)
            .min(bins.len() - 1);
        match &mut bins[index] {
            Some((low, high)) => {
                if y < low.1 {
                    *low = (x, y);
                }
                if y > high.1 {
                    *high = (x, y);
                }
            }
            None => bins[index] = Some(((x, y), (x, y))),
        }
    }
    let mut result = bins
        .into_iter()
        .flatten()
        .flat_map(|(a, b)| [a, b])
        .collect::<Vec<_>>();
    if let Some(p) = points.first() {
        result.push(*p);
    }
    if let Some(p) = points.last() {
        result.push(*p);
    }
    result.sort_by(|a, b| a.0.total_cmp(&b.0));
    result.dedup();
    result
}
/// Smooth a series with a centered time-window mean.
///
/// The window spans ten percent of the observed x range. Inputs with fewer than
/// three points are returned unchanged; other inputs are sorted by x before
/// smoothing. The returned points therefore need not retain the input order.
pub fn moving_average(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let span = (sorted.last().unwrap().0 - sorted.first().unwrap().0).max(1e-9);
    let half = span * 0.05;
    let mut out = Vec::with_capacity(sorted.len());
    let mut lo = 0usize;
    let mut hi = 0usize;
    let mut sum = 0.0;
    for i in 0..sorted.len() {
        let x = sorted[i].0;
        while hi < sorted.len() && sorted[hi].0 <= x + half {
            sum += sorted[hi].1;
            hi += 1;
        }
        while lo < hi && sorted[lo].0 < x - half {
            sum -= sorted[lo].1;
            lo += 1;
        }
        let n = (hi - lo) as f64;
        if n > 0.0 {
            out.push((x, sum / n));
        }
    }
    out
}
/// One interpolated SMA sample per terminal column so Chart draws a continuous line.
pub fn column_line(points: &[(f64, f64)], columns: usize, bounds: [f64; 2]) -> Vec<(f64, f64)> {
    let smooth = moving_average(points);
    if smooth.is_empty() || columns == 0 {
        return smooth;
    }
    let n = columns.max(2);
    let span = (bounds[1] - bounds[0]).max(1e-9);
    let mut j = 0;
    (0..n)
        .map(|i| {
            let x = bounds[0] + span * i as f64 / (n - 1) as f64;
            while j + 1 < smooth.len() && smooth[j + 1].0 < x {
                j += 1;
            }
            let y = if j + 1 < smooth.len() {
                let (x0, y0) = smooth[j];
                let (x1, y1) = smooth[j + 1];
                if (x1 - x0).abs() > 1e-12 {
                    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
                } else {
                    y0
                }
            } else {
                smooth[j].1
            };
            (x, y)
        })
        .collect()
}
/// Map a loss onto a chart row. Log scale is used when the trend spans decades so
/// late improvement is not flattened under an early spike.
pub fn trend_row(y: f64, low: f64, high: f64, height: u16, log: bool) -> u16 {
    if height == 0 {
        return 0;
    }
    let (a, b, v) = if log && low > 0.0 && high > 0.0 {
        (low.ln(), high.ln(), y.max(low).ln())
    } else {
        (low, high, y)
    };
    let t = ((v - a) / (b - a).max(1e-9)).clamp(0.0, 1.0);
    ((1.0 - t) * (height.saturating_sub(1) as f64)).round() as u16
}
