use super::*;

pub(crate) fn extract_probes(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
) -> Result<ProbeStage, RecipeError> {
    let ym = mean(y);
    let ys = stddev(y);
    let yz: Vec<f64> = y
        .iter()
        .map(|v| if ys > 1e-9 { (v - ym) / ys } else { 0.0 })
        .collect();
    let mut z = vec![0.0; n * p];
    let mut z_unc = vec![0.0; n * p];
    let mut probes = vec![[0.0; 14]; p];
    let mut table_cols = vec![[0.0; 6]; p];
    for j in 0..p {
        let col: Vec<f64> = (0..n).map(|i| x[i * p + j]).collect();
        let m = mean(&col);
        let s = stddev(&col);
        let unc: Vec<f64> = col
            .iter()
            .map(|v| if s > 1e-9 { (v - m) / s } else { 0.0 })
            .collect();
        let clp: Vec<f64> = unc.iter().map(|v| v.clamp(-8.0, 8.0)).collect();
        for i in 0..n {
            z[i * p + j] = clp[i];
            z_unc[i * p + j] = unc[i];
        }
        let mut sorted = unc.clone();
        sorted.sort_by(f64::total_cmp);
        let q05 = quantile(&sorted, 0.05);
        let q25 = quantile(&sorted, 0.25);
        let q50 = quantile(&sorted, 0.5);
        let q75 = quantile(&sorted, 0.75);
        let q95 = quantile(&sorted, 0.95);
        let (q10, q90) = probe_bounds(quantile(&sorted, 0.10), quantile(&sorted, 0.90))?;
        let tail = ((q95 - q05) / (q75 - q25).max(1e-6)).clamp(0.0, 20.0) / 10.0;
        let skew = mean(&unc.iter().map(|v| v.powi(3)).collect::<Vec<_>>());
        let kurt =
            (mean(&unc.iter().map(|v| v.powi(4)).collect::<Vec<_>>()) - 3.0).clamp(-3.0, 50.0);
        let mode = unc.iter().filter(|v| (**v - q50).abs() < 0.05).count() as f64 / n as f64;
        let pear = if ys > 1e-9 {
            unc.iter().zip(&yz).map(|(a, b)| a * b).sum::<f64>().abs() / n as f64
        } else {
            0.0
        };
        let ranks = recipe_argsort(y);
        let mut yr = vec![0.0; n];
        for (r, &i) in ranks.iter().enumerate() {
            yr[i] = r as f64;
        }
        let order = recipe_argsort(&clp);
        let rank_y: Vec<f64> = order.iter().map(|&i| yr[i]).collect();
        let xi = if ys > 1e-9 && n > 1 {
            (1.0 - 3.0 * rank_y.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>()
                / (n - 1) as f64
                / (n + 1).max(2) as f64)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let yrz: Vec<f64> = yr
            .iter()
            .map(|v| (v - 0.5 * (n - 1) as f64) / (((n * n - 1) as f64 / 12.0).sqrt().max(1e-9)))
            .collect();
        let spearman = if ys > 1e-9 {
            (order
                .iter()
                .enumerate()
                .map(|(r, &i)| {
                    (r as f64 - 0.5 * (n - 1) as f64)
                        / (((n * n - 1) as f64 / 12.0).sqrt().max(1e-9))
                        * yrz[i]
                })
                .sum::<f64>()
                / n as f64)
                .abs()
        } else {
            0.0
        };
        table_cols[j] = [skew, kurt, tail, mode, pear, spearman];
        let q50c = q50.clamp(-8.0, 8.0);
        let q25c = quantile(&sorted, 0.25).clamp(-8.0, 8.0);
        let q75c = quantile(&sorted, 0.75).clamp(-8.0, 8.0);
        let transforms: [Vec<f64>; 13] = [
            clp.iter().map(|v| v.asinh()).collect(),
            clp.iter().map(|v| v.atan()).collect(),
            clp.iter()
                .map(|v| erf_approx(v / std::f64::consts::SQRT_2))
                .collect(),
            clp.iter().map(|v| v.signum() * v.abs().cbrt()).collect(),
            clp.iter().map(|v| v.signum() * v.abs().ln_1p()).collect(),
            clp.iter().map(|v| (0.75 * v).tanh()).collect(),
            clp.iter().map(|v| (-v.abs()).exp_m1()).collect(),
            clp.iter().map(|v| v.clamp(q10, q90)).collect(),
            clp.iter()
                .map(|v| if *v >= q50c { 1.0 } else { 0.0 })
                .collect(),
            clp.iter()
                .map(|v| (v - q75c).max(0.0) - (q25c - v).max(0.0))
                .collect(),
            clp.iter()
                .map(|v| (0.5 * std::f64::consts::PI * v).sin())
                .collect(),
            clp.iter().map(|v| v - v.floor()).collect(),
            clp.clone(),
        ];
        for k in 0..13 {
            probes[j][k] = abs_corr_with_y(&transforms[k], &yz);
        }
        probes[j][12] = xi;
        probes[j][13] = pear;
        let _ = q10;
    }
    let row_corr = row_summary_corr(&z, n, p, &yz);
    Ok(ProbeStage {
        y_scaled: y.to_vec(),
        z,
        z_unc,
        y_z: yz,
        probes,
        table_cols,
        row_corr,
    })
}

fn erf_approx(x: f64) -> f64 {
    let sign = x.signum();
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-a * a).exp())
}

fn row_summary_corr(z: &[f64], n: usize, p: usize, yz: &[f64]) -> [f64; 8] {
    let mut rows: Vec<[f64; 8]> = Vec::with_capacity(n);
    for i in 0..n {
        let r: Vec<f64> = (0..p).map(|j| z[i * p + j]).collect();
        let m = mean(&r);
        let sd = (mean(&r.iter().map(|v| (v - m).powi(2)).collect::<Vec<_>>()) + 1e-8).sqrt();
        let l1 = mean(&r.iter().map(|v| v.abs()).collect::<Vec<_>>());
        let l2 = (mean(&r.iter().map(|v| v * v).collect::<Vec<_>>()) + 1e-8).sqrt();
        let lo = r.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let skew =
            mean(&r.iter().map(|v| ((v - m) / sd).powi(3)).collect::<Vec<_>>()).clamp(-10.0, 10.0);
        rows.push([m, sd, l1, l2, lo, hi, skew, 0.0]);
    }
    std::array::from_fn(|k| abs_corr_with_y(&rows.iter().map(|r| r[k]).collect::<Vec<_>>(), yz))
}
