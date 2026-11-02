use super::*;

pub(super) fn building_blocks(s: &ProbeStage, n: usize, p: usize) -> BlockStage {
    let max_probe = |k: usize| s.probes.iter().map(|r| r[k]).fold(0.0_f64, f64::max);
    let row_gain = s.row_corr.iter().copied().fold(0.0_f64, f64::max);
    let mut gain = [0.0; 16];
    gain[0] = max_probe(0);
    gain[1] = max_probe(1);
    gain[2] = max_probe(2);
    gain[3] = max_probe(3);
    gain[4] = max_probe(4);
    gain[5] = row_gain;
    gain[6] = max_probe(5);
    gain[7] = max_probe(6);
    gain[8] = max_probe(7);
    gain[9] = max_probe(8);
    gain[10] = max_probe(9);
    gain[11] = max_probe(10);
    gain[12] = max_probe(11);
    gain[11] = (0..p)
        .map(|j| {
            let sin = (0..n)
                .map(|i| (0.5 * std::f64::consts::PI * s.z[i * p + j]).sin())
                .collect::<Vec<_>>();
            let cos = (0..n)
                .map(|i| (0.5 * std::f64::consts::PI * s.z[i * p + j]).cos())
                .collect::<Vec<_>>();
            abs_corr_with_y(&sin, &s.y_z).max(abs_corr_with_y(&cos, &s.y_z))
        })
        .fold(0.0, f64::max);
    if p >= 2 && stddev(&s.y_scaled) > 1e-9 {
        let mut idx: Vec<usize> = (0..p).collect();
        idx.sort_by(|&a, &b| {
            s.probes[b][0]
                .max(s.probes[b][8])
                .max(s.probes[b][12])
                .total_cmp(&s.probes[a][0].max(s.probes[a][8]).max(s.probes[a][12]))
        });
        idx.truncate(p.min(6));
        let (mut h, mut a, mut d) = (Vec::new(), Vec::new(), Vec::new());
        for ia in 0..idx.len() {
            for ib in ia + 1..idx.len() {
                let mut hv = Vec::with_capacity(n);
                let mut av = Vec::with_capacity(n);
                let mut dv = Vec::with_capacity(n);
                for i in 0..n {
                    let x = s.z[i * p + idx[ia]];
                    let y = s.z[i * p + idx[ib]];
                    hv.push(x.hypot(y));
                    av.push(x.atan2(y));
                    dv.push((x - y) / (1.0 + x.abs() + y.abs()));
                }
                h.push(abs_corr_with_y(&hv, &s.y_z));
                a.push(abs_corr_with_y(&av, &s.y_z));
                d.push(abs_corr_with_y(&dv, &s.y_z));
            }
        }
        gain[13] = h.into_iter().fold(0.0, f64::max);
        gain[14] = a.into_iter().fold(0.0, f64::max);
        gain[15] = d.into_iter().fold(0.0, f64::max);
    }
    let high = {
        let mut v = gain;
        v[14] = 0.0;
        v[15] = 0.0;
        v
    };
    let lin = max_probe(13);
    let denom = high.iter().copied().fold(lin, f64::max).max(0.02);
    let mut contrast = [0.0; 16];
    contrast[0] = (high[0] - lin) / denom;
    contrast[1] = (high[1] - lin) / denom;
    contrast[2] = (high[2] - lin) / denom;
    contrast[3] = (high[3] - lin) / denom;
    contrast[4] = (high[5] - lin) / denom;
    contrast[5] = mean(&s.row_corr) / denom;
    contrast[6] = gini(&s.row_corr);
    contrast[7] = (high[7] - high[6]) / denom;
    contrast[8] = (high[8] - lin) / denom;
    contrast[9] = (high[9] - high[2]) / denom;
    contrast[10] = (high[10] - lin) / denom;
    contrast[11] = (high[11] - lin) / denom;
    contrast[12] = (high[12] - lin) / denom;
    contrast[13] = (high[13] - lin) / denom;
    let mut het = [0.0; 8];
    if stddev(&s.y_scaled) > 1e-9 {
        let bj = (0..p)
            .max_by(|&a, &b| s.probes[a][0].total_cmp(&s.probes[b][0]))
            .unwrap_or(0);
        let resid: Vec<f64> = (0..n)
            .map(|i| (s.y_z[i] - s.probes[bj][0] * s.z[i * p + bj]).abs())
            .collect();
        let rs = stddev(&resid);
        let rz: Vec<f64> = resid
            .iter()
            .map(|v| {
                if rs > 1e-9 {
                    (v - mean(&resid)) / rs
                } else {
                    0.0
                }
            })
            .collect();
        let hc: Vec<f64> = (0..p)
            .map(|j| abs_corr_with_y(&(0..n).map(|i| s.z[i * p + j]).collect::<Vec<_>>(), &rz))
            .collect();
        let mut sorted = hc.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        het[0] = sorted.first().copied().unwrap_or(0.0);
        het[1] = mean(&sorted[..sorted.len().min(3)]);
        het[2] = mean(&hc);
        het[3] = gini(&hc);
        let y_hi: Vec<f64> = s
            .y_z
            .iter()
            .map(|v| if *v > 1.0 { 1.0 } else { 0.0 })
            .collect();
        let y_lo: Vec<f64> = s
            .y_z
            .iter()
            .map(|v| if *v < -1.0 { 1.0 } else { 0.0 })
            .collect();
        let mut u = Vec::new();
        let mut l = Vec::new();
        for j in 0..p {
            let xu: Vec<f64> = (0..n)
                .map(|i| if s.z[i * p + j] > 1.0 { 1.0 } else { 0.0 })
                .collect();
            let xl: Vec<f64> = (0..n)
                .map(|i| if s.z[i * p + j] < -1.0 { 1.0 } else { 0.0 })
                .collect();
            u.push(abs_corr_with_y(&xu, &y_hi));
            l.push(abs_corr_with_y(&xl, &y_lo));
        }
        het[4] = u.iter().copied().fold(0.0, f64::max);
        het[5] = mean(&u);
        het[6] = l.iter().copied().fold(0.0, f64::max);
        het[7] = mean(&l);
    }
    let mc = target_simplex(&s.z, &s.y_scaled, n, p);
    let mut vec = [0.0_f32; BUILDING_BLOCK_WIDTH];
    for i in 0..16 {
        vec[i] = squash(high[i]) as f32;
    }
    for i in 0..8 {
        vec[16 + i] = squash(s.row_corr[i]) as f32;
    }
    for i in 0..16 {
        vec[24 + i] = contrast[i].clamp(-1.0, 1.0) as f32;
    }
    for i in 0..8 {
        vec[40 + i] = squash(het[i]) as f32;
    }
    for i in 0..20 {
        vec[48 + i] = mc[i] as f32;
    }
    let mut cols = vec![[0.0_f32; 7]; p];
    for (j, col) in cols.iter_mut().enumerate() {
        *col = [
            s.probes[j][0] as f32,
            s.probes[j][1] as f32,
            s.probes[j][2] as f32,
            s.probes[j][8] as f32,
            s.probes[j][9] as f32,
            s.probes[j][11] as f32,
            s.probes[j][12] as f32,
        ];
    }
    BlockStage {
        raw_gain: gain,
        vector: vec,
        column: cols,
    }
}

fn target_simplex(z: &[f64], y: &[f64], n: usize, p: usize) -> [f64; 20] {
    let mut out = [0.0; 20];
    let mut labels: Vec<f64> = y
        .iter()
        .map(|v| (v * 100_000.0).round() / 100_000.0)
        .collect();
    labels.sort_by(f64::total_cmp);
    labels.dedup_by(|a, b| *a == *b);
    let counts: Vec<usize> = labels
        .iter()
        .map(|label| y.iter().filter(|v| (**v - *label).abs() <= 1e-4).count())
        .collect();
    let k = labels.len();
    let discrete = 2 <= k && k <= 32.min((n / 3).max(2));
    if discrete {
        let priors: Vec<f64> = counts.iter().map(|c| *c as f64 / n as f64).collect();
        out[0] = (k.min(64) as f64).log2() / 6.0;
        out[1] =
            -priors.iter().map(|v| v * v.max(1e-12).ln()).sum::<f64>() / (k as f64).ln().max(1e-9);
        out[2] = priors.iter().copied().fold(f64::INFINITY, f64::min)
            / priors.iter().copied().fold(0.0, f64::max).max(1e-9);
        out[3] = gini(&priors);
        let mut centroids = vec![0.0; k * p];
        for (ci, label) in labels.iter().enumerate() {
            let rows: Vec<usize> = (0..n).filter(|&i| (y[i] - *label).abs() <= 1e-4).collect();
            for j in 0..p {
                centroids[ci * p + j] =
                    mean(&rows.iter().map(|&i| z[i * p + j]).collect::<Vec<_>>());
            }
        }
        let mut fisher = vec![0.0; p];
        for j in 0..p {
            let between = (0..k)
                .map(|ci| priors[ci] * centroids[ci * p + j].powi(2))
                .sum::<f64>();
            fisher[j] = squash(between / (1.0 - between).max(1e-4));
        }
        let mut fs = fisher.clone();
        fs.sort_by(|a, b| b.total_cmp(a));
        let fisher_count = fs.len().min(4);
        out[4..4 + fisher_count].copy_from_slice(&fs[..fisher_count]);
        out[8] = mean(&fisher);
        out[9] = gini(&fisher);
        if k >= 2 {
            let mut gram = vec![0.0; k * k];
            for a in 0..k {
                for b in 0..k {
                    gram[a * k + b] = (0..p)
                        .map(|j| {
                            priors[a].sqrt()
                                * centroids[a * p + j]
                                * priors[b].sqrt()
                                * centroids[b * p + j]
                        })
                        .sum();
                }
            }
            let (e, _) = feat_math::heads::jacobi_eigen(&mut gram, k);
            let mut singular: Vec<f64> = e.into_iter().map(|v| squash(v.max(0.0).sqrt())).collect();
            singular.sort_by(|a, b| b.total_cmp(a));
            let singular_count = singular.len().min(6);
            out[10..10 + singular_count].copy_from_slice(&singular[..singular_count]);
        }
    }
    let ym = mean(y);
    let ys = stddev(y).max(1e-9);
    let yz: Vec<f64> = y.iter().map(|v| (v - ym) / ys).collect();
    let mut sorted = yz.clone();
    sorted.sort_by(f64::total_cmp);
    let q10 = quantile(&sorted, 0.10);
    let q25 = quantile(&sorted, 0.25);
    let q75 = quantile(&sorted, 0.75);
    let q90 = quantile(&sorted, 0.90);
    out[16] = squash(mean(&yz.iter().map(|v| v.powi(3)).collect::<Vec<_>>()));
    out[17] = squash((mean(&yz.iter().map(|v| v.powi(4)).collect::<Vec<_>>()) - 3.0) / 4.0);
    out[18] = squash(mean(
        &yz.iter().map(|v| v.asinh().powi(2)).collect::<Vec<_>>(),
    ));
    out[19] = squash((q90 - q10) / (q75 - q25).max(1e-4) / 3.0);
    let _ = (z, n, p);
    out
}

pub(super) fn report_descriptors(
    s: &ProbeStage,
    b: &BlockStage,
    n: usize,
    p: usize,
) -> ReportStage {
    let u: [f64; 16] = std::array::from_fn(|i| squash(b.raw_gain[i]) as f32 as f64);
    let probe = [
        u[4],
        u[0],
        u[3],
        u[6],
        u[1],
        u[7],
        u[10],
        u[9],
        u[11],
        u[7],
        u[5],
        u[13],
        u[11],
        0.01,
        u[0].max(u[4]),
        u[9],
        u[9].max(u[13]),
        u[9].max(u[13]) * 1.05,
        u[13] * 0.9,
        u[13] * 0.95,
        u[0].max(u[6]) * 1.05,
        u[4] * 0.85,
        u[9].max(u[13]) * 0.85,
    ];
    let mut raw_gains = [0.0; 23];
    let mut present = [false; 23];
    let mut se = [0.0; 23];
    let mut lam = [0.0; 23];
    for i in 0..23 {
        raw_gains[i] = (probe[i] * 100_000.0).round() / 100_000.0;
        present[i] = i <= 16 || i == 21;
        if present[i] {
            se[i] = ((1.0 / (n.max(16) as f64).sqrt()) * 100_000.0).round() / 100_000.0;
            lam[i] = 0.3;
        }
    }
    let mut gains = raw_gains;
    for i in 0..23 {
        if !present[i] {
            gains[i] = 0.0;
        }
    }
    let sg: [f64; 23] = std::array::from_fn(|i| squash(gains[i]));
    let mut mid = [0.0; 23];
    let active: Vec<usize> = (0..23).filter(|&i| present[i]).collect();
    for &i in &active {
        let less = active
            .iter()
            .filter(|&&j| raw_gains[j] < raw_gains[i])
            .count() as f64;
        let eq = active
            .iter()
            .filter(|&&j| raw_gains[j] == raw_gains[i])
            .count() as f64;
        mid[i] = (less + 0.5 * (eq - 1.0)) / (active.len() - 1).max(1) as f64;
    }
    let contrasts = report_contrasts(&raw_gains, &present);
    let mut raw_spec = vec![[0.0_f32; 9]; p];
    for (j, row) in raw_spec.iter_mut().enumerate() {
        let c = b.column[j];
        let vals = [c[0], c[3], c[4], c[0], c[1], c[5], c[2], c[6], c[3]];
        for k in 0..9 {
            if k != 7 {
                row[k] = vals[k];
            }
        }
    }
    let spec: Vec<[f32; 9]> = raw_spec
        .iter()
        .map(|row| std::array::from_fn(|k| squash(f64::from(row[k])) as f32))
        .collect();
    let mut lin = Vec::new();
    let mut macro_nl = Vec::new();
    let mut large_nl = Vec::new();
    let mut xis = Vec::new();
    let mut best = Vec::new();
    for row in &raw_spec {
        let v: Vec<f64> = row.iter().map(|x| squash(f64::from(*x))).collect();
        lin.push(v[0].max(0.0));
        macro_nl.push(v[3].max(v[4]).max(v[6]).max(0.0));
        large_nl.push(v[1].max(v[2]).max(v[5]).max(v[8]).max(0.0));
        xis.push(0.0);
        best.push(v.into_iter().fold(0.0_f64, f64::max));
    }
    let top1 = best.iter().copied().fold(0.0, f64::max);
    let mut best_sorted = best.clone();
    best_sorted.sort_by(|a, b| b.total_cmp(a));
    let mut summary = [0.0; 16];
    if p > 0 {
        summary = [
            lin.iter().copied().fold(0.0, f64::max),
            mean(&lin),
            gini(&lin),
            macro_nl.iter().copied().fold(0.0, f64::max),
            mean(&macro_nl),
            large_nl.iter().copied().fold(0.0, f64::max),
            mean(&large_nl),
            0.0,
            0.0,
            0.0,
            top1,
            mean(&best_sorted[..best_sorted.len().min(3)]),
            gini(&best),
            macro_nl
                .iter()
                .zip(&lin)
                .map(|(a, b)| a - b)
                .fold(f64::NEG_INFINITY, f64::max),
            large_nl
                .iter()
                .zip(&lin)
                .map(|(a, b)| a - b)
                .fold(f64::NEG_INFINITY, f64::max),
            lin.iter().map(|v| -*v).fold(f64::NEG_INFINITY, f64::max),
        ];
    }
    let mean_y = mean(&s.y_scaled);
    let std_y = stddev(&s.y_scaled);
    let shape = [
        0.0,
        (n.max(1) as f64).log10() / 6.0,
        (p.max(1) as f64).log2() / 12.0,
        ((p.max(1) as f64 / n.max(1) as f64).log10()) / 4.0,
    ];
    let macro_idx = [0, 1, 2, 3, 4, 5, 13, 14, 21];
    let larger_idx = [6, 7, 8, 9, 10, 11, 12, 15, 16];
    let fine_idx = [17, 18, 19, 20, 22];
    let macro_vals: Vec<f64> = macro_idx.iter().map(|&i| sg[i]).collect();
    let large_vals: Vec<f64> = larger_idx.iter().map(|&i| sg[i]).collect();
    let fine_vals: Vec<f64> = fine_idx.iter().map(|&i| sg[i]).collect();
    let group = [
        macro_vals.iter().copied().fold(0.0, f64::max),
        large_vals.iter().copied().fold(0.0, f64::max),
        0.0,
        mean(&macro_vals),
        mean(&large_vals),
        0.0,
        mean(&active.iter().map(|&i| squash(se[i])).collect::<Vec<_>>()),
        1.0,
        mean(&active.iter().map(|&i| lam[i]).collect::<Vec<_>>()),
    ];
    let mut v = [0.0_f32; REPORT_WIDTH];
    let mut at = 0;
    for x in sg {
        v[at] = x as f32;
        at += 1;
    }
    for &is_present in present.iter().take(23) {
        v[at] = if is_present { 1.0 } else { 0.0 };
        at += 1;
    }
    for x in mid {
        v[at] = x as f32;
        at += 1;
    }
    for x in contrasts {
        v[at] = x as f32;
        at += 1;
    }
    v[at] = squash(mean_y) as f32;
    v[at + 1] = squash(std_y) as f32;
    at += 2;
    for x in shape {
        v[at] = x as f32;
        at += 1;
    }
    for x in summary {
        v[at] = x as f32;
        at += 1;
    }
    for x in group {
        v[at] = x as f32;
        at += 1;
    }
    let _ = fine_vals;
    ReportStage {
        vector: v,
        gains: sg,
        ses: std::array::from_fn(|i| squash(se[i])),
        lambdas: lam,
        present,
        column_spectrum: spec,
    }
}

fn report_contrasts(g: &[f64; 23], present: &[bool; 23]) -> [f64; 16] {
    let max = |idx: &[usize]| idx.iter().map(|&i| g[i]).fold(f64::NEG_INFINITY, f64::max);
    let s = (0..23)
        .filter(|&i| present[i])
        .map(|i| g[i])
        .fold(0.0_f64, f64::max);
    if s < 0.01 {
        return [0.0; 16];
    }
    let sub = |a: f64, b: f64| ((a - b) / s).clamp(-1.0, 1.0);
    let _ = present;
    [
        s.clamp(-1.0, 1.0),
        (g[0] / s).clamp(-1.0, 1.0),
        (max(&[0, 1, 2, 3, 4, 5, 14]) / s).clamp(-1.0, 1.0),
        sub(max(&[1, 2, 4]), g[0]),
        sub(max(&[3, 5, 9]), g[0]),
        (g[14] / s).clamp(-1.0, 1.0),
        (g[13] / s).clamp(-1.0, 1.0),
        sub(g[0], g[21]),
        sub(g[7], g[6]),
        sub(g[8], max(&[0, 4])),
        sub(g[11], g[0]),
        sub(g[12], max(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14])),
        sub(max(&[15, 16]), max(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14])),
        sub(g[12], max(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14])),
        0.0,
        0.0,
    ]
}

pub(super) fn table_descriptors(s: &ProbeStage, n: usize, p: usize) -> TableStage {
    let mut corr = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..p {
            corr[a * p + b] = if a == b {
                1.0
            } else {
                (0..n)
                    .map(|i| s.z_unc[i * p + a] * s.z_unc[i * p + b])
                    .sum::<f64>()
                    / n as f64
            };
        }
    }
    let mut max_corr = vec![0.0; p];
    let mut mean_corr = vec![0.0; p];
    for j in 0..p {
        let vals: Vec<f64> = (0..p)
            .filter(|&k| k != j)
            .map(|k| corr[j * p + k].abs())
            .collect();
        max_corr[j] = vals.iter().copied().fold(0.0, f64::max);
        mean_corr[j] = mean(&vals);
    }
    let mut eig = if p == 1 {
        vec![1.0]
    } else {
        let (e, _) = feat_math::heads::jacobi_eigen(&mut corr, p);
        e.into_iter().map(|v| v.max(0.0)).collect::<Vec<_>>()
    };
    eig.sort_by(|a, b| b.total_cmp(a));
    let es = eig.iter().sum::<f64>().max(1e-9);
    let shares: Vec<f64> = eig.iter().map(|v| v / es).collect();
    let eff = (es * es / (eig.iter().map(|v| v * v).sum::<f64>().max(1e-9))) / p as f64;
    let entropy = if p == 1 {
        1.0
    } else {
        -shares
            .iter()
            .filter(|v| **v > 1e-12)
            .map(|v| v * v.ln())
            .sum::<f64>()
            / (p as f64).ln().max(1e-9)
    };
    let skew: Vec<f64> = s.table_cols.iter().map(|v| v[0]).collect();
    let kurt: Vec<f64> = s.table_cols.iter().map(|v| v[1]).collect();
    let tail: Vec<f64> = s.table_cols.iter().map(|v| v[2]).collect();
    let mode: Vec<f64> = s.table_cols.iter().map(|v| v[3]).collect();
    let pear: Vec<f64> = s.table_cols.iter().map(|v| v[4]).collect();
    let spear: Vec<f64> = s.table_cols.iter().map(|v| v[5]).collect();
    let mut sr = pear.clone();
    sr.sort_by(|a, b| b.total_cmp(a));
    let mut ss = spear.clone();
    ss.sort_by(|a, b| b.total_cmp(a));
    let mut v = [0.0_f32; TABLE_WIDTH];
    v[0] = eff as f32;
    v[1] = entropy as f32;
    let ask = mean(&skew.iter().map(|x| x.abs()).collect::<Vec<_>>());
    v[2] = (ask / (1.0 + ask)) as f32;
    v[3] = mean(&mode) as f32;
    v[4] = sr.first().copied().unwrap_or(0.0) as f32;
    v[5] = mean(&pear) as f32;
    v[6] = gini(&pear) as f32;
    v[7] = ((n.max(1) as f64).log10() / 6.0) as f32;
    v[8] = shares.first().copied().unwrap_or(1.0) as f32;
    v[9] = shares.iter().take(3).sum::<f64>() as f32;
    v[10] = shares.iter().take(5).sum::<f64>() as f32;
    v[11] = max_corr.iter().copied().fold(0.0, f64::max) as f32;
    v[12] = mean(&mean_corr) as f32;
    v[13] = mean(&tail) as f32;
    v[14] = ss.first().copied().unwrap_or(0.0) as f32;
    v[15] = spear
        .iter()
        .zip(&pear)
        .map(|(a, b)| a - b)
        .fold(0.0, f64::max) as f32;
    let mut col = vec![[0.0_f32; 8]; p];
    for j in 0..p {
        col[j] = [
            squash(skew[j]) as f32,
            squash(kurt[j]) as f32,
            tail[j] as f32,
            mode[j] as f32,
            max_corr[j] as f32,
            pear[j] as f32,
            spear[j] as f32,
            0.0,
        ];
    }
    TableStage {
        vector: v,
        column: col,
    }
}

pub(super) fn inter_descriptors(
    r: &ReportStage,
    _t: &TableStage,
    b: &BlockStage,
    p: usize,
) -> [f32; INTERACTION_WIDTH] {
    let g = r.gains;
    let m = r.present;
    let bb: [f64; 16] = std::array::from_fn(|i| squash(if i >= 14 { 0.0 } else { b.raw_gain[i] }));
    let fam: [Vec<f64>; 5] = [
        [g[0], g[1], g[2], g[4], bb[0], bb[1], bb[2], bb[3], bb[4]].to_vec(),
        [g[3], g[5], g[9], bb[6], bb[7], bb[13]].to_vec(),
        [g[6], g[7], g[15], g[16], g[17], bb[9], bb[10]].to_vec(),
        [g[8], g[12], bb[11], bb[12], 0.0].to_vec(),
        [g[10], g[11], g[18], g[19], g[20], bb[5], bb[15]].to_vec(),
    ];
    let mut active: Vec<f64> = g
        .iter()
        .enumerate()
        .filter(|(i, _)| m[*i])
        .map(|(_, value)| *value)
        .chain(bb.iter().copied().filter(|v| *v > 0.0))
        .filter(|v| v.is_finite())
        .collect();
    let smax = active.iter().copied().fold(0.02, f64::max);
    let mut out = [0.0_f32; INTERACTION_WIDTH];
    for fi in 0..5 {
        let vals: Vec<f64> = fam[fi]
            .iter()
            .copied()
            .filter(|v| v.is_finite() && v.abs() > 1e-9)
            .collect();
        if !vals.is_empty() {
            let mx = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sd = stddev(&vals);
            out[fi * 4] = mx.clamp(-1.0, 1.0) as f32;
            out[fi * 4 + 1] = mean(&vals).clamp(-1.0, 1.0) as f32;
            out[fi * 4 + 2] = (sd / smax).clamp(0.0, 1.0) as f32;
            out[fi * 4 + 3] = (mx / smax).clamp(-1.0, 1.0) as f32;
        }
    }
    let lin = g[0].max(bb[4] * 0.5).max(0.0);
    let best1 = g[14].max(lin).max(0.01);
    let tail = g[1].max(g[2]).max(bb[0]).max(bb[3]);
    let sharp = g[7].max(g[15]).max(bb[9]);
    let smooth = g[4].max(bb[1]).max(bb[2]).max(bb[6]);
    let pair = g[11].max(bb[13]).max(bb[14]).max(bb[15]);
    let row = g[10].max(bb[5]);
    let per = g[8].max(bb[11]).max(bb[12]);
    let dlin = if m[0] && m[21] { g[0] - g[21] } else { 0.0 };
    let entropy = if active.len() > 1 {
        let max = active.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ex: Vec<f64> = active
            .iter()
            .map(|v| (8.0 * v / smax - max).exp())
            .collect();
        let sum = ex.iter().sum::<f64>().max(1e-12);
        -ex.iter()
            .map(|v| {
                let q = v / sum;
                q * q.max(1e-12).ln()
            })
            .sum::<f64>()
            / (active.len() as f64).ln().max(1e-9)
    } else {
        0.0
    };
    let ratios = [
        (tail - lin) / smax,
        (sharp - smooth) / smax,
        (g[16] - g[15]) / smax,
        0.0,
        (pair - best1) / smax,
        (row - best1) / smax,
        squash(row / (lin + 0.02)),
        (per - out[0] as f64) / smax,
        0.0,
        0.0,
        dlin / smax,
        entropy,
    ];
    for i in 0..12 {
        out[20 + i] = if i == 6 {
            ratios[i] as f32
        } else {
            ratios[i].clamp(-1.0, 1.0) as f32
        };
    }
    let mut ga = Vec::new();
    let mut sea = Vec::new();
    let mut la = Vec::new();
    for i in 0..23 {
        if m[i] {
            ga.push(g[i]);
            sea.push(r.ses[i].max(1e-3));
            la.push(r.lambdas[i]);
        }
    }
    if !ga.is_empty() {
        let weights: Vec<f64> = sea.iter().map(|v| 1.0 / (v * v + 1e-4)).collect();
        let ts: Vec<f64> = ga.iter().zip(&sea).map(|(a, b)| a / b).collect();
        out[32] = (weights.iter().zip(&ga).map(|(w, g)| w * g).sum::<f64>()
            / weights.iter().sum::<f64>().max(1e-9))
        .clamp(-1.0, 1.0) as f32;
        out[33] = squash(ts.iter().copied().fold(0.0, f64::max) / 10.0) as f32;
        out[34] = squash(mean(&ts) / 10.0) as f32;
        out[35] = squash(stddev(&ts) / 10.0) as f32;
        out[36] = ts.iter().filter(|v| **v > 2.0).count() as f32 / ts.len() as f32;
        out[37] = mean(&la).clamp(0.0, 1.0) as f32;
        out[38] = stddev(&la).clamp(0.0, 1.0) as f32;
        out[39] = corr(&ga, &la) as f32;
    }
    if p > 0 {
        let mut lin_c = Vec::new();
        let mut step = Vec::new();
        let mut xi = Vec::new();
        let mut asinh = Vec::new();
        let mut nonlin = Vec::new();
        for j in 0..p {
            let cs = r.column_spectrum[j];
            let cb = b.column[j];
            lin_c.push(f64::from(cs[0]));
            step.push(f64::from(cb[3].max(cs[1])));
            xi.push(f64::from(cb[6]));
            asinh.push(f64::from(cb[0]));
            nonlin.push(cb[1..].iter().copied().fold(0.0_f32, f32::max) as f64);
        }
        out[40] = corr(&lin_c, &step) as f32;
        out[41] = corr(&lin_c, &xi) as f32;
        out[42] = corr(&lin_c, &asinh) as f32;
        let mut orth: Vec<f64> = nonlin
            .iter()
            .zip(&lin_c)
            .map(|(a, b)| (a - b).max(0.0))
            .collect();
        orth.sort_by(|a, b| b.total_cmp(a));
        out[43] = squash(orth[0]) as f32;
        out[44] = squash(mean(&orth[..orth.len().min(3)])) as f32;
        let mut ent = 0.0;
        for j in 0..p {
            let vals = b.column[j]
                .iter()
                .map(|v| f64::from(*v).max(1e-6))
                .collect::<Vec<_>>();
            let sm = vals.iter().sum::<f64>();
            ent += -vals
                .iter()
                .map(|v| {
                    let q = v / sm;
                    q * q.ln()
                })
                .sum::<f64>()
                / (7.0_f64).ln();
        }
        out[45] = (ent / p as f64).clamp(0.0, 1.0) as f32;
        out[46] = squash(gini(&nonlin) / (gini(&lin_c) + 0.05)) as f32;
        out[47] = squash(
            f64::from(b.vector[5])
                / (nonlin
                    .iter()
                    .copied()
                    .fold(0.02, f64::max)
                    .max(lin_c.iter().copied().fold(0.0, f64::max))),
        ) as f32;
    }
    active.clear();
    out
}
