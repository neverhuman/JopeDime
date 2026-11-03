use super::super::*;

pub(crate) fn building_blocks(s: &ProbeStage, n: usize, p: usize) -> BlockStage {
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
