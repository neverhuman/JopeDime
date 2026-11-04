use super::super::*;

pub(crate) fn table_descriptors(s: &ProbeStage, n: usize, p: usize) -> TableStage {
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
