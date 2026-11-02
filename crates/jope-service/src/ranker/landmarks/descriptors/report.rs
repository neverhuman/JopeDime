use super::*;

pub(crate) fn report_descriptors(
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
