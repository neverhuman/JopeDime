use super::*;

pub(crate) fn inter_descriptors(
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
