use super::types::GpData;
use crate::dsl::Expr;
use crate::metrics::{corr_std, sse, standardize};

pub(super) struct Eval {
    pub(super) gain: f32,
    pub(super) sg: f32,
    pub(super) lc: f32,
    pub(super) lib_corr: f32,
    pub(super) maxcorr: f32,
    pub(super) nodes: usize,
    pub(super) target_corr: f32,
    pub(super) valid: bool,
}

pub(super) fn sketch(vstd: &[f32], sub: &[usize]) -> Vec<f32> {
    let m = sub.len();
    let mut s: Vec<f32> = sub.iter().map(|&i| vstd[i]).collect();
    let mean = s.iter().sum::<f32>() / m as f32;
    let mut var = 0f32;
    for x in &s {
        let d = x - mean;
        var += d * d;
    }
    let sd = (var / m as f32).sqrt();
    if sd > 1e-9 {
        for x in &mut s {
            *x = (*x - mean) / sd;
        }
    } else {
        s.fill(0.0);
    }
    s
}

fn surv_scan(cand: &[f32], surv: &[f32], mlen: usize, use_len: usize) -> f32 {
    let k = surv.len().checked_div(mlen).unwrap_or(0);
    let inv = 1.0 / use_len as f32;
    let mut maxc = 0f32;
    for j in 0..k {
        let row = &surv[j * mlen..j * mlen + use_len];
        let mut d = 0f32;
        for t in 0..use_len {
            d += cand[t] * row[t];
        }
        let c = (d * inv).abs();
        if c > maxc {
            maxc = c;
        }
    }
    maxc
}

/// EXACT |corr| scan vs the flat survivor store [K*n]; returns count of survivors with |corr|>=tau.
pub(super) fn exact_scan_count(cand: &[f32], surv_v: &[f32], n: usize, tau: f32) -> usize {
    let k = surv_v.len().checked_div(n).unwrap_or(0);
    let mut cnt = 0usize;
    for j in 0..k {
        if corr_std(cand, &surv_v[j * n..(j + 1) * n]) >= tau {
            cnt += 1;
        }
    }
    cnt
}

fn corr_std_holdout(v: &[f32], ref_v: &[f32], holdout: &[usize]) -> f32 {
    let n = holdout.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0f32;
    for &i in holdout {
        sum += v[i];
    }
    let mean = sum / n as f32;
    let mut var = 0f32;
    for &i in holdout {
        let d = v[i] - mean;
        var += d * d;
    }
    let sd = var.sqrt();
    if sd < 1e-9 {
        return 0.0;
    }
    let mut sum_ref = 0f32;
    for &i in holdout {
        sum_ref += ref_v[i];
    }
    let mean_ref = sum_ref / n as f32;
    let mut cov = 0f32;
    let mut var_ref = 0f32;
    for &i in holdout {
        let d = v[i] - mean;
        let dr = ref_v[i] - mean_ref;
        cov += d * dr;
        var_ref += dr * dr;
    }
    let sd_ref = var_ref.sqrt();
    if sd_ref < 1e-9 {
        return 0.0;
    }
    (cov / (sd * sd_ref)).abs()
}

fn split_gain_holdout(v: &[f32], gd: &GpData, holdout: &[usize]) -> f32 {
    let n = gd.x.n;
    let mut fin: Vec<usize> = Vec::with_capacity(n);
    let mut miss_r: Vec<f32> = Vec::new();
    for &i in holdout {
        if v[i].is_finite() {
            fin.push(i);
        } else {
            miss_r.push(gd.r[i]);
        }
    }
    if fin.len() < 50 {
        return 0.0;
    }
    fin.sort_unstable_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let m = fin.len();
    let mut ps = vec![0f32; m + 1];
    let mut pss = vec![0f32; m + 1];
    for j in 0..m {
        let rv = gd.r[fin[j]];
        ps[j + 1] = ps[j] + rv;
        pss[j + 1] = pss[j] + rv * rv;
    }
    let tot_s = ps[m];
    let tot_ss = pss[m];
    let sse_group = |s: f32, ss: f32, cnt: f32| -> f32 {
        if cnt < 1.0 {
            0.0
        } else {
            ss - s * s / cnt
        }
    };
    let sse_miss = sse(&miss_r);
    let mut best = f32::INFINITY;
    let steps = 31usize;
    for t in 1..steps {
        let k = (m * t / steps).clamp(1, m - 1);
        let ls = ps[k];
        let lss = pss[k];
        let lc = k as f32;
        let rs = tot_s - ls;
        let rss = tot_ss - lss;
        let rc = (m - k) as f32;
        let s = sse_group(ls, lss, lc) + sse_group(rs, rss, rc) + sse_miss;
        if s < best {
            best = s;
        }
    }
    if !best.is_finite() {
        return 0.0;
    }
    ((gd.r_sse - best) / gd.r_sse.max(1e-9)).max(0.0)
}

pub(super) fn evaluate(
    e: &Expr,
    gd: &GpData,
    folds: &[Vec<usize>],
    surv: &[f32],
    m_full: usize,
    m_fit: usize,
    gain_floor: f32,
    lin_w: f32,
) -> Eval {
    let nodes = e.nodes();
    let v = e.eval(&gd.x);
    let finite = v.iter().filter(|x| x.is_finite()).count();
    if (finite as f32 / gd.x.n as f32) < 0.5 {
        return invalid_eval(nodes);
    }
    let (vstd, ok) = standardize(&v);
    if !ok {
        return invalid_eval(nodes);
    }
    // CV-penalized fitness: score the feature on each (per-generation, reshuffled) fold and harshly
    // penalize cross-fold inconsistency. A feature that spikes on one slice but not others gets a low
    // gain and dies. Legacy path (cv_k==1) evaluates the single holdout slice with zero penalty.
    let mut sigs: [f32; 16] = [0.0; 16];
    let kf = folds.len().clamp(1, 16);
    let mut sg = 0.0f32;
    let mut lc = 0.0f32;
    for (fi, fold) in folds.iter().take(kf).enumerate() {
        let sg_f = split_gain_holdout(&v, gd, fold);
        let lc_f = corr_std_holdout(&vstd, &gd.r_std, fold);
        if fi == 0 {
            sg = sg_f;
            lc = lc_f;
        }
        sigs[fi] = sg_f + lin_w * lc_f;
    }
    let mean = sigs[..kf].iter().sum::<f32>() / kf as f32;
    let signal = if gd.cv_penalty > 0.0 && kf > 1 {
        let var = sigs[..kf]
            .iter()
            .map(|s| (s - mean) * (s - mean))
            .sum::<f32>()
            / kf as f32;
        mean - gd.cv_penalty * var.sqrt()
    } else {
        mean
    };
    if signal < gain_floor {
        return Eval {
            gain: signal,
            sg,
            lc,
            lib_corr: 0.0,
            maxcorr: 0.0,
            nodes,
            target_corr: 0.0,
            valid: true,
        };
    }
    let mut lib_corr = 0f32;
    for l in &gd.lib_std {
        let cc = corr_std_holdout(&vstd, l, &gd.holdout);
        if cc > lib_corr {
            lib_corr = cc;
        }
    }
    let sk = sketch(&vstd, &gd.sub);
    let surv_corr = surv_scan(&sk, surv, m_full, m_fit.min(m_full));
    let maxcorr = lib_corr.max(surv_corr);
    let tcorr = corr_std_holdout(&vstd, &gd.y_std, &gd.holdout);
    Eval {
        gain: signal,
        sg,
        lc,
        lib_corr,
        maxcorr,
        nodes,
        target_corr: tcorr,
        valid: true,
    }
}

fn invalid_eval(nodes: usize) -> Eval {
    Eval {
        gain: 0.0,
        sg: 0.0,
        lc: 0.0,
        lib_corr: 1.0,
        maxcorr: 1.0,
        nodes,
        target_corr: 0.0,
        valid: false,
    }
}

#[inline]
pub(super) fn fit_of(ev: &Eval, tau: f32, mu: f32) -> f32 {
    if !ev.valid {
        return -1e9;
    }
    ev.gain * (1.0 - ev.maxcorr / tau).max(0.0) - mu * ev.nodes as f32
}
