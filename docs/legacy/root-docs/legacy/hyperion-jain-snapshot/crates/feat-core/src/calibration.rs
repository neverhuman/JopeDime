//! Post-hoc probability calibration, fit on OOF predictions and applied before holdout scoring. Monotone
//! calibrators (isotonic/Platt/temperature) are AUC-preserving — they cannot hurt the ranking metric while
//! fixing logloss/Brier and making member probabilities comparable for the logistic/BMA stackers.

#[derive(Clone, Debug)]
pub enum Calibrator {
    Identity,
    Platt { a: f64, b: f64 },              // p = σ(a·s + b)
    Temperature { t: f64 },                // p = σ(logit(s)/t)
    Isotonic { x: Vec<f32>, y: Vec<f32> }, // monotone PAV map (sorted score breakpoints → prob)
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
}
fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

impl Calibrator {
    pub fn apply(&self, s: f32) -> f32 {
        match self {
            Calibrator::Identity => s.clamp(0.0, 1.0),
            Calibrator::Platt { a, b } => sigmoid(*a as f32 * s + *b as f32),
            Calibrator::Temperature { t } => sigmoid(logit(s) / *t as f32),
            Calibrator::Isotonic { x, y } => iso_apply(x, y, s),
        }
    }
    pub fn apply_vec(&self, s: &[f32]) -> Vec<f32> {
        s.iter().map(|&v| self.apply(v)).collect()
    }
}

fn iso_apply(xs: &[f32], ys: &[f32], s: f32) -> f32 {
    if xs.is_empty() {
        return s.clamp(0.0, 1.0);
    }
    if s <= xs[0] {
        return ys[0];
    }
    if s >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    let (mut lo, mut hi) = (0usize, xs.len() - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= s {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (x0, x1, y0, y1) = (xs[lo], xs[hi], ys[lo], ys[hi]);
    if (x1 - x0).abs() < 1e-12 {
        return y0;
    }
    (y0 + (s - x0) / (x1 - x0) * (y1 - y0)).clamp(0.0, 1.0)
}

/// Isotonic regression via Pool-Adjacent-Violators (a monotone non-decreasing score→prob map).
pub fn fit_isotonic(score: &[f32], y: &[f32]) -> Calibrator {
    let n = score.len();
    if n == 0 {
        return Calibrator::Identity;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| score[a].partial_cmp(&score[b]).unwrap());
    let (mut val, mut wt, mut cnt) = (Vec::<f64>::new(), Vec::<f64>::new(), Vec::<usize>::new());
    for &i in &idx {
        let (mut v, mut w, mut c) = (y[i] as f64, 1.0f64, 1usize);
        while let Some(&pv) = val.last() {
            if pv <= v + 1e-12 {
                break;
            }
            let pw = wt.pop().unwrap();
            let pc = cnt.pop().unwrap();
            val.pop();
            v = (pv * pw + v * w) / (pw + w);
            w += pw;
            c += pc;
        }
        val.push(v);
        wt.push(w);
        cnt.push(c);
    }
    let (mut xs, mut ys) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let mut pos = 0usize;
    for (b, &c) in cnt.iter().enumerate() {
        let fv = val[b] as f32;
        for k in 0..c {
            xs.push(score[idx[pos + k]]);
            ys.push(fv);
        }
        pos += c;
    }
    Calibrator::Isotonic { x: xs, y: ys }
}

/// Platt scaling: fit p = σ(a·s + b) by Newton on logloss with Platt's smoothed targets.
pub fn fit_platt(score: &[f32], y: &[f32]) -> Calibrator {
    let n = score.len();
    if n == 0 {
        return Calibrator::Identity;
    }
    let np = y.iter().filter(|&&v| v > 0.5).count() as f64;
    let nn = n as f64 - np;
    let (hi, lo) = ((np + 1.0) / (np + 2.0), 1.0 / (nn + 2.0));
    let t: Vec<f64> = y.iter().map(|&v| if v > 0.5 { hi } else { lo }).collect();
    let s: Vec<f64> = score.iter().map(|&v| v as f64).collect();
    let (mut a, mut b) = (1.0f64, 0.0f64);
    for _ in 0..60 {
        let (mut g0, mut g1, mut h00, mut h01, mut h11) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            let p = 1.0 / (1.0 + (-(a * s[i] + b)).exp());
            let d = p - t[i];
            g0 += d * s[i];
            g1 += d;
            let w = (p * (1.0 - p)).max(1e-9);
            h00 += w * s[i] * s[i];
            h01 += w * s[i];
            h11 += w;
        }
        h00 += 1e-9;
        h11 += 1e-9;
        let det = h00 * h11 - h01 * h01;
        if det.abs() < 1e-18 {
            break;
        }
        let da = (g0 * h11 - g1 * h01) / det;
        let db = (h00 * g1 - h01 * g0) / det;
        a -= da;
        b -= db;
        if da.abs() + db.abs() < 1e-10 {
            break;
        }
    }
    Calibrator::Platt { a, b }
}

/// Temperature scaling on logits: p = σ(logit(s)/t), `t` by golden-section search on logloss.
pub fn fit_temperature(score: &[f32], y: &[f32]) -> Calibrator {
    if score.is_empty() {
        return Calibrator::Identity;
    }
    let logits: Vec<f32> = score.iter().map(|&p| logit(p)).collect();
    let loss = |t: f32| -> f64 {
        let mut l = 0.0;
        for i in 0..score.len() {
            let p = (sigmoid(logits[i] / t) as f64).clamp(1e-9, 1.0 - 1e-9);
            l -= if y[i] > 0.5 { p.ln() } else { (1.0 - p).ln() };
        }
        l
    };
    let (mut a, mut b) = (0.05f32, 20.0f32);
    let gr = 0.618_034f32;
    let (mut c, mut d) = (b - gr * (b - a), a + gr * (b - a));
    let (mut fc, mut fd) = (loss(c), loss(d));
    for _ in 0..60 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = loss(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = loss(d);
        }
    }
    Calibrator::Temperature {
        t: ((a + b) / 2.0) as f64,
    }
}

pub fn brier(p: &[f32], y: &[f32]) -> f64 {
    let n = p.len().max(1) as f64;
    p.iter()
        .zip(y)
        .map(|(&pi, &yi)| ((pi - yi) as f64).powi(2))
        .sum::<f64>()
        / n
}

/// Expected Calibration Error over `bins` equal-width probability bins.
pub fn ece(p: &[f32], y: &[f32], bins: usize) -> f64 {
    let n = p.len();
    if n == 0 {
        return 0.0;
    }
    let mut bs = vec![(0f64, 0f64, 0usize); bins];
    for i in 0..n {
        let b = ((p[i].clamp(0.0, 0.999_999) * bins as f32) as usize).min(bins - 1);
        bs[b].0 += p[i] as f64;
        bs[b].1 += y[i] as f64;
        bs[b].2 += 1;
    }
    let mut e = 0.0;
    for (sp, sy, c) in bs {
        if c > 0 {
            e += (c as f64 / n as f64) * (sp / c as f64 - sy / c as f64).abs();
        }
    }
    e
}

fn logloss_clamped(p: &[f32], y: &[f32]) -> f64 {
    let mut l = 0.0;
    for (&pi, &yi) in p.iter().zip(y) {
        let pi = (pi as f64).clamp(1e-9, 1.0 - 1e-9);
        l -= if yi > 0.5 { pi.ln() } else { (1.0 - pi).ln() };
    }
    l / p.len().max(1) as f64
}

#[derive(Clone, Debug)]
pub struct CalibReport {
    pub method: String,
    pub brier: f64,
    pub ece: f64,
    pub logloss: f64,
}

/// Choose the best calibrator by **inner 3-fold-CV logloss** on the OOF predictions (so we don't pick an
/// isotonic fit that overfits small n), then refit the winner on all OOF. Isotonic only competes when
/// `n >= min_iso_n`. Returns the fitted calibrator + a report on the (held-out-calibrated) OOF.
pub fn fit_best_calibrator(oof: &[f32], y: &[f32], min_iso_n: usize) -> (Calibrator, CalibReport) {
    let n = oof.len();
    if n < 20 {
        return (
            Calibrator::Identity,
            report("identity", &Calibrator::Identity, oof, y),
        );
    }
    // candidate fitters
    type Fitter = fn(&[f32], &[f32]) -> Calibrator;
    let mut cands: Vec<(&str, Fitter)> = vec![
        ("identity", |_, _| Calibrator::Identity),
        ("platt", fit_platt),
        ("temperature", fit_temperature),
    ];
    if n >= min_iso_n {
        cands.push(("isotonic", fit_isotonic));
    }
    // 3-fold inner CV by index mod 3
    let mut best = (f64::INFINITY, 0usize);
    for (ci, (_, f)) in cands.iter().enumerate() {
        let mut tot = 0.0;
        for fold in 0..3 {
            let (mut tr_s, mut tr_y, mut va_s, mut va_y) = (vec![], vec![], vec![], vec![]);
            for i in 0..n {
                if i % 3 == fold {
                    va_s.push(oof[i]);
                    va_y.push(y[i]);
                } else {
                    tr_s.push(oof[i]);
                    tr_y.push(y[i]);
                }
            }
            let cal = f(&tr_s, &tr_y);
            tot += logloss_clamped(&cal.apply_vec(&va_s), &va_y);
        }
        if tot < best.0 {
            best = (tot, ci);
        }
    }
    let (name, fitter) = cands[best.1];
    let cal = fitter(oof, y);
    let rep = report(name, &cal, oof, y);
    (cal, rep)
}

fn report(name: &str, cal: &Calibrator, oof: &[f32], y: &[f32]) -> CalibReport {
    let p = cal.apply_vec(oof);
    CalibReport {
        method: name.into(),
        brier: brier(&p, y),
        ece: ece(&p, y, 10),
        logloss: logloss_clamped(&p, y),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::auc;

    #[test]
    fn isotonic_is_monotone_and_auc_preserving() {
        // miscalibrated scores (squashed) but monotone in y on average
        let n = 400;
        let score: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 0.3 + 0.35).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                if (i as f32 / n as f32) > 0.5 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let cal = fit_isotonic(&score, &y);
        let p = cal.apply_vec(&score);
        // monotone non-decreasing
        for w in p.windows(2) {
            assert!(w[1] + 1e-6 >= w[0]);
        }
        // AUC preserved (monotone map)
        assert!((auc(&y, &p) - auc(&y, &score)).abs() < 1e-9);
    }

    #[test]
    fn platt_improves_logloss_on_miscalibrated() {
        let n = 600;
        let y: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();
        // overconfident scores: push toward 0/1
        let score: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 0.02 } else { 0.98 })
            .collect();
        let before = logloss_clamped(&score, &y);
        let cal = fit_platt(&score, &y);
        let after = logloss_clamped(&cal.apply_vec(&score), &y);
        assert!(after <= before + 1e-9);
    }

    #[test]
    fn best_calibrator_runs() {
        let n = 300;
        let y: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();
        let score: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 0.3 } else { 0.7 }).collect();
        let (_cal, rep) = fit_best_calibrator(&score, &y, 100);
        assert!(rep.brier.is_finite() && rep.ece.is_finite());
    }
}
