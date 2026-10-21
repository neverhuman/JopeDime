// Organizer-side, post-freeze. Build and run: see README.md in the parent directory and the header of Opus5p5_Extra.contract.txt.
// Organizer-side post-freeze contract driver for the General Regressor Challenge (O6).
// Interface, state and determinism checks on tiny invented arrays; no tuning, no gate on accuracy.
use submission::{Regressor, RegressorApi};
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn table(n: usize, p: usize, seed: u64, nan_rate: f64) -> (Vec<f64>, Vec<f64>) {
    // returns (train table with target last, feature-only matrix)
    let mut r = Lcg(seed);
    let mut t = Vec::with_capacity(n * (p + 1));
    let mut x = Vec::with_capacity(n * p);
    for _ in 0..n {
        let mut row = Vec::with_capacity(p);
        for _ in 0..p {
            let v = r.next() * 4.0 - 2.0;
            row.push(if r.next() < nan_rate { f64::NAN } else { v });
        }
        let g = |j: usize| if j < p && !row[j].is_nan() { row[j] } else { 0.0 };
        let y = 3.0 * g(0) + (2.0 * g(1)).sin() + if g(2) > 0.5 { 2.0 } else { 0.0 } + 0.1 * (r.next() - 0.5);
        t.extend_from_slice(&row);
        t.push(y * 1000.0 + 5e4);
        x.extend_from_slice(&row);
    }
    (t, x)
}

fn close(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-10 * 1f64.max(x.abs()).max(y.abs()))
}

fn main() {
    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut check = |name: &str, ok: bool| {
        if ok { pass += 1; println!("PASS  {name}"); } else { fail += 1; println!("FAIL  {name}"); }
    };

    let (t, x) = table(300, 5, 1, 0.05);
    let p = 5usize;

    // valid finite arrays
    let mut m = Regressor::new(7);
    let r = m.fit(&t, 300, p + 1);
    check("fit valid table", r.is_ok());
    let pr = m.predict(&x, 300, p).unwrap();
    check("predict length and finite", pr.len() == 300 && pr.iter().all(|v| v.is_finite()));

    // wrong length / overflow / empty
    let mut e = Regressor::new(1);
    check("fit wrong length -> Err", e.fit(&t[..t.len() - 1], 300, p + 1).is_err());
    check("fit n_rows*n_cols overflow -> Err", e.fit(&t, usize::MAX, 2).is_err());
    check("fit n_rows = 0 -> Err", e.fit(&[], 0, 3).is_err());
    check("fit n_cols = 0 -> Err", e.fit(&[], 3, 0).is_err());
    check("predict wrong length -> Err", m.predict(&x[..x.len() - 1], 300, p).is_err());
    check("predict overflow -> Err", m.predict(&x, usize::MAX, p).is_err());

    // target column handling: prediction width is P, not C
    check("predict with width C (target placeholder) -> Err", m.predict(&t, 300, p + 1).is_err());

    // infinite / nonfinite
    let mut ti = t.clone(); ti[3] = f64::INFINITY;
    check("fit infinite feature -> Err", Regressor::new(1).fit(&ti, 300, p + 1).is_err());
    let mut tn = t.clone(); tn[p] = f64::NAN;
    check("fit NaN target -> Err", Regressor::new(1).fit(&tn, 300, p + 1).is_err());
    let mut tin = t.clone(); tin[p] = f64::NEG_INFINITY;
    check("fit infinite target -> Err", Regressor::new(1).fit(&tin, 300, p + 1).is_err());
    let mut xi = x.clone(); xi[7] = f64::NEG_INFINITY;
    check("predict infinite feature -> Err", m.predict(&xi, 300, p).is_err());

    // one-column table (zero features)
    let ys = [1.0, 2.0, 6.0];
    let mut z = Regressor::new(3);
    check("fit one-column table", z.fit(&ys, 3, 1).is_ok());
    let zp = z.predict(&[], 4, 0).unwrap();
    check("zero-feature predict returns n_rows mean values", zp.len() == 4 && zp.iter().all(|v| (v - 3.0).abs() < 1e-12));
    check("zero-feature predict n_rows=0 -> empty", z.predict(&[], 0, 0).map(|v| v.is_empty()).unwrap_or(false));

    // single row, two rows
    let mut s1 = Regressor::new(3);
    check("fit single-row table", s1.fit(&[1.0, f64::NAN, 4.5], 1, 3).is_ok());
    check("single-row predict finite", s1.predict(&[0.0, 1.0, f64::NAN, f64::NAN], 2, 2).map(|v| v.len() == 2 && v.iter().all(|a| a.is_finite())).unwrap_or(false));
    let mut s2 = Regressor::new(3);
    check("fit two-row table", s2.fit(&[1.0, 0.0, 2.0, -1.0], 2, 2).is_ok() && s2.predict(&[0.5], 1, 1).map(|v| v[0].is_finite()).unwrap_or(false));

    // constant target
    let mut ct = t.clone();
    for i in 0..300 { ct[i * (p + 1) + p] = -7.25; }
    let mut c = Regressor::new(1);
    check("constant target fits", c.fit(&ct, 300, p + 1).is_ok());
    check("constant target predicts the constant", c.predict(&x, 300, p).map(|v| v.iter().all(|a| *a == -7.25)).unwrap_or(false));

    // NaN features: whole column and whole row missing, constant + repeated columns, repeated rows, signed zeros
    let n = 120;
    let mut tt = Vec::new();
    let mut r = Lcg(9);
    for i in 0..n {
        let a = r.next();
        let row = [f64::NAN, 4.0, a, a, if i % 2 == 0 { 0.0 } else { -0.0 }];
        let rr: Vec<f64> = if i % 10 == 0 { vec![f64::NAN; 5] } else { row.to_vec() };
        tt.extend_from_slice(&rr);
        tt.push(2.0 * a + 1.0);
    }
    for _ in 0..20 { let first: Vec<f64> = tt[6..12].to_vec(); tt.extend_from_slice(&first); }
    let mut d = Regressor::new(11);
    check("degenerate columns / missing rows / repeated rows fit", d.fit(&tt, n + 20, 6).is_ok());
    let q = [f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, 1.0, 4.0, 0.3, 0.3, -0.0];
    check("degenerate predict finite", d.predict(&q, 2, 5).map(|v| v.iter().all(|a| a.is_finite())).unwrap_or(false));

    // lifecycle
    let un = Regressor::new(5);
    check("predict before fit -> Err", un.predict(&x, 300, p).is_err());
    check("predict before fit (empty batch) -> Err", un.predict(&[], 0, p).is_err());
    let mut lf = Regressor::new(5);
    lf.fit(&t, 300, p + 1).unwrap();
    check("refit with invalid input -> Err", lf.fit(&tn, 300, p + 1).is_err());
    check("predict after failed refit -> Err", lf.predict(&x, 300, p).is_err());
    check("predict different width -> Err", m.predict(&x, 250, 6).is_err() && m.predict(&x[..1200], 300, 4).is_err());
    check("zero-row prediction -> empty", m.predict(&[], 0, p).map(|v| v.is_empty()).unwrap_or(false));

    // determinism and batch invariance
    let mut m2 = Regressor::new(7);
    m2.fit(&t, 300, p + 1).unwrap();
    let pr2 = m2.predict(&x, 300, p).unwrap();
    check("same seed, fresh instance -> identical predictions", pr == pr2);
    let pr3 = m.predict(&x, 300, p).unwrap();
    check("repeated predict identical", pr == pr3);
    let mut batched = Vec::new();
    for (a, b) in [(0usize, 1usize), (1, 64), (64, 65), (65, 200), (200, 300)] {
        batched.extend(m.predict(&x[a * p..b * p], b - a, p).unwrap());
    }
    check("full vs batched predictions within tolerance", close(&pr, &batched));
    let perm: Vec<usize> = (0..300).map(|i| (i * 7 + 3) % 300).collect();
    let mut xp = Vec::new();
    for &i in &perm { xp.extend_from_slice(&x[i * p..(i + 1) * p]); }
    let pp = m.predict(&xp, 300, p).unwrap();
    let expect: Vec<f64> = perm.iter().map(|&i| pr[i]).collect();
    check("permuted rows -> permuted outputs", close(&pp, &expect));
    let (t2, x2) = table(200, 3, 2, 0.0);
    let mut refit = Regressor::new(7);
    refit.fit(&t, 300, p + 1).unwrap();
    refit.fit(&t2, 200, 4).unwrap();
    let mut fresh = Regressor::new(7);
    fresh.fit(&t2, 200, 4).unwrap();
    check("refit on second task == fresh instance", refit.predict(&x2, 200, 3).unwrap() == fresh.predict(&x2, 200, 3).unwrap());
    let other = { let mut o = Regressor::new(8); o.fit(&t, 300, p + 1).unwrap(); o.predict(&x, 300, p).unwrap() };
    check("different seed still finite", other.iter().all(|v| v.is_finite()));

    println!("\ncontract checks: {pass} passed, {fail} failed");

    // informational smoke (not an admission gate): fit/predict wall time and held-out RMSE vs mean
    for &(n, p, nan) in &[(2_000usize, 8usize, 0.0f64), (20_000, 20, 0.02), (100_000, 10, 0.0)] {
        let (tr, _) = table(n, p, 100 + n as u64, nan);
        let (te_t, te_x) = table(n / 4, p, 200 + n as u64, nan);
        let mut mm = Regressor::new(42);
        let t0 = Instant::now();
        mm.fit(&tr, n, p + 1).unwrap();
        let fit_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let ph = mm.predict(&te_x, n / 4, p).unwrap();
        let pred_s = t1.elapsed().as_secs_f64();
        let yt: Vec<f64> = (0..n / 4).map(|i| te_t[i * (p + 1) + p]).collect();
        let ybar = (0..n).map(|i| tr[i * (p + 1) + p]).sum::<f64>() / n as f64;
        let rmse = (ph.iter().zip(&yt).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / yt.len() as f64).sqrt();
        let rmse0 = (yt.iter().map(|b| (ybar - b).powi(2)).sum::<f64>() / yt.len() as f64).sqrt();
        println!("smoke n={n:>7} p={p:>3}: fit {fit_s:7.2}s  predict {pred_s:6.3}s  rmse {rmse:9.3}  mean-baseline {rmse0:9.3}");
    }
    if fail > 0 { std::process::exit(1); }
}
