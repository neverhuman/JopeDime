use super::eval::{evaluate, exact_scan_count, fit_of, sketch, Eval};
use super::random::{crossover, grow, mutate};
use super::types::{GpConfig, GpData, SurvivorRec};
use crate::canary::{self, CanarySpec};
use crate::dsl::Expr;
use crate::metrics::standardize;
use crate::progress::{Event, Sink, WinnerFeature};
use crate::rng::{seeded, Rng, Rng64};
use rayon::prelude::*;
use std::time::Instant;

/// Run Phase 1. Returns the (formula-deduped) survivor universe, sorted by signal descending.
pub fn run_gp(gd: &GpData, cfg: &GpConfig, sink: &dyn Sink, t0: Instant) -> Vec<SurvivorRec> {
    let n = gd.x.n;
    let c = gd.x.c;
    let maxd = cfg.depth;
    let tau = cfg.tau;
    let mu = cfg.mu;
    let nisl = cfg.islands.max(1);
    let isl_pop = (cfg.pop / nisl).max(4);
    let msub = cfg.subsample.min(n);
    let m_fit = cfg.subsample_fit;
    let gain_floor = 0.00002f32;
    let migrants = (isl_pop / 20).max(2);

    let icfg: Vec<(f32, f32)> = (0..nisl)
        .map(|i| match i % 3 {
            0 => (tau, 0.70),
            1 => (tau, 0.50),
            _ => ((tau * 0.6).max(0.40), 0.85),
        })
        .collect();

    let mut rng = seeded(cfg.seed);
    let initd = maxd.saturating_sub(1).max(1);
    let mut islands: Vec<Vec<Expr>> = (0..nisl)
        .map(|_| (0..isl_pop).map(|_| grow(c, &mut rng, 0, initd)).collect())
        .collect();

    let mut surv: Vec<SurvivorRec> = Vec::new();
    let mut surv_sk: Vec<f32> = Vec::new(); // [K*msub]
    let mut surv_v: Vec<f32> = Vec::new(); // [K*n]
    let mut admitted = 0u64;

    {
        let mut e = Event::new(1, "start", t0.elapsed().as_secs_f64());
        e.gens = Some(cfg.gens as u64);
        e.survivors = Some(0);
        sink.emit(&e);
    }

    for g in 0..cfg.gens {
        if sink.is_cancelled() {
            break;
        }
        // Fresh fold partition every generation (anti-lock-in): a feature must generalize across
        // many different random splits over the run to keep scoring well and survive.
        let gen_folds = make_gen_folds(gd, cfg.seed, g);
        let evals: Vec<Vec<Eval>> = islands
            .iter()
            .map(|pv| {
                pv.par_iter()
                    .map(|e| {
                        evaluate(
                            e, gd, &gen_folds, &surv_sk, msub, m_fit, gain_floor, cfg.lin_w,
                        )
                    })
                    .collect()
            })
            .collect();

        let orders: Vec<Vec<usize>> = (0..nisl)
            .map(|k| {
                let tau_k = icfg[k].0;
                let mut o: Vec<usize> = (0..islands[k].len()).collect();
                o.sort_unstable_by(|&a, &b| {
                    fit_of(&evals[k][b], tau_k, mu)
                        .partial_cmp(&fit_of(&evals[k][a], tau_k, mu))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                o
            })
            .collect();

        let (bk, bi) = best_expr(&orders, &evals, &icfg, mu);
        {
            let mut e = Event::new(1, "gen", t0.elapsed().as_secs_f64());
            e.gen = Some(g as u64);
            e.gens = Some(cfg.gens as u64);
            e.survivors = Some(surv.len() as u64);
            e.admitted = Some(admitted);
            e.best_signal = Some(evals[bk][bi].gain as f64);
            e.best_maxcorr = Some(evals[bk][bi].maxcorr as f64);
            let winners = live_winners(&surv, &islands, &evals, &orders, 5, gd, gain_floor, tau);
            if !winners.is_empty() {
                e.winners = Some(winners);
            }
            sink.emit(&e);
        }

        admit_survivors(
            &islands,
            &evals,
            &orders,
            gd,
            cfg,
            &mut surv,
            &mut surv_sk,
            &mut surv_v,
            &mut admitted,
            gain_floor,
            tau,
            n,
        );

        if g == cfg.gens - 1 {
            break;
        }
        islands = next_generation(
            islands, &orders, &evals, &icfg, &mut rng, cfg, initd, maxd, c, migrants, mu, g,
        );
    }

    let out = dedupe_survivors(&surv);
    {
        let mut e = Event::new(1, "done", t0.elapsed().as_secs_f64());
        e.survivors = Some(out.len() as u64);
        e.admitted = Some(admitted);
        let winners = top_clean_winners(&out, 5, &gd.canary_specs);
        if !winners.is_empty() {
            e.winners = Some(winners);
        }
        sink.emit(&e);
    }
    out
}

/// Build this generation's CV folds. Legacy (cv_k<=1): the single fixed holdout slice. Otherwise
/// reshuffle the subsample with a per-generation seed and round-robin into `cv_k` disjoint folds,
/// so the fold partition differs every generation (features can't lock onto one split).
fn make_gen_folds(gd: &GpData, seed: u64, g: usize) -> Vec<Vec<usize>> {
    // n-adaptive: keep each fold >= MIN_FOLD rows so the per-fold signal is reliable. On tiny data
    // this collapses to k=1 (legacy single slice, zero effective penalty) rather than scoring on
    // noise -- small folds otherwise turn the consistency penalty into random damage (regression n<~200).
    const MIN_FOLD: usize = 64;
    let k = gd.cv_k.min(16).min((gd.sub.len() / MIN_FOLD).max(1));
    if k <= 1 {
        return vec![gd.holdout.clone()];
    }
    let mut idx = gd.sub.clone();
    let mut rng = seeded(seed ^ 0xF01D_5EED ^ (g as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    for i in (1..idx.len()).rev() {
        let j = rng.gen_range(0..=i);
        idx.swap(i, j);
    }
    let mut folds: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &v) in idx.iter().enumerate() {
        folds[i % k].push(v);
    }
    folds
}

#[derive(Debug)]
struct WinnerCandidate {
    formula: String,
    lift: f64,
    stability: f64,
    canary: bool,
}

/// Top-`k` clean survivors as final cockpit/SDK winner features: sorted by `signal` desc, deduped
/// by formula. `lift` = signal, `stability` = raw target correlation. Canary refs are skipped
/// defensively here; survivor admission already keeps them out of the downstream model path.
fn top_clean_winners(
    surv: &[SurvivorRec],
    k: usize,
    canary_specs: &[CanarySpec],
) -> Vec<WinnerFeature> {
    let mut candidates = Vec::with_capacity(surv.len().min(k));
    for s in surv {
        if expr_refs_canary(&s.expr, canary_specs) {
            continue;
        }
        candidates.push(WinnerCandidate {
            formula: s.formula.clone(),
            lift: s.signal as f64,
            stability: s.target_corr as f64,
            canary: false,
        });
    }
    collect_winners(candidates, k)
}

/// Live phase-1 winners. Clean entries come from already-admitted survivors; canary-contaminated
/// entries are read from the current population only and flagged for the cockpit purge animation.
/// They are never admitted into `surv`, and `build_augmented_from_search` keeps a second quarantine
/// guard before model selection.
#[allow(clippy::too_many_arguments)]
fn live_winners(
    surv: &[SurvivorRec],
    islands: &[Vec<Expr>],
    evals: &[Vec<Eval>],
    orders: &[Vec<usize>],
    k: usize,
    gd: &GpData,
    gain_floor: f32,
    tau: f32,
) -> Vec<WinnerFeature> {
    let mut candidates = Vec::with_capacity(surv.len().min(k) + k);
    for s in surv {
        if expr_refs_canary(&s.expr, &gd.canary_specs) {
            continue;
        }
        candidates.push(WinnerCandidate {
            formula: s.formula.clone(),
            lift: s.signal as f64,
            stability: s.target_corr as f64,
            canary: false,
        });
    }
    if !gd.canary_specs.is_empty() {
        for kidx in 0..orders.len() {
            for &i in orders[kidx].iter().take((islands[kidx].len() / 3).max(1)) {
                let ev = &evals[kidx][i];
                if ev.gain < gain_floor || ev.lib_corr >= tau {
                    continue;
                }
                if !expr_refs_canary(&islands[kidx][i], &gd.canary_specs) {
                    continue;
                }
                candidates.push(WinnerCandidate {
                    formula: islands[kidx][i].format(&gd.names),
                    lift: ev.gain as f64,
                    stability: ev.target_corr as f64,
                    canary: true,
                });
            }
        }
    }
    collect_winners(candidates, k)
}

fn collect_winners(mut candidates: Vec<WinnerCandidate>, k: usize) -> Vec<WinnerFeature> {
    candidates.sort_unstable_by(|a, b| {
        b.lift
            .partial_cmp(&a.lift)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.formula.cmp(&b.formula))
    });
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for c in candidates {
        if out.len() >= k {
            break;
        }
        if seen.insert(c.formula.clone()) {
            out.push(WinnerFeature {
                formula: c.formula,
                lift: c.lift,
                stability: c.stability,
                canary: c.canary,
            });
        }
    }
    out
}

fn expr_refs_canary(expr: &Expr, canary_specs: &[CanarySpec]) -> bool {
    if canary_specs.is_empty() {
        return false;
    }
    let mut refs = Vec::new();
    expr.col_refs(&mut refs);
    canary::expr_refs_canary(&refs, canary_specs)
}

fn best_expr(
    orders: &[Vec<usize>],
    evals: &[Vec<Eval>],
    icfg: &[(f32, f32)],
    mu: f32,
) -> (usize, usize) {
    let (mut bk, mut bi, mut bf) = (0usize, 0usize, f32::NEG_INFINITY);
    for k in 0..orders.len() {
        let i = orders[k][0];
        let f = fit_of(&evals[k][i], icfg[k].0, mu);
        if f > bf {
            bf = f;
            bk = k;
            bi = i;
        }
    }
    (bk, bi)
}

#[allow(clippy::too_many_arguments)]
fn admit_survivors(
    islands: &[Vec<Expr>],
    evals: &[Vec<Eval>],
    orders: &[Vec<usize>],
    gd: &GpData,
    cfg: &GpConfig,
    surv: &mut Vec<SurvivorRec>,
    surv_sk: &mut Vec<f32>,
    surv_v: &mut Vec<f32>,
    admitted: &mut u64,
    gain_floor: f32,
    tau: f32,
    n: usize,
) {
    let mut cand: Vec<(usize, usize)> = Vec::new();
    for k in 0..orders.len() {
        for &i in orders[k].iter().take((islands[k].len() / 3).max(1)) {
            let ev = &evals[k][i];
            if ev.gain < gain_floor || ev.lib_corr >= tau {
                continue;
            }
            if expr_refs_canary(&islands[k][i], &gd.canary_specs) {
                continue;
            }
            cand.push((k, i));
        }
    }
    cand.sort_unstable_by(|&(ak, ai), &(bk, bi)| {
        evals[bk][bi]
            .gain
            .partial_cmp(&evals[ak][ai].gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let scans: Vec<(Vec<f32>, usize)> = cand
        .par_iter()
        .map(|&(k, i)| {
            let (vstd, ok) = standardize(&islands[k][i].eval(&gd.x));
            if !ok {
                return (Vec::new(), usize::MAX);
            }
            let cnt0 = exact_scan_count(&vstd, surv_v, n, tau);
            (vstd, cnt0)
        })
        .collect();
    let mut new_v: Vec<f32> = Vec::new();
    for (ci, &(k, i)) in cand.iter().enumerate() {
        if surv.len() >= cfg.max_surv {
            break;
        }
        let (vstd, cnt0) = &scans[ci];
        if *cnt0 != 0 || vstd.is_empty() {
            continue;
        }
        if exact_scan_count(vstd, &new_v, n, tau) != 0 {
            continue;
        }
        let ev = &evals[k][i];
        let e = &islands[k][i];
        surv.push(SurvivorRec {
            expr: e.clone(),
            formula: e.format(&gd.names),
            signal: ev.gain,
            proxy_gain: ev.sg,
            lin_corr: ev.lc,
            maxcorr_library: ev.lib_corr,
            target_corr: ev.target_corr,
            depth: e.depth(),
            nodes: ev.nodes,
        });
        surv_v.extend_from_slice(vstd);
        surv_sk.extend_from_slice(&sketch(vstd, &gd.sub));
        new_v.extend_from_slice(vstd);
        *admitted += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn next_generation(
    islands: Vec<Vec<Expr>>,
    orders: &[Vec<usize>],
    evals: &[Vec<Eval>],
    icfg: &[(f32, f32)],
    rng: &mut Rng64,
    cfg: &GpConfig,
    initd: usize,
    maxd: usize,
    c: usize,
    migrants: usize,
    mu: f32,
    generation: usize,
) -> Vec<Vec<Expr>> {
    let nisl = islands.len();
    let mig = generation > 0 && generation.is_multiple_of(cfg.migrate_every);
    let mut next: Vec<Vec<Expr>> = Vec::with_capacity(nisl);
    for k in 0..nisl {
        let (tau_k, xover_k) = icfg[k];
        let evk = &evals[k];
        let ork = &orders[k];
        let np_target = islands[k].len();
        let elite = (np_target / 20).max(1);
        let mut np: Vec<Expr> = ork
            .iter()
            .take(elite)
            .map(|&i| islands[k][i].clone())
            .collect();
        if mig {
            let src = (k + nisl - 1) % nisl;
            for &i in orders[src].iter().take(migrants) {
                np.push(islands[src][i].clone());
            }
        }
        while np.len() < np_target {
            let ai = tournament(ork, evk, tau_k, mu, rng);
            let pa = &islands[k][ai];
            let child = if rng.gen_bool(xover_k as f64) {
                let bi2 = tournament(ork, evk, tau_k, mu, rng);
                crossover(pa, &islands[k][bi2], rng, maxd)
            } else if rng.gen_bool(0.8) {
                mutate(pa, c, rng, maxd)
            } else {
                grow(c, rng, 0, initd)
            };
            np.push(child);
        }
        next.push(np);
    }
    next
}

fn dedupe_survivors(surv: &[SurvivorRec]) -> Vec<SurvivorRec> {
    let mut order: Vec<usize> = (0..surv.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        surv[b]
            .signal
            .partial_cmp(&surv[a].signal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for &i in &order {
        if seen.insert(surv[i].formula.clone()) {
            out.push(surv[i].clone());
        }
    }
    out
}

fn tournament(ork: &[usize], evk: &[Eval], tau_k: f32, mu: f32, rng: &mut Rng64) -> usize {
    let mut b = ork[rng.gen_range(0..ork.len())];
    for _ in 0..6 {
        let cidx = ork[rng.gen_range(0..ork.len())];
        if fit_of(&evk[cidx], tau_k, mu) > fit_of(&evk[b], tau_k, mu) {
            b = cidx;
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canary::{CanaryProfile, CanarySpec};

    fn spec() -> CanarySpec {
        CanarySpec {
            version: 1,
            name: "CANARY_FEATURE_0000__x".into(),
            source_name: "x".into(),
            source_index: 0,
            canary_index: 0,
            search_width: 2,
            seed: 7,
            profile: CanaryProfile::Continuous {
                missing_rate: 0.0,
                quantiles: vec![0.0, 1.0],
            },
        }
    }

    fn survivor(expr: Expr, formula: &str, signal: f32) -> SurvivorRec {
        SurvivorRec {
            expr,
            formula: formula.into(),
            signal,
            proxy_gain: signal,
            lin_corr: 0.0,
            maxcorr_library: 0.0,
            target_corr: signal,
            depth: 1,
            nodes: 1,
        }
    }

    fn eval(gain: f32, target_corr: f32) -> Eval {
        Eval {
            gain,
            sg: gain,
            lc: 0.0,
            lib_corr: 0.0,
            maxcorr: 0.0,
            nodes: 1,
            target_corr,
            valid: true,
        }
    }

    #[test]
    fn live_winners_emit_canary_candidates_with_flag() {
        let canary = spec();
        let surv = vec![
            survivor(Expr::Col(0), "clean_a", 0.50),
            survivor(Expr::Col(1), "clean_b", 0.40),
        ];
        let gd = GpData {
            x: crate::data::Matrix::new(vec![0.0; 9], 3, 3),
            names: vec!["x".into(), "y".into(), "CANARY_FEATURE_0000__x".into()],
            r: vec![0.0; 3],
            r_std: vec![0.0; 3],
            r_sse: 0.0,
            y_std: vec![0.0; 3],
            lib_std: Vec::new(),
            sub: vec![0, 1, 2],
            holdout: vec![0, 1, 2],
            cv_k: 1,
            cv_penalty: 0.0,
            canary_specs: vec![canary],
        };
        let islands = vec![vec![Expr::Col(2)]];
        let evals = vec![vec![eval(0.90, 0.31)]];
        let orders = vec![vec![0]];

        let winners = live_winners(&surv, &islands, &evals, &orders, 2, &gd, 0.00002, 0.5);

        assert_eq!(
            winners
                .iter()
                .map(|w| w.formula.as_str())
                .collect::<Vec<_>>(),
            vec!["CANARY_FEATURE_0000__x", "clean_a"]
        );
        assert!(winners[0].canary);
        assert!(!winners[1].canary);
    }

    #[test]
    fn final_top_winners_skip_canary_refs_and_backfill() {
        let canary = spec();
        let surv = vec![
            survivor(Expr::Col(2), "CANARY_FEATURE_0000__x", 0.90),
            survivor(Expr::Col(0), "clean_a", 0.50),
            survivor(Expr::Col(1), "clean_b", 0.40),
        ];

        let winners = top_clean_winners(&surv, 2, &[canary]);

        assert_eq!(
            winners
                .iter()
                .map(|w| w.formula.as_str())
                .collect::<Vec<_>>(),
            vec!["clean_a", "clean_b"]
        );
        assert!(winners.iter().all(|w| !w.canary));
    }
}
