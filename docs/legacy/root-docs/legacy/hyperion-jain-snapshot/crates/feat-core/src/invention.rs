//! Phase 6 — INVENT: evolutionary search over feat-math pipeline genomes.
//!
//! Runs after selection (folds + champion columns exist) and before the phase-5 model comparison.
//! Every candidate is EXECUTED by the feat-math Rust interpreter; the search streams live phase-6
//! events (per-generation stats + full emitted Python on every improvement + token credit). After
//! the run, the top-3 distinct genomes get PROPER fold-pure CV on the pipeline's own folds via
//! `build_oof_member`, so they land in the leaderboard and CV heatmap like any other member — but
//! they are deliberately NOT fuser/GES candidates (their artifact is the emitted standalone code,
//! not a servable native model).

use crate::backend::TrainSpec;
use crate::data::Matrix;
use crate::ensemble::build_oof_member;
use crate::learners::math::MathBackend;
use crate::metrics::{metric_name, task_score};
use crate::progress::{Event, Sink};
use crate::task::TaskKind;
use feat_math::evolve::{self, EvolveConfig, GenStats, Observer, Scored};
use feat_math::transpile::Emitted;
use std::collections::HashSet;
use std::sync::Mutex;
use std::time::Instant;

/// How many leaders get an HONEST real-head OOF CV *during* the run (for a truthful live
/// "best algorithm" score). A real-head OOF CV is EXPENSIVE (~seconds each — the genome's real
/// gpr/boost head with its own params), so keep this small; the search itself stays fast on the
/// ridge surrogate.
const HONEST_LIVE_BUDGET: usize = 4;
/// Cumulative wall-clock cap on the in-run honest CVs (belt-and-suspenders with the count budget):
/// once exceeded, the live champion just holds and the search runs on unimpeded.
const HONEST_LIVE_SECS: f64 = 45.0;
/// How many top surrogate genomes the end-of-run finalist gate scans (widened from 12 so the
/// genuinely-best real-head genome isn't gate-kept out by a weak surrogate ranking).
const FINAL_SCAN_TOP_K: usize = 16;
/// Max distinct finalists kept for the leaderboard + champion pick (also the scan's early-exit).
const MAX_FINALISTS: usize = 5;
/// Wall-clock cap on the end-of-run honest finalist gate — so a few pathologically slow real
/// heads can never hang the stage; it stops and ships the best of whatever it honestly tested.
const FINAL_GATE_SECS: f64 = 75.0;

/// Invention budget. `gens == 0` disables the stage entirely.
#[derive(Clone, Copy, Debug)]
pub struct InventionConfig {
    pub gens: usize,
    pub pop: usize,
}

impl Default for InventionConfig {
    fn default() -> Self {
        InventionConfig { gens: 0, pop: 64 }
    }
}

#[derive(Clone, Debug)]
pub struct InventedFinalist {
    /// Leaderboard/member name (`math_<hash8>`).
    pub name: String,
    /// Canonical genome DSL.
    pub dsl: String,
    /// Proper fold-pure CV mean (task metric) from build_oof_member.
    pub cv_mean: f64,
    /// Per-fold CV scores (the CV-heatmap row).
    pub fold_scores: Vec<f64>,
    /// Holdout score of the full-train refit.
    pub holdout: f64,
    /// Holdout predictions of the full-train refit (row order = hold CSV).
    pub holdout_preds: Vec<f32>,
    /// Full standalone Python artifact.
    pub python: String,
    /// Full standalone Rust artifact.
    pub rust: String,
}

#[derive(Clone, Debug, Default)]
pub struct InventionReport {
    /// Finalists ranked by fold-pure CV score, best first.
    pub finalists: Vec<InventedFinalist>,
    pub evals: u64,
    pub tokens_total: f64,
}

/// Row cap for the evolution's inner CV — keeps each genome eval fast; finalists get proper CV on
/// the full selection frame afterwards.
const SEARCH_ROWS: usize = 1500;

fn fnv_hash8(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")[..8].to_string()
}

fn rank_finalists_by_cv(finalists: &mut [InventedFinalist]) {
    finalists.sort_by(|a, b| {
        b.cv_mean
            .total_cmp(&a.cv_mean)
            .then_with(|| a.name.cmp(&b.name))
    });
}

/// Honestly OOF-CV one genome's REAL head on the full selection frame and package it as a
/// finalist. This is the SAME estimator the end-of-run gate uses — reused so the live "best
/// algorithm" score and the final champion come from one source of truth (not the ridge surrogate).
#[allow(clippy::too_many_arguments)]
fn honest_finalist(
    dsl: &str,
    x: &Matrix,
    y: &[f32],
    xh: &Matrix,
    yh: &[f32],
    cols: &[u32],
    folds: &[(Vec<usize>, Vec<usize>)],
    cv_spec: &TrainSpec,
    ho_spec: &TrainSpec,
    task: TaskKind,
    n_classes: usize,
) -> Option<InventedFinalist> {
    let be = MathBackend::from_dsl(dsl)?;
    let name = format!("math_{}", fnv_hash8(dsl));
    let member = build_oof_member(
        &name, &be, x, y, xh, folds, cols, cv_spec, ho_spec, task, n_classes,
    );
    let holdout = task_score(task, yh, &member.infer, n_classes);
    let (python, rust) = feat_math::GenomeIr::parse(dsl)
        .ok()
        .and_then(|g| feat_math::transpile::emit(&g, &Default::default()).ok())
        .map(|e| (e.python, e.rust))
        .unwrap_or_default();
    Some(InventedFinalist {
        name,
        dsl: dsl.to_string(),
        cv_mean: member.mean_auc,
        fold_scores: member.fold_auc,
        holdout,
        holdout_preds: member.infer,
        python,
        rust,
    })
}

/// Live honest-champion tracker: the best genome whose REAL head has been honestly OOF-CV'd so
/// far, plus the deterministic gen-stride throttle/budget and the memo of every honest CV (reused
/// by the final gate so no genome is CV'd twice).
struct HonestState {
    /// Generation of the last honest CV — the throttle is by GENERATION (deterministic), never
    /// wall-clock, so which leaders get honestly tested is reproducible run-to-run.
    last_gen: Option<usize>,
    budget: usize,
    /// Cumulative seconds spent on in-run honest CVs (wall-clock cap).
    spent_secs: f64,
    /// Best-real-AUC finalist so far — what the live "best algorithm" headline shows.
    champ: Option<InventedFinalist>,
    /// Every genome honestly CV'd during the run (deduped) — reused by the end-of-run gate.
    memo: Vec<InventedFinalist>,
    seen: HashSet<String>,
}

/// Bridge from the evolve observer to the jain progress sink (phase-6 events).
struct SinkBridge<'a> {
    sink: &'a dyn Sink,
    t0: Instant,
    metric: &'static str,
    gens: usize,
    /// Final (evals, tokens_total) captured from the last generation for the report.
    totals: Mutex<(u64, f64)>,
    // --- Honest real-head CV of the surrogate's leaders DURING the run, so the live score is the
    // real tested AUC (not the ridge surrogate) and the final pick reuses these honest evals. ---
    honest: Mutex<HonestState>,
    /// Run an honest CV at most once every this-many generations (≈ gens / budget).
    gen_stride: usize,
    x: &'a Matrix,
    y: &'a [f32],
    xh: &'a Matrix,
    yh: &'a [f32],
    cols: &'a [u32],
    folds: &'a [(Vec<usize>, Vec<usize>)],
    cv_spec: &'a TrainSpec,
    ho_spec: &'a TrainSpec,
    task: TaskKind,
    n_classes: usize,
}

impl<'a> SinkBridge<'a> {
    fn base(&self, event: &str, s: &GenStats) -> Event {
        let mut e = Event::new(6, event, self.t0.elapsed().as_secs_f64());
        e.gen = Some(s.gen as u64);
        e.gens = Some(self.gens as u64);
        e.done_frac = Some(((s.gen + 1) as f64 / self.gens.max(1) as f64).min(1.0));
        e.evals = Some(s.evals);
        e.evals_per_sec = Some(s.evals_per_sec);
        e.tokens_total = Some(s.tokens_total);
        e.tokens_per_sec = Some(s.tokens_per_sec);
        // The live "best algorithm" score is the HONEST real-head OOF CV of the current champion,
        // NOT the ridge-surrogate best (`s.best_cv`) — so the number the user watches is the real
        // tested AUC and matches the final `done` champion. `None` until the first honest CV lands.
        if let Some(c) = self.honest.lock().unwrap().champ.as_ref() {
            e.best_val_auc = Some(c.cv_mean);
            e.model = Some(c.name.clone());
        }
        e.metric = Some(self.metric.to_string());
        e
    }

    /// If the throttle + budget allow, honestly CV this leader's REAL head and promote it to the
    /// live champion when it is the best real-AUC seen. The throttle is by GENERATION (deterministic).
    fn maybe_honest_cv(&self, gen: usize, dsl: &str) {
        // Decide under a short lock; never hold the lock during the (seconds-long) real-head CV.
        {
            let mut h = self.honest.lock().unwrap();
            if h.budget == 0 || h.spent_secs > HONEST_LIVE_SECS || h.seen.contains(dsl) {
                return;
            }
            let due = h.last_gen.is_none_or(|g| gen >= g + self.gen_stride);
            if !due {
                return;
            }
            h.last_gen = Some(gen);
            h.budget -= 1;
            h.seen.insert(dsl.to_string());
        }
        let started = Instant::now();
        let done = honest_finalist(
            dsl,
            self.x,
            self.y,
            self.xh,
            self.yh,
            self.cols,
            self.folds,
            self.cv_spec,
            self.ho_spec,
            self.task,
            self.n_classes,
        );
        let dt = started.elapsed().as_secs_f64();
        let mut h = self.honest.lock().unwrap();
        h.spent_secs += dt;
        if let Some(f) = done {
            if h.champ.as_ref().is_none_or(|c| f.cv_mean > c.cv_mean) {
                h.champ = Some(f.clone());
            }
            h.memo.push(f);
        }
    }
}

impl<'a> Observer for SinkBridge<'a> {
    fn on_generation(&self, s: &GenStats) {
        let mut e = self.base("gen", s);
        // Downsample the population fitness for the live visuals (≤48 values, sorted desc).
        let mut pf = s.pop_fitness.clone();
        if pf.len() > 48 {
            let step = pf.len() as f64 / 48.0;
            pf = (0..48).map(|i| pf[(i as f64 * step) as usize]).collect();
        }
        e.pop_fitness = Some(pf);
        self.sink.emit(&e);
        *self.totals.lock().unwrap() = (s.evals, s.tokens_total);
    }
    fn on_improvement(&self, best: &Scored, emitted: &Emitted, s: &GenStats) {
        // Honestly test this surrogate leader's REAL head (throttled) so the live champion is the
        // real tested AUC, not the surrogate mirage.
        self.maybe_honest_cv(s.gen, &best.dsl);
        let mut e = self.base("improved", s);
        // Show the honest champion's code/genome. Until the first honest CV lands, fall back to the
        // surrogate leader's emitted code (base() leaves best_val_auc = None in that brief window).
        let champ = self.honest.lock().unwrap().champ.clone();
        if let Some(c) = champ {
            e.genome = Some(c.dsl);
            e.best_code = Some(c.python);
            e.model = Some(c.name);
        } else {
            e.genome = Some(best.dsl.clone());
            e.best_code = Some(emitted.python.clone());
            e.model = Some(format!("math_{}", fnv_hash8(&best.dsl)));
        }
        self.sink.emit(&e);
    }
    fn on_eval(&self, dsl: &str, score: f64, code: Option<&str>) {
        // Lightweight, EPHEMERAL live-churn event (WebProgressSink broadcasts it without
        // persisting): a single candidate's DSL + its score, plus (on the code cadence) its
        // emitted Python so the panel can cycle real algorithms as they are tried.
        let mut e = Event::new(6, "eval", self.t0.elapsed().as_secs_f64());
        e.genome = Some(dsl.to_string());
        e.model = Some(format!("math_{}", fnv_hash8(dsl)));
        e.eval_score = Some(score);
        if let Some(c) = code {
            e.best_code = Some(c.to_string());
        }
        self.sink.emit(&e);
    }
    fn is_cancelled(&self) -> bool {
        self.sink.is_cancelled()
    }
}

/// Run the invention stage. `x`/`xh` are the augmented train/holdout matrices; `cols` the selected
/// (champion) columns; `folds` the pipeline's fixed CV folds.
#[allow(clippy::too_many_arguments)]
pub fn run_invention(
    inv: &InventionConfig,
    x: &Matrix,
    y: &[f32],
    xh: &Matrix,
    yh: &[f32],
    cols: &[u32],
    folds: &[(Vec<usize>, Vec<usize>)],
    cv_spec: &TrainSpec,
    ho_spec: &TrainSpec,
    task: TaskKind,
    n_classes: usize,
    seed: u64,
    sink: &dyn Sink,
    t0: Instant,
) -> Option<InventionReport> {
    if inv.gens == 0 || !matches!(task, TaskKind::Binary | TaskKind::Regression) || cols.is_empty()
    {
        // Never skip silently: tell the cockpit WHY the new-math invention stage did not run so the
        // user isn't left wondering where the live algorithm search went.
        let reason = if inv.gens == 0 {
            "New-math invention is off at this effort level — raise the effort (Medium or higher) to invent standalone algorithms.".to_string()
        } else if !matches!(task, TaskKind::Binary | TaskKind::Regression) {
            format!(
                "New-math invention supports binary or numeric (regression) targets; this run is multiclass ({n_classes} classes). Pick a two-class or numeric target to watch it invent algorithms."
            )
        } else {
            "New-math invention had no engineered feature columns to search over.".to_string()
        };
        let mut e = Event::new(6, "skipped", t0.elapsed().as_secs_f64());
        e.note = Some(reason);
        sink.emit(&e);
        return None;
    }
    let metric = metric_name(task);

    // --- Build the search frame: selected columns, row-subsampled (seeded, deterministic). ---
    let n = x.n;
    let take = n.min(SEARCH_ROWS);
    let mut order: Vec<usize> = (0..n).collect();
    // xorshift-style seeded shuffle (deterministic, dependency-free)
    let mut s = seed | 1;
    for i in (1..order.len()).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        order.swap(i, (s as usize) % (i + 1));
    }
    order.truncate(take);
    let rows: feat_math::Rows = order
        .iter()
        .map(|&i| cols.iter().map(|&c| x.at(i, c as usize) as f64).collect())
        .collect();
    let ys: Vec<f64> = order.iter().map(|&i| y[i] as f64).collect();

    // --- Task-aware fitness: AUC on continuous scores for binary, R² for regression. ---
    let metric_fn = move |yt: &[f64], yp: &[f64]| -> f64 {
        let yt32: Vec<f32> = yt.iter().map(|v| *v as f32).collect();
        let yp32: Vec<f32> = yp.iter().map(|v| *v as f32).collect();
        task_score(task, &yt32, &yp32, n_classes)
    };

    let mut e = Event::new(6, "start", t0.elapsed().as_secs_f64());
    e.gens = Some(inv.gens as u64);
    e.rows = Some(take as u64);
    e.cols = Some(cols.len() as u64);
    e.metric = Some(metric.to_string());
    sink.emit(&e);

    let bridge = SinkBridge {
        sink,
        t0,
        metric,
        gens: inv.gens,
        totals: Mutex::new((0, 0.0)),
        honest: Mutex::new(HonestState {
            last_gen: None,
            budget: HONEST_LIVE_BUDGET,
            spent_secs: 0.0,
            champ: None,
            memo: Vec::new(),
            seen: HashSet::new(),
        }),
        gen_stride: (inv.gens / HONEST_LIVE_BUDGET).max(1),
        x,
        y,
        xh,
        yh,
        cols,
        folds,
        cv_spec,
        ho_spec,
        task,
        n_classes,
    };
    // Hard wall-clock bound so the stage never runs away on wide/slow frames: fast datasets finish
    // all `gens` well under budget; large ones stop early and still stream + CV whatever they found.
    // Derived from the generation budget (higher effort ⇒ more gens ⇒ a longer, but capped, run).
    let time_budget_secs = (inv.gens as f64 * 2.0).clamp(30.0, 480.0);
    let cfg = EvolveConfig {
        pop: inv.pop.max(8),
        gens: inv.gens,
        folds: 3,
        seed,
        max_transforms: 6,
        time_budget_secs,
        seed_genomes: Vec::new(),
    };
    let ranked = evolve::evolve(&rows, &ys, &metric_fn, &cfg, &bridge);
    let (evals, tokens_total) = *bridge.totals.lock().unwrap();

    // --- Final champion gate: the candidate pool is the genomes ALREADY honestly CV'd during the
    // run (reused from the live honest-champion memo — no double work) PLUS a WIDENED scan of the
    // surrogate's top-K (CV'd now). The champion is the honest real-head OOF-CV max across that
    // pool — so the surrogate can't gate-keep the genuinely-best real model out of contention, and
    // the winner equals the honest score watched live. Both estimators are the SAME (real head,
    // full-frame fold-pure OOF), so the numbers are directly comparable. ---
    // Add a finalist only if it's a NEW genome AND a new (CV, holdout) signature (identical
    // performance ⇒ effectively the same model — don't waste a slot).
    fn add_distinct(
        f: InventedFinalist,
        finalists: &mut Vec<InventedFinalist>,
        seen_dsl: &mut HashSet<String>,
        seen_sig: &mut Vec<(i64, i64)>,
    ) {
        if !seen_dsl.insert(f.dsl.clone()) {
            return;
        }
        let sig = (
            (f.cv_mean * 1e4).round() as i64,
            (f.holdout * 1e4).round() as i64,
        );
        if seen_sig.contains(&sig) {
            return;
        }
        seen_sig.push(sig);
        finalists.push(f);
    }

    let memo: Vec<InventedFinalist> = std::mem::take(&mut bridge.honest.lock().unwrap().memo);
    let mut finalists: Vec<InventedFinalist> = Vec::new();
    let mut seen_dsl: HashSet<String> = HashSet::new();
    let mut seen_sig: Vec<(i64, i64)> = Vec::new();
    // (a) reuse the leaders honestly CV'd DURING the run (no recompute).
    for f in memo {
        add_distinct(f, &mut finalists, &mut seen_dsl, &mut seen_sig);
    }
    // (b) widen: honestly CV the surrogate's top genomes not already tested — bounded by a distinct
    // count AND a wall-clock guard so slow real heads can't hang the stage.
    let gate_t0 = Instant::now();
    for scored in ranked.iter().take(FINAL_SCAN_TOP_K) {
        if finalists.len() >= MAX_FINALISTS || gate_t0.elapsed().as_secs_f64() > FINAL_GATE_SECS {
            break;
        }
        if seen_dsl.contains(&scored.dsl) {
            continue;
        }
        if let Some(f) = honest_finalist(
            &scored.dsl,
            x,
            y,
            xh,
            yh,
            cols,
            folds,
            cv_spec,
            ho_spec,
            task,
            n_classes,
        ) {
            add_distinct(f, &mut finalists, &mut seen_dsl, &mut seen_sig);
        }
    }
    // Champion = the honest-CV max across the tested pool.
    rank_finalists_by_cv(&mut finalists);
    finalists.truncate(MAX_FINALISTS);
    // Stream the finalists as phase-5 leaderboard members (leaderboard rows + CV-heatmap folds).
    for f in &finalists {
        let mut e = Event::new(5, "model", t0.elapsed().as_secs_f64());
        e.model = Some(f.name.clone());
        e.metric = Some(metric.to_string());
        e.best_val_auc = Some(f.cv_mean);
        e.best_holdout_auc = Some(f.holdout);
        e.folds = Some(f.fold_scores.clone());
        e.genome = Some(f.dsl.clone());
        sink.emit(&e);
    }

    let mut e = Event::new(6, "done", t0.elapsed().as_secs_f64());
    e.done_frac = Some(1.0);
    e.metric = Some(metric.to_string());
    if let Some(best) = finalists.first() {
        e.model = Some(best.name.clone());
        e.best_val_auc = Some(best.cv_mean);
        e.best_holdout_auc = Some(best.holdout);
        e.genome = Some(best.dsl.clone());
        e.best_code = Some(best.python.clone());
    }
    sink.emit(&e);

    Some(InventionReport {
        finalists,
        evals,
        tokens_total,
    })
}

#[cfg(test)]
mod tests {
    use super::{rank_finalists_by_cv, InventedFinalist};

    fn finalist(name: &str, cv_mean: f64, holdout: f64) -> InventedFinalist {
        InventedFinalist {
            name: name.to_string(),
            dsl: format!("(pipe X ({name}))"),
            cv_mean,
            fold_scores: vec![cv_mean],
            holdout,
            holdout_preds: vec![holdout as f32],
            python: format!("# {name}\n"),
            rust: String::new(),
        }
    }

    #[test]
    fn finalist_ranking_uses_fold_pure_cv_not_gate_order_or_holdout() {
        let mut finalists = vec![
            finalist("math_a", 0.6924, 0.99),
            finalist("math_c", 0.8736, 0.50),
            finalist("math_b", 0.8100, 0.95),
        ];

        rank_finalists_by_cv(&mut finalists);

        assert_eq!(finalists[0].name, "math_c");
        assert_eq!(
            finalists
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>(),
            vec!["math_c", "math_b", "math_a"]
        );
    }

    #[test]
    fn finalist_ranking_ties_by_name_not_holdout() {
        let mut finalists = vec![
            finalist("math_z", 0.8736, 0.99),
            finalist("math_a", 0.8736, 0.10),
            finalist("math_m", 0.8100, 0.95),
        ];

        rank_finalists_by_cv(&mut finalists);

        assert_eq!(
            finalists
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>(),
            vec!["math_a", "math_z", "math_m"]
        );
    }
}
