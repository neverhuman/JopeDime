use super::*;

/// k-fold CV of one genome: full refit inside every fold, scored by `metric` (higher = better).
/// Returns None for genomes that fail to fit or produce non-finite scores.
/// Widest transform knobs are clamped during search so the transformed frame stays cheap to build.
pub(super) const SEARCH_MAX_DEGREE: i64 = 4;
pub(super) const SEARCH_MAX_N: i64 = 64;

/// The heart of the engine's speed: during search we DO NOT train each genome's real (expensive)
/// head. We replace it with a cheap closed-form RIDGE surrogate and measure how linearly-predictable
/// the genome's TRANSFORMS have made the target. That is the whole bet — the transforms' job is to
/// linearize structure (cheb/poly/rff/quantile all do exactly this), so ridge is a principled and
/// millisecond-fast fitness. Only the top-K leaders are later gated to their real head + proper CV.
/// Also clamps the widest transform knobs so building the frame stays cheap.
pub(super) fn search_surrogate(g: &DslExpr) -> DslExpr {
    let mut c = g.clone();
    for st in &mut c.stages {
        for (key, cap) in [("degree", SEARCH_MAX_DEGREE), ("n", SEARCH_MAX_N)] {
            if let Some(v) = st.params.get_mut(key) {
                if v.as_u64().map(|r| r as i64 > cap).unwrap_or(false) {
                    *v = ParamValue::Int(cap);
                }
            }
        }
    }
    match c.stages.last_mut() {
        Some(last) if HEAD_OPS.contains(&last.op.as_str()) => {
            last.op = "ridge".to_string();
            last.params.clear();
        }
        // No head present (all-transform genome) — append a ridge so it is scorable.
        _ => c.stages.push(Stage {
            op: "ridge".to_string(),
            params: BTreeMap::new(),
        }),
    }
    c
}

/// Lite head budgets for the v2 race: `(head op, param overrides)` applied by `lite_budget`.
/// One table so the knobs are tunable in one place. Heads absent from the table
/// (ridge / ridge_gcv / logistic / ksmooth / stack / ensemble / ensemble_ridge_gbm) are already
/// cheap at race row counts and pass through unchanged.
pub const LITE_HEAD_BUDGETS: [(&str, &[(&str, i64)]); 6] = [
    ("histgbm2", &[("rounds", 40), ("bags", 1)]),
    ("histgbm", &[("rounds", 25)]),
    ("gbm", &[("rounds", 25)]),
    ("gpr", &[("steps", 5), ("m", 96)]),
    ("krr", &[("m", 96)]),
    ("elasticnet", &[("sweeps", 15)]),
];

/// Lite-budget variant of a genome for rungs 1-2 of the race: KEEPS the genome's real head — the
/// whole point, since the ridge surrogate is head-blind — but caps the head's compute budget via
/// `LITE_HEAD_BUDGETS` (tuning knobs like leaves/lr/kernel are preserved) and applies the same
/// transform clamps as `search_surrogate`. A lab study showed this low-fidelity probe of the true
/// objective scores Spearman 0.731 vs honest CV (surrogate: 0.373) at a fraction of full-fit cost.
pub(super) fn lite_budget(g: &DslExpr) -> DslExpr {
    let mut c = g.clone();
    for st in &mut c.stages {
        for (key, cap) in [("degree", SEARCH_MAX_DEGREE), ("n", SEARCH_MAX_N)] {
            if let Some(v) = st.params.get_mut(key) {
                if v.as_u64().map(|r| r as i64 > cap).unwrap_or(false) {
                    *v = ParamValue::Int(cap);
                }
            }
        }
    }
    match c.stages.last_mut() {
        Some(last) if HEAD_OPS.contains(&last.op.as_str()) => {
            if let Some((_, overrides)) = LITE_HEAD_BUDGETS.iter().find(|(op, _)| *op == last.op) {
                for (key, val) in overrides.iter() {
                    last.params
                        .insert((*key).to_string(), ParamValue::Int(*val));
                }
            }
        }
        // No head present (all-transform genome) — append a ridge so it is scorable.
        _ => c.stages.push(Stage {
            op: "ridge".to_string(),
            params: BTreeMap::new(),
        }),
    }
    c
}

/// The static emitted-token estimate (Python + Rust) of each genome, in input order, computed
/// in parallel. Genomes the emitter cannot support estimate `0.0`, as the serial loops did.
pub(super) fn token_estimates<'a>(genomes: impl Iterator<Item = &'a DslExpr>) -> Vec<f64> {
    let genomes: Vec<&DslExpr> = genomes.collect();
    genomes
        .into_par_iter()
        .map(|genome| {
            crate::tokens::try_estimate_tokens(
                &GenomeIr::Pipeline(genome.clone()),
                &EmitOptions::default(),
            )
            .map(|e| (e.py + e.rs) as f64)
            .unwrap_or(0.0)
        })
        .collect()
}

pub(super) fn score_genome(
    g: &DslExpr,
    x: &Rows,
    y: &[f64],
    assign: &[usize],
    k: usize,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
) -> Option<Vec<f64>> {
    let genome = GenomeIr::Pipeline(search_surrogate(g));
    let mut fold_scores = Vec::with_capacity(k);
    for fold in 0..k {
        let (mut xtr, mut ytr, mut xte, mut yte) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..y.len() {
            if assign[i] == fold {
                xte.push(x[i].clone());
                yte.push(y[i]);
            } else {
                xtr.push(x[i].clone());
                ytr.push(y[i]);
            }
        }
        if xte.is_empty() || xtr.is_empty() {
            return None;
        }
        let m = model::fit(&genome, &xtr, &ytr).ok()?;
        let pred = m.predict(&xte).ok()?;
        let s = metric(&yte, &pred);
        if !s.is_finite() {
            return None;
        }
        fold_scores.push(s);
    }
    Some(fold_scores)
}

// B10 lexicase: begin
/// [`score_genome`], also returning every row's out-of-fold prediction, by row (`NaN` for a row
/// that is in no fold). The program, folds, fits, predictions and metric calls are
/// `score_genome`'s, in the same order, so the fold scores are the same bits.
pub(super) fn score_genome_predictions(
    g: &DslExpr,
    x: &Rows,
    y: &[f64],
    assign: &[usize],
    k: usize,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
) -> ScoredCases {
    let genome = GenomeIr::Pipeline(search_surrogate(g));
    let mut fold_scores = Vec::with_capacity(k);
    let mut predictions = vec![f64::NAN; y.len()];
    for fold in 0..k {
        let (mut xtr, mut ytr, mut xte, mut yte) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut held_out = Vec::new();
        for i in 0..y.len() {
            if assign[i] == fold {
                xte.push(x[i].clone());
                yte.push(y[i]);
                held_out.push(i);
            } else {
                xtr.push(x[i].clone());
                ytr.push(y[i]);
            }
        }
        if xte.is_empty() || xtr.is_empty() {
            return None;
        }
        let m = model::fit(&genome, &xtr, &ytr).ok()?;
        let pred = m.predict(&xte).ok()?;
        let s = metric(&yte, &pred);
        if !s.is_finite() {
            return None;
        }
        fold_scores.push(s);
        for (&row, &value) in held_out.iter().zip(&pred) {
            predictions[row] = value;
        }
    }
    Some((fold_scores, predictions))
}

/// Fold scores with a per-row vector beside them (out-of-fold predictions or residual cases), or
/// `None` where the legacy scorer gives `None`.
pub(super) type ScoredCases = Option<(Vec<f64>, Vec<f64>)>;

/// [`score_genome`], also returning the signed out-of-fold residual `y[r] - prediction` of each
/// row of `case_rows`, in that order.
pub(super) fn score_genome_cases(
    g: &DslExpr,
    x: &Rows,
    y: &[f64],
    assign: &[usize],
    k: usize,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    case_rows: &[usize],
) -> ScoredCases {
    let (fold_scores, predictions) = score_genome_predictions(g, x, y, assign, k, metric)?;
    let residuals = case_rows
        .iter()
        .map(|&row| y[row] - predictions[row])
        .collect();
    Some((fold_scores, residuals))
}

/// The surrogate search's legacy scoring of `unseen` under lexicase selection:
/// [`score_genome_cases`] in parallel, keeping each scored candidate's cases in `cases`.
#[allow(clippy::too_many_arguments)]
pub(super) fn score_unseen_with_cases(
    unseen: Vec<(String, DslExpr)>,
    x: &Rows,
    y: &[f64],
    assign: &[usize],
    k: usize,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    case_rows: &[usize],
    cases: &mut HashMap<String, Vec<f64>>,
) -> Vec<(String, DslExpr, Option<Vec<f64>>)> {
    let scored: Vec<(String, DslExpr, ScoredCases)> = unseen
        .into_par_iter()
        .map(|(key, g)| {
            let scored = score_genome_cases(&g, x, y, assign, k, metric, case_rows);
            (key, g, scored)
        })
        .collect();
    scored
        .into_iter()
        .map(|(key, g, scored)| {
            let scores = scored.map(|(scores, residuals)| {
                cases.insert(key.clone(), residuals);
                scores
            });
            (key, g, scores)
        })
        .collect()
}
// B10 lexicase: end

// B11 ridge scoring: begin
/// The certified scorer of one surrogate search: the surrogate memo and the fold-scoped stage
/// cache ([`RidgeScoring::Certified`]). Every score it returns is [`score_genome`]'s, bit for bit.
/// Its keys pin the surrogate and the stage chain, not the data: they are sound because one scorer
/// serves one search run, whose rows, labels, folds and case rows are fixed.
pub(super) struct CertifiedScoring {
    pub(super) cache: crate::stage_cache::StageCache,
    /// A surrogate's exact key ([`exact_surrogate_key`]) to its fold scores and, when lexicase
    /// cases are kept, its residuals on the run's case rows (fixed for the run).
    pub(super) memo: HashMap<Vec<u8>, ScoredCases>,
    pub(super) candidates: u64,
    pub(super) reuses: u64,
}

impl CertifiedScoring {
    pub(super) fn new() -> Self {
        Self {
            cache: crate::stage_cache::StageCache::new(CERTIFIED_STAGE_CACHE_BYTES),
            memo: HashMap::new(),
            candidates: 0,
            reuses: 0,
        }
    }

    /// Scores `unseen` as the legacy scorer does, keeping each candidate's lexicase cases on
    /// `case_rows` when given. Each distinct surrogate not yet in the memo is scored once, in
    /// parallel.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn score_unseen(
        &mut self,
        unseen: Vec<(String, DslExpr)>,
        x: &Rows,
        y: &[f64],
        assign: &[usize],
        k: usize,
        metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
        case_rows: Option<&[usize]>,
        cases: &mut HashMap<String, Vec<f64>>,
    ) -> Vec<(String, DslExpr, Option<Vec<f64>>)> {
        let surrogates: Vec<DslExpr> = unseen.iter().map(|(_, g)| search_surrogate(g)).collect();
        let exact: Vec<Vec<u8>> = surrogates.iter().map(exact_surrogate_key).collect();
        let mut queued = std::collections::HashSet::new();
        let todo: Vec<(Vec<u8>, DslExpr)> = surrogates
            .into_iter()
            .zip(&exact)
            .filter(|(_, id)| !self.memo.contains_key(*id) && queued.insert((*id).clone()))
            .map(|(surrogate, id)| (id.clone(), surrogate))
            .collect();
        let cache = &self.cache;
        let scored: Vec<(Vec<u8>, ScoredCases)> = todo
            .into_par_iter()
            .map(|(id, surrogate)| {
                let genome = GenomeIr::Pipeline(surrogate);
                let scored =
                    score_surrogate_cached(&genome, x, y, assign, k, metric, cache, case_rows);
                (id, scored)
            })
            .collect();
        let computed = scored.len() as u64;
        self.memo.extend(scored);
        self.candidates += unseen.len() as u64;
        self.reuses += unseen.len() as u64 - computed;
        unseen
            .into_iter()
            .zip(exact)
            .map(|((key, g), id)| {
                let entry = &self.memo[&id];
                if let (Some(_), Some((_, residuals))) = (case_rows, entry) {
                    cases.insert(key.clone(), residuals.clone());
                }
                let scores = entry.as_ref().map(|(scores, _)| scores.clone());
                (key, g, scores)
            })
            .collect()
    }

    pub(super) fn receipt(&self, generation: usize) -> ScoringReceipt {
        let (stage_hits, stage_misses, _) = self.cache.stats();
        ScoringReceipt {
            generation,
            candidates: self.candidates,
            surrogate_reuses: self.reuses,
            stage_hits,
            stage_misses,
        }
    }
}

/// The exact key of a surrogate: its stage count, then each stage's exact encoding (op and every
/// parameter, floats as their IEEE-754 bits). Unlike canonical text, it separates surrogates that
/// print alike but differ in any parameter bit.
pub(super) fn exact_surrogate_key(surrogate: &DslExpr) -> Vec<u8> {
    let mut key = (surrogate.stages.len() as u64).to_le_bytes().to_vec();
    for stage in &surrogate.stages {
        crate::stage_cache::write_exact_stage(stage, &mut key);
    }
    key
}

/// [`score_genome_predictions`] for an already built surrogate, each fold fitted through the
/// stage cache: the same folds, the same fits (bit-identical, [`model::fit_with_stage_cache`]),
/// predictions and metric calls. With `case_rows`, also the residuals `y - prediction` on those
/// rows, in that order, as [`score_genome_cases`] computes them; otherwise an empty vector.
#[allow(clippy::too_many_arguments)]
pub(super) fn score_surrogate_cached(
    surrogate: &GenomeIr,
    x: &Rows,
    y: &[f64],
    assign: &[usize],
    k: usize,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    cache: &crate::stage_cache::StageCache,
    case_rows: Option<&[usize]>,
) -> ScoredCases {
    let keep = case_rows.is_some();
    let mut fold_scores = Vec::with_capacity(k);
    let mut predictions = if keep {
        vec![f64::NAN; y.len()]
    } else {
        Vec::new()
    };
    for fold in 0..k {
        let (mut xtr, mut ytr, mut xte, mut yte) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut held_out = Vec::new();
        for i in 0..y.len() {
            if assign[i] == fold {
                xte.push(x[i].clone());
                yte.push(y[i]);
                held_out.push(i);
            } else {
                xtr.push(x[i].clone());
                ytr.push(y[i]);
            }
        }
        if xte.is_empty() || xtr.is_empty() {
            return None;
        }
        let ctx = crate::stage_cache::StageCacheCtx {
            context_tag: "evolve-surrogate".to_string(),
            folds: k,
            fold,
        };
        let m = model::fit_with_stage_cache(surrogate, &xtr, &ytr, cache, &ctx).ok()?;
        let pred = m.predict(&xte).ok()?;
        let s = metric(&yte, &pred);
        if !s.is_finite() {
            return None;
        }
        fold_scores.push(s);
        if keep {
            for (&row, &value) in held_out.iter().zip(&pred) {
                predictions[row] = value;
            }
        }
    }
    let residuals = case_rows.map_or_else(Vec::new, |rows| {
        rows.iter().map(|&row| y[row] - predictions[row]).collect()
    });
    Some((fold_scores, residuals))
}
// B11 ridge scoring: end

// ---------------------------------------------------------------------------
// Fault policy (kernel2 WP-1.5): the scorers and bookkeeping of the non-legacy policies. Under
// `FaultPolicy::Legacy` none of this runs.
// ---------------------------------------------------------------------------

/// One evaluation under a non-legacy [`FaultPolicy`].
pub(super) enum Admitted {
    /// Scored: the policy's fold scores, and the fault totals of every pass.
    Scored(Vec<f64>, FaultTotals),
    /// Not scored, for a reason the legacy scorers share: an empty fold, a fit or prediction
    /// error, a score that is not finite (a penalized one included), or the budget or a
    /// cancellation stopping the evaluation.
    Failed,
    /// Rejected by the policy.
    Rejected(Rejection),
}

/// The fit a non-legacy policy runs: [`fit_traced`], except in tests.
pub(super) type TracedFit<'fit> = dyn Fn(&GenomeIr, &Rows, &[f64]) -> anyhow::Result<(model::Model, PipelineTelemetry)>
    + Sync
    + 'fit;

/// The scope check of a non-legacy policy: the program as it is fitted, at the input width. A set
/// without rows is left to the folds, which fail on it as the legacy scorers do.
pub(super) fn scope_rejection(genome: &GenomeIr, x: &Rows) -> Option<Rejection> {
    let width = x.first()?.len();
    validate_scope(genome, width).err().map(Rejection::Scope)
}

/// One fold under a non-legacy policy: fit and predict with the traced entry points and add both
/// passes to `faults`. `stop` runs between the fit and the prediction, where [`score_rung`] checks
/// its budget. `Err` ends the evaluation: a Hard fault rejects it, anything else fails it.
pub(super) fn traced_fold(
    genome: &GenomeIr,
    (xtr, ytr): (&Rows, &[f64]),
    (xte, yte): (&Rows, &[f64]),
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    fit: &TracedFit<'_>,
    stop: &dyn Fn() -> bool,
    faults: &mut FaultTotals,
) -> Result<f64, Admitted> {
    let hard = |faults: &FaultTotals| Admitted::Rejected(Rejection::HardFault(faults.hard_classes));
    let (fitted, training) = fit(genome, xtr, ytr).map_err(|_| Admitted::Failed)?;
    faults.add_pass(&training);
    if faults.is_hard() {
        return Err(hard(faults));
    }
    if stop() {
        return Err(Admitted::Failed);
    }
    let (predictions, held_out) = predict_traced(&fitted, xte).map_err(|_| Admitted::Failed)?;
    faults.add_pass(&held_out);
    if faults.is_hard() {
        return Err(hard(faults));
    }
    let score = metric(yte, &predictions);
    if score.is_finite() {
        Ok(score)
    } else {
        Err(Admitted::Failed)
    }
}

/// The end of an evaluation whose every fold was scored without a Hard fault: the fold scores as
/// `policy` keeps them.
pub(super) fn admit(policy: FaultPolicy, fold_scores: Vec<f64>, faults: FaultTotals) -> Admitted {
    let scores = match policy {
        FaultPolicy::Penalize {
            lambda_domain,
            lambda_saturation,
            ..
        } => {
            let penalized = fold_scores
                .iter()
                .map(|&score| faults.penalize(score, lambda_domain, lambda_saturation))
                .collect::<Option<Vec<f64>>>();
            match penalized {
                Some(scores) => scores,
                None => return Admitted::Failed,
            }
        }
        FaultPolicy::Legacy | FaultPolicy::RejectHard => fold_scores,
    };
    Admitted::Scored(scores, faults)
}

/// [`score_genome`] under a non-legacy [`FaultPolicy`]: the same program, folds and metric, with
/// the scope check before anything is fitted and every fold run by [`traced_fold`].
#[allow(clippy::too_many_arguments)]
pub(super) fn score_genome_admitted(
    g: &DslExpr,
    x: &Rows,
    y: &[f64],
    assign: &[usize],
    k: usize,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    policy: FaultPolicy,
    fit: &TracedFit<'_>,
) -> Admitted {
    let genome = GenomeIr::Pipeline(search_surrogate(g));
    if let Some(rejection) = scope_rejection(&genome, x) {
        return Admitted::Rejected(rejection);
    }
    let mut faults = FaultTotals::default();
    let mut fold_scores = Vec::with_capacity(k);
    for fold in 0..k {
        let (mut xtr, mut ytr, mut xte, mut yte) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..y.len() {
            if assign[i] == fold {
                xte.push(x[i].clone());
                yte.push(y[i]);
            } else {
                xtr.push(x[i].clone());
                ytr.push(y[i]);
            }
        }
        if xte.is_empty() || xtr.is_empty() {
            return Admitted::Failed;
        }
        match traced_fold(
            &genome,
            (&xtr, &ytr),
            (&xte, &yte),
            metric,
            fit,
            &|| false,
            &mut faults,
        ) {
            Ok(score) => fold_scores.push(score),
            Err(end) => return end,
        }
    }
    admit(policy, fold_scores, faults)
}

/// What a non-legacy [`FaultPolicy`] knows about the candidates of one run: the fault totals of
/// every scored evaluation, per canonical program and rung (the surrogate search uses rung 0), and
/// the programs it rejected. It is only looked up by program, never iterated, so the order of its
/// hash maps cannot reach a result.
pub(super) struct Admission {
    pub(super) policy: FaultPolicy,
    pub(super) faults: HashMap<String, [Option<FaultTotals>; 3]>,
    pub(super) rejected: std::collections::HashSet<String>,
}

/// The fault counters of one generation's [`GenStats`]; all zero under `Legacy`.
#[derive(Default)]
pub(super) struct FaultCounters {
    pub(super) hard_rejected_share: f64,
    pub(super) hard_rejected: u64,
    pub(super) pop_domain_rate: f64,
    pub(super) elite_domain_rate: f64,
}

impl Admission {
    /// The admission state of a run under `policy`, or `None` under `Legacy`, which runs the
    /// search unchanged. Panics when the policy cannot be run ([`FaultPolicy::validate`]).
    pub(super) fn start(policy: FaultPolicy) -> Option<Self> {
        if matches!(policy, FaultPolicy::Legacy) {
            return None;
        }
        if let Err(reason) = policy.validate() {
            panic!("{reason}");
        }
        Some(Self {
            policy,
            faults: HashMap::new(),
            rejected: std::collections::HashSet::new(),
        })
    }

    /// Records one evaluation of `key` at `rung` and returns the scores the search keeps: the
    /// policy's scores, or `None` when the evaluation failed or was rejected. A rejection is
    /// reported to `obs`.
    pub(super) fn record(
        &mut self,
        key: &str,
        rung: usize,
        admitted: Admitted,
        obs: &dyn Observer,
    ) -> Option<Vec<f64>> {
        match admitted {
            Admitted::Scored(scores, totals) => {
                self.faults.entry(key.to_string()).or_default()[rung] = Some(totals);
                Some(scores)
            }
            Admitted::Failed => None,
            Admitted::Rejected(reason) => {
                obs.on_reject(key, &reason);
                self.rejected.insert(key.to_string());
                None
            }
        }
    }

    /// The fault totals of `key`'s scored evaluation at `rung`; zero when there is none.
    pub(super) fn totals(&self, key: &str, rung: usize) -> FaultTotals {
        self.faults
            .get(key)
            .and_then(|rungs| rungs[rung])
            .unwrap_or_default()
    }

    /// The faulted cells of each ranked candidate, in order, when the policy breaks ties.
    /// `rung_of` names the rung each candidate's score comes from.
    pub(super) fn tie_break_cells(
        &self,
        ranked: &[Scored],
        rung_of: impl Fn(&str) -> usize,
    ) -> Option<Vec<u64>> {
        self.policy.breaks_ties().then(|| {
            ranked
                .iter()
                .map(|s| self.totals(&s.dsl, rung_of(&s.dsl)).faulted_cells())
                .collect()
        })
    }

    /// One generation's counters: `population` is the generation's candidates, `ranked` their
    /// ranking and `elites` how many ranked candidates pass into the next generation. `rung_of`
    /// names the rung each ranked candidate's score comes from.
    pub(super) fn counters(
        &self,
        population: &[DslExpr],
        ranked: &[Scored],
        elites: usize,
        rung_of: impl Fn(&str) -> usize,
    ) -> FaultCounters {
        let mut seen = std::collections::HashSet::new();
        let (mut distinct, mut rejected) = (0_u64, 0_u64);
        for genome in population {
            let key = genome.to_string();
            let was_rejected = self.rejected.contains(&key);
            if seen.insert(key) {
                distinct += 1;
                rejected += u64::from(was_rejected);
            }
        }
        let mean_domain_rate = |candidates: &[Scored]| {
            if candidates.is_empty() {
                return 0.0;
            }
            candidates
                .iter()
                .map(|s| self.totals(&s.dsl, rung_of(&s.dsl)).domain_rate())
                .sum::<f64>()
                / candidates.len() as f64
        };
        FaultCounters {
            hard_rejected_share: if distinct == 0 {
                0.0
            } else {
                rejected as f64 / distinct as f64
            },
            hard_rejected: self.rejected.len() as u64,
            pop_domain_rate: mean_domain_rate(ranked),
            elite_domain_rate: mean_domain_rate(&ranked[..elites.min(ranked.len())]),
        }
    }
}

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------
