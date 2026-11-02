use super::*;

/// The variance-penalised fitness the gate, the league and the exact-CV campaign rank
/// regression by: `max(oof_r2, -1) - 0.25 * min(population_sd(fold_r2), 1)`, over finite folds
/// only. Copied from `crates/jope-campaigns/src/commands/exact_cv.rs::robust_fitness` so the
/// benchmark ranks by the same number the campaign does, without depending on the campaign crate.
pub fn robust_fitness(oof_r2: f64, fold_r2: &[f64]) -> f64 {
    let finite: Vec<f64> = fold_r2.iter().copied().filter(|v| v.is_finite()).collect();
    let population_sd = if finite.len() < 2 {
        0.0
    } else {
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        (finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / finite.len() as f64).sqrt()
    };
    oof_r2.max(-1.0) - 0.25 * population_sd.min(1.0)
}

/// The binary campaign lane's twin of [`robust_fitness`], on the Gini scale (`2·AUC − 1`: the
/// same "0 = chance, 1 = perfect" geometry as R², so the fold-variance penalty carries over at
/// a compatible scale). Folds whose held-out rows were one class have undefined AUC (passed as
/// NaN, or simply omitted) and are excluded from the penalty. Copied from
/// `crates/jope-campaigns/src/commands/exact_cv.rs::robust_fitness_binary`.
pub fn robust_fitness_binary(pooled_oof_auc: f64, fold_auc: &[f64]) -> f64 {
    let gini = 2.0 * pooled_oof_auc.clamp(0.0, 1.0) - 1.0;
    let fold_gini: Vec<f64> = fold_auc
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|auc| 2.0 * auc - 1.0)
        .collect();
    let population_sd = if fold_gini.len() < 2 {
        0.0
    } else {
        let mean = fold_gini.iter().sum::<f64>() / fold_gini.len() as f64;
        (fold_gini
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / fold_gini.len() as f64)
            .sqrt()
    };
    gini - 0.25 * population_sd.min(1.0)
}

/// The same penalty for a score already on a `[0, 1]` scale (multiclass macro-F1):
/// `oof - 0.25 * min(population_sd(folds), 1)`. No campaign lane defines a multiclass fitness,
/// so this is the benchmark's own definition; it is the regression rule without the `−1` floor.
pub fn robust_fitness_bounded(oof: f64, folds: &[f64]) -> f64 {
    let finite: Vec<f64> = folds.iter().copied().filter(|v| v.is_finite()).collect();
    let population_sd = if finite.len() < 2 {
        0.0
    } else {
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        (finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / finite.len() as f64).sqrt()
    };
    oof - 0.25 * population_sd.min(1.0)
}

/// The task's variance-penalised deep fitness: the pooled out-of-fold primary score penalised
/// by the population SD of the per-fold scores.
pub fn robust_fitness_for(task: Task, oof_score: f64, fold_scores: &[f64]) -> f64 {
    match task {
        Task::Regression => robust_fitness(oof_score, fold_scores),
        Task::Binary => robust_fitness_binary(oof_score, fold_scores),
        Task::Multiclass => robust_fitness_bounded(oof_score, fold_scores),
    }
}

/// One fold of a deep evaluation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FoldResult {
    /// Held-out fold.
    pub fold: usize,
    /// [`crate::cuts::Cut::cut_id`].
    pub cut_id: String,
    /// The program scored on this fold (differs per fold only when the search was re-run per
    /// cut; empty for an adapter, and when producing the program failed).
    #[serde(default)]
    pub dsl: String,
    /// The score.
    pub outcome: Outcome,
}

/// A program (or a per-cut program) scored on every cut, pooled out of fold.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Deep {
    /// `ok` when every fold scored and the pooled score is defined, else `rejected`.
    pub status: String,
    /// Why it was rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Per-fold outcomes, fold order.
    pub folds: Vec<FoldResult>,
    /// Per-fold primary scores, fold order, folds where it is defined only.
    pub fold_score: Vec<f64>,
    /// Pooled out-of-fold primary score over every row, in row order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oof_score: Option<f64>,
    /// Every pooled out-of-fold metric ([`Task::metrics`]).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub oof: Metrics,
    /// Population SD of `fold_score`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fold_sd: Option<f64>,
    /// [`robust_fitness_for`] of the above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robust_fitness: Option<f64>,
    /// Wall seconds summed over folds (fit + predict).
    pub seconds: f64,
    /// Wall seconds spent producing per-fold programs (0 unless the search was re-run per cut).
    #[serde(default)]
    pub produce_seconds: f64,
}

impl Deep {
    fn rejected(
        reason: String,
        folds: Vec<FoldResult>,
        fold_score: Vec<f64>,
        seconds: f64,
    ) -> Self {
        Self {
            status: "rejected".to_owned(),
            reason: Some(reason),
            folds,
            fold_score,
            oof_score: None,
            oof: Metrics::new(),
            fold_sd: None,
            robust_fitness: None,
            seconds,
            produce_seconds: 0.0,
        }
    }
}

/// Score one program on every cut; folds run in parallel.
pub fn evaluate_deep(dsl: &str, x: &Rows, y: &[f64], cuts: &[Cut], task: Task) -> Deep {
    let programs: Vec<std::result::Result<String, String>> =
        cuts.iter().map(|_| Ok(dsl.to_owned())).collect();
    evaluate_deep_programs(&programs, x, y, cuts, task)
}

/// Score `programs[i]` on `cuts[i]` for every cut and pool out of fold. An `Err` program is a
/// fold whose program could not be produced (a failed per-cut search): that fold is a rejection
/// naming the cause, never a parse attempt on the error text. `programs.len()` must equal
/// `cuts.len()`; a mismatch is reported as a rejected deep score, never a panic.
pub fn evaluate_deep_programs(
    programs: &[std::result::Result<String, String>],
    x: &Rows,
    y: &[f64],
    cuts: &[Cut],
    task: Task,
) -> Deep {
    if programs.len() != cuts.len() {
        return Deep::rejected(
            format!("{} programs for {} cuts", programs.len(), cuts.len()),
            Vec::new(),
            Vec::new(),
            0.0,
        );
    }
    let outcomes: Vec<Outcome> = cuts
        .par_iter()
        .zip(programs.par_iter())
        .map(|(cut, program)| match program {
            Ok(dsl) => evaluate(dsl, x, y, cut, task),
            Err(reason) => Outcome::rejected(format!("search: {reason}"), 0.0, 0.0, cut.hold.len()),
        })
        .collect();
    let labels: Vec<String> = programs
        .iter()
        .map(|p| p.clone().unwrap_or_default())
        .collect();
    pool(cuts, &labels, outcomes, y, task)
}

/// Pool one outcome per cut out of fold: every row's held-out prediction in row order, scored
/// for `task`, penalised by the fold variance. `labels[i]` is recorded as fold `i`'s `dsl`
/// (the program scored there; empty for an adapter or a fold whose program could not be made).
/// Any rejected fold, any uncovered row or an undefined pooled score rejects the whole thing.
pub fn pool(
    cuts: &[Cut],
    labels: &[String],
    outcomes: Vec<Outcome>,
    y: &[f64],
    task: Task,
) -> Deep {
    if labels.len() != cuts.len() || outcomes.len() != cuts.len() {
        return Deep::rejected(
            format!(
                "{} labels and {} outcomes for {} cuts",
                labels.len(),
                outcomes.len(),
                cuts.len()
            ),
            Vec::new(),
            Vec::new(),
            0.0,
        );
    }
    let seconds = outcomes
        .iter()
        .map(|o| o.fit_seconds + o.predict_seconds)
        .sum();
    let mut oof = vec![f64::NAN; y.len()];
    let mut fold_score = Vec::new();
    let mut all_ok = true;
    let mut folds = Vec::with_capacity(cuts.len());
    for ((cut, label), outcome) in cuts.iter().zip(labels).zip(outcomes) {
        if outcome.is_ok() {
            for (i, &row) in cut.hold.iter().enumerate() {
                oof[row] = outcome.predictions[i];
            }
            if let Some(score) = outcome.score {
                fold_score.push(score);
            }
        } else {
            all_ok = false;
        }
        folds.push(FoldResult {
            fold: cut.fold,
            cut_id: cut.cut_id.clone(),
            dsl: label.clone(),
            outcome: Outcome {
                predictions: Vec::new(),
                ..outcome
            },
        });
    }
    if !all_ok {
        return Deep::rejected("a fold was rejected".to_owned(), folds, fold_score, seconds);
    }
    if oof.iter().any(|v| !v.is_finite()) {
        return Deep::rejected(
            "the cuts do not cover every row".to_owned(),
            folds,
            fold_score,
            seconds,
        );
    }
    let (oof_score, oof_metrics) = match score(task, y, &oof) {
        Ok((Some(score), metrics, _)) => (score, metrics),
        Ok((None, _, note)) => {
            return Deep::rejected(
                format!("pooled: {}", note.unwrap_or_default()),
                folds,
                fold_score,
                seconds,
            )
        }
        Err(reason) => {
            return Deep::rejected(format!("pooled: {reason}"), folds, fold_score, seconds)
        }
    };
    let fold_sd = if fold_score.len() < 2 {
        0.0
    } else {
        let mean = fold_score.iter().sum::<f64>() / fold_score.len() as f64;
        (fold_score.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / fold_score.len() as f64)
            .sqrt()
    };
    Deep {
        status: "ok".to_owned(),
        reason: None,
        folds,
        robust_fitness: Some(robust_fitness_for(task, oof_score, &fold_score)),
        fold_score,
        oof_score: Some(oof_score),
        oof: oof_metrics,
        fold_sd: Some(fold_sd),
        seconds,
        produce_seconds: 0.0,
    }
}

/// `dsl` on every cut as `Ok` programs (the shape [`evaluate_deep_programs`] wants).
pub fn fixed_programs(dsl: &str, cuts: &[Cut]) -> Vec<std::result::Result<String, String>> {
    cuts.iter().map(|_| Ok(dsl.to_owned())).collect()
}

/// Whether `dsl` parses; the error otherwise (for validating caller programs up front).
pub fn parse_check(dsl: &str) -> Result<GenomeIr> {
    GenomeIr::parse(dsl).with_context(|| format!("program {dsl:?}"))
}
