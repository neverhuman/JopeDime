use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn final_cv(
    config: &Config,
    train_path: &str,
    _x: &Rows,
    y: &[f64],
    top: &mut [TopGene],
) -> Result<usize, String> {
    let fold_count = if config.strict_pooled_search() {
        config.final_cv_folds
    } else {
        usable_folds(y.len(), config.final_cv_folds)
    };
    let folds = kfold(
        y,
        fold_count,
        seed_from(&format!(
            "screen:{}:{}:finalcv",
            config.dataset_hash, config.master_seed
        )),
    );
    if config.strict_pooled_search() {
        validate_strict_fold_plan(y.len(), &folds)?;
    }
    let mut evals = 0usize;
    let fold_train: Vec<Vec<usize>> = (0..folds.len())
        .map(|f| {
            folds
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != f)
                .flat_map(|(_, ids)| ids.iter().copied())
                .collect()
        })
        .collect();
    let pairs: Vec<(usize, usize)> = (0..top.len())
        .flat_map(|g| (0..folds.len()).map(move |f| (g, f)))
        .collect();
    // Per-fold R^2 is not merely unstable on a one-row fold, it is UNDEFINED:
    // ss_tot is 0, r2() returns None, and the caller scores the clip floor. With
    // leave-one-out that happens on EVERY fold, so every gene ties at -1.0 and
    // the winner falls through to the lexicographically smallest hash. Pooling
    // is therefore not optional below two rows per fold -- it is the only
    // defined estimator, so force it regardless of the flag.
    let degenerate_fold = folds.iter().any(|f| f.len() < 2);
    if config.pooled_oof || degenerate_fold {
        // Pooled out-of-fold R^2: one ratio computed over every row once, rather
        // than an average of per-fold R^2 values that each carry their own
        // ss_tot denominator. feat_math documents the pooled form as the primary
        // CV estimate; the per-fold mean is unstable on small folds and rewards
        // a gene that drew one lucky slice.
        type FoldPrediction = ((usize, usize), Result<Vec<f64>, EvalFailure>);
        let fold_preds: Vec<FoldPrediction> = pairs
            .par_iter()
            .map(|&(g, f)| {
                (
                    (g, f),
                    eval_predictions(train_path, &top[g].canonical, &fold_train[f], &folds[f]),
                )
            })
            .collect();
        for (g, gene) in top.iter_mut().enumerate() {
            let mut oof_true: Vec<f64> = Vec::new();
            let mut oof_pred: Vec<f64> = Vec::new();
            let mut prediction_views: Vec<Option<&[f64]>> = Vec::with_capacity(folds.len());
            let mut failures = 0usize;
            for (f, fold) in folds.iter().enumerate() {
                let entry = fold_preds
                    .iter()
                    .find(|((gg, ff), _)| *gg == g && *ff == f)
                    .map(|(_, r)| r);
                match entry {
                    Some(Ok(pred)) if pred.len() == fold.len() => {
                        prediction_views.push(Some(pred.as_slice()));
                        for (k, &row) in fold.iter().enumerate() {
                            oof_true.push(y[row]);
                            oof_pred.push(pred[k]);
                        }
                    }
                    _ => {
                        prediction_views.push(None);
                        failures += 1;
                    }
                }
            }
            gene.cv_failed_folds = failures;
            gene.cv7_mean_r2 = if config.strict_pooled_search() {
                complete_pooled_oof_r2(y, &folds, &prediction_views)
            } else if oof_true.is_empty() {
                None
            } else {
                r2(&oof_true, &oof_pred).map(|v| v.max(-1.0))
            };
            evals += folds.len();
        }
        return Ok(evals);
    }

    let fold_scores: Vec<((usize, usize), Result<f64, EvalFailure>)> = pairs
        .par_iter()
        .map(|&(g, f)| {
            (
                (g, f),
                eval_checked(train_path, &top[g].canonical, &fold_train[f], &folds[f]),
            )
        })
        .collect();
    let cv_scores: Vec<(Option<f64>, usize)> = (0..top.len())
        .map(|g| {
            let mine: Vec<&Result<f64, EvalFailure>> = fold_scores
                .iter()
                .filter(|((gg, _), _)| *gg == g)
                .map(|(_, v)| v)
                .collect();
            let failures = mine.iter().filter(|r| r.is_err()).count();
            // Legacy behaviour scores an infrastructure failure as the clip
            // floor, which is indistinguishable from a genuinely bad model.
            // Dropping those folds instead is a real change, so it stays behind
            // a flag: with it off, recorded runs replay bit-for-bit.
            let s: Vec<f64> = if config.drop_failed_folds {
                mine.iter()
                    .filter_map(|r| r.as_ref().ok().copied())
                    .collect()
            } else {
                mine.iter().map(|r| *r.as_ref().unwrap_or(&-1.0)).collect()
            };
            let mean = if s.is_empty() {
                None
            } else {
                Some(s.iter().sum::<f64>() / s.len() as f64)
            };
            (mean, failures)
        })
        .collect();
    for (gene, (cv, failures)) in top.iter_mut().zip(cv_scores) {
        gene.cv7_mean_r2 = cv;
        gene.cv_failed_folds = failures;
        evals += folds.len();
    }
    Ok(evals)
}
