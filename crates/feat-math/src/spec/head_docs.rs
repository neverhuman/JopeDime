use super::*;

pub fn head_fn_name(head: &PlannedHead) -> &'static str {
    match head {
        PlannedHead::Ridge { .. } => "ridge_regression",
        PlannedHead::RidgeGcv => "gcv_ridge_regression",
        PlannedHead::ElasticNet { .. } => "elastic_net",
        PlannedHead::Gbm(_) => "gradient_boosted_trees",
        PlannedHead::Blend { .. } => "blended_regression",
        PlannedHead::HistGbm2 { .. } => "leafwise_boosted_trees",
        PlannedHead::Krr { .. } => "kernel_ridge_regression",
        PlannedHead::Gpr { .. } => "gaussian_process_ard_regression",
        PlannedHead::GprKernel { .. } => "gaussian_process_composite_kernel_regression",
        PlannedHead::Ksmooth { .. } => "nadaraya_watson_regression",
        PlannedHead::Stack => "greedy_ensemble_selection",
        PlannedHead::Portfolio(_) => "internally_validated_ensemble",
    }
}

/// Emitted-artifact tag for one kernel primitive (used in the kernel-terms
/// literal both emitters bake into the composite-GP head).
pub fn kernel_prim_tag(prim: KernelPrim) -> &'static str {
    match prim {
        KernelPrim::Rbf => "squared_exponential",
        KernelPrim::Rq => "rational_quadratic",
        KernelPrim::Periodic => "periodic",
        KernelPrim::Linear => "linear",
    }
}

/// One-sentence description of a kernel primitive for the docs.
fn kernel_prim_doc(prim: KernelPrim) -> &'static str {
    match prim {
        KernelPrim::Rbf => {
            "a squared-exponential factor with per-feature \
            automatic-relevance lengthscales"
        }
        KernelPrim::Rq => {
            "a rational-quadratic factor (1 + r^2/(2 alpha))^-alpha \
            with per-feature lengthscales and a learned mixing exponent alpha, a \
            scale mixture of squared exponentials"
        }
        KernelPrim::Periodic => {
            "a periodic factor exp(-2 sum_d sin^2(pi (x_d - \
            x'_d)/period) / l^2) with a learned shared period and lengthscale \
            (MacKay, 1998)"
        }
        KernelPrim::Linear => {
            "a homogeneous linear factor (x . x')/p, giving a \
            Bayesian-linear-regression component"
        }
    }
}

pub fn head_doc(head: &PlannedHead) -> String {
    match head {
        PlannedHead::Ridge { lambda_frac } => format!(
            "ridge regression (Hoerl & Kennard, 1970) on standardized columns with penalty \
             max({lf}*n, 1e-3); the symmetric positive-definite normal equations are solved by a \
             Cholesky factorization and predictions are clipped to [-60, 60].",
            lf = crate::mathexpr::fmt_f64(*lambda_frac)
        ),
        PlannedHead::RidgeGcv => "ridge regression on standardized columns whose penalty is \
             chosen by generalized cross-validation, GCV(lambda) = mean squared residual / \
             max(1 - tr(S_lambda)/n, 1e-3)^2 (Golub, Heath & Wahba, 1979), over a fixed 25-point \
             logarithmic grid from 1e-5 to 1e3; when rows and columns both exceed 384 the 384 \
             columns most correlated with the target are kept (a train-only screen), one \
             cyclic-Jacobi eigendecomposition of the smaller of the p x p standardized Gram \
             matrix and the n x n inner-product matrix (they share their nonzero spectrum) \
             scores every grid penalty in closed form, the winning penalty is refit on all \
             training rows, and predictions are clipped to [-60, 60]."
            .to_string(),
        PlannedHead::ElasticNet { alpha, l1_ratio, sweeps } => format!(
            "elastic net regression (Zou & Hastie, 2005) with alpha={a}, l1_ratio={l}, fit by at \
             most {sweeps} sweeps of cyclic coordinate descent with soft-thresholding (Friedman, \
             Hastie & Tibshirani, 2010); predictions are clipped to [-60, 60].",
            a = crate::mathexpr::fmt_f64(*alpha),
            l = crate::mathexpr::fmt_f64(*l1_ratio)
        ),
        PlannedHead::Gbm(g) => format!(
            "gradient boosted regression trees (Friedman, 2001) with histogram splits on training \
             quantile bins and second-order leaf values -g/(h + lambda) as in modern boosting \
             systems (Chen & Guestrin, 2016): {rounds} rounds, depth {depth}, learning rate {lr}, \
             lambda {lam}, {bins} bins, min leaf {ml}.",
            rounds = g.rounds,
            depth = g.depth,
            lr = crate::mathexpr::fmt_f64(g.lr),
            lam = crate::mathexpr::fmt_f64(g.lambda),
            bins = g.bins,
            ml = g.min_leaf
        ),
        PlannedHead::Blend { lambda_frac, gbm } => format!(
            "a convex blend w*linear + (1-w)*trees of ridge regression (lambda fraction {lf}) and \
             gradient boosted trees ({rounds} rounds, depth {depth}); w is picked from \
             {{0, 0.25, 0.5, 0.75, 1}} on an internal 80/20 holdout by root mean squared error and \
             both members are then refit on all training rows.",
            lf = crate::mathexpr::fmt_f64(*lambda_frac),
            rounds = gbm.rounds,
            depth = gbm.depth
        ),
        PlannedHead::HistGbm2 {
            leaves,
            lr,
            bags,
            rounds,
            row_subsample,
            col_subsample,
            min_leaf,
            val_pct,
            val_shuffle,
            huber_delta,
            stop_mae,
        } => {
            let col_doc = if *col_subsample < 1.0 {
                format!(
                    " and a column-sampling share of {}",
                    crate::mathexpr::fmt_f64(*col_subsample)
                )
            } else {
                String::new()
            };
            let loss_doc = if *huber_delta > 0.0 {
                format!(
                    " Gradients are the residuals clipped at {} — the Huber loss (Huber, 1964).",
                    crate::mathexpr::fmt_f64(*huber_delta)
                )
            } else {
                String::new()
            };
            let stop_metric = if *stop_mae {
                "mean absolute error"
            } else {
                "root mean squared error"
            };
            let split_kind = if *val_shuffle {
                "rows chosen by a seeded shuffle"
            } else {
                "the contiguous tail rows"
            };
            format!(
                "gradient boosted regression trees (Friedman, 2001) grown leaf-wise / best-first \
                 to at most {leaves} leaves with exact split search over pre-sorted feature \
                 columns, in the manner of LightGBM (Ke et al., 2017); leaf values are the \
                 second-order step -g/(h + 1) (Chen & Guestrin, 2016), learning rate {lr}, \
                 minimum leaf size {min_leaf}, with a per-tree row-sampling share of \
                 {rs}{col_doc}.{loss_doc} {bags} bagged replicate(s) (Breiman, 1996) each stop \
                 early by {stop_metric} on an internal {val_pct}% holdout ({split_kind}; \
                 patience 50 rounds, at most {rounds}) and predictions average the replicates.",
                lr = crate::mathexpr::fmt_f64(*lr),
                rs = crate::mathexpr::fmt_f64(*row_subsample)
            )
        }
        PlannedHead::Krr { m } => format!(
            "kernel ridge regression (Saunders, Gammerman & Vovk, 1998) with a Gaussian kernel on \
             features scaled by the inverse training standard deviation; the kernel width gamma \
             (5 values scaled by 1/d) and ridge penalty lambda (5 values) are chosen by root mean \
             squared error on a contiguous 80/20 holdout using at most 256 seeded kernel centers, \
             the final coefficient system (K + lambda I) a = y over at most {m} seeded centers is \
             solved by Cholesky factorization, and predictions k(x)^T a are clipped to [-60, 60]."
        ),
        PlannedHead::Gpr { m, steps } => format!(
            "Gaussian-process regression (Rasmussen & Williams, 2006) with a squared-exponential \
             automatic-relevance-determination kernel sf2 * exp(-0.5 * sum_d (x_d - x'_d)^2 / ls_d^2) \
             on features scaled by the inverse training standard deviation; the per-feature \
             lengthscales ls_d, signal variance sf2, and noise variance sn2 are learned by {steps} \
             steps of gradient ascent on the log marginal likelihood, whose gradient \
             0.5 * trace((a a^T - Ky^-1) dKy/dtheta) with a = Ky^-1 y is evaluated with one \
             Cholesky-based inverse per step on at most 256 seeded rows; the predictive coefficients \
             are then solved on up to {m} seeded inducing rows and predictions k(x)^T a are clipped \
             to [-60, 60]."
        ),
        PlannedHead::GprKernel { m, steps, source, spec } => {
            let mut seen: Vec<KernelPrim> = Vec::new();
            for term in spec {
                for prim in term {
                    if !seen.contains(prim) {
                        seen.push(*prim);
                    }
                }
            }
            let prim_docs: Vec<&str> = seen.iter().map(|p| kernel_prim_doc(*p)).collect();
            format!(
                "Gaussian-process regression (Rasmussen & Williams, 2006) whose covariance is the \
                 composed kernel \"{source}\" — a sum of products of primitives in the spirit of \
                 kernel structure search (Duvenaud et al., 2013): {prims}. Each sum term carries \
                 a learned amplitude; every factor's shape parameters and the noise variance are \
                 learned on features scaled by the inverse training standard deviation via \
                 {steps} steps of gradient ascent on the log marginal likelihood, whose gradient \
                 0.5 * trace((a a^T - Ky^-1) dKy/dtheta) with a = Ky^-1 y is evaluated with one \
                 Cholesky-based inverse per step on at most 256 seeded rows; the predictive \
                 coefficients are then solved on up to {m} seeded inducing rows and predictions \
                 k(x)^T a are clipped to [-60, 60].",
                prims = prim_docs.join("; ")
            )
        }
        PlannedHead::Ksmooth { k, gamma } => format!(
            "Nadaraya-Watson kernel regression (Nadaraya, 1964; Watson, 1964): each prediction \
             is the Gaussian-weighted average of the targets of the {k} nearest training rows in \
             standardized feature space, with weights exp(-{g} * (d^2 - d^2_min)/p) over squared \
             Euclidean distances (ties broken by the row index); subtracting the nearest \
             distance d^2_min keeps the largest weight at exactly 1 so the denominator never \
             vanishes, and predictions are clipped to [-60, 60].",
            g = crate::mathexpr::fmt_f64(*gamma)
        ),
        PlannedHead::Stack => format!(
            "greedy ensemble selection with replacement (Caruana, Niculescu-Mizil, Crew & Ksikes, \
             2004) over a fixed five-model library: ridge regression at two penalties, an elastic \
             net, leaf-wise boosted trees ({leaves} leaves, learning rate {lr}, {bags} bagged \
             replicates), and Gaussian kernel ridge regression (at most {m} centers). Out-of-fold \
             predictions from contiguous 5-fold splits guide up to 20 greedy additions maximizing \
             the R^2 of the running mean; final weights are selection frequencies and each \
             selected model is refit on all training rows.",
            leaves = crate::boost2::DEFAULT_LEAVES,
            lr = crate::mathexpr::fmt_f64(crate::boost2::DEFAULT_LR),
            bags = crate::boost2::DEFAULT_BAGS,
            m = crate::krr::DEFAULT_M
        ),
        PlannedHead::Portfolio(p) => format!(
            "an internally validated ensemble of {n} component pipelines, each fitted on the \
             training rows and measured with held-out predictions from seeded shuffled \
             {folds}-fold splits (repeated on smaller samples for stability), every component \
             refit inside each fold so no fitted statistic sees its validation rows. Component \
             activation depends only on the training sample's row count, column count and \
             duplicate-row share; per-fold predictions are clipped to the fold's padded target \
             range; the components are then combined by a non-negative ridge blend of their \
             deviations from the training mean, solved by exact cyclic coordinate descent (at \
             most 500 sweeps) with a shrinkage penalty that grows with the component count and \
             shrinks with the sample size, falling back to the single best-measured component \
             when the blend loses. Components with meaningful weight are refit on all training \
             rows, a refit whose training error exceeds 1.05 standard deviations of the target \
             is replaced by the mean predictor, and every prediction is clipped to the padded \
             training target range.",
            n = p.members.iter().filter(|m| !m.hidden).count(),
            folds = p.folds
        ),
    }
}

pub fn head_short(head: &PlannedHead) -> &'static str {
    match head {
        PlannedHead::Ridge { .. } => "ridge regression",
        PlannedHead::RidgeGcv => "GCV-tuned ridge regression",
        PlannedHead::ElasticNet { .. } => "elastic net regression",
        PlannedHead::Gbm(_) => "gradient boosted trees",
        PlannedHead::Blend { .. } => "a ridge / boosted-trees blend",
        PlannedHead::HistGbm2 { .. } => "leaf-wise boosted trees",
        PlannedHead::Krr { .. } => "kernel ridge regression",
        PlannedHead::Gpr { .. } => "Gaussian-process regression",
        PlannedHead::GprKernel { .. } => "Gaussian-process regression with a composed kernel",
        PlannedHead::Ksmooth { .. } => "a nearest-neighbor kernel smoother",
        PlannedHead::Stack => "a greedy model ensemble",
        PlannedHead::Portfolio(_) => "an internally validated model ensemble",
    }
}

/// Stable per-kind key: also the base of the emitted target-transform
/// function names.
pub fn target_fn_name(kind: TargetKind) -> &'static str {
    match kind {
        TargetKind::RankGauss => "normal_scores_target",
        TargetKind::Log1p => "shifted_log_target",
        TargetKind::Winsor => "winsorized_target",
        TargetKind::ZScore => "standardized_target",
    }
}

/// True for kinds whose inverse carries its own train-range prediction clamp
/// instead of the historical ±60 wrap (which silently destroys any target
/// whose scale exceeds it).
pub fn target_range_aware(kind: TargetKind) -> bool {
    matches!(kind, TargetKind::ZScore)
}

/// Rust fitted-state struct name (None for the stateless kind).
pub fn target_struct_name(kind: TargetKind) -> Option<&'static str> {
    match kind {
        TargetKind::RankGauss => Some("NormalScoresTarget"),
        TargetKind::Log1p => None,
        TargetKind::Winsor => Some("WinsorizedTarget"),
        TargetKind::ZScore => Some("StandardizedTarget"),
    }
}

/// One-sentence math description of a target transform, used in docs.
pub fn target_doc(target: &PlannedTarget) -> String {
    match target.kind {
        TargetKind::RankGauss => "replaces each training-target value by the standard normal \
            quantile of its mid-rank (Acklam's rational probit approximation), a normal-scores \
            transform; the inverse interpolates linearly through the fitted table of distinct \
            (score, value) pairs, clamped to the extreme training values."
            .to_string(),
        TargetKind::Log1p => "maps each training-target value through the shifted logarithm \
            ln(1 + max(y, -0.999999)); the inverse is exp(z) - 1 with the exponent clipped to \
            [-745, 700]."
            .to_string(),
        TargetKind::Winsor => format!(
            "winsorizes the training target by clipping to its [{q}, 1-{q}] quantiles, limiting \
             the leverage of extreme targets (Tukey & McLaughlin, 1963); the inverse is the \
             identity, so predictions stay drawn toward the bulk of the target.",
            q = crate::mathexpr::fmt_f64(target.q)
        ),
        TargetKind::ZScore => "standardizes the training target to zero mean and unit variance \
            (standard deviation floored at 1e-9); the inverse maps predictions back through \
            z * sigma + mu and clamps them to the training-target range widened by 5% or the \
            [-60, 60] window, whichever is wider."
            .to_string(),
    }
}

/// Short phrase for titles.
pub fn target_short(kind: TargetKind) -> &'static str {
    match kind {
        TargetKind::RankGauss => "a normal-scores target transform",
        TargetKind::Log1p => "a shifted-log target transform",
        TargetKind::Winsor => "a winsorized target",
        TargetKind::ZScore => "a standardized (z-score) target transform",
    }
}
