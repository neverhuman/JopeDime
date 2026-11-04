#[cfg(test)]
pub(crate) fn gbm_parameter_keys(gbm: &crate::spec::GbmPlan) -> &'static [&'static str] {
    let crate::spec::GbmPlan {
        rounds: _,
        depth: _,
        lr: _,
        lambda: _,
        bins: _,
        min_leaf: _,
        max_features: _,
    } = gbm;
    &[
        "rounds",
        "depth",
        "lr",
        "lambda",
        "bins",
        "min_leaf",
        "max_features",
    ]
}

#[cfg(test)]
pub(crate) fn planned_head_parameter_keys(
    head: &crate::spec::PlannedHead,
) -> &'static [&'static str] {
    use crate::spec::PlannedHead;

    match head {
        PlannedHead::Ridge { lambda_frac: _ } => &["lambda_frac"],
        PlannedHead::RidgeGcv => &[],
        PlannedHead::ElasticNet {
            alpha: _,
            l1_ratio: _,
            sweeps: _,
        } => &["alpha", "l1_ratio", "sweeps"],
        PlannedHead::Gbm(gbm) => gbm_parameter_keys(gbm),
        PlannedHead::Blend {
            lambda_frac: _,
            gbm,
        } => {
            let _ = gbm_parameter_keys(gbm);
            &[
                "lambda_frac",
                "rounds",
                "depth",
                "lr",
                "lambda",
                "bins",
                "min_leaf",
                "max_features",
            ]
        }
        PlannedHead::HistGbm2 {
            leaves: _,
            lr: _,
            bags: _,
            rounds: _,
            row_subsample: _,
            col_subsample: _,
            min_leaf: _,
            val_pct: _,
            val_shuffle: _,
            huber_delta: _,
            stop_mae: _,
        } => &[
            "leaves",
            "lr",
            "bags",
            "rounds",
            "row_subsample",
            "col_subsample",
            "min_leaf",
            "val_pct",
            "val_shuffle",
            "huber_delta",
            "stop_mae",
        ],
        PlannedHead::Krr { m: _ } => &["m"],
        PlannedHead::Gpr { m: _, steps: _ } => &["m", "steps"],
        PlannedHead::GprKernel {
            m: _,
            steps: _,
            source: _,
            spec: _,
        } => &["m", "steps", "kernel"],
        PlannedHead::Ksmooth { k: _, gamma: _ } => &["k", "gamma"],
        PlannedHead::Stack => &[],
        PlannedHead::Portfolio(plan) => {
            // `members` is a collection of nested operation/head plans, whose
            // fields are covered by those type-wide inventories. `fallback`
            // is a derived member index, not an independently tunable value.
            let crate::spec::PortfolioPlan {
                folds: _,
                repeats_auto: _,
                members: _,
                fallback: _,
            } = plan;
            &["folds", "repeats_auto"]
        }
    }
}

#[cfg(test)]
pub(crate) fn evolve_config_parameter_keys(
    config: &crate::evolve::EvolveConfig,
) -> &'static [&'static str] {
    // `seed_genomes` contains complete starter programs, not parameter fields.
    let crate::evolve::EvolveConfig {
        pop: _,
        gens: _,
        folds: _,
        seed: _,
        max_transforms: _,
        time_budget_secs: _,
        seed_genomes: _,
        fitness: _,
        target_score: _,
        rank_robust: _,
        hard_ten_priors: _,
        expansion_vocab: _,
        fault_policy: _,
        selection: _,
        ridge_scoring: _,
    } = config;
    &[
        "pop",
        "gens",
        "folds",
        "seed",
        "max_transforms",
        "time_budget_secs",
        "fitness",
        "target_score",
        "rank_robust",
        "hard_ten_priors",
        "expansion_vocab",
        "fault_policy",
        "selection",
        "ridge_scoring",
    ]
}

#[cfg(test)]
pub(crate) fn planned_target_parameter_keys(
    target: &crate::spec::PlannedTarget,
) -> &'static [&'static str] {
    let crate::spec::PlannedTarget { kind: _, q: _ } = target;
    &["q"]
}

/// Returns the catalog identifiers for the configurable fields of each planned
/// operation. Expression/source payloads and fitted state are intentionally
/// excluded: they define operation content or learned results, not tunable
/// settings. Context prefixes distinguish repeated names such as `k`, `gamma`,
/// `seed`, and `bins` across operations and model heads.
#[cfg(test)]
pub(crate) fn planned_op_parameter_keys(op: &crate::spec::PlannedOp) -> Vec<&'static str> {
    use crate::spec::PlannedOp;

    match op {
        PlannedOp::Standardize
        | PlannedOp::RobustScale
        | PlannedOp::Rank
        | PlannedOp::Blom
        | PlannedOp::PolyInteract
        | PlannedOp::RowStats
        | PlannedOp::Sigmoid
        | PlannedOp::NormRow
        | PlannedOp::Polar
        | PlannedOp::RowSpectral
        | PlannedOp::RowDiffStats => Vec::new(),
        PlannedOp::QuantileBins { bins: _ } => vec!["op.quantile_bins.bins"],
        PlannedOp::BinOnehot { bins: _ } => vec!["op.bin_onehot.bins"],
        PlannedOp::Hinge { knots: _ } => vec!["op.hinge.knots"],
        PlannedOp::WinsorClip { q: _ } => vec!["op.winsor_clip.q"],
        PlannedOp::Cheb { degree: _ } => vec!["op.cheb.degree"],
        PlannedOp::RffCos {
            n: _,
            gamma: _,
            seed: _,
        } => vec!["op.rff_cos.n", "op.rff_cos.gamma", "op.rff_cos.seed"],
        PlannedOp::RffSin {
            n: _,
            gamma: _,
            seed: _,
        } => vec!["op.rff_sin.n", "op.rff_sin.gamma", "op.rff_sin.seed"],
        PlannedOp::RandProj { k: _, seed: _ } => vec!["op.rand_proj.k", "op.rand_proj.seed"],
        PlannedOp::LandmarkDist {
            k: _,
            gamma: _,
            seed: _,
        } => vec![
            "op.landmark_dist.k",
            "op.landmark_dist.gamma",
            "op.landmark_dist.seed",
        ],
        PlannedOp::ObliqueHinge { n: _, seed: _ } => {
            vec!["op.oblique_hinge.n", "op.oblique_hinge.seed"]
        }
        PlannedOp::PairProd { topk: _ } => vec!["op.pair_prod.topk"],
        PlannedOp::TargetBinEncode { bins: _, prior: _ } => {
            vec!["op.target_bin_encode.bins", "op.target_bin_encode.prior"]
        }
        PlannedOp::Step { t: _ } => vec!["op.step.t"],
        PlannedOp::Lag { k: _ } => vec!["op.lag.k"],
        PlannedOp::Roll { k: _ } => vec!["op.roll.k"],
        PlannedOp::FftMag { k: _ } => vec!["op.fft_mag.k"],
        PlannedOp::SvdProject {
            k: _,
            iters: _,
            seed: _,
        } => vec![
            "op.svd_project.k",
            "op.svd_project.iters",
            "op.svd_project.seed",
        ],
        PlannedOp::Nystrom {
            k: _,
            gamma: _,
            seed: _,
        } => vec!["op.nystrom.k", "op.nystrom.gamma", "op.nystrom.seed"],
        PlannedOp::Symlog { c: _ } => vec!["op.symlog.c"],
        PlannedOp::SelectCorr { k: _, by_var: _ } => {
            vec!["op.select_corr.k", "op.select_corr.by_var"]
        }
        PlannedOp::Impute {
            strategy: _,
            value: _,
            add_indicator: _,
        } => vec![
            "op.impute.strategy",
            "op.impute.value",
            "op.impute.add_indicator",
        ],
        PlannedOp::PairRatio { topk: _ } => vec!["op.pair_ratio.topk"],
        PlannedOp::Spline { knots: _ } => vec!["op.spline.knots"],
        PlannedOp::MapExpr { source: _, expr: _ }
        | PlannedOp::DeriveExpr { source: _, expr: _ } => Vec::new(),
        PlannedOp::RowVectorV2 { state: _ } => Vec::new(),
        PlannedOp::RowVectorInputV2 { width: _ } => vec!["op.row_vector_input_v2.width"],
        PlannedOp::ScalarV2 {
            program: _,
            carrier: _,
            width: _,
        } => vec!["op.scalar_v2.carrier", "op.scalar_v2.width"],
    }
}

#[cfg(test)]
pub(crate) fn planned_stage_parameter_keys(stage: &crate::spec::PlannedStage) -> Vec<&'static str> {
    let crate::spec::PlannedStage { op, append: _ } = stage;
    let mut keys = planned_op_parameter_keys(op);
    keys.push("stage.append");
    keys
}
