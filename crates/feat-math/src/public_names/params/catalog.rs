use super::PublicName;

/// Public descriptions for tunable fields in planned operations, targets,
/// model heads, and search configuration.
pub const PARAMETERS: &[PublicName] = &[
    PublicName {
        key: "alpha",
        public_name: "regularization strength",
        purpose: "control how strongly model coefficients are kept small",
    },
    PublicName {
        key: "l1_ratio",
        public_name: "sparsity balance",
        purpose: "balance coefficient selection with coefficient shrinkage",
    },
    PublicName {
        key: "sweeps",
        public_name: "coordinate passes",
        purpose: "set how many passes are used to fit the model",
    },
    PublicName {
        key: "lambda_frac",
        public_name: "ridge penalty fraction",
        purpose: "set the strength of the ridge penalty",
    },
    PublicName {
        key: "rounds",
        public_name: "boosting rounds",
        purpose: "set how many successive trees are fitted",
    },
    PublicName {
        key: "depth",
        public_name: "tree depth",
        purpose: "limit how many levels each tree can use",
    },
    PublicName {
        key: "lr",
        public_name: "learning rate",
        purpose: "set how much each fitted tree changes the prediction",
    },
    PublicName {
        key: "lambda",
        public_name: "coefficient penalty",
        purpose: "limit the size of tree leaf values",
    },
    PublicName {
        key: "bins",
        public_name: "split search bins",
        purpose: "set how many candidate cut points are considered",
    },
    PublicName {
        key: "min_leaf",
        public_name: "minimum leaf size",
        purpose: "require each tree leaf to contain enough rows",
    },
    PublicName {
        key: "max_features",
        public_name: "feature limit",
        purpose: "cap the number of features considered by a tree",
    },
    PublicName {
        key: "leaves",
        public_name: "leaf limit",
        purpose: "set the maximum number of leaves in each tree",
    },
    PublicName {
        key: "bags",
        public_name: "tree bag count",
        purpose: "set how many independently sampled tree groups are fitted",
    },
    PublicName {
        key: "row_subsample",
        public_name: "row sampling share",
        purpose: "set the share of training rows used for each tree",
    },
    PublicName {
        key: "col_subsample",
        public_name: "feature sampling share",
        purpose: "set the share of features considered for each tree",
    },
    PublicName {
        key: "val_pct",
        public_name: "validation share",
        purpose: "set the share of training rows held aside for early stopping",
    },
    PublicName {
        key: "val_shuffle",
        public_name: "validation shuffle setting",
        purpose: "choose whether rows are shuffled before validation is selected",
    },
    PublicName {
        key: "huber_delta",
        public_name: "robust loss cutoff",
        purpose: "set where the fitting loss changes from squared to robust error",
    },
    PublicName {
        key: "stop_mae",
        public_name: "early stopping metric",
        purpose: "choose whether early stopping tracks mean absolute error",
    },
    PublicName {
        key: "m",
        public_name: "landmark count",
        purpose: "set how many training rows represent the kernel model",
    },
    PublicName {
        key: "steps",
        public_name: "kernel fitting steps",
        purpose: "set how many optimization steps fit the kernel model",
    },
    PublicName {
        key: "kernel",
        public_name: "kernel structure",
        purpose: "choose how similarity between rows is calculated",
    },
    PublicName {
        key: "k",
        public_name: "neighbor count",
        purpose: "set how many nearby training rows contribute to a prediction",
    },
    PublicName {
        key: "gamma",
        public_name: "kernel width",
        purpose: "set how quickly the influence of nearby rows decreases",
    },
    PublicName {
        key: "folds",
        public_name: "validation fold count",
        purpose: "set how many data partitions are used for validation",
    },
    PublicName {
        key: "repeats_auto",
        public_name: "automatic validation repeats",
        purpose: "allow the number of validation repeats to follow the data size",
    },
    PublicName {
        key: "q",
        public_name: "tail clipping fraction",
        purpose: "set how much of each target tail is clipped",
    },
    PublicName {
        key: "pop",
        public_name: "candidate count",
        purpose: "set how many candidate models are considered in each round",
    },
    PublicName {
        key: "gens",
        public_name: "search round count",
        purpose: "set how many rounds the search runs",
    },
    PublicName {
        key: "seed",
        public_name: "repeatability seed",
        purpose: "make randomized search choices repeatable",
    },
    PublicName {
        key: "max_transforms",
        public_name: "feature stage limit",
        purpose: "cap the number of feature processing stages before a model",
    },
    PublicName {
        key: "time_budget_secs",
        public_name: "time budget",
        purpose: "limit how long the search runs",
    },
    PublicName {
        key: "fitness",
        public_name: "scoring mode",
        purpose: "choose how candidate models are evaluated",
    },
    PublicName {
        key: "target_score",
        public_name: "early stopping score",
        purpose: "stop the search when a candidate reaches the configured score",
    },
    PublicName {
        key: "rank_robust",
        public_name: "stability-aware ranking",
        purpose: "include fold-to-fold score variation when ranking candidates",
    },
    PublicName {
        key: "hard_ten_priors",
        public_name: "challenging starter set",
        purpose: "include high-regularization and simple baseline starting models",
    },
    PublicName {
        key: "expansion_vocab",
        public_name: "expanded transform choices",
        purpose: "include the expanded set of feature transformations in search",
    },
    PublicName {
        key: "fault_policy",
        public_name: "candidate fault policy",
        purpose: "choose how invalid transformations affect candidate scoring",
    },
    PublicName {
        key: "selection",
        public_name: "candidate selection method",
        purpose: "choose how candidates are selected for the next round",
    },
    PublicName {
        key: "ridge_scoring",
        public_name: "ridge scoring method",
        purpose: "choose how ridge surrogate scores are computed",
    },
    // PlannedOp parameters use variant-qualified keys. Fields such as `k`,
    // `gamma`, and `seed` have different meanings across transforms, so their
    // public descriptions must remain tied to the operation that consumes them.
    PublicName {
        key: "op.quantile_bins.bins",
        public_name: "quantile bin count",
        purpose: "set how many value groups are formed from the training data",
    },
    PublicName {
        key: "op.bin_onehot.bins",
        public_name: "one-hot bin count",
        purpose: "set how many quantile groups receive separate output columns",
    },
    PublicName {
        key: "op.hinge.knots",
        public_name: "hinge knot count",
        purpose: "set how many data-derived breakpoints form the piecewise curves",
    },
    PublicName {
        key: "op.winsor_clip.q",
        public_name: "feature tail clip fraction",
        purpose: "set the fraction clipped from each feature tail",
    },
    PublicName {
        key: "op.cheb.degree",
        public_name: "polynomial degree",
        purpose: "set the highest polynomial order used for each feature",
    },
    PublicName {
        key: "op.rff_cos.n",
        public_name: "cosine feature count",
        purpose: "set how many cosine features approximate the smooth similarity",
    },
    PublicName {
        key: "op.rff_cos.gamma",
        public_name: "cosine similarity width",
        purpose: "set how quickly similarity decreases with distance",
    },
    PublicName {
        key: "op.rff_cos.seed",
        public_name: "cosine feature seed",
        purpose: "make randomized cosine features repeatable",
    },
    PublicName {
        key: "op.rff_sin.n",
        public_name: "sine feature count",
        purpose: "set how many sine features approximate the smooth similarity",
    },
    PublicName {
        key: "op.rff_sin.gamma",
        public_name: "sine similarity width",
        purpose: "set how quickly similarity decreases with distance",
    },
    PublicName {
        key: "op.rff_sin.seed",
        public_name: "sine feature seed",
        purpose: "make randomized sine features repeatable",
    },
    PublicName {
        key: "op.rand_proj.k",
        public_name: "projection count",
        purpose: "set how many projected feature directions are retained",
    },
    PublicName {
        key: "op.rand_proj.seed",
        public_name: "projection seed",
        purpose: "make the randomized projection repeatable",
    },
    PublicName {
        key: "op.landmark_dist.k",
        public_name: "landmark count",
        purpose: "set how many reference rows define the distance features",
    },
    PublicName {
        key: "op.landmark_dist.gamma",
        public_name: "landmark similarity width",
        purpose: "set how quickly landmark similarity decreases with distance",
    },
    PublicName {
        key: "op.landmark_dist.seed",
        public_name: "landmark selection seed",
        purpose: "make the randomized landmark selection repeatable",
    },
    PublicName {
        key: "op.oblique_hinge.n",
        public_name: "projected hinge count",
        purpose: "set how many learned directions receive piecewise thresholds",
    },
    PublicName {
        key: "op.oblique_hinge.seed",
        public_name: "projected hinge seed",
        purpose: "make the randomized directions repeatable",
    },
    PublicName {
        key: "op.pair_prod.topk",
        public_name: "feature product limit",
        purpose: "limit how many strongly related feature pairs are combined",
    },
    PublicName {
        key: "op.target_bin_encode.bins",
        public_name: "target encoding bin count",
        purpose: "set how many feature groups receive target-based values",
    },
    PublicName {
        key: "op.target_bin_encode.prior",
        public_name: "target encoding prior strength",
        purpose: "set how strongly each group is smoothed toward the overall target mean",
    },
    PublicName {
        key: "op.step.t",
        public_name: "step threshold",
        purpose: "set the cutoff above which a value becomes one",
    },
    PublicName {
        key: "op.lag.k",
        public_name: "lag distance",
        purpose: "set how many row positions back a lagged value is read",
    },
    PublicName {
        key: "op.roll.k",
        public_name: "rotation distance",
        purpose: "set how many positions each row is rotated",
    },
    PublicName {
        key: "op.fft_mag.k",
        public_name: "frequency limit",
        purpose: "set the highest frequency included in each row summary",
    },
    PublicName {
        key: "op.svd_project.k",
        public_name: "component count",
        purpose: "set how many projected directions are retained",
    },
    PublicName {
        key: "op.svd_project.iters",
        public_name: "projection iteration count",
        purpose: "set how many refinement steps estimate the components",
    },
    PublicName {
        key: "op.svd_project.seed",
        public_name: "component seed",
        purpose: "make the randomized component estimate repeatable",
    },
    PublicName {
        key: "op.nystrom.k",
        public_name: "kernel landmark count",
        purpose: "set how many reference rows approximate the kernel",
    },
    PublicName {
        key: "op.nystrom.gamma",
        public_name: "landmark kernel width",
        purpose: "set how quickly kernel similarity decreases with distance",
    },
    PublicName {
        key: "op.nystrom.seed",
        public_name: "kernel landmark seed",
        purpose: "make the randomized landmark selection repeatable",
    },
    PublicName {
        key: "op.symlog.c",
        public_name: "signed log scale",
        purpose: "set the scale at which signed logarithmic compression changes",
    },
    PublicName {
        key: "op.select_corr.k",
        public_name: "screened feature count",
        purpose: "set how many features pass the correlation screen",
    },
    PublicName {
        key: "op.select_corr.by_var",
        public_name: "feature screening basis",
        purpose: "choose whether features are ranked by target relation or training variation",
    },
    PublicName {
        key: "op.impute.strategy",
        public_name: "replacement method",
        purpose: "choose mean, median, or constant replacement for missing values",
    },
    PublicName {
        key: "op.impute.value",
        public_name: "constant replacement value",
        purpose: "set the replacement used when the constant method is selected",
    },
    PublicName {
        key: "op.impute.add_indicator",
        public_name: "missingness marker setting",
        purpose: "choose whether output columns mark the locations of missing values",
    },
    PublicName {
        key: "op.pair_ratio.topk",
        public_name: "ratio feature limit",
        purpose: "limit how many target-related feature ratios are formed",
    },
    PublicName {
        key: "op.spline.knots",
        public_name: "spline knot count",
        purpose: "set how many training quantiles form the spline basis",
    },
    PublicName {
        key: "op.row_vector_input_v2.width",
        public_name: "row input width",
        purpose: "declare the feature width expected by the row-vector operation",
    },
    PublicName {
        key: "op.scalar_v2.carrier",
        public_name: "formula evaluation scope",
        purpose: "choose whether the formula evaluates once per row or for each input column",
    },
    PublicName {
        key: "op.scalar_v2.width",
        public_name: "formula input width",
        purpose: "declare the feature width accepted by the scalar formula",
    },
    PublicName {
        key: "stage.append",
        public_name: "append transform output",
        purpose: "choose whether a stage adds output columns or replaces its input columns",
    },
];
