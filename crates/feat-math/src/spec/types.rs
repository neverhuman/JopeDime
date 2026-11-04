use super::*;

#[derive(Clone, Debug)]
pub enum Plan {
    Pipeline {
        /// Optional target-side transform (always the FIRST stage of the
        /// source pipeline; same split semantics as `model.rs::split_target`).
        target: Option<PlannedTarget>,
        stages: Vec<PlannedStage>,
        head: PlannedHead,
    },
    Symbolic {
        source: String,
        expr: MathAst,
        affine: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetKind {
    RankGauss,
    Log1p,
    Winsor,
    ZScore,
}

#[derive(Clone, Debug)]
pub struct PlannedTarget {
    pub kind: TargetKind,
    /// Winsor tail fraction, pre-clamped to [0, 0.25] exactly as
    /// `target_transform.rs::fit_target` clamps it; unused by other kinds.
    pub q: f64,
}

#[derive(Clone, Debug)]
pub struct PlannedStage {
    pub op: PlannedOp,
    pub append: bool,
}

#[derive(Clone, Debug)]
pub enum PlannedOp {
    Standardize,
    RobustScale,
    Rank,
    Blom,
    QuantileBins {
        bins: usize,
    },
    BinOnehot {
        bins: usize,
    },
    Hinge {
        knots: usize,
    },
    WinsorClip {
        q: f64,
    },
    Cheb {
        degree: usize,
    },
    PolyInteract,
    RffCos {
        n: usize,
        gamma: f64,
        seed: u64,
    },
    RffSin {
        n: usize,
        gamma: f64,
        seed: u64,
    },
    RandProj {
        k: usize,
        seed: u64,
    },
    LandmarkDist {
        k: usize,
        gamma: f64,
        seed: u64,
    },
    ObliqueHinge {
        n: usize,
        seed: u64,
    },
    RowStats,
    PairProd {
        topk: usize,
    },
    TargetBinEncode {
        bins: usize,
        prior: f64,
    },
    Sigmoid,
    /// Per-column hard indicator `1.0 if v > t else 0.0` at a fixed parameter
    /// threshold `t` (default 0.5). Stateless — the honest replacement for the
    /// `clip((v - t) * 1e15, 0, 1)` idiom.
    Step {
        t: f64,
    },
    NormRow,
    Polar,
    Lag {
        k: usize,
    },
    Roll {
        k: usize,
    },
    /// `k` is the raw request; the artifact clamps to [1, d/2] at fit time
    /// (the training width is unknown at emission).
    FftMag {
        k: usize,
    },
    RowSpectral,
    RowDiffStats,
    /// `k` is pre-clamped to [1, 64]; the artifact applies `.min(cols)` at fit.
    SvdProject {
        k: usize,
        iters: usize,
        seed: u64,
    },
    /// `k` is pre-clamped to [1, 128]; the artifact applies `.min(n)` at fit.
    Nystrom {
        k: usize,
        gamma: f64,
        seed: u64,
    },
    MapExpr {
        source: String,
        expr: MathAst,
    },
    DeriveExpr {
        source: String,
        expr: MathAst,
    },
    /// Signed shifted-log compression asinh(v / c); `c` pre-clamped to
    /// [1e-9, 1e9]. Stateless.
    Symlog {
        c: f64,
    },
    /// Top-k column screening; `k` pre-clamped to [1, 512], the artifact
    /// applies `.min(cols)` at fit. `by_var` false = |pearson(x_j, y)|
    /// (target-aware), true = train variance.
    SelectCorr {
        k: usize,
        by_var: bool,
    },
    /// Zero-as-missing imputation. `strategy`: 0=mean, 1=median, 2=const;
    /// `value` is the const fill (pre-clamped to [-1e12, 1e12], passed but
    /// unused for mean/median); `add_indicator` appends one 0/1 missingness
    /// column per train-zero-containing column.
    Impute {
        strategy: usize,
        value: f64,
        add_indicator: bool,
    },
    /// Target-guided regularized ratios x_a * x_b / (x_b^2 + eps) over the
    /// screened top-32 columns; `topk` pre-clamped to [1, 64].
    PairRatio {
        topk: usize,
    },
    /// Natural cubic spline basis at train quantile knots; `knots` pre-clamped
    /// to [3, 12].
    Spline {
        knots: usize,
    },
    // B12: constructed only by the explicit v2 row-vector planner.
    RowVectorV2 {
        state: crate::kernel::row_vector::FittedStage,
    },
    RowVectorInputV2 {
        width: usize,
    },
    // B05–B07: explicit scalar-v2 raw-input stage, replacing the feature frame.
    ScalarV2 {
        program: crate::kernel::scalar_program::ScalarExpressionProgram,
        carrier: crate::kernel::scalar_program::ScalarCarrier,
        width: usize,
    },
}

#[derive(Clone, Debug)]
pub struct GbmPlan {
    pub rounds: usize,
    pub depth: usize,
    pub lr: f64,
    pub lambda: f64,
    pub bins: usize,
    pub min_leaf: usize,
    pub max_features: usize,
}

#[derive(Clone, Debug)]
pub enum PlannedHead {
    Ridge {
        lambda_frac: f64,
    },
    /// `ridge_gcv`: parameter-free ridge whose penalty is chosen by generalized
    /// cross-validation over the fixed grid logspace(-5, 3, 25); fully resolved
    /// by construction.
    RidgeGcv,
    ElasticNet {
        alpha: f64,
        l1_ratio: f64,
        sweeps: usize,
    },
    Gbm(GbmPlan),
    Blend {
        lambda_frac: f64,
        gbm: GbmPlan,
    },
    /// `histgbm2`: leaf-wise Newton boosting with exact splits and bagging.
    /// Defaults replay the historical head bit-for-bit.
    HistGbm2 {
        leaves: usize,
        lr: f64,
        bags: usize,
        rounds: usize,
        row_subsample: f64,
        col_subsample: f64,
        min_leaf: usize,
        val_pct: usize,
        val_shuffle: bool,
        /// Huber gradient clip; 0.0 means plain squared loss.
        huber_delta: f64,
        /// Early-stop the holdout on MAE instead of RMSE.
        stop_mae: bool,
    },
    /// `krr`: Gaussian kernel ridge regression with grid-selected width/penalty.
    Krr {
        m: usize,
    },
    /// `gpr`: Gaussian-process regression with a learned ARD kernel.
    Gpr {
        m: usize,
        steps: usize,
    },
    /// `gpr kernel="..."`: the composite-kernel GP (sum/product of primitives).
    GprKernel {
        m: usize,
        steps: usize,
        source: String,
        spec: KernelSpec,
    },
    /// `ksmooth`: Nadaraya-Watson kernel-weighted nearest-neighbor smoother.
    Ksmooth {
        k: usize,
        gamma: f64,
    },
    /// `stack`: greedy ensemble selection over the fixed five-model library.
    Stack,
    /// `portfolio` (compact head): the pinned member registry expanded to a
    /// fixed component roster, gated at fit time on train stats, measured by
    /// seeded shuffled repeated K-fold OOF predictions, and combined by a
    /// non-negative ridge blend in deviation-from-mean space.
    Portfolio(PortfolioPlan),
}

#[derive(Clone, Debug)]
pub struct PortfolioPlan {
    pub folds: usize,
    pub repeats_auto: bool,
    /// Registry members after preset/add/drop resolution, in registry order.
    pub members: Vec<PlannedMember>,
    /// Index (0-based, into `members`) of the small-sample fallback
    /// component; when `ridge_gcv_raw` was dropped from the roster an extra
    /// hidden member is appended for this role (never gated in).
    pub fallback: usize,
}

#[derive(Clone, Debug)]
pub struct PlannedMember {
    /// Registry name — plan metadata only, NEVER emitted (banned vocabulary).
    pub name: String,
    pub gate: crate::members::Gate,
    pub proj_view: bool,
    /// Hidden members exist only for the small-n fallback: they are emitted
    /// as components but never appended to the gated roster.
    pub hidden: bool,
    pub kind: PlannedMemberKind,
}

#[derive(Clone, Debug)]
pub enum PlannedMemberKind {
    Pipeline {
        stages: Vec<PlannedStage>,
        head: Box<PlannedHead>,
    },
    GroupMean,
}
