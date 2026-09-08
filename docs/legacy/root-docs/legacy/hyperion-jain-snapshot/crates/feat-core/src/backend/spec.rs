use serde::{Deserialize, Serialize};

/// What the model predicts and the loss it trains (model-side mirror of `task::TaskKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskKind {
    BinaryClass,
    MultiClass,
    MultiLabel,
    Regression,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossKind {
    Logloss,
    CrossEntropy,
    MultiLogloss,
    MultiClass,
    Rmse,
    Mae,
    Quantile,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ObjectiveSpec {
    pub task: TaskKind,
    pub loss: LossKind,
}

impl ObjectiveSpec {
    /// Derive the model-side objective from the inferred data task.
    pub fn from_task(t: &crate::task::TaskSpec) -> Self {
        Self::from_kind(t.kind)
    }

    pub fn from_kind(kind: crate::task::TaskKind) -> Self {
        use crate::task::TaskKind as TK;
        let (task, loss) = match kind {
            TK::Binary => (TaskKind::BinaryClass, LossKind::Logloss),
            TK::MultiClass => (TaskKind::MultiClass, LossKind::MultiClass),
            TK::MultiLabel => (TaskKind::MultiLabel, LossKind::Logloss),
            TK::Regression => (TaskKind::Regression, LossKind::Rmse),
        };
        ObjectiveSpec { task, loss }
    }
}

/// Metric used for fixed-fold early stopping / model selection (never the blind holdout).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvalMetric {
    Auc,
    Logloss,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoostingType {
    Plain,
    Ordered,
    Goss,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrowPolicy {
    Symmetric,
    DepthWise,
    LossGuide,
    LeafWise,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendKind {
    CatBoost,
    XGBoost,
    LightGbm,
    Mock,
    Glm,
    Rff,
    Knn,
    Mlp,
    SmallNBma,
    RustPfn,
    /// An invented pipeline genome executed by the feat-math interpreter.
    Invented,
}

#[cfg(feature = "ci-smoke")]
pub const CI_SMOKE_BACKEND: &str = "ci_smoke";

/// Training hyperparameters. Defaults preserve the original CatBoost-era behavior.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TrainSpec {
    pub iters: usize,
    pub depth: usize,
    pub lr: f64,
    pub thread_count: usize,
    pub seed: u64,
    pub objective: ObjectiveSpec,
    pub eval_metric: EvalMetric,
    pub early_stopping_rounds: Option<usize>,
    pub use_best_iteration: bool,
    pub l2: f64,
    pub row_subsample: f64,
    pub col_subsample: f64,
    pub border_count: usize,
    pub boosting_type: BoostingType,
    pub grow_policy: GrowPolicy,
    pub random_strength: Option<f64>,
    pub bagging_temperature: Option<f64>,
    pub class_weights: Option<[f64; 2]>,
    pub scale_pos_weight: Option<f64>,
}

impl Default for TrainSpec {
    fn default() -> Self {
        TrainSpec {
            iters: 200,
            depth: 6,
            lr: 0.05,
            thread_count: 1,
            seed: 0,
            objective: ObjectiveSpec {
                task: TaskKind::BinaryClass,
                loss: LossKind::Logloss,
            },
            eval_metric: EvalMetric::Auc,
            early_stopping_rounds: None,
            use_best_iteration: false,
            l2: 3.0,
            row_subsample: 1.0,
            col_subsample: 1.0,
            border_count: 128,
            boosting_type: BoostingType::Plain,
            grow_policy: GrowPolicy::Symmetric,
            random_strength: None,
            bagging_temperature: None,
            class_weights: None,
            scale_pos_weight: None,
        }
    }
}

impl TrainSpec {
    /// Build from the five basic CatBoost-era fields; v2 fields take behavior-preserving defaults.
    pub fn from_basic_fields(
        iters: usize,
        depth: usize,
        lr: f64,
        thread_count: usize,
        seed: u64,
    ) -> Self {
        TrainSpec {
            iters,
            depth,
            lr,
            thread_count,
            seed,
            ..Default::default()
        }
    }
}

/// Dataset size regime, used to pick winning default hyperparameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SizeRegime {
    Tiny,
    Small,
    Medium,
    Large,
}

#[derive(Clone, Copy, Debug)]
pub struct DataRegime {
    pub size: SizeRegime,
    pub wide: bool,
    pub imbalanced: bool,
}

pub fn infer_regime(n: usize, p: usize, prevalence: f64) -> DataRegime {
    let size = match n {
        0..=2_000 => SizeRegime::Tiny,
        2_001..=10_000 => SizeRegime::Small,
        10_001..=100_000 => SizeRegime::Medium,
        _ => SizeRegime::Large,
    };
    let wide = p > 1000 || (p as f64 / n.max(1) as f64) > 0.5;
    let imbalanced = prevalence.min(1.0 - prevalence) <= 0.10;
    DataRegime {
        size,
        wide,
        imbalanced,
    }
}

/// Winning out-of-the-box defaults per regime + backend.
pub fn auto_defaults(
    regime: DataRegime,
    backend: BackendKind,
    prevalence: f64,
    seed: u64,
    thread_count: usize,
) -> TrainSpec {
    let (iters, lr, depth, l2, row, col, border, es) = match regime.size {
        SizeRegime::Tiny => (3000, 0.025, 4, 10.0, 0.85, 0.80, 64, 400),
        SizeRegime::Small => (2500, 0.035, 5, 6.0, 0.85, 0.85, 128, 300),
        SizeRegime::Medium => (1800, 0.045, 6, 3.0, 0.80, 0.90, 128, 250),
        SizeRegime::Large => (1200, 0.060, 7, 1.0, 0.70, 0.80, 255, 150),
    };
    let grow = match (backend, regime.size) {
        (BackendKind::LightGbm, SizeRegime::Large) => GrowPolicy::LeafWise,
        (BackendKind::XGBoost, SizeRegime::Large) => GrowPolicy::LossGuide,
        _ => GrowPolicy::Symmetric,
    };
    let boosting = match (backend, regime.size) {
        (BackendKind::CatBoost, SizeRegime::Tiny) | (BackendKind::CatBoost, SizeRegime::Small) => {
            BoostingType::Ordered
        }
        _ => BoostingType::Plain,
    };
    let mut s = TrainSpec {
        iters,
        depth,
        lr,
        thread_count,
        seed,
        l2,
        row_subsample: row,
        col_subsample: col,
        border_count: border,
        boosting_type: boosting,
        grow_policy: grow,
        early_stopping_rounds: Some(es),
        use_best_iteration: true,
        ..Default::default()
    };
    if regime.wide {
        s.depth = s.depth.min(5);
        s.l2 = s.l2.max(12.0);
        s.col_subsample = s.col_subsample.min(0.50);
        s.border_count = s.border_count.min(64);
    }
    if regime.imbalanced {
        let pos = prevalence.clamp(1e-6, 1.0 - 1e-6);
        let neg = 1.0 - pos;
        s.scale_pos_weight = Some(neg / pos);
        s.class_weights = Some([0.5 / neg, 0.5 / pos]);
    }
    s
}

/// Build task-aware, regime-adaptive training specs from the dataset shape.
pub fn auto_specs(
    n: usize,
    p: usize,
    prevalence: f64,
    task: crate::task::TaskKind,
    seed: u64,
    cv_threads: usize,
    ho_threads: usize,
) -> (TrainSpec, TrainSpec, TrainSpec) {
    let regime = infer_regime(n, p, prevalence);
    let obj = ObjectiveSpec::from_kind(task);
    let mut ho = auto_defaults(regime, BackendKind::CatBoost, prevalence, seed, ho_threads);
    ho.objective = obj;
    let mut cv = ho;
    cv.iters = (ho.iters / 2).max(100);
    cv.thread_count = cv_threads;
    let resid = cv;
    (cv, ho, resid)
}
