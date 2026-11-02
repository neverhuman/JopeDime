use crate::dsl::{ParamValue, Stage};
use crate::metrics::{mae, rmse};

pub const DEFAULT_LEAVES: usize = 31;
pub const DEFAULT_LR: f64 = 0.05;
/// Bag count default; sized so the emitted pure-Python twin stays inside its
/// training budget (measured in battle-transpile golden tests).
pub const DEFAULT_BAGS: usize = 3;

pub const MIN_LEAF: usize = 3;
pub const LAMBDA: f64 = 1.0;
pub const SUBSAMPLE: f64 = 0.7;
pub const PATIENCE: usize = 50;
/// Default (and historical hard cap on) boosting rounds; the `rounds` param
/// can now raise this to `ROUNDS_CAP`.
pub const MAX_ROUNDS: usize = 500;
/// Hard cap on the searchable `rounds` parameter.
pub const ROUNDS_CAP: usize = 1000;
pub const MAX_FEATURES: usize = 256;
/// PCG32 stream id for the per-bag row-subsampling generator.
pub const BAG_STREAM: u64 = 1442;
/// Default per-tree column share (1.0 = no column subsampling, and — key for
/// backward compatibility — zero extra RNG draws).
pub const DEFAULT_COL_SUBSAMPLE: f64 = 1.0;
/// Default internal-holdout share in percent (n_tr = n*(100-val_pct)/100;
/// 20 reproduces the historical contiguous 80/20 split bit-for-bit).
pub const DEFAULT_VAL_PCT: usize = 20;
/// PCG32 seed for the optional shuffled holdout split (`val_shuffle=true`);
/// the stream is the row count, so the split is deterministic per dataset size.
pub const HOLDOUT_SEED: u64 = 4127;

/// Default Huber clip when `loss=huber` is requested without a delta.
pub const DEFAULT_HUBER_DELTA: f64 = 0.1;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HistGbm2Params {
    pub leaves: usize,
    pub lr: f64,
    pub bags: usize,
    /// Huber gradient clip; <= 0 means plain squared loss (the default).
    pub huber_delta: f64,
    /// Early-stop the holdout on MAE instead of RMSE.
    pub stop_mae: bool,
    /// Boosting-round budget (early stopping usually ends sooner).
    pub rounds: usize,
    /// Per-tree row-sampling share.
    pub row_subsample: f64,
    /// Per-tree column-sampling share (1.0 = off, no RNG draws).
    pub col_subsample: f64,
    /// Minimum rows per leaf for a split to qualify.
    pub min_leaf: usize,
    /// Internal-holdout share in percent of n.
    pub val_pct: usize,
    /// Seeded-shuffle the holdout split instead of the contiguous tail.
    pub val_shuffle: bool,
}

impl Default for HistGbm2Params {
    fn default() -> Self {
        HistGbm2Params {
            leaves: DEFAULT_LEAVES,
            lr: DEFAULT_LR,
            bags: DEFAULT_BAGS,
            huber_delta: 0.0,
            stop_mae: false,
            rounds: MAX_ROUNDS,
            row_subsample: SUBSAMPLE,
            col_subsample: DEFAULT_COL_SUBSAMPLE,
            min_leaf: MIN_LEAF,
            val_pct: DEFAULT_VAL_PCT,
            val_shuffle: false,
        }
    }
}

impl HistGbm2Params {
    pub fn from(stage: &Stage) -> HistGbm2Params {
        let huber = matches!(
            stage.params.get("loss"),
            Some(ParamValue::Str(s)) if s == "huber"
        );
        HistGbm2Params {
            leaves: stage.get_usize("leaves", DEFAULT_LEAVES).clamp(2, 255),
            lr: stage.get_f64("lr", DEFAULT_LR),
            bags: stage.get_usize("bags", DEFAULT_BAGS).clamp(1, 10),
            huber_delta: if huber {
                stage.get_f64("delta", DEFAULT_HUBER_DELTA).clamp(1e-6, 1e6)
            } else {
                0.0
            },
            stop_mae: matches!(
                stage.params.get("stop"),
                Some(ParamValue::Str(s)) if s == "mae"
            ),
            rounds: stage.get_usize("rounds", MAX_ROUNDS).clamp(1, ROUNDS_CAP),
            row_subsample: stage.get_f64("row_subsample", SUBSAMPLE).clamp(0.5, 1.0),
            col_subsample: stage
                .get_f64("col_subsample", DEFAULT_COL_SUBSAMPLE)
                .clamp(0.5, 1.0),
            min_leaf: stage.get_usize("min_leaf", MIN_LEAF).clamp(1, 20),
            val_pct: stage.get_usize("val_pct", DEFAULT_VAL_PCT).clamp(5, 40),
            val_shuffle: matches!(
                stage.params.get("val_shuffle"),
                Some(ParamValue::Bool(true))
            ),
        }
    }
}

/// Holdout early-stopping criterion (see the module notes).
pub(crate) fn holdout_error(y: &[f64], p: &[f64], stop_mae: bool) -> f64 {
    if stop_mae {
        mae(y, p)
    } else {
        rmse(y, p)
    }
}
