//! `histgbm2` — leaf-wise (best-first) Newton-boosted trees with EXACT split
//! search over pre-sorted feature indices, per-tree row subsampling, early
//! stopping on an internal holdout, and bagging over a few deterministic seeds.
//!
//! Searchable depth/capacity knobs (all defaults replay the historical
//! behavior bit-for-bit): `leaves` (2-255), `rounds` (up to 1000, early
//! stopping with patience 50 usually ends sooner), `row_subsample` (0.5-1.0),
//! `col_subsample` (0.5-1.0; per-tree feature masks drawn after the row mask,
//! so 1.0 consumes no extra randomness), `min_leaf` (1-20), `val_pct`
//! (holdout share in percent; 20 = the historical contiguous 80/20 split) and
//! `val_shuffle` (seeded-shuffled holdout rows instead of the contiguous tail).
//!
//! Losses (`loss=` param, squared by default):
//! - squared: g_i = f_i - y_i, h_i = 1, leaf value -Σg/(Σh + lambda).
//! - huber (`loss=huber delta=0.1`): Newton boosting with the Huber loss
//!   (Huber, 1964). The gradient is the residual clipped at delta,
//!   g_i = clip(f_i - y_i, -delta, delta), and the curvature is the constant
//!   h_i = 1 (the LightGBM convention for Huber), so the tree machinery —
//!   count-based hessian sums and the same gain formula — is unchanged and a
//!   single outlier can pull any leaf by at most delta/(h + lambda).
//!
//! Early stopping (`stop=` param): the holdout criterion is RMSE by default
//! or MAE with `stop=mae` (less dominated by tail rows on noisy targets).
//!
//! Every arithmetic step is ordered identically in the emitted Python and Rust
//! artifacts (battle-transpile), so predictions are bit-identical across the
//! three implementations. All randomness is PCG32 keyed by the bag index.

use crate::dsl::{ParamValue, Stage};
use crate::metrics::{mae, rmse};
use crate::rng::Pcg32;
use crate::Rows;

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
fn holdout_error(y: &[f64], p: &[f64], stop_mae: bool) -> f64 {
    if stop_mae {
        mae(y, p)
    } else {
        rmse(y, p)
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum BoostNode {
    Leaf(f64),
    Split {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
}

pub type BoostTree = Vec<BoostNode>;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HistGbm2Model {
    pub init: f64,
    pub lr: f64,
    pub n_feat: usize,
    /// One tree list per bag; prediction is init + lr * mean over bags of the
    /// summed leaf values.
    pub bags: Vec<Vec<BoostTree>>,
}

impl HistGbm2Model {
    pub fn fit(x: &Rows, y: &[f64], cfg: &HistGbm2Params) -> HistGbm2Model {
        let n = x.len().min(y.len());
        let cols = if x.is_empty() { 0 } else { x[0].len() };
        let n_feat = cols.min(MAX_FEATURES);
        if n < 12 || n_feat == 0 {
            let m = mean(&y[..n.max(1).min(y.len())]);
            return HistGbm2Model {
                init: m,
                lr: cfg.lr,
                n_feat: 0,
                bags: Vec::new(),
            };
        }
        // Internal holdout split: contiguous tail by default (bit-identical to
        // the historical 80/20), or a seeded shuffle with `val_shuffle=true`.
        let n_tr = (n * (100 - cfg.val_pct)) / 100;
        let n_hold = n - n_tr;
        let mut split_idx: Vec<usize> = (0..n).collect();
        if cfg.val_shuffle {
            let mut rng = Pcg32::new(HOLDOUT_SEED, n as u64);
            for i in (1..n).rev() {
                let j = (rng.next_u32() as usize) % (i + 1);
                split_idx.swap(i, j);
            }
        }
        let mut tr_idx = split_idx[..n_tr].to_vec();
        tr_idx.sort_unstable();
        let mut hold_idx = split_idx[n_tr..].to_vec();
        hold_idx.sort_unstable();
        let xtr: Rows = tr_idx.iter().map(|&i| x[i].clone()).collect();
        let ytr: Vec<f64> = tr_idx.iter().map(|&i| y[i]).collect();
        let xh: Rows = hold_idx.iter().map(|&i| x[i].clone()).collect();
        let y_hold: Vec<f64> = hold_idx.iter().map(|&i| y[i]).collect();
        let init = mean(&ytr);
        // Feature-major copy of the training values the split search and the partition read
        // (`xcol[f * n_tr + i] == xtr[i][f]`): one contiguous column per feature instead of a
        // row-pointer chase per cell. Same values, same comparisons.
        let mut xcol = vec![0.0f64; n_feat * n_tr];
        for (i, row) in xtr.iter().enumerate() {
            for f in 0..n_feat {
                xcol[f * n_tr + i] = row[f];
            }
        }
        // Pre-sort feature indices once; ties break on the row index so the
        // order matches a stable value sort in every implementation.
        let order: Vec<Vec<usize>> = (0..n_feat)
            .map(|f| {
                let mut idx: Vec<usize> = (0..n_tr).collect();
                idx.sort_by(|&a, &b| {
                    if xtr[a][f] < xtr[b][f] {
                        std::cmp::Ordering::Less
                    } else if xtr[a][f] > xtr[b][f] {
                        std::cmp::Ordering::Greater
                    } else {
                        a.cmp(&b)
                    }
                });
                idx
            })
            .collect();
        let mut bags: Vec<Vec<BoostTree>> = Vec::with_capacity(cfg.bags);
        for bag in 1..=cfg.bags {
            let mut rng = Pcg32::new(bag as u64, BAG_STREAM);
            let mut f_tr = vec![init; n_tr];
            let mut f_hold = vec![init; n_hold];
            let mut best_score = holdout_error(&y_hold, &f_hold, cfg.stop_mae);
            let mut best_round: i64 = -1;
            let mut trees: Vec<BoostTree> = Vec::new();
            for round_index in 0..cfg.rounds {
                let in_sample: Vec<bool> = (0..n_tr)
                    .map(|_| rng.next_f64() < cfg.row_subsample)
                    .collect();
                // Column subsampling draws AFTER the row draws (and only when
                // active), so col_subsample=1.0 replays the historical stream.
                let col_ok: Vec<bool> = if cfg.col_subsample < 1.0 {
                    let mut ok: Vec<bool> = (0..n_feat)
                        .map(|_| rng.next_f64() < cfg.col_subsample)
                        .collect();
                    if !ok.iter().any(|&b| b) {
                        for v in ok.iter_mut() {
                            *v = true;
                        }
                    }
                    ok
                } else {
                    vec![true; n_feat]
                };
                // Huber: g = clip(residual, -delta, delta); squared: g = residual.
                let grad: Vec<f64> = if cfg.huber_delta > 0.0 {
                    (0..n_tr)
                        .map(|i| (f_tr[i] - ytr[i]).clamp(-cfg.huber_delta, cfg.huber_delta))
                        .collect()
                } else {
                    (0..n_tr).map(|i| f_tr[i] - ytr[i]).collect()
                };
                let tree = grow_leafwise(
                    &xcol,
                    n_tr,
                    &order,
                    &grad,
                    &in_sample,
                    &col_ok,
                    n_feat,
                    cfg.leaves,
                    cfg.min_leaf,
                );
                for (i, ft) in f_tr.iter_mut().enumerate() {
                    *ft += cfg.lr * route(&tree, &xtr[i]);
                }
                for (h, fh) in f_hold.iter_mut().enumerate() {
                    *fh += cfg.lr * route(&tree, &xh[h]);
                }
                trees.push(tree);
                let r = holdout_error(&y_hold, &f_hold, cfg.stop_mae);
                if r < best_score {
                    best_score = r;
                    best_round = round_index as i64;
                }
                if round_index as i64 - best_round >= PATIENCE as i64 {
                    break;
                }
            }
            trees.truncate((best_round + 1) as usize);
            bags.push(trees);
        }
        HistGbm2Model {
            init,
            lr: cfg.lr,
            n_feat,
            bags,
        }
    }

    pub fn predict(&self, x: &Rows) -> Vec<f64> {
        x.iter()
            .map(|row| {
                if self.bags.is_empty() {
                    return self.init;
                }
                let mut acc = 0.0;
                for trees in &self.bags {
                    let mut s = 0.0;
                    for tree in trees {
                        s += route(tree, row);
                    }
                    acc += s;
                }
                self.init + self.lr * acc / self.bags.len() as f64
            })
            .collect()
    }
}

/// A grown-but-unsplit leaf: its rows (per feature, in sorted order), gradient
/// sums, and its best prospective split (gain 0 when none qualifies).
struct LeafGroup {
    node: usize,
    lists: Vec<Vec<usize>>,
    g: f64,
    h: f64,
    gain: f64,
    feature: usize,
    threshold: f64,
}

/// `xcol` is the feature-major training matrix: `xcol[f * n_rows + i]` is row `i`, feature `f`.
#[allow(clippy::too_many_arguments)]
fn grow_leafwise(
    xcol: &[f64],
    n_rows: usize,
    order: &[Vec<usize>],
    grad: &[f64],
    in_sample: &[bool],
    col_ok: &[bool],
    n_feat: usize,
    max_leaves: usize,
    min_leaf: usize,
) -> BoostTree {
    let mut lists: Vec<Vec<usize>> = Vec::with_capacity(n_feat);
    for feature_order in order.iter().take(n_feat) {
        lists.push(
            feature_order
                .iter()
                .copied()
                .filter(|&i| in_sample[i])
                .collect(),
        );
    }
    let mut g = 0.0;
    for &i in &lists[0] {
        g += grad[i];
    }
    let h = lists[0].len() as f64;
    let mut tree: BoostTree = vec![BoostNode::Leaf(-g / (h + LAMBDA))];
    let mut root = LeafGroup {
        node: 0,
        lists,
        g,
        h,
        gain: 0.0,
        feature: 0,
        threshold: 0.0,
    };
    find_best_split(xcol, n_rows, grad, &mut root, col_ok, n_feat, min_leaf);
    let mut pool = vec![root];
    let mut leaf_count = 1usize;
    while leaf_count < max_leaves {
        // Highest-gain leaf splits next; strict > keeps the earliest on ties.
        let mut pick = usize::MAX;
        let mut pick_gain = 0.0f64;
        for (k, rec) in pool.iter().enumerate() {
            if rec.gain > pick_gain {
                pick_gain = rec.gain;
                pick = k;
            }
        }
        if pick == usize::MAX {
            break;
        }
        let rec = pool.remove(pick);
        let feat = rec.feature;
        let thr = rec.threshold;
        let mut left_lists: Vec<Vec<usize>> = Vec::with_capacity(n_feat);
        let mut right_lists: Vec<Vec<usize>> = Vec::with_capacity(n_feat);
        let split_col = &xcol[feat * n_rows..(feat + 1) * n_rows];
        // Every feature's list holds the same rows, so the child sizes are known up front and
        // each child list is allocated once at its exact size (stable partition, same order).
        let n_left = rec.lists[0]
            .iter()
            .filter(|&&i| split_col[i] <= thr)
            .count();
        let n_right = rec.lists[0].len() - n_left;
        for f in 0..n_feat {
            let mut l = Vec::with_capacity(n_left);
            let mut r = Vec::with_capacity(n_right);
            for &i in &rec.lists[f] {
                if split_col[i] <= thr {
                    l.push(i);
                } else {
                    r.push(i);
                }
            }
            left_lists.push(l);
            right_lists.push(r);
        }
        // Child sums are recomputed over the feature-0 ordering so every
        // implementation accumulates in the same sequence.
        let mut gl = 0.0;
        for &i in &left_lists[0] {
            gl += grad[i];
        }
        let hl = left_lists[0].len() as f64;
        let mut gr = 0.0;
        for &i in &right_lists[0] {
            gr += grad[i];
        }
        let hr = right_lists[0].len() as f64;
        let li = tree.len();
        tree.push(BoostNode::Leaf(-gl / (hl + LAMBDA)));
        let ri = tree.len();
        tree.push(BoostNode::Leaf(-gr / (hr + LAMBDA)));
        tree[rec.node] = BoostNode::Split {
            feature: feat,
            threshold: thr,
            left: li,
            right: ri,
        };
        let mut lrec = LeafGroup {
            node: li,
            lists: left_lists,
            g: gl,
            h: hl,
            gain: 0.0,
            feature: 0,
            threshold: 0.0,
        };
        find_best_split(xcol, n_rows, grad, &mut lrec, col_ok, n_feat, min_leaf);
        pool.push(lrec);
        let mut rrec = LeafGroup {
            node: ri,
            lists: right_lists,
            g: gr,
            h: hr,
            gain: 0.0,
            feature: 0,
            threshold: 0.0,
        };
        find_best_split(xcol, n_rows, grad, &mut rrec, col_ok, n_feat, min_leaf);
        pool.push(rrec);
        leaf_count += 1;
    }
    tree
}

/// Exact split search over the leaf's pre-sorted rows. Only boundaries between
/// distinct values qualify; strict > with ascending (feature, position) scan
/// order means the lowest feature/threshold index wins ties. Features with
/// `col_ok[f] == false` are skipped (per-tree column subsampling).
fn find_best_split(
    xcol: &[f64],
    n_rows: usize,
    grad: &[f64],
    rec: &mut LeafGroup,
    col_ok: &[bool],
    n_feat: usize,
    min_leaf: usize,
) {
    rec.gain = 0.0;
    let m = rec.lists[0].len();
    if m < 2 * min_leaf {
        return;
    }
    let parent = rec.g * rec.g / (rec.h + LAMBDA);
    let mut best_gain = 1e-9;
    for f in 0..n_feat {
        if !col_ok[f] {
            continue;
        }
        let lst = &rec.lists[f];
        let col = &xcol[f * n_rows..(f + 1) * n_rows];
        let mut gl = 0.0;
        for t in 0..(m - 1) {
            let i = lst[t];
            gl += grad[i];
            let cnt = t + 1;
            if cnt < min_leaf {
                continue;
            }
            if m - cnt < min_leaf {
                break;
            }
            let v = col[i];
            if v < col[lst[t + 1]] {
                let hl = cnt as f64;
                let hr = (m - cnt) as f64;
                let gr = rec.g - gl;
                let gain = gl * gl / (hl + LAMBDA) + gr * gr / (hr + LAMBDA) - parent;
                if gain > best_gain {
                    best_gain = gain;
                    rec.gain = gain;
                    rec.feature = f;
                    rec.threshold = v;
                }
            }
        }
    }
}

pub fn route(tree: &[BoostNode], row: &[f64]) -> f64 {
    let mut idx = 0usize;
    loop {
        match &tree[idx] {
            BoostNode::Leaf(v) => return *v,
            BoostNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                let v = if *feature < row.len() {
                    row[*feature]
                } else {
                    0.0
                };
                idx = if v <= *threshold { *left } else { *right };
            }
        }
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::r2;
    use crate::model::cv_r2;
    use crate::GenomeIr;

    fn toy() -> (Rows, Vec<f64>) {
        // Same toy as model.rs tests: y = 2*x0 - x1 + 0.3*sin(6*x2).
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut rng = Pcg32::new(1, 1);
        for _ in 0..240 {
            let a = rng.next_f64();
            let b = rng.next_f64();
            let c = rng.next_f64();
            x.push(vec![a, b, c]);
            y.push(2.0 * a - b + 0.3 * (6.0 * c).sin());
        }
        (x, y)
    }

    #[test]
    fn histgbm2_is_deterministic() {
        let (x, y) = toy();
        let cfg = HistGbm2Params::default();
        let m1 = HistGbm2Model::fit(&x, &y, &cfg);
        let m2 = HistGbm2Model::fit(&x, &y, &cfg);
        let p1 = m1.predict(&x);
        let p2 = m2.predict(&x);
        assert_eq!(p1, p2);
        assert!(r2(&y, &p1) > 0.8, "train r2={}", r2(&y, &p1));
    }

    #[test]
    fn histgbm2_beats_histgbm_on_toy_cv() {
        let (x, y) = toy();
        let old = cv_r2(
            &GenomeIr::parse("(pipe X (histgbm))").unwrap(),
            &x,
            &y,
            5,
            "toy",
        )
        .unwrap();
        let new = cv_r2(
            &GenomeIr::parse("(pipe X (histgbm2))").unwrap(),
            &x,
            &y,
            5,
            "toy",
        )
        .unwrap();
        println!(
            "accuracy[toy cv oof_r2] histgbm={} histgbm2={}",
            old.oof_r2, new.oof_r2
        );
        assert!(
            new.oof_r2 >= old.oof_r2,
            "histgbm2 oof={} < histgbm oof={}",
            new.oof_r2,
            old.oof_r2
        );
    }

    /// Backward compatibility: a genome that spells out every new parameter at
    /// its default must score bit-identically to the bare `(histgbm2)`.
    #[test]
    fn histgbm2_explicit_defaults_match_bare_head() {
        let (x, y) = toy();
        let bare = GenomeIr::parse("(pipe X (histgbm2))").unwrap();
        let full = GenomeIr::parse(
            "(pipe X (histgbm2 leaves=31 lr=0.05 bags=3 rounds=500 row_subsample=0.7 \
             col_subsample=1.0 min_leaf=3 val_pct=20 val_shuffle=false))",
        )
        .unwrap();
        let p1 = crate::model::fit_predict(&bare, &x, &y, &x).unwrap();
        let p2 = crate::model::fit_predict(&full, &x, &y, &x).unwrap();
        assert_eq!(p1, p2, "explicit defaults drifted from the bare head");
    }

    #[test]
    fn histgbm2_deep_config_is_deterministic_and_learns() {
        let (x, y) = toy();
        let g = GenomeIr::parse(
            "(pipe X (histgbm2 leaves=127 rounds=300 lr=0.05 bags=2 row_subsample=0.8 \
             col_subsample=0.7 min_leaf=2 val_pct=15 val_shuffle=true))",
        )
        .unwrap();
        let p1 = crate::model::fit_predict(&g, &x, &y, &x).unwrap();
        let p2 = crate::model::fit_predict(&g, &x, &y, &x).unwrap();
        assert_eq!(p1, p2);
        assert!(r2(&y, &p1) > 0.8, "train r2={}", r2(&y, &p1));
        // The shuffled 15% holdout actually moves the split point.
        let cfg = HistGbm2Params::from(match &g {
            GenomeIr::Pipeline(e) => e.stages.last().unwrap(),
            _ => unreachable!(),
        });
        assert_eq!(cfg.val_pct, 15);
        assert!(cfg.val_shuffle);
        assert_eq!(cfg.rounds, 300);
        assert_eq!(cfg.min_leaf, 2);
    }

    #[test]
    fn histgbm2_param_clamps_hold() {
        let g = GenomeIr::parse(
            "(pipe X (histgbm2 leaves=4096 rounds=9999 row_subsample=0.1 col_subsample=0.0 \
             min_leaf=500 val_pct=90))",
        )
        .unwrap();
        let cfg = HistGbm2Params::from(match &g {
            GenomeIr::Pipeline(e) => e.stages.last().unwrap(),
            _ => unreachable!(),
        });
        assert_eq!(cfg.leaves, 255);
        assert_eq!(cfg.rounds, 1000);
        assert_eq!(cfg.row_subsample, 0.5);
        assert_eq!(cfg.col_subsample, 0.5);
        assert_eq!(cfg.min_leaf, 20);
        assert_eq!(cfg.val_pct, 40);
    }

    #[test]
    fn histgbm2_small_n_falls_back_to_mean() {
        let x: Rows = (0..6).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let m = HistGbm2Model::fit(&x, &y, &HistGbm2Params::default());
        assert!(m.bags.is_empty());
        let p = m.predict(&x);
        assert!(p.iter().all(|v| (*v - 2.5).abs() < 1e-12));
    }
}
