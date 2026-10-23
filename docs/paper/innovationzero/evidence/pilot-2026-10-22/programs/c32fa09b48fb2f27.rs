//! Regression model: mid-rank transform, then pairwise products, then leaf-wise boosted trees.
//!
//! This program trains a regression model on train.csv, in which every row holds the feature
//! columns followed by the target as the last column, and then writes one prediction per line for
//! the rows of test_features.csv. All arithmetic is IEEE-754 double precision in a fixed
//! evaluation order, so repeated runs give identical bits.
//!
//! The feature pipeline has 2 stages, applied in order; each stage is marked "replaced" when it
//! overwrites the columns it receives or "appended" when it keeps them and adds its outputs
//! alongside, and after every stage non-finite cells become 0.0 and each row is capped at 512
//! columns. Stage 1 (mid_rank_transform, replaced) replaces each value by its mid-rank within the
//! training column, i.e. the average of the strict and non-strict ranks scaled to [0, 1], an
//! empirical distribution-function transform. Stage 2 (pairwise_products, appended) forms
//! degree-two polynomial features: the square of every column plus pairwise products, capped at
//! 512 outputs.
//!
//! The head (leafwise_boosted_trees) is gradient boosted regression trees (Friedman, 2001) grown
//! leaf-wise / best-first to at most 31 leaves with exact split search over pre-sorted feature
//! columns, in the manner of LightGBM (Ke et al., 2017); leaf values are the second-order step
//! -g/(h + 1) (Chen & Guestrin, 2016), learning rate 0.05, minimum leaf size 3, with a per-tree
//! row-sampling share of 0.7. 3 bagged replicate(s) (Breiman, 1996) each stop early by root mean
//! squared error on an internal 20% holdout (the contiguous tail rows; patience 50 rounds, at
//! most 500) and predictions average the replicates. Seeded stages draw weights from a permuted
//! congruential generator (PCG XSH-RR 64/32; O'Neill, 2014) keyed by the stage seed and the
//! input/output widths, so training is fully deterministic. Missing or malformed CSV cells parse
//! as 0.0 and short rows are padded with zeros to the training width.
//!
//! Build: rustc -O model.rs
//! Usage: ./model train.csv test_features.csv predictions.csv

#![forbid(unsafe_code)]
#![allow(dead_code)]

/// Permuted congruential generator, XSH-RR 64/32 variant (O'Neill, 2014).
struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    fn new(seed: u64, stream: u64) -> Pcg32 {
        let mut rng = Pcg32 { state: 0, inc: (stream << 1) | 1 };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
}

impl Pcg32 {
    /// Uniform in [0, 1) from 53 random bits: ((hi<<32|lo) >> 11) * 2^-53.
    fn next_f64(&mut self) -> f64 {
        let hi = (self.next_u32() as u64) << 32;
        let lo = self.next_u32() as u64;
        (((hi | lo) >> 11) as f64) * (1.0 / 9007199254740992.0)
    }
}

/// Replace non-finite values with 0.0 in every row.
fn sanitize(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| row.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect())
        .collect()
}

/// Stage epilogue: optionally keep the inputs, cap width at 512, zero non-finite cells.
fn finish_stage(x: &[Vec<f64>], mut out: Vec<Vec<f64>>, append: bool) -> Vec<Vec<f64>> {
    if append {
        for (i, row) in out.iter_mut().enumerate() {
            let mut merged = x[i].clone();
            merged.append(row);
            *row = merged;
        }
    }
    for row in out.iter_mut() {
        if row.len() > 512 {
            row.truncate(512);
        }
        for v in row.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        if row.is_empty() {
            row.push(0.0);
        }
    }
    out
}

/// Column j of x in ascending order.
fn sorted_column(x: &[Vec<f64>], j: usize) -> Vec<f64> {
    let mut v: Vec<f64> = x.iter().map(|row| row[j]).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v
}

/// Count of elements strictly less than v (hand-rolled binary search).
fn lower_bound(sorted_values: &[f64], v: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = sorted_values.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if sorted_values[mid] < v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Count of elements less than or equal to v (hand-rolled binary search).
fn upper_bound(sorted_values: &[f64], v: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = sorted_values.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if sorted_values[mid] <= v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Average of the strict and non-strict ranks of v, scaled to [0, 1].
fn mid_rank(sorted_values: &[f64], v: f64) -> f64 {
    let lb = lower_bound(sorted_values, v);
    let ub = upper_bound(sorted_values, v);
    (lb + ub) as f64 / (2.0 * sorted_values.len() as f64)
}

fn mean_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for v in values {
        total += *v;
    }
    total / values.len() as f64
}

/// Root mean squared error over the common prefix.
fn rmse(y: &[f64], p: &[f64]) -> f64 {
    let n = y.len().min(p.len());
    if n == 0 {
        return 0.0;
    }
    let mut ss = 0.0;
    for i in 0..n {
        let d = y[i] - p[i];
        ss += d * d;
    }
    (ss / n as f64).sqrt()
}

struct MidRankTransform {
    sorted_cols: Vec<Vec<f64>>,
}

/// This transform replaces each value by its mid-rank within the training column, i.e. the
/// average of the strict and non-strict ranks scaled to [0, 1], an empirical
/// distribution-function transform.
fn mid_rank_transform_fit(x: &[Vec<f64>]) -> MidRankTransform {
    MidRankTransform { sorted_cols: (0..x[0].len()).map(|j| sorted_column(x, j)).collect() }
}

fn mid_rank_transform_apply(state: &MidRankTransform, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, v)| mid_rank(&state.sorted_cols[j], *v))
                .collect()
        })
        .collect();
    finish_stage(x, out, append)
}

struct PairwiseProducts {
    pairs: Vec<(usize, usize)>,
}

/// This transform forms degree-two polynomial features: the square of every column plus
/// pairwise products, capped at 512 outputs.
fn pairwise_products_fit(x: &[Vec<f64>]) -> PairwiseProducts {
    let cols = x[0].len();
    let mut pairs: Vec<(usize, usize)> = (0..cols).map(|i| (i, i)).collect();
    let mut full = false;
    for i in 0..cols {
        if full {
            break;
        }
        for j in (i + 1)..cols {
            if pairs.len() >= 512 {
                full = true;
                break;
            }
            pairs.push((i, j));
        }
    }
    PairwiseProducts { pairs }
}

fn pairwise_products_apply(state: &PairwiseProducts, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| state.pairs.iter().map(|&(a, b)| row[a] * row[b]).collect())
        .collect();
    finish_stage(x, out, append)
}

/// One node of a leaf-wise regression tree; feature -1 marks a leaf.
struct BoostNode {
    feature: i64,
    threshold: f64,
    left: usize,
    right: usize,
    value: f64,
}

/// A grown-but-unsplit leaf: its rows per feature (in pre-sorted order) and
/// its best prospective split (gain 0 when none qualifies).
struct LeafGroup {
    node: usize,
    lists: Vec<Vec<usize>>,
    gain: f64,
    feature: usize,
    threshold: f64,
}

/// Early-stopping criterion: mean absolute error when `absolute`, else root
/// mean squared error.
fn holdout_error(y: &[f64], p: &[f64], absolute: bool) -> f64 {
    if absolute {
        let n = y.len().min(p.len());
        if n == 0 {
            return 0.0;
        }
        let mut s = 0.0;
        for i in 0..n {
            s += (y[i] - p[i]).abs();
        }
        return s / n as f64;
    }
    rmse(y, p)
}

/// Exact split search over the pre-sorted rows of one leaf.
///
/// Only boundaries between distinct values qualify; the strict > comparison
/// with an ascending (feature, position) scan keeps the lowest feature and
/// threshold on ties. Features with col_ok[f] false are skipped (per-tree
/// column subsampling). Returns (gain, feature, threshold); gain 0 means no
/// split clears the 1e-9 floor.
fn leafwise_best_split(x: &[Vec<f64>], grad: &[f64], lists: &[Vec<usize>], g: f64, h: f64,
                       col_ok: &[bool], n_feat: usize, min_leaf: usize, lam: f64) -> (f64, usize, f64) {
    let m = lists[0].len();
    if m < 2 * min_leaf {
        return (0.0, 0, 0.0);
    }
    let parent = g * g / (h + lam);
    let mut best_gain = 1e-9;
    let mut out = (0.0, 0usize, 0.0);
    for f in 0..n_feat {
        if !col_ok[f] {
            continue;
        }
        let lst = &lists[f];
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
            let v = x[i][f];
            if v < x[lst[t + 1]][f] {
                let hl = cnt as f64;
                let hr = (m - cnt) as f64;
                let gr = g - gl;
                let gain = gl * gl / (hl + lam) + gr * gr / (hr + lam) - parent;
                if gain > best_gain {
                    best_gain = gain;
                    out = (gain, f, v);
                }
            }
        }
    }
    out
}

/// Grow one regression tree leaf-wise (best-first): the leaf with the highest
/// gain splits next. Leaf values are the second-order step -g/(h + lam).
fn leafwise_tree(x: &[Vec<f64>], order: &[Vec<usize>], grad: &[f64], in_sample: &[bool],
                 col_ok: &[bool], n_feat: usize, max_leaves: usize, min_leaf: usize,
                 lam: f64) -> Vec<BoostNode> {
    let mut lists: Vec<Vec<usize>> = Vec::new();
    for f in 0..n_feat {
        let mut kept = Vec::new();
        for &i in &order[f] {
            if in_sample[i] {
                kept.push(i);
            }
        }
        lists.push(kept);
    }
    let mut g = 0.0;
    for &i in &lists[0] {
        g += grad[i];
    }
    let h = lists[0].len() as f64;
    let mut tree = vec![BoostNode { feature: -1, threshold: 0.0, left: 0, right: 0, value: -g / (h + lam) }];
    let (gain, feat, thr) = leafwise_best_split(x, grad, &lists, g, h, col_ok, n_feat, min_leaf, lam);
    let mut pool = vec![LeafGroup { node: 0, lists, gain, feature: feat, threshold: thr }];
    let mut leaf_count = 1usize;
    while leaf_count < max_leaves {
        let mut pick = usize::MAX;
        let mut pick_gain = 0.0;
        for k in 0..pool.len() {
            if pool[k].gain > pick_gain {
                pick_gain = pool[k].gain;
                pick = k;
            }
        }
        if pick == usize::MAX {
            break;
        }
        let group = pool.remove(pick);
        let feat = group.feature;
        let thr = group.threshold;
        let mut left_lists: Vec<Vec<usize>> = Vec::new();
        let mut right_lists: Vec<Vec<usize>> = Vec::new();
        for f in 0..n_feat {
            let mut left = Vec::new();
            let mut right = Vec::new();
            for &i in &group.lists[f] {
                if x[i][feat] <= thr {
                    left.push(i);
                } else {
                    right.push(i);
                }
            }
            left_lists.push(left);
            right_lists.push(right);
        }
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
        tree.push(BoostNode { feature: -1, threshold: 0.0, left: 0, right: 0, value: -gl / (hl + lam) });
        let ri = tree.len();
        tree.push(BoostNode { feature: -1, threshold: 0.0, left: 0, right: 0, value: -gr / (hr + lam) });
        tree[group.node] = BoostNode { feature: feat as i64, threshold: thr, left: li, right: ri, value: 0.0 };
        let (bg, bf, bt) = leafwise_best_split(x, grad, &left_lists, gl, hl, col_ok, n_feat, min_leaf, lam);
        pool.push(LeafGroup { node: li, lists: left_lists, gain: bg, feature: bf, threshold: bt });
        let (bg, bf, bt) = leafwise_best_split(x, grad, &right_lists, gr, hr, col_ok, n_feat, min_leaf, lam);
        pool.push(LeafGroup { node: ri, lists: right_lists, gain: bg, feature: bf, threshold: bt });
        leaf_count += 1;
    }
    tree
}

/// Follow split nodes (go left when value <= threshold) to a leaf value.
fn leafwise_route(tree: &[BoostNode], row: &[f64]) -> f64 {
    let mut idx = 0usize;
    loop {
        let node = &tree[idx];
        if node.feature < 0 {
            return node.value;
        }
        let f = node.feature as usize;
        let v = if f < row.len() { row[f] } else { 0.0 };
        idx = if v <= node.threshold { node.left } else { node.right };
    }
}

struct LeafwiseBoostedTrees {
    init: f64,
    lr: f64,
    bags: Vec<Vec<Vec<BoostNode>>>,
}

/// This head is gradient boosted regression trees (Friedman, 2001) grown leaf-wise / best-first
/// to at most 31 leaves with exact split search over pre-sorted feature columns, in the manner
/// of LightGBM (Ke et al., 2017); leaf values are the second-order step -g/(h + 1) (Chen &
/// Guestrin, 2016), learning rate 0.05, minimum leaf size 3, with a per-tree row-sampling share
/// of 0.7. 3 bagged replicate(s) (Breiman, 1996) each stop early by root mean squared error on
/// an internal 20% holdout (the contiguous tail rows; patience 50 rounds, at most 500) and
/// predictions average the replicates.
fn leafwise_boosted_trees_fit(x: &[Vec<f64>], y: &[f64], max_leaves: usize, lr: f64,
                              n_bags: usize, rounds: i64, row_share: f64, col_share: f64,
                              min_leaf: usize, holdout_pct: usize, shuffled_split: bool,
                              robust_clip: f64, absolute_error: bool) -> LeafwiseBoostedTrees {
    let n = x.len().min(y.len());
    let cols = x[0].len();
    let n_feat = cols.min(256);
    if n < 12 || n_feat == 0 {
        let m = mean_of(&y[..n.max(1).min(y.len())]);
        return LeafwiseBoostedTrees { init: m, lr, bags: Vec::new() };
    }
    // Internal holdout split: the contiguous tail by default, or rows chosen
    // by a seeded shuffle.
    let n_tr = (n * (100 - holdout_pct)) / 100;
    let n_hold = n - n_tr;
    let mut split_idx: Vec<usize> = (0..n).collect();
    if shuffled_split {
        let mut rng = Pcg32::new(4127, n as u64);
        for i in (1..n).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            split_idx.swap(i, j);
        }
    }
    let mut tr_idx = split_idx[..n_tr].to_vec();
    tr_idx.sort();
    let mut hold_idx = split_idx[n_tr..].to_vec();
    hold_idx.sort();
    let xtr: Vec<Vec<f64>> = tr_idx.iter().map(|&i| x[i].clone()).collect();
    let ytr: Vec<f64> = tr_idx.iter().map(|&i| y[i]).collect();
    let xh: Vec<Vec<f64>> = hold_idx.iter().map(|&i| x[i].clone()).collect();
    let y_hold: Vec<f64> = hold_idx.iter().map(|&i| y[i]).collect();
    let init = mean_of(&ytr);
    // Pre-sort feature indices once; ties break on the row index so the order
    // matches a stable value sort exactly.
    let mut order: Vec<Vec<usize>> = Vec::new();
    for f in 0..n_feat {
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
        order.push(idx);
    }
    let mut bags: Vec<Vec<Vec<BoostNode>>> = Vec::new();
    for bag in 1..(n_bags + 1) {
        let mut rng = Pcg32::new(bag as u64, 1442);
        let mut f_tr = vec![init; n_tr];
        let mut f_hold = vec![init; n_hold];
        let mut best_err = holdout_error(&y_hold, &f_hold, absolute_error);
        let mut best_round: i64 = -1;
        let mut trees: Vec<Vec<BoostNode>> = Vec::new();
        for round_index in 0i64..rounds {
            let mut in_sample = Vec::new();
            for _ in 0..n_tr {
                in_sample.push(rng.next_f64() < row_share);
            }
            // Column masks are drawn AFTER the row mask and only when active,
            // so a share of 1.0 consumes no randomness at all.
            let col_ok: Vec<bool> = if col_share < 1.0 {
                let mut ok = Vec::new();
                for _ in 0..n_feat {
                    ok.push(rng.next_f64() < col_share);
                }
                if !ok.iter().any(|&b| b) {
                    for v in ok.iter_mut() {
                        *v = true;
                    }
                }
                ok
            } else {
                vec![true; n_feat]
            };
            let mut grad = Vec::new();
            if robust_clip > 0.0 {
                for i in 0..n_tr {
                    grad.push((f_tr[i] - ytr[i]).clamp(-robust_clip, robust_clip));
                }
            } else {
                for i in 0..n_tr {
                    grad.push(f_tr[i] - ytr[i]);
                }
            }
            let tree = leafwise_tree(&xtr, &order, &grad, &in_sample, &col_ok, n_feat,
                                     max_leaves, min_leaf, 1.0);
            for i in 0..n_tr {
                f_tr[i] += lr * leafwise_route(&tree, &xtr[i]);
            }
            for hh in 0..n_hold {
                f_hold[hh] += lr * leafwise_route(&tree, &xh[hh]);
            }
            trees.push(tree);
            let r = holdout_error(&y_hold, &f_hold, absolute_error);
            if r < best_err {
                best_err = r;
                best_round = round_index;
            }
            if round_index - best_round >= 50 {
                break;
            }
        }
        trees.truncate((best_round + 1) as usize);
        bags.push(trees);
    }
    LeafwiseBoostedTrees { init, lr, bags }
}

fn leafwise_boosted_trees_predict(state: &LeafwiseBoostedTrees, x: &[Vec<f64>]) -> Vec<f64> {
    let mut predictions = Vec::new();
    for row in x {
        if state.bags.is_empty() {
            predictions.push(state.init);
            continue;
        }
        let mut acc = 0.0;
        for trees in &state.bags {
            let mut s = 0.0;
            for tree in trees {
                s += leafwise_route(tree, row);
            }
            acc += s;
        }
        predictions.push(state.init + state.lr * acc / state.bags.len() as f64);
    }
    predictions
}

/// Feature stages applied in order, then the leafwise_boosted_trees head.
struct Model {
    stage1: MidRankTransform,
    stage2: PairwiseProducts,
    head: LeafwiseBoostedTrees,
}

impl Model {
    /// Fit every stage in order on the training rows.
    fn fit(x: &[Vec<f64>], y: &[f64]) -> Model {
        if x.is_empty() || y.len() < x.len() {
            eprintln!("training data is empty or target is short");
            std::process::exit(2);
        }
        let x = sanitize(x);
        let stage1 = mid_rank_transform_fit(&x);
        let x = mid_rank_transform_apply(&stage1, &x, false);
        let stage2 = pairwise_products_fit(&x);
        let x = pairwise_products_apply(&stage2, &x, true);
        let head = leafwise_boosted_trees_fit(&x, y, 31, 0.05, 3, 500, 0.7, 1.0, 3, 20, false, 0.0, false);
        Model { stage1, stage2, head }
    }

    /// Apply the fitted stages, then the regression head.
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let x = sanitize(x);
        let x = mid_rank_transform_apply(&self.stage1, &x, false);
        let x = pairwise_products_apply(&self.stage2, &x, true);
        leafwise_boosted_trees_predict(&self.head, &x)
    }

    /// Coefficient of determination R^2 of predict against y.
    fn score(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let predictions = self.predict(x);
        let n = y.len().min(predictions.len());
        if n == 0 {
            return 0.0;
        }
        let mut mean = 0.0;
        for i in 0..n {
            mean += y[i];
        }
        mean /= n as f64;
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for i in 0..n {
            let d = y[i] - predictions[i];
            ss_res += d * d;
            let t = y[i] - mean;
            ss_tot += t * t;
        }
        if ss_tot <= 0.0 {
            return 0.0;
        }
        1.0 - ss_res / ss_tot
    }
}

/// Parse one CSV cell; blank, malformed, or non-finite cells become 0.0.
fn parse_cell(cell: &str) -> f64 {
    let text = cell.trim();
    if text.is_empty() || text.contains('_') {
        return 0.0;
    }
    match text.parse::<f64>() {
        Ok(value) => {
            if value.is_finite() {
                value
            } else {
                0.0
            }
        }
        Err(_) => 0.0,
    }
}

fn looks_numeric(cell: &str) -> bool {
    let text = cell.trim();
    text.is_empty() || (!text.contains('_') && text.parse::<f64>().is_ok())
}

/// Read CSV rows of floats, skipping blank lines and one header line if present.
fn read_csv(path: &str) -> Vec<Vec<f64>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("cannot read {}: {}", path, err);
            std::process::exit(2);
        }
    };
    let mut records: Vec<Vec<&str>> = Vec::new();
    for line in text.lines() {
        let cells: Vec<&str> = line.split(',').collect();
        if cells.iter().all(|cell| cell.trim().is_empty()) {
            continue;
        }
        records.push(cells);
    }
    if !records.is_empty() && !records[0].iter().all(|cell| looks_numeric(cell)) {
        records.remove(0);
    }
    records
        .iter()
        .map(|cells| cells.iter().map(|cell| parse_cell(cell)).collect())
        .collect()
}

/// Pad short rows with 0.0 (and trim long ones) to the training feature width.
fn pad(mut row: Vec<f64>, width: usize) -> Vec<f64> {
    while row.len() < width {
        row.push(0.0);
    }
    row.truncate(width);
    row
}

/// One shortest round-trip decimal per line (matches Python's repr()).
fn write_predictions(path: &str, predictions: &[f64]) {
    let mut out = String::new();
    for value in predictions {
        out.push_str(&format!("{:?}\n", value));
    }
    if let Err(err) = std::fs::write(path, out) {
        eprintln!("cannot write {}: {}", path, err);
        std::process::exit(2);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: model train.csv test_features.csv predictions.csv");
        std::process::exit(2);
    }
    let train_rows = read_csv(&args[1]);
    if train_rows.is_empty() {
        eprintln!("training data is empty");
        std::process::exit(2);
    }
    let y_train: Vec<f64> = train_rows.iter().map(|row| row[row.len() - 1]).collect();
    let features: Vec<Vec<f64>> = train_rows.iter().map(|row| row[..row.len() - 1].to_vec()).collect();
    let mut width = 0usize;
    for row in &features {
        if row.len() > width {
            width = row.len();
        }
    }
    let mut x_train: Vec<Vec<f64>> = Vec::new();
    for row in features {
        x_train.push(pad(row, width));
    }
    let mut x_test: Vec<Vec<f64>> = Vec::new();
    for row in read_csv(&args[2]) {
        x_test.push(pad(row, width));
    }
    let model = Model::fit(&x_train, &y_train);
    let predictions = model.predict(&x_test);
    write_predictions(&args[3], &predictions);
}
