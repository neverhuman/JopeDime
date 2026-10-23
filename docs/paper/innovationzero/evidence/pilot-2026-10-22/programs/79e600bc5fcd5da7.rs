//! Regression model: pairwise products, then robust scaling, then gradient boosted trees.
//!
//! This program trains a regression model on train.csv, in which every row holds the feature
//! columns followed by the target as the last column, and then writes one prediction per line for
//! the rows of test_features.csv. All arithmetic is IEEE-754 double precision in a fixed
//! evaluation order, so repeated runs give identical bits.
//!
//! The feature pipeline has 2 stages, applied in order; each stage is marked "replaced" when it
//! overwrites the columns it receives or "appended" when it keeps them and adds its outputs
//! alongside, and after every stage non-finite cells become 0.0 and each row is capped at 512
//! columns. Stage 1 (pairwise_products, appended) forms degree-two polynomial features: the
//! square of every column plus pairwise products, capped at 512 outputs. Stage 2 (robust_scaler,
//! appended) centers each column at its training median and divides by the interquartile range,
//! the outlier-resistant scaling of exploratory data analysis (Tukey, 1977).
//!
//! The head (gradient_boosted_trees) is gradient boosted regression trees (Friedman, 2001) with
//! histogram splits on training quantile bins and second-order leaf values -g/(h + lambda) as in
//! modern boosting systems (Chen & Guestrin, 2016): 122 rounds, depth 3, learning rate 0.1,
//! lambda 1.0, 64 bins, min leaf 10. Seeded stages draw weights from a permuted congruential
//! generator (PCG XSH-RR 64/32; O'Neill, 2014) keyed by the stage seed and the input/output
//! widths, so training is fully deterministic. Missing or malformed CSV cells parse as 0.0 and
//! short rows are padded with zeros to the training width.
//!
//! Build: rustc -O model.rs
//! Usage: ./model train.csv test_features.csv predictions.csv

#![forbid(unsafe_code)]
#![allow(dead_code)]

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

/// Linearly interpolated quantile of an ascending slice (the numpy 'linear' rule).
fn quantile(sorted_values: &[f64], q: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let pos = q * (sorted_values.len() - 1) as f64;
    let lo = pos as usize;
    let hi = (lo + 1).min(sorted_values.len() - 1);
    let frac = pos - lo as f64;
    sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
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

/// One node of a regression tree; feature -1 marks a leaf holding `value`.
struct TreeNode {
    feature: i64,
    threshold: i64,
    left: usize,
    right: usize,
    value: f64,
}

/// Quantile cut points for one feature, consecutive duplicates collapsed.
fn tree_bin_edges(x: &[Vec<f64>], f: usize, n: usize, bins: usize) -> Vec<f64> {
    let mut vals: Vec<f64> = (0..n).map(|r| x[r][f]).collect();
    vals.sort_by(|a, b| a.total_cmp(b));
    let mut edges = Vec::new();
    for q in 1..bins {
        let idx = ((vals.len() as f64) * (q as f64 / bins as f64)) as usize;
        edges.push(vals[idx.min(vals.len() - 1)]);
    }
    edges.dedup();
    edges
}

/// Flat row-major matrix of per-feature bin indices.
fn tree_bin_rows(x: &[Vec<f64>], rows: usize, n_feat: usize, edges: &[Vec<f64>]) -> Vec<i64> {
    let mut out = vec![0i64; rows * n_feat];
    for r in 0..rows {
        for f in 0..n_feat {
            let v = if f < x[r].len() { x[r][f] } else { 0.0 };
            out[r * n_feat + f] = lower_bound(&edges[f], v) as i64;
        }
    }
    out
}

/// Greedy histogram split search; returns the index of the new node.
/// Leaf values are the second-order step -g/(h + lam).
fn grow_tree(bins: &[i64], grad: &[f64], hess: &[f64], rows: Vec<usize>, n_feat: usize,
             nbins: usize, max_depth: usize, min_leaf: usize, lam: f64, depth: usize,
             tree: &mut Vec<TreeNode>, leaf_of: &mut [usize]) -> usize {
    let mut g = 0.0;
    for &i in &rows {
        g += grad[i];
    }
    let mut h = 0.0;
    for &i in &rows {
        h += hess[i];
    }
    let leaf_value = -g / (h + lam);
    if depth >= max_depth || rows.len() < 2 * min_leaf {
        let idx = tree.len();
        tree.push(TreeNode { feature: -1, threshold: 0, left: 0, right: 0, value: leaf_value });
        for &i in &rows {
            leaf_of[i] = idx;
        }
        return idx;
    }
    let mut best_gain = 0.0;
    let mut best_feat: i64 = -1;
    let mut best_thr: i64 = 0;
    let parent = g * g / (h + lam);
    for f in 0..n_feat {
        let mut gh = vec![0.0; nbins];
        let mut hh = vec![0.0; nbins];
        for &i in &rows {
            let b = bins[i * n_feat + f] as usize;
            gh[b] += grad[i];
            hh[b] += hess[i];
        }
        let mut gl = 0.0;
        let mut hl = 0.0;
        for b in 0..(nbins - 1) {
            gl += gh[b];
            hl += hh[b];
            let gr = g - gl;
            let hr = h - hl;
            if hl < 1.0 || hr < 1.0 {
                continue;
            }
            let gain = gl * gl / (hl + lam) + gr * gr / (hr + lam) - parent;
            if gain > best_gain {
                best_gain = gain;
                best_feat = f as i64;
                best_thr = b as i64;
            }
        }
    }
    if best_feat < 0 || best_gain <= 1e-9 {
        let idx = tree.len();
        tree.push(TreeNode { feature: -1, threshold: 0, left: 0, right: 0, value: leaf_value });
        for &i in &rows {
            leaf_of[i] = idx;
        }
        return idx;
    }
    let mut left_rows = Vec::new();
    let mut right_rows = Vec::new();
    for &i in &rows {
        if bins[i * n_feat + best_feat as usize] <= best_thr {
            left_rows.push(i);
        } else {
            right_rows.push(i);
        }
    }
    if left_rows.len() < min_leaf || right_rows.len() < min_leaf {
        let idx = tree.len();
        tree.push(TreeNode { feature: -1, threshold: 0, left: 0, right: 0, value: leaf_value });
        for &i in &rows {
            leaf_of[i] = idx;
        }
        return idx;
    }
    let idx = tree.len();
    tree.push(TreeNode { feature: -1, threshold: 0, left: 0, right: 0, value: 0.0 });
    let left = grow_tree(bins, grad, hess, left_rows, n_feat, nbins, max_depth, min_leaf, lam,
                         depth + 1, tree, leaf_of);
    let right = grow_tree(bins, grad, hess, right_rows, n_feat, nbins, max_depth, min_leaf, lam,
                          depth + 1, tree, leaf_of);
    tree[idx] = TreeNode { feature: best_feat, threshold: best_thr, left, right, value: 0.0 };
    idx
}

/// Follow split nodes to a leaf and return its value.
fn route_tree(tree: &[TreeNode], bins: &[i64], r: usize, n_feat: usize) -> f64 {
    let mut idx = 0usize;
    loop {
        let node = &tree[idx];
        if node.feature < 0 {
            return node.value;
        }
        idx = if bins[r * n_feat + node.feature as usize] <= node.threshold {
            node.left
        } else {
            node.right
        };
    }
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

struct RobustScaler {
    medians: Vec<f64>,
    scales: Vec<f64>,
}

/// This transform centers each column at its training median and divides by the interquartile
/// range, the outlier-resistant scaling of exploratory data analysis (Tukey, 1977).
fn robust_scaler_fit(x: &[Vec<f64>]) -> RobustScaler {
    let mut medians = Vec::new();
    let mut scales = Vec::new();
    for j in 0..x[0].len() {
        let s = sorted_column(x, j);
        medians.push(quantile(&s, 0.5));
        let iqr = quantile(&s, 0.75) - quantile(&s, 0.25);
        scales.push(iqr.max(1e-9));
    }
    RobustScaler { medians, scales }
}

fn robust_scaler_apply(state: &RobustScaler, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, v)| (v - state.medians[j]) / state.scales[j])
                .collect()
        })
        .collect();
    finish_stage(x, out, append)
}

struct GradientBoostedTrees {
    edges: Vec<Vec<f64>>,
    trees: Vec<Vec<TreeNode>>,
    init: f64,
    lr: f64,
    n_feat: usize,
    fallback: f64,
}

/// This head is gradient boosted regression trees (Friedman, 2001) with histogram splits on
/// training quantile bins and second-order leaf values -g/(h + lambda) as in modern boosting
/// systems (Chen & Guestrin, 2016): 122 rounds, depth 3, learning rate 0.1, lambda 1.0, 64
/// bins, min leaf 10.
fn gradient_boosted_trees_fit(x: &[Vec<f64>], y: &[f64], rounds: usize, max_depth: usize, lr: f64,
                              lam: f64, bins: usize, min_leaf: usize, max_features: usize) -> GradientBoostedTrees {
    let n = x.len().min(y.len());
    let cols = x[0].len();
    let n_feat = cols.min(max_features);
    if n < 4 || n_feat == 0 {
        let m = mean_of(&y[..n.max(1).min(y.len())]);
        return GradientBoostedTrees {
            edges: Vec::new(),
            trees: Vec::new(),
            init: m,
            lr,
            n_feat: 0,
            fallback: m,
        };
    }
    let edges: Vec<Vec<f64>> = (0..n_feat).map(|f| tree_bin_edges(x, f, n, bins)).collect();
    let train_bins = tree_bin_rows(x, n, n_feat, &edges);
    let init = mean_of(&y[..n]);
    let mut f_train = vec![init; n];
    let mut trees = Vec::new();
    for _ in 0..rounds {
        let grad: Vec<f64> = (0..n).map(|i| f_train[i] - y[i]).collect();
        let hess = vec![1.0; n];
        let mut tree: Vec<TreeNode> = Vec::new();
        let mut train_leaf = vec![0usize; n];
        grow_tree(&train_bins, &grad, &hess, (0..n).collect(), n_feat, bins, max_depth, min_leaf,
                  lam, 0, &mut tree, &mut train_leaf);
        for i in 0..n {
            f_train[i] += lr * tree[train_leaf[i]].value;
        }
        trees.push(tree);
    }
    GradientBoostedTrees { edges, trees, init, lr, n_feat, fallback: init }
}

fn gradient_boosted_trees_predict(state: &GradientBoostedTrees, x: &[Vec<f64>]) -> Vec<f64> {
    if state.n_feat == 0 {
        return vec![state.fallback; x.len()];
    }
    let bins = tree_bin_rows(x, x.len(), state.n_feat, &state.edges);
    let mut predictions = Vec::new();
    for r in 0..x.len() {
        let mut value = state.init;
        for tree in &state.trees {
            value += state.lr * route_tree(tree, &bins, r, state.n_feat);
        }
        predictions.push(value);
    }
    predictions
}

/// Feature stages applied in order, then the gradient_boosted_trees head.
struct Model {
    stage1: PairwiseProducts,
    stage2: RobustScaler,
    head: GradientBoostedTrees,
}

impl Model {
    /// Fit every stage in order on the training rows.
    fn fit(x: &[Vec<f64>], y: &[f64]) -> Model {
        if x.is_empty() || y.len() < x.len() {
            eprintln!("training data is empty or target is short");
            std::process::exit(2);
        }
        let x = sanitize(x);
        let stage1 = pairwise_products_fit(&x);
        let x = pairwise_products_apply(&stage1, &x, true);
        let stage2 = robust_scaler_fit(&x);
        let x = robust_scaler_apply(&stage2, &x, true);
        let head = gradient_boosted_trees_fit(&x, y, 122, 3, 0.1, 1.0, 64, 10, 256);
        Model { stage1, stage2, head }
    }

    /// Apply the fitted stages, then the regression head.
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let x = sanitize(x);
        let x = pairwise_products_apply(&self.stage1, &x, true);
        let x = robust_scaler_apply(&self.stage2, &x, true);
        gradient_boosted_trees_predict(&self.head, &x)
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
