#![forbid(unsafe_code)]

// Original, dependency-free general regressor. The last training column is y.
// This file deliberately contains no transport, filesystem access, or benchmark code.

pub trait RegressorApi: Sized {
    fn new(seed: u64) -> Self;

    fn fit(
        &mut self,
        train: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(), String>;

    fn predict(
        &self,
        features: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Result<Vec<f64>, String>;
}

const MAX_ROWS: usize = 32_768;
const MAX_FEATURES: usize = 128;
const BASES: usize = 4;
const BINS: usize = 64;
const RATE: f64 = 0.07;

pub struct Regressor {
    seed: u64,
    state: Option<Fitted>,
}

struct Fitted {
    width: usize,
    prep: Preparation,
    coefficients: Vec<f64>,
    trees: Vec<Tree>,
    linear_weight: f64,
    tree_weight: f64,
}

struct Random {
    state: u64,
}

impl Random {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        // Rejection avoids modulo bias; all callers supply positive n.
        let bound = n as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let z = self.next();
            if z >= threshold {
                return (z % bound) as usize;
            }
        }
    }

    fn shuffle(&mut self, values: &mut [usize]) {
        for i in (1..values.len()).rev() {
            let j = self.below(i + 1);
            values.swap(i, j);
        }
    }
}

fn sampled_rows(n: usize, rng: &mut Random) -> Vec<usize> {
    let size = n.min(MAX_ROWS);
    let mut rows: Vec<usize> = (0..size).collect();
    for i in size..n {
        let j = rng.below(i + 1);
        if j < size {
            rows[j] = i;
        }
    }
    rng.shuffle(&mut rows);
    rows
}

#[derive(Clone)]
struct Feature {
    column: usize,
    magnitude: f64,
    center: f64,
    inverse_sd: f64,
    basis_mean: [f64; BASES],
    basis_inverse_sd: [f64; BASES],
    cuts: Vec<f64>,
}

impl Feature {
    fn raw_basis(&self, x: f64) -> [f64; BASES] {
        if x.is_nan() {
            return [0.0, 0.0, 0.0, 1.0];
        }
        // Divide before subtraction, including for opposite finite extremes.
        let normalized = (x / self.magnitude).clamp(-1.0e100, 1.0e100);
        let z = ((normalized - self.center) * self.inverse_sd).clamp(-10.0, 10.0);
        [z, z * z, z.max(0.0), 0.0]
    }

    fn basis(&self, x: f64) -> [f64; BASES] {
        let mut result = self.raw_basis(x);
        for k in 0..BASES {
            result[k] = ((result[k] - self.basis_mean[k]) * self.basis_inverse_sd[k]).clamp(-32.0, 32.0);
        }
        result
    }

    fn bin(&self, x: f64) -> u8 {
        if x.is_nan() {
            return 0;
        }
        let mut lo = 0;
        let mut hi = self.cuts.len();
        while lo < hi {
            let middle = lo + (hi - lo) / 2;
            if x <= self.cuts[middle] {
                hi = middle;
            } else {
                lo = middle + 1;
            }
        }
        (lo + 1) as u8
    }
}

struct Preparation {
    features: Vec<Feature>,
    target_magnitude: f64,
    target_mean: f64,
}

struct Matrix {
    n: usize,
    linear: Vec<f64>,
    bins: Vec<u8>,
}

// Supervised screening uses only internal training rows. Retain a random
// complement so a zero marginal correlation does not exclude every interaction.
fn choose_features(
    table: &[f64],
    cols: usize,
    rows: &[usize],
    rng: &mut Random,
) -> Vec<usize> {
    let p = cols - 1;
    if p <= MAX_FEATURES {
        return (0..p).collect();
    }
    let count = rows.len().min(384);
    let sample = &rows[..count];
    let mut target_scale: f64 = 0.0;
    for &r in sample {
        target_scale = target_scale.max(table[r * cols + p].abs());
    }
    if target_scale == 0.0 {
        target_scale = 1.0;
    }
    let mut mean_y = 0.0;
    for &r in sample {
        mean_y += table[r * cols + p] / target_scale / count as f64;
    }
    let mut ranked: Vec<(f64, usize)> = Vec::new();
    let mut random_columns: Vec<usize> = Vec::new();
    for j in 0..p {
        if random_columns.len() < 32 {
            random_columns.push(j);
        } else {
            let slot = rng.below(j + 1);
            if slot < 32 {
                random_columns[slot] = j;
            }
        }
        let mut magnitude: f64 = 0.0;
        let mut observed = 0usize;
        for &r in sample {
            let x = table[r * cols + j];
            if !x.is_nan() {
                magnitude = magnitude.max(x.abs());
                observed += 1;
            }
        }
        if magnitude == 0.0 {
            magnitude = 1.0;
        }
        let mut center = 0.0;
        if observed > 0 {
            for &r in sample {
                let x = table[r * cols + j];
                if !x.is_nan() {
                    center += x / magnitude / observed as f64;
                }
            }
        }
        let mut sums = [0.0; 3];
        let mut squares = [0.0; 3];
        let mut products = [0.0; 3];
        for &r in sample {
            let x = table[r * cols + j];
            let y = table[r * cols + p] / target_scale - mean_y;
            let values = if x.is_nan() {
                [0.0, 0.0, 1.0]
            } else {
                let z = x / magnitude - center;
                [z, z * z, 0.0]
            };
            for k in 0..3 {
                sums[k] += values[k];
                squares[k] += values[k] * values[k];
                products[k] += values[k] * y;
            }
        }
        let mut score: f64 = 0.0;
        for k in 0..3 {
            let variance = squares[k] - sums[k] * sums[k] / count as f64;
            if variance > 0.0 {
                score = score.max(products[k] * products[k] / variance);
            }
        }
        let position = ranked.iter().position(|&(s, _)| score > s).unwrap_or(ranked.len());
        if position < 96 {
            ranked.insert(position, (score, j));
            if ranked.len() > 96 {
                ranked.pop();
            }
        }
    }
    let mut selected: Vec<usize> = ranked.iter().map(|&(_, j)| j).collect();
    for j in random_columns {
        if !selected.contains(&j) {
            selected.push(j);
        }
    }
    selected.sort_unstable();
    selected
}

fn prepare(
    table: &[f64],
    cols: usize,
    rows: &[usize],
    selected: &[usize],
) -> (Preparation, Matrix, Vec<f64>) {
    let n = rows.len();
    let mut magnitude: f64 = 0.0;
    for &r in rows {
        magnitude = magnitude.max(table[r * cols + cols - 1].abs());
    }
    if magnitude == 0.0 {
        magnitude = 1.0;
    }
    let mut target_mean = 0.0;
    for &r in rows {
        target_mean += table[r * cols + cols - 1] / magnitude / n as f64;
    }
    let y: Vec<f64> = rows.iter()
        .map(|&r| table[r * cols + cols - 1] / magnitude - target_mean)
        .collect();
    let mut matrix = Matrix {
        n,
        linear: vec![0.0; n * selected.len() * BASES],
        bins: vec![0; n * selected.len()],
    };
    let mut features = Vec::new();
    for (j, &column) in selected.iter().enumerate() {
        let mut sorted: Vec<f64> = rows.iter()
            .map(|&r| table[r * cols + column])
            .filter(|x| !x.is_nan())
            .collect();
        sorted.sort_unstable_by(|a, b| a.total_cmp(b));
        let mut feature = Feature {
            column,
            magnitude: 1.0,
            center: 0.0,
            inverse_sd: 1.0,
            basis_mean: [0.0; BASES],
            basis_inverse_sd: [0.0; BASES],
            cuts: Vec::new(),
        };
        if !sorted.is_empty() {
            feature.magnitude = sorted[0].abs().max(sorted[sorted.len() - 1].abs());
            if feature.magnitude == 0.0 {
                feature.magnitude = 1.0;
            }
            for &x in &sorted {
                feature.center += x / feature.magnitude / sorted.len() as f64;
            }
            let mut variance = 0.0;
            for &x in &sorted {
                let z = x / feature.magnitude - feature.center;
                variance += z * z / sorted.len() as f64;
            }
            if variance > 0.0 {
                feature.inverse_sd = 1.0 / variance.sqrt();
            }
            for k in 1..BINS {
                let at = k * sorted.len() / BINS;
                if at > 0 && at < sorted.len() && sorted[at - 1] < sorted[at] {
                    // Threshold at an observed value avoids midpoint overflow,
                    // rounded-equal midpoints, and subnormal underflow.
                    let cut = sorted[at - 1];
                    if feature.cuts.last().map_or(true, |&last| cut > last) {
                        feature.cuts.push(cut);
                    }
                }
            }
        }
        for (i, &r) in rows.iter().enumerate() {
            let x = table[r * cols + column];
            matrix.bins[j * n + i] = feature.bin(x);
            let values = feature.raw_basis(x);
            for k in 0..BASES {
                matrix.linear[(j * BASES + k) * n + i] = values[k];
                feature.basis_mean[k] += values[k] / n as f64;
            }
        }
        for k in 0..BASES {
            let offset = (j * BASES + k) * n;
            let mut variance = 0.0;
            for i in 0..n {
                let centered = matrix.linear[offset + i] - feature.basis_mean[k];
                variance += centered * centered / n as f64;
            }
            let inverse = if variance > 0.0 { 1.0 / variance.sqrt() } else { 0.0 };
            feature.basis_inverse_sd[k] = inverse;
            for i in 0..n {
                matrix.linear[offset + i] =
                    ((matrix.linear[offset + i] - feature.basis_mean[k]) * inverse).clamp(-32.0, 32.0);
            }
        }
        features.push(feature);
    }
    (Preparation { features, target_magnitude: magnitude, target_mean }, matrix, y)
}

fn encode(table: &[f64], cols: usize, rows: &[usize], prep: &Preparation) -> Matrix {
    let n = rows.len();
    let mut matrix = Matrix {
        n,
        linear: vec![0.0; n * prep.features.len() * BASES],
        bins: vec![0; n * prep.features.len()],
    };
    for (j, feature) in prep.features.iter().enumerate() {
        for (i, &r) in rows.iter().enumerate() {
            let x = table[r * cols + feature.column];
            matrix.bins[j * n + i] = feature.bin(x);
            let values = feature.basis(x);
            for k in 0..BASES {
                matrix.linear[(j * BASES + k) * n + i] = values[k];
            }
        }
    }
    matrix
}

fn ridge(matrix: &Matrix, y: &[f64], lambda: f64) -> Vec<f64> {
    let n = matrix.n;
    let d = matrix.linear.len() / n;
    let mut coefficients = vec![0.0; d];
    if d == 0 {
        return coefficients;
    }
    let mut residual = y.to_vec();
    let mut norms = vec![0.0; d];
    for j in 0..d {
        for i in 0..n {
            let x = matrix.linear[j * n + i];
            norms[j] += x * x;
        }
    }
    let mut energy = 0.0;
    for &value in y {
        energy += value * value / n as f64;
    }
    let tolerance = energy.sqrt() * 1.0e-7;
    let passes = (40_000_000usize / n.max(1) / d.max(1)).clamp(4, 32);
    for _ in 0..passes {
        let mut movement: f64 = 0.0;
        for j in 0..d {
            if norms[j] == 0.0 {
                continue;
            }
            let offset = j * n;
            let mut product = 0.0;
            for i in 0..n {
                product += matrix.linear[offset + i] * residual[i];
            }
            let next = (product + norms[j] * coefficients[j]) / (norms[j] + lambda * n as f64);
            let change = next - coefficients[j];
            coefficients[j] = next;
            movement = movement.max(change.abs());
            for i in 0..n {
                residual[i] -= change * matrix.linear[offset + i];
            }
        }
        if movement <= tolerance {
            break;
        }
    }
    coefficients
}

fn linear_predictions(matrix: &Matrix, coefficients: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; matrix.n];
    for (j, &coefficient) in coefficients.iter().enumerate() {
        for i in 0..matrix.n {
            result[i] += coefficient * matrix.linear[j * matrix.n + i];
        }
    }
    result
}

struct Node {
    feature: usize,
    cut: u8,
    missing_left: bool,
    left: usize,
    right: usize,
    value: f64,
}

struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn prediction(&self, bins: &[u8], stride: usize, row: usize) -> f64 {
        let mut at = 0;
        loop {
            let node = &self.nodes[at];
            if node.feature == usize::MAX {
                return node.value;
            }
            let bin = bins[node.feature * stride + row];
            let left = if bin == 0 { node.missing_left } else { bin <= node.cut };
            at = if left { node.left } else { node.right };
        }
    }
}

struct Builder<'a> {
    matrix: &'a Matrix,
    residual: &'a [f64],
    features: &'a [usize],
    bin_counts: &'a [usize],
    min_leaf: usize,
    penalty: f64,
    min_gain: f64,
    max_depth: usize,
    nodes: Vec<Node>,
}

impl<'a> Builder<'a> {
    fn node(&mut self, rows: &mut [usize], depth: usize) -> usize {
        let mut total = 0.0;
        for &i in rows.iter() {
            total += self.residual[i];
        }
        let position = self.nodes.len();
        self.nodes.push(Node {
            feature: usize::MAX,
            cut: 0,
            missing_left: true,
            left: 0,
            right: 0,
            value: (total / (rows.len() as f64 + self.penalty)).clamp(-4.0, 4.0),
        });
        if depth >= self.max_depth || rows.len() < 2 * self.min_leaf {
            return position;
        }
        let parent = total * total / (rows.len() as f64 + self.penalty);
        let mut best_gain = self.min_gain;
        let mut best_feature = usize::MAX;
        let mut best_cut = 0u8;
        let mut best_missing_left = true;
        for &j in self.features {
            let mut sums = [0.0; BINS + 1];
            let mut counts = [0usize; BINS + 1];
            let offset = j * self.matrix.n;
            for &i in rows.iter() {
                let bin = self.matrix.bins[offset + i] as usize;
                sums[bin] += self.residual[i];
                counts[bin] += 1;
            }
            let mut finite_sum = 0.0;
            let mut finite_count = 0usize;
            for cut in 0..self.bin_counts[j] - 1 {
                if cut > 0 {
                    finite_sum += sums[cut];
                    finite_count += counts[cut];
                }
                for missing_left in [true, false] {
                    let count = finite_count + if missing_left { counts[0] } else { 0 };
                    if count < self.min_leaf || rows.len() - count < self.min_leaf {
                        continue;
                    }
                    let sum = finite_sum + if missing_left { sums[0] } else { 0.0 };
                    let other = total - sum;
                    let gain = sum * sum / (count as f64 + self.penalty)
                        + other * other / ((rows.len() - count) as f64 + self.penalty)
                        - parent;
                    if gain > best_gain {
                        best_gain = gain;
                        best_feature = j;
                        best_cut = cut as u8;
                        best_missing_left = missing_left;
                    }
                }
            }
        }
        if best_feature == usize::MAX {
            return position;
        }
        let mut split = 0;
        for i in 0..rows.len() {
            let bin = self.matrix.bins[best_feature * self.matrix.n + rows[i]];
            let left = if bin == 0 { best_missing_left } else { bin <= best_cut };
            if left {
                rows.swap(i, split);
                split += 1;
            }
        }
        let (left_rows, right_rows) = rows.split_at_mut(split);
        let left = self.node(left_rows, depth + 1);
        let right = self.node(right_rows, depth + 1);
        self.nodes[position].feature = best_feature;
        self.nodes[position].cut = best_cut;
        self.nodes[position].missing_left = best_missing_left;
        self.nodes[position].left = left;
        self.nodes[position].right = right;
        position
    }
}

fn depth_for(n: usize) -> usize {
    if n < 256 { 4 } else { 6 }
}

fn round_limit(n: usize, p: usize) -> usize {
    let depth = depth_for(n);
    let columns = p.min(48);
    let sample = n.min(8192);
    let estimate = sample * columns * depth
        + (1usize << depth) * columns * (BINS + 1)
        + n * depth;
    (480_000_000usize / estimate.max(1)).clamp(16, 320)
}

struct Booster {
    rng: Random,
    rows: Vec<usize>,
    columns: Vec<usize>,
    prediction: Vec<f64>,
    bin_counts: Vec<usize>,
}

impl Booster {
    fn new(n: usize, prep: &Preparation, seed: u64) -> Self {
        Self {
            rng: Random::new(seed),
            rows: (0..n).collect(),
            columns: (0..prep.features.len()).collect(),
            prediction: vec![0.0; n],
            bin_counts: prep.features.iter().map(|f| f.cuts.len() + 2).collect(),
        }
    }

    fn step(&mut self, matrix: &Matrix, y: &[f64]) -> Tree {
        let n = matrix.n;
        let residual: Vec<f64> = y.iter().zip(self.prediction.iter()).map(|(a, b)| a - b).collect();
        let sample = n.min(8192);
        for i in 0..sample {
            let j = i + self.rng.below(n - i);
            self.rows.swap(i, j);
        }
        self.rng.shuffle(&mut self.columns);
        let features = &self.columns[..self.columns.len().min(48)];
        let mut energy = 0.0;
        for &i in &self.rows[..sample] {
            energy += residual[i] * residual[i] / sample as f64;
        }
        let mut builder = Builder {
            matrix,
            residual: &residual,
            features,
            bin_counts: &self.bin_counts,
            min_leaf: (n / 2048).clamp(3, 16),
            penalty: (sample as f64 * 0.001).max(1.0),
            min_gain: energy * 1.0e-10,
            max_depth: depth_for(n),
            nodes: Vec::new(),
        };
        builder.node(&mut self.rows[..sample], 0);
        let tree = Tree { nodes: builder.nodes };
        for i in 0..n {
            self.prediction[i] += RATE * tree.prediction(&matrix.bins, n, i);
        }
        tree
    }
}

// Exact minimization of a two-coordinate quadratic on the convex hull of
// mean, linear, and tree predictions. Candidates cover vertices, edges, interior.
fn blend(linear: &[f64], tree: &[f64], target: &[f64]) -> (f64, f64, f64) {
    let mut ll = 0.0;
    let mut tt = 0.0;
    let mut lt = 0.0;
    let mut yl = 0.0;
    let mut yt = 0.0;
    for i in 0..target.len() {
        ll += linear[i] * linear[i];
        tt += tree[i] * tree[i];
        lt += linear[i] * tree[i];
        yl += target[i] * linear[i];
        yt += target[i] * tree[i];
    }
    let mut candidates = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
    if ll > 0.0 {
        candidates.push(((yl / ll).clamp(0.0, 1.0), 0.0));
    }
    if tt > 0.0 {
        candidates.push((0.0, (yt / tt).clamp(0.0, 1.0)));
    }
    let edge = ll + tt - 2.0 * lt;
    if edge > 0.0 {
        let a = ((yl - yt - lt + tt) / edge).clamp(0.0, 1.0);
        candidates.push((a, 1.0 - a));
    }
    let determinant = ll * tt - lt * lt;
    if determinant > 1.0e-12 * ll * tt {
        let a = (yl * tt - yt * lt) / determinant;
        let b = (yt * ll - yl * lt) / determinant;
        if a >= 0.0 && b >= 0.0 && a + b <= 1.0 {
            candidates.push((a, b));
        }
    }
    let mut best = (0.0, 0.0, f64::INFINITY);
    for (a, b) in candidates {
        let mut loss = 0.0;
        for i in 0..target.len() {
            let error = target[i] - a * linear[i] - b * tree[i];
            loss += error * error / target.len() as f64;
        }
        if loss < best.2 {
            best = (a, b, loss);
        }
    }
    best
}

fn train_model(table: &[f64], cols: usize, n: usize, seed: u64) -> Fitted {
    let mut rng = Random::new(seed);
    let rows = sampled_rows(n, &mut rng);
    let validation_count = if rows.len() >= 40 { (rows.len() / 5).min(2048) } else { 0 };
    let (validation_rows, learning_rows) = rows.split_at(validation_count);
    let selected = choose_features(table, cols, learning_rows, &mut rng);
    let (prep, matrix, y) = prepare(table, cols, learning_rows, &selected);
    let tree_seed = seed ^ 0xa0761d6478bd642f;
    let mut lambda = 0.03;
    let mut linear_weight = if selected.is_empty() { 0.0 } else { 0.5 };
    let mut tree_weight = if selected.is_empty() { 0.0 } else { 0.5 };
    let mut rounds = if selected.is_empty() { 0 } else { round_limit(matrix.n, selected.len()).min(64) };
    if validation_count > 0 && !selected.is_empty() {
        let validation = encode(table, cols, validation_rows, &prep);
        // Validation labels only define scoring units and selection. They never
        // enter any training matrix, feature statistic, or component predictor.
        let mut scoring_magnitude = prep.target_magnitude;
        for &r in validation_rows {
            scoring_magnitude = scoring_magnitude.max(table[r * cols + cols - 1].abs());
        }
        let ratio = prep.target_magnitude / scoring_magnitude;
        let target: Vec<f64> = validation_rows.iter()
            .map(|&r| table[r * cols + cols - 1] / scoring_magnitude - prep.target_mean * ratio)
            .collect();
        let zero = vec![0.0; validation_count];
        let mut best_linear = zero.clone();
        let mut best_loss = f64::INFINITY;
        for penalty in [1.0, 0.03, 0.001] {
            let coefficients = ridge(&matrix, &y, penalty);
            let predictions: Vec<f64> = linear_predictions(&validation, &coefficients)
                .into_iter().map(|x| x * ratio).collect();
            let (weight, _, loss) = blend(&predictions, &zero, &target);
            if loss < best_loss {
                best_loss = loss;
                lambda = penalty;
                linear_weight = weight;
                tree_weight = 0.0;
                best_linear = predictions;
            }
        }
        rounds = 0;
        let mut booster = Booster::new(matrix.n, &prep, tree_seed);
        let mut tree_prediction = zero;
        let limit = round_limit(matrix.n, selected.len());
        for iteration in 1..=limit {
            let tree = booster.step(&matrix, &y);
            for i in 0..validation_count {
                tree_prediction[i] += RATE * ratio * tree.prediction(&validation.bins, validation.n, i);
            }
            if iteration % 8 == 0 || iteration == limit {
                let (a, b, loss) = blend(&best_linear, &tree_prediction, &target);
                if loss < best_loss * (1.0 - 1.0e-6) {
                    best_loss = loss;
                    linear_weight = a;
                    tree_weight = b;
                    rounds = iteration;
                }
                if iteration >= rounds + 64 {
                    break;
                }
            }
        }
    }
    // Refit the chosen procedure on the entire bounded sample, including the
    // internal validation rows. Feature identities and all choices are frozen.
    let (prep, matrix, y) = prepare(table, cols, &rows, &selected);
    let coefficients = if linear_weight > 0.0 {
        ridge(&matrix, &y, lambda)
    } else {
        vec![0.0; selected.len() * BASES]
    };
    let mut trees = Vec::new();
    if tree_weight > 0.0 && rounds > 0 {
        let mut booster = Booster::new(matrix.n, &prep, tree_seed);
        for _ in 0..rounds {
            trees.push(booster.step(&matrix, &y));
        }
    }
    Fitted { width: cols - 1, prep, coefficients, trees, linear_weight, tree_weight }
}

impl RegressorApi for Regressor {
    fn new(seed: u64) -> Self {
        Self { seed, state: None }
    }

    fn fit(&mut self, train: &[f64], n_rows: usize, n_cols: usize) -> Result<(), String> {
        self.state = None;
        if n_rows == 0 || n_cols == 0 {
            return Err("Training requires at least one row and one target column".to_string());
        }
        let length = n_rows.checked_mul(n_cols).ok_or_else(|| "Training shape overflow".to_string())?;
        if train.len() != length {
            return Err("Training shape does not match the supplied buffer".to_string());
        }
        for row in train.chunks_exact(n_cols) {
            if !row[n_cols - 1].is_finite() {
                return Err("The final training column must contain finite targets".to_string());
            }
            for &x in &row[..n_cols - 1] {
                if x.is_infinite() {
                    return Err("Infinite feature values are invalid".to_string());
                }
            }
        }
        self.state = Some(train_model(train, n_cols, n_rows, self.seed));
        Ok(())
    }

    fn predict(&self, features: &[f64], n_rows: usize, n_features: usize) -> Result<Vec<f64>, String> {
        let model = self.state.as_ref().ok_or_else(|| "The regressor is not fitted".to_string())?;
        if n_features != model.width {
            return Err("Prediction width does not match the fitted feature count".to_string());
        }
        let length = n_rows.checked_mul(n_features).ok_or_else(|| "Prediction shape overflow".to_string())?;
        if features.len() != length {
            return Err("Prediction shape does not match the supplied buffer".to_string());
        }
        for &x in features {
            if x.is_infinite() {
                return Err("Infinite feature values are invalid".to_string());
            }
        }
        let mut predictions = Vec::new();
        predictions.try_reserve_exact(n_rows).map_err(|_| "Prediction allocation failed".to_string())?;
        let mut bins = vec![0; model.prep.features.len()];
        for i in 0..n_rows {
            let mut linear = 0.0;
            for (j, feature) in model.prep.features.iter().enumerate() {
                let x = features[i * n_features + feature.column];
                bins[j] = feature.bin(x);
                let basis = feature.basis(x);
                for k in 0..BASES {
                    linear += model.coefficients[j * BASES + k] * basis[k];
                }
            }
            let mut nonlinear = 0.0;
            for tree in &model.trees {
                nonlinear += RATE * tree.prediction(&bins, 1, 0);
            }
            let normalized = model.prep.target_mean
                + model.linear_weight * linear + model.tree_weight * nonlinear;
            if !normalized.is_finite() {
                return Err("Nonfinite numerical prediction".to_string());
            }
            let limit = f64::MAX / model.prep.target_magnitude;
            let restored = normalized.clamp(-limit, limit) * model.prep.target_magnitude;
            // Rounding at MAX / magnitude can make the multiplication overflow.
            let prediction = if restored.is_finite() { restored }
                else if normalized < 0.0 { -f64::MAX } else { f64::MAX };
            predictions.push(prediction);
        }
        Ok(predictions)
    }
}
