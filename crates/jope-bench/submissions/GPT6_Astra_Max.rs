#![forbid(unsafe_code)]

// Original, dependency-free general regressor. The last training column is y.
// This file deliberately contains no transport, filesystem access, or benchmark code.

pub trait RegressorApi: Sized {
    fn new(seed: u64) -> Self;

    fn fit(&mut self, train: &[f64], n_rows: usize, n_cols: usize) -> Result<(), String>;

    fn predict(
        &self,
        features: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Result<Vec<f64>, String>;
}

#[path = "GPT6_Astra_Max/prep.rs"]
mod prep;
#[path = "GPT6_Astra_Max/tree.rs"]
mod tree;

pub(super) use prep::{choose_features, encode, linear_predictions, prepare, ridge};
pub(super) use tree::{Booster, Tree};

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
            result[k] =
                ((result[k] - self.basis_mean[k]) * self.basis_inverse_sd[k]).clamp(-32.0, 32.0);
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
    let validation_count = if rows.len() >= 40 {
        (rows.len() / 5).min(2048)
    } else {
        0
    };
    let (validation_rows, learning_rows) = rows.split_at(validation_count);
    let selected = choose_features(table, cols, learning_rows, &mut rng);
    let (prep, matrix, y) = prepare(table, cols, learning_rows, &selected);
    let tree_seed = seed ^ 0xa0761d6478bd642f;
    let mut lambda = 0.03;
    let mut linear_weight = if selected.is_empty() { 0.0 } else { 0.5 };
    let mut tree_weight = if selected.is_empty() { 0.0 } else { 0.5 };
    let mut rounds = if selected.is_empty() {
        0
    } else {
        round_limit(matrix.n, selected.len()).min(64)
    };
    if validation_count > 0 && !selected.is_empty() {
        let validation = encode(table, cols, validation_rows, &prep);
        // Validation labels only define scoring units and selection. They never
        // enter any training matrix, feature statistic, or component predictor.
        let mut scoring_magnitude = prep.target_magnitude;
        for &r in validation_rows {
            scoring_magnitude = scoring_magnitude.max(table[r * cols + cols - 1].abs());
        }
        let ratio = prep.target_magnitude / scoring_magnitude;
        let target: Vec<f64> = validation_rows
            .iter()
            .map(|&r| table[r * cols + cols - 1] / scoring_magnitude - prep.target_mean * ratio)
            .collect();
        let zero = vec![0.0; validation_count];
        let mut best_linear = zero.clone();
        let mut best_loss = f64::INFINITY;
        for penalty in [1.0, 0.03, 0.001] {
            let coefficients = ridge(&matrix, &y, penalty);
            let predictions: Vec<f64> = linear_predictions(&validation, &coefficients)
                .into_iter()
                .map(|x| x * ratio)
                .collect();
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
                tree_prediction[i] +=
                    RATE * ratio * tree.prediction(&validation.bins, validation.n, i);
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
    Fitted {
        width: cols - 1,
        prep,
        coefficients,
        trees,
        linear_weight,
        tree_weight,
    }
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
        let length = n_rows
            .checked_mul(n_cols)
            .ok_or_else(|| "Training shape overflow".to_string())?;
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

    fn predict(
        &self,
        features: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Result<Vec<f64>, String> {
        let model = self
            .state
            .as_ref()
            .ok_or_else(|| "The regressor is not fitted".to_string())?;
        if n_features != model.width {
            return Err("Prediction width does not match the fitted feature count".to_string());
        }
        let length = n_rows
            .checked_mul(n_features)
            .ok_or_else(|| "Prediction shape overflow".to_string())?;
        if features.len() != length {
            return Err("Prediction shape does not match the supplied buffer".to_string());
        }
        for &x in features {
            if x.is_infinite() {
                return Err("Infinite feature values are invalid".to_string());
            }
        }
        let mut predictions = Vec::new();
        predictions
            .try_reserve_exact(n_rows)
            .map_err(|_| "Prediction allocation failed".to_string())?;
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
                + model.linear_weight * linear
                + model.tree_weight * nonlinear;
            if !normalized.is_finite() {
                return Err("Nonfinite numerical prediction".to_string());
            }
            let limit = f64::MAX / model.prep.target_magnitude;
            let restored = normalized.clamp(-limit, limit) * model.prep.target_magnitude;
            // Rounding at MAX / magnitude can make the multiplication overflow.
            let prediction = if restored.is_finite() {
                restored
            } else if normalized < 0.0 {
                -f64::MAX
            } else {
                f64::MAX
            };
            predictions.push(prediction);
        }
        Ok(predictions)
    }
}
