//! jope emit.
//!
//! This program trains a regression model on train.csv, in which every row holds the feature
//! columns followed by the target as the last column, and then writes one prediction per line for
//! the rows of test_features.csv. All arithmetic is IEEE-754 double precision in a fixed
//! evaluation order, so repeated runs give identical bits.
//!
//! The feature pipeline has 1 stage, applied in order; each stage is marked "replaced" when it
//! overwrites the columns it receives or "appended" when it keeps them and adds its outputs
//! alongside, and after every stage non-finite cells become 0.0 and each row is capped at 512
//! columns. Stage 1 (standard_scaler, replaced) centers each column by its training mean and
//! divides by the standard deviation (1/n convention, floored at 1e-9), clamping the score to
//! [-20, 20].
//!
//! The head (ridge_regression) is ridge regression (Hoerl & Kennard, 1970) on standardized
//! columns with penalty max(0.02*n, 1e-3); the symmetric positive-definite normal equations are
//! solved by a Cholesky factorization and predictions are clipped to [-60, 60]. Seeded stages
//! draw weights from a permuted congruential generator (PCG XSH-RR 64/32; O'Neill, 2014) keyed by
//! the stage seed and the input/output widths, so training is fully deterministic. Missing or
//! malformed CSV cells parse as 0.0 and short rows are padded with zeros to the training width.
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

/// Per-column mean and standard deviation (1/n convention), floored at 1e-9.
fn std_fit(x: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len().max(1) as f64;
    let p = x[0].len();
    let mut means = vec![0.0; p];
    let mut scales = vec![1.0; p];
    for j in 0..p {
        let mut s = 0.0;
        for row in x {
            s += row[j];
        }
        means[j] = s / n;
        let mut v = 0.0;
        for row in x {
            let d = row[j] - means[j];
            v += d * d;
        }
        scales[j] = (v / n).sqrt().max(1e-9);
    }
    (means, scales)
}

/// Solve the symmetric positive-definite system A w = b.
///
/// A (flattened row-major, overwritten) is factored in place as L L^T, then a
/// forward and a backward triangular solve produce w. A tiny 1e-6 pivot stands
/// in whenever a diagonal entry falls under 1e-12.
fn cholesky_solve(a: &mut [f64], b: &[f64], p: usize) -> Vec<f64> {
    for i in 0..p {
        for j in 0..(i + 1) {
            let mut s = a[i * p + j];
            for k in 0..j {
                s -= a[i * p + k] * a[j * p + k];
            }
            if i == j {
                a[i * p + j] = if s > 1e-12 { s.sqrt() } else { 1e-6 };
            } else {
                a[i * p + j] = s / a[j * p + j];
            }
        }
    }
    let mut z = vec![0.0; p];
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * p + k] * z[k];
        }
        z[i] = s / a[i * p + i];
    }
    let mut w = vec![0.0; p];
    for i in (0..p).rev() {
        let mut s = z[i];
        for k in (i + 1)..p {
            s -= a[k * p + i] * w[k];
        }
        w[i] = s / a[i * p + i];
    }
    w
}

struct StandardScaler {
    means: Vec<f64>,
    scales: Vec<f64>,
}

/// This transform centers each column by its training mean and divides by the standard
/// deviation (1/n convention, floored at 1e-9), clamping the score to [-20, 20].
fn standard_scaler_fit(x: &[Vec<f64>]) -> StandardScaler {
    let (means, scales) = std_fit(x);
    StandardScaler { means, scales }
}

fn standard_scaler_apply(state: &StandardScaler, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, v)| ((v - state.means[j]) / state.scales[j]).clamp(-20.0, 20.0))
                .collect()
        })
        .collect();
    finish_stage(x, out, append)
}

struct RidgeRegression {
    means: Vec<f64>,
    scales: Vec<f64>,
    w: Vec<f64>,
}

/// This head is ridge regression (Hoerl & Kennard, 1970) on standardized columns with penalty
/// max(0.02*n, 1e-3); the symmetric positive-definite normal equations are solved by a Cholesky
/// factorization and predictions are clipped to [-60, 60].
fn ridge_regression_fit(x: &[Vec<f64>], y: &[f64], lambda_frac: f64) -> RidgeRegression {
    let n = x.len().min(y.len());
    let p = x[0].len();
    let (means, scales) = std_fit(x);
    let lam = (lambda_frac * n as f64).max(1e-3);
    let mut y_mean = 0.0;
    for i in 0..n {
        y_mean += y[i];
    }
    y_mean /= n.max(1) as f64;
    let mut ata = vec![0.0; p * p];
    let mut atb = vec![0.0; p];
    let mut xs_row = vec![0.0; p];
    for r in 0..n {
        let row = &x[r];
        for j in 0..p {
            xs_row[j] = ((row[j] - means[j]) / scales[j]).clamp(-20.0, 20.0);
        }
        let yr = y[r] - y_mean;
        for i in 0..p {
            let xi = xs_row[i];
            atb[i] += xi * yr;
            for j in i..p {
                ata[i * p + j] += xi * xs_row[j];
            }
        }
    }
    for i in 0..p {
        ata[i * p + i] += lam;
        for j in 0..i {
            ata[i * p + j] = ata[j * p + i];
        }
    }
    let coeffs = cholesky_solve(&mut ata, &atb, p);
    let mut w = Vec::with_capacity(p + 1);
    w.push(y_mean);
    w.extend(coeffs);
    RidgeRegression { means, scales, w }
}

fn ridge_regression_predict(state: &RidgeRegression, x: &[Vec<f64>]) -> Vec<f64> {
    let p = state.means.len();
    let mut predictions = Vec::new();
    for row in x {
        let mut z = state.w[0];
        for j in 0..p.min(row.len()) {
            let xs = ((row[j] - state.means[j]) / state.scales[j]).clamp(-20.0, 20.0);
            z += state.w[j + 1] * xs;
        }
        predictions.push(z.clamp(-60.0, 60.0));
    }
    predictions
}

/// Feature stages applied in order, then the ridge_regression head.
struct Model {
    stage1: StandardScaler,
    head: RidgeRegression,
}

impl Model {
    /// Fit every stage in order on the training rows.
    fn fit(x: &[Vec<f64>], y: &[f64]) -> Model {
        if x.is_empty() || y.len() < x.len() {
            eprintln!("training data is empty or target is short");
            std::process::exit(2);
        }
        let x = sanitize(x);
        let stage1 = standard_scaler_fit(&x);
        let x = standard_scaler_apply(&stage1, &x, false);
        let head = ridge_regression_fit(&x, y, 0.02);
        Model { stage1, head }
    }

    /// Apply the fitted stages, then the regression head.
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let x = sanitize(x);
        let x = standard_scaler_apply(&self.stage1, &x, false);
        ridge_regression_predict(&self.head, &x)
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
