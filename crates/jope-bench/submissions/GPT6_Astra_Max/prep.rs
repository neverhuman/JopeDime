use super::*;

pub(crate) fn choose_features(
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
        let position = ranked
            .iter()
            .position(|&(s, _)| score > s)
            .unwrap_or(ranked.len());
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

pub(crate) fn prepare(
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
    let y: Vec<f64> = rows
        .iter()
        .map(|&r| table[r * cols + cols - 1] / magnitude - target_mean)
        .collect();
    let mut matrix = Matrix {
        n,
        linear: vec![0.0; n * selected.len() * BASES],
        bins: vec![0; n * selected.len()],
    };
    let mut features = Vec::new();
    for (j, &column) in selected.iter().enumerate() {
        let mut sorted: Vec<f64> = rows
            .iter()
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
            let inverse = if variance > 0.0 {
                1.0 / variance.sqrt()
            } else {
                0.0
            };
            feature.basis_inverse_sd[k] = inverse;
            for i in 0..n {
                matrix.linear[offset + i] = ((matrix.linear[offset + i] - feature.basis_mean[k])
                    * inverse)
                    .clamp(-32.0, 32.0);
            }
        }
        features.push(feature);
    }
    (
        Preparation {
            features,
            target_magnitude: magnitude,
            target_mean,
        },
        matrix,
        y,
    )
}

pub(crate) fn encode(table: &[f64], cols: usize, rows: &[usize], prep: &Preparation) -> Matrix {
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

pub(crate) fn ridge(matrix: &Matrix, y: &[f64], lambda: f64) -> Vec<f64> {
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

pub(crate) fn linear_predictions(matrix: &Matrix, coefficients: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; matrix.n];
    for (j, &coefficient) in coefficients.iter().enumerate() {
        for i in 0..matrix.n {
            result[i] += coefficient * matrix.linear[j * matrix.n + i];
        }
    }
    result
}
