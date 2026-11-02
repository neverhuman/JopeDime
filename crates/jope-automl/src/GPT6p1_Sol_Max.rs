#![forbid(unsafe_code)]

//! An original, dependency-free general regressor.
//! Authoring: mathematical reasoning; no numerical experiments or borrowed source.
//! The standalone source is frozen before repository integration or execution.

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

const MAX_ROWS: usize = 24_576;
const MAX_FEATURES: usize = 96;
const MAX_CANDIDATES: usize = 512;
const QUANTILE_BINS: usize = 64;

mod random;
use random::{sample_indices, Random};
mod tree;
use tree::{growth, tree_limit, Growth, Tree};
mod selection;
use selection::{choose, fit_final};

pub struct Regressor {
    seed: u64,
    fitted: Option<Fitted>,
}

struct Fitted {
    width: usize,
    features: Vec<Feature>,
    scale: TargetScale,
    linear: Linear,
    trees: Vec<Tree>,
    mixture: f64,
}

#[derive(Clone)]
struct Feature {
    column: usize,
    magnitude: f64,
    center: f64,
    inverse_sd: f64,
    missing_rate: f64,
    cuts: Vec<f64>,
}

#[derive(Clone, Copy)]
struct TargetScale {
    magnitude: f64,
    center: f64,
    sd: f64,
}

struct Prepared {
    n: usize,
    fit_n: usize,
    features: Vec<Feature>,
    scale: TargetScale,
    // Both matrices are column-major. Linear columns come in value/missing pairs.
    bins: Vec<u8>,
    values: Vec<f64>,
    y: Vec<f64>,
}

struct Linear {
    intercept: f64,
    coefficients: Vec<f64>,
}

struct Choice {
    penalty: f64,
    trees: usize,
    mixture: f64,
}

impl TargetScale {
    fn learn(table: &[f64], width: usize, rows: &[usize]) -> Self {
        let mut magnitude: f64 = 0.0;
        for &row in rows {
            magnitude = magnitude.max(table[row * width + width - 1].abs());
        }
        if magnitude == 0.0 {
            return Self {
                magnitude: 0.0,
                center: 0.0,
                sd: 0.0,
            };
        }
        let mut center = 0.0;
        let mut moment = 0.0;
        for (i, &row) in rows.iter().enumerate() {
            let value = table[row * width + width - 1] / magnitude;
            let delta = value - center;
            center += delta / (i + 1) as f64;
            moment += delta * (value - center);
        }
        Self {
            magnitude,
            center,
            sd: (moment / rows.len() as f64).max(0.0).sqrt(),
        }
    }

    fn encode(&self, value: f64) -> f64 {
        if self.magnitude == 0.0 || self.sd <= 1e-150 {
            0.0
        } else {
            (value / self.magnitude - self.center) / self.sd
        }
    }

    fn decode(&self, value: f64) -> f64 {
        if self.magnitude == 0.0 {
            return 0.0;
        }
        let unit = self.center + self.sd * value.clamp(-1e6, 1e6);
        let result = self.magnitude * unit;
        if result.is_finite() {
            result
        } else if unit < 0.0 {
            -f64::MAX
        } else {
            f64::MAX
        }
    }
}

impl Feature {
    fn linear(&self, value: f64) -> (f64, f64) {
        if value.is_nan() {
            (0.0, 1.0 - self.missing_rate)
        } else {
            // Clipping before multiplication also keeps an extreme new row finite.
            let scaled = (value / self.magnitude).clamp(-1e6, 1e6);
            (
                ((scaled - self.center) * self.inverse_sd).clamp(-20.0, 20.0),
                -self.missing_rate,
            )
        }
    }

    fn bin(&self, value: f64) -> u8 {
        if value.is_nan() {
            return 0;
        }
        let mut low = 0;
        let mut high = self.cuts.len();
        while low < high {
            let middle = low + (high - low) / 2;
            if value <= self.cuts[middle] {
                high = middle;
            } else {
                low = middle + 1;
            }
        }
        (low + 1) as u8
    }

    fn learn(table: &[f64], width: usize, rows: &[usize], column: usize) -> Self {
        let mut finite = Vec::new();
        let mut magnitude: f64 = 0.0;
        for &row in rows {
            let x = table[row * width + column];
            if !x.is_nan() {
                magnitude = magnitude.max(x.abs());
                finite.push(x);
            }
        }
        // A positive magnitude is also defined for zero and completely missing columns.
        if magnitude == 0.0 {
            magnitude = 1.0;
        }
        let mut center = 0.0;
        let mut moment = 0.0;
        for (i, &x) in finite.iter().enumerate() {
            let value = x / magnitude;
            let delta = value - center;
            center += delta / (i + 1) as f64;
            moment += delta * (value - center);
        }
        let sd = (moment / finite.len().max(1) as f64).max(0.0).sqrt();
        let inverse_sd = if sd > 1e-12 { 1.0 / sd } else { 0.0 };
        finite.sort_unstable_by(|a, b| a.total_cmp(b));
        let mut cuts = Vec::new();
        if finite.len() > 1 {
            let maximum = finite[finite.len() - 1];
            for j in 1..QUANTILE_BINS {
                let k = (finite.len() / QUANTILE_BINS) * j
                    + ((finite.len() % QUANTILE_BINS) * j) / QUANTILE_BINS;
                let cut = finite[k.min(finite.len() - 1)];
                if cut < maximum && cuts.last().map_or(true, |last| cut > *last) {
                    cuts.push(cut);
                }
            }
        }
        Self {
            column,
            magnitude,
            center,
            inverse_sd,
            missing_rate: 1.0 - finite.len() as f64 / rows.len() as f64,
            cuts,
        }
    }
}

fn feature_columns(
    table: &[f64],
    width: usize,
    rows: &[usize],
    scale: TargetScale,
    rng: &mut Random,
) -> Vec<usize> {
    let p = width - 1;
    if p <= MAX_FEATURES {
        return (0..p).collect();
    }
    let candidates = sample_indices(p, MAX_CANDIDATES, rng);
    let pilot = &rows[..rows.len().min(512)];
    let mut ranking = Vec::new();
    for column in candidates {
        let mut magnitude: f64 = 0.0;
        for &row in pilot {
            let x = table[row * width + column];
            if !x.is_nan() {
                magnitude = magnitude.max(x.abs());
            }
        }
        magnitude = magnitude.max(f64::MIN_POSITIVE);
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut observed, mut missing, mut missing_y, mut total_y, mut total_yy) =
            (0usize, 0usize, 0.0, 0.0, 0.0);
        for &row in pilot {
            let y = scale.encode(table[row * width + width - 1]);
            total_y += y;
            total_yy += y * y;
            let x = table[row * width + column];
            if x.is_nan() {
                missing += 1;
                missing_y += y;
            } else {
                let x = x / magnitude;
                observed += 1;
                sx += x;
                sy += y;
                sxx += x * x;
                syy += y * y;
                sxy += x * y;
            }
        }
        let count = observed.max(1) as f64;
        let vx = (sxx - sx * sx / count).max(0.0);
        let vy = (syy - sy * sy / count).max(0.0);
        let covariance = sxy - sx * sy / count;
        let mut score = covariance * covariance / (vx * vy + 1e-30);
        let count = pilot.len() as f64;
        let missing_variance = missing as f64 * (1.0 - missing as f64 / count);
        let target_variance = (total_yy - total_y * total_y / count).max(0.0);
        let missing_covariance = missing_y - missing as f64 * total_y / count;
        score = score.max(
            missing_covariance * missing_covariance / (missing_variance * target_variance + 1e-30),
        );
        ranking.push((score, column));
    }
    ranking.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let favored = (MAX_FEATURES * 3 / 4).min(ranking.len());
    let mut result: Vec<usize> = ranking[..favored].iter().map(|x| x.1).collect();
    let mut remainder: Vec<usize> = ranking[favored..].iter().map(|x| x.1).collect();
    rng.shuffle(&mut remainder);
    result.extend(remainder.into_iter().take(MAX_FEATURES - favored));
    result.sort_unstable();
    result
}

impl Prepared {
    fn new(table: &[f64], width: usize, rows: &[usize], fit_n: usize, seed: u64) -> Self {
        let scale = TargetScale::learn(table, width, &rows[..fit_n]);
        let mut rng = Random::new(seed ^ 0x27d4eb2f165667c5);
        let columns = feature_columns(table, width, &rows[..fit_n], scale, &mut rng);
        let features: Vec<Feature> = columns
            .into_iter()
            .map(|column| Feature::learn(table, width, &rows[..fit_n], column))
            .collect();
        let n = rows.len();
        let mut bins = vec![0u8; n * features.len()];
        let mut values = vec![0.0; n * features.len() * 2];
        for (j, feature) in features.iter().enumerate() {
            for (i, &row) in rows.iter().enumerate() {
                let value = table[row * width + feature.column];
                bins[j * n + i] = feature.bin(value);
                let (continuous, missing) = feature.linear(value);
                values[(2 * j) * n + i] = continuous;
                values[(2 * j + 1) * n + i] = missing;
            }
        }
        let y = rows
            .iter()
            .map(|&row| scale.encode(table[row * width + width - 1]))
            .collect();
        Self {
            n,
            fit_n,
            features,
            scale,
            bins,
            values,
            y,
        }
    }
}

impl Linear {
    fn learn(data: &Prepared, penalty: f64) -> Self {
        let dimension = 2 * data.features.len();
        let n = data.fit_n;
        let mut coefficients = vec![0.0; dimension];
        let mut intercept = data.y[..n].iter().sum::<f64>() / n as f64;
        let mut residual: Vec<f64> = data.y[..n].iter().map(|y| y - intercept).collect();
        let norms: Vec<f64> = (0..dimension)
            .map(|j| {
                data.values[j * data.n..j * data.n + n]
                    .iter()
                    .map(|x| x * x)
                    .sum()
            })
            .collect();
        let passes = (40_000_000usize / n.saturating_mul(dimension.max(1)).max(1)).clamp(6, 48);
        for _ in 0..passes {
            let shift = residual.iter().sum::<f64>() / n as f64;
            intercept += shift;
            for r in &mut residual {
                *r -= shift;
            }
            let mut change = shift.abs();
            for j in 0..dimension {
                if norms[j] < 1e-24 {
                    continue;
                }
                let x = &data.values[j * data.n..j * data.n + n];
                let mut numerator = coefficients[j] * norms[j];
                for i in 0..n {
                    numerator += x[i] * residual[i];
                }
                let updated = (numerator / (norms[j] + penalty * n as f64)).clamp(-1e4, 1e4);
                let delta = updated - coefficients[j];
                coefficients[j] = updated;
                change = change.max(delta.abs());
                for i in 0..n {
                    residual[i] -= delta * x[i];
                }
            }
            if change < 1e-7 {
                break;
            }
        }
        Self {
            intercept,
            coefficients,
        }
    }

    fn prepared_predictions(&self, data: &Prepared) -> Vec<f64> {
        let mut result = vec![self.intercept; data.n];
        for (j, &coefficient) in self.coefficients.iter().enumerate() {
            for (i, prediction) in result.iter_mut().enumerate() {
                *prediction += coefficient * data.values[j * data.n + i];
            }
        }
        result
    }
}

impl RegressorApi for Regressor {
    fn new(seed: u64) -> Self {
        Self { seed, fitted: None }
    }

    fn fit(&mut self, train: &[f64], n_rows: usize, n_cols: usize) -> Result<(), String> {
        // Invalidate first, including on shape errors and unsuccessful refits.
        self.fitted = None;
        if n_rows == 0 || n_cols == 0 {
            return Err(String::from(
                "fitting requires at least one row and the target column",
            ));
        }
        let length = n_rows
            .checked_mul(n_cols)
            .ok_or_else(|| String::from("training dimensions overflow"))?;
        if train.len() != length {
            return Err(String::from("training length does not match shape"));
        }
        for row in train.chunks_exact(n_cols) {
            for &x in &row[..n_cols - 1] {
                if x.is_infinite() {
                    return Err(String::from("infinite training feature"));
                }
            }
            if !row[n_cols - 1].is_finite() {
                return Err(String::from("target must be finite"));
            }
        }
        let mut rng = Random::new(self.seed ^ 0xd6e8feb86659fd93);
        let rows = sample_indices(n_rows, MAX_ROWS, &mut rng);
        let n = rows.len();
        let choice = if n >= 64 && n_cols > 1 {
            let validation_n = 16usize.max(n / 5);
            // All screening, scales, and quantiles here use only the inner fit rows.
            let data = Prepared::new(train, n_cols, &rows, n - validation_n, self.seed);
            choose(&data, self.seed)
        } else {
            // A fixed conservative policy when a meaningful holdout is unavailable.
            Choice {
                penalty: if n < 12 { 1.0 } else { 0.01 },
                trees: if n >= 12 && n_cols > 1 { 96 } else { 0 },
                mixture: 0.5,
            }
        };
        let data = Prepared::new(train, n_cols, &rows, n, self.seed);
        let choice = if data.scale.sd <= 1e-150 || data.features.is_empty() {
            Choice {
                penalty: choice.penalty,
                trees: 0,
                mixture: 0.0,
            }
        } else {
            choice
        };
        self.fitted = Some(fit_final(data, n_cols - 1, choice, self.seed));
        Ok(())
    }

    fn predict(
        &self,
        features: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Result<Vec<f64>, String> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| String::from("regressor is not fitted"))?;
        if n_features != fitted.width {
            return Err(String::from("prediction feature width changed"));
        }
        let length = n_rows
            .checked_mul(n_features)
            .ok_or_else(|| String::from("prediction dimensions overflow"))?;
        if features.len() != length {
            return Err(String::from("prediction length does not match shape"));
        }
        for &value in features {
            if value.is_infinite() {
                return Err(String::from("infinite prediction feature"));
            }
        }
        let mut result = Vec::new();
        result
            .try_reserve(n_rows)
            .map_err(|_| String::from("prediction allocation failed"))?;
        let mut bins = vec![0u8; fitted.features.len()];
        for row in 0..n_rows {
            let mut prediction = fitted.linear.intercept;
            for (j, feature) in fitted.features.iter().enumerate() {
                let value = features[row * n_features + feature.column];
                let (continuous, missing) = feature.linear(value);
                prediction += fitted.linear.coefficients[2 * j] * continuous
                    + fitted.linear.coefficients[2 * j + 1] * missing;
                bins[j] = feature.bin(value);
            }
            let mut correction = 0.0;
            for tree in &fitted.trees {
                correction += tree.predict_bins(&bins);
            }
            prediction += fitted.mixture * correction;
            let value = fitted.scale.decode(prediction);
            if !value.is_finite() {
                return Err(String::from("numerical prediction failure"));
            }
            result.push(value);
        }
        Ok(result)
    }
}
