#![forbid(unsafe_code)]

//! An original, dependency-free general regressor.
//! Authoring: mathematical reasoning; no numerical experiments or borrowed source.
//! The standalone source is frozen before repository integration or execution.

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

const MAX_ROWS: usize = 24_576;
const MAX_FEATURES: usize = 96;
const MAX_CANDIDATES: usize = 512;
const QUANTILE_BINS: usize = 64;

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

struct Tree {
    nodes: Vec<Node>,
}

struct Node {
    feature: usize,
    threshold: u8,
    missing_left: bool,
    left: usize,
    right: usize,
    value: f64,
}

struct Growth {
    depth: usize,
    min_leaf: usize,
    penalty: f64,
    rate: f64,
}

struct Choice {
    penalty: f64,
    trees: usize,
    mixture: f64,
}

struct Random {
    state: u64,
}

impl Random {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        // A fixed integer mixing construction, not a source of external entropy.
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn index(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    fn shuffle(&mut self, values: &mut [usize]) {
        for i in (1..values.len()).rev() {
            let j = self.index(i + 1);
            values.swap(i, j);
        }
    }
}

fn sample_indices(total: usize, maximum: usize, rng: &mut Random) -> Vec<usize> {
    let count = total.min(maximum);
    let mut selected: Vec<usize> = (0..count).collect();
    for i in count..total {
        let j = rng.index(i + 1);
        if j < count {
            selected[j] = i;
        }
    }
    rng.shuffle(&mut selected);
    selected
}

impl TargetScale {
    fn learn(table: &[f64], width: usize, rows: &[usize]) -> Self {
        let mut magnitude: f64 = 0.0;
        for &row in rows {
            magnitude = magnitude.max(table[row * width + width - 1].abs());
        }
        if magnitude == 0.0 {
            return Self { magnitude: 0.0, center: 0.0, sd: 0.0 };
        }
        let mut center = 0.0;
        let mut moment = 0.0;
        for (i, &row) in rows.iter().enumerate() {
            let value = table[row * width + width - 1] / magnitude;
            let delta = value - center;
            center += delta / (i + 1) as f64;
            moment += delta * (value - center);
        }
        Self { magnitude, center, sd: (moment / rows.len() as f64).max(0.0).sqrt() }
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
            (((scaled - self.center) * self.inverse_sd).clamp(-20.0, 20.0), -self.missing_rate)
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
    table: &[f64], width: usize, rows: &[usize], scale: TargetScale, rng: &mut Random,
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
        score = score.max(missing_covariance * missing_covariance
            / (missing_variance * target_variance + 1e-30));
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
        let features: Vec<Feature> = columns.into_iter()
            .map(|column| Feature::learn(table, width, &rows[..fit_n], column)).collect();
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
        let y = rows.iter().map(|&row| scale.encode(table[row * width + width - 1])).collect();
        Self { n, fit_n, features, scale, bins, values, y }
    }
}

impl Linear {
    fn learn(data: &Prepared, penalty: f64) -> Self {
        let dimension = 2 * data.features.len();
        let n = data.fit_n;
        let mut coefficients = vec![0.0; dimension];
        let mut intercept = data.y[..n].iter().sum::<f64>() / n as f64;
        let mut residual: Vec<f64> = data.y[..n].iter().map(|y| y - intercept).collect();
        let norms: Vec<f64> = (0..dimension).map(|j| {
            data.values[j * data.n..j * data.n + n].iter().map(|x| x * x).sum()
        }).collect();
        let passes = (40_000_000usize / n.saturating_mul(dimension.max(1)).max(1)).clamp(6, 48);
        for _ in 0..passes {
            let shift = residual.iter().sum::<f64>() / n as f64;
            intercept += shift;
            for r in &mut residual { *r -= shift; }
            let mut change = shift.abs();
            for j in 0..dimension {
                if norms[j] < 1e-24 { continue; }
                let x = &data.values[j * data.n..j * data.n + n];
                let mut numerator = coefficients[j] * norms[j];
                for i in 0..n { numerator += x[i] * residual[i]; }
                let updated = (numerator / (norms[j] + penalty * n as f64)).clamp(-1e4, 1e4);
                let delta = updated - coefficients[j];
                coefficients[j] = updated;
                change = change.max(delta.abs());
                for i in 0..n { residual[i] -= delta * x[i]; }
            }
            if change < 1e-7 { break; }
        }
        Self { intercept, coefficients }
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

impl Tree {
    fn predict_prepared(&self, data: &Prepared, row: usize) -> f64 {
        let mut index = 0;
        loop {
            let node = &self.nodes[index];
            if node.left == usize::MAX { return node.value; }
            let bin = data.bins[node.feature * data.n + row];
            let left = if bin == 0 { node.missing_left } else { bin <= node.threshold };
            index = if left { node.left } else { node.right };
        }
    }

    fn predict_bins(&self, bins: &[u8]) -> f64 {
        let mut index = 0;
        loop {
            let node = &self.nodes[index];
            if node.left == usize::MAX { return node.value; }
            let bin = bins[node.feature];
            let left = if bin == 0 { node.missing_left } else { bin <= node.threshold };
            index = if left { node.left } else { node.right };
        }
    }

    fn learn(data: &Prepared, residual: &[f64], rng: &mut Random, growth: &Growth) -> Self {
        let mut rows = Vec::new();
        for i in 0..data.fit_n {
            if rng.index(100) < 85 { rows.push(i); }
        }
        if rows.len() < 2 * growth.min_leaf { rows = (0..data.fit_n).collect(); }
        let mut columns: Vec<usize> = (0..data.features.len()).collect();
        rng.shuffle(&mut columns);
        columns.truncate(tree_width(data.features.len()));
        let mut tree = Self { nodes: Vec::new() };
        tree.grow(data, residual, rows, &columns, 0, growth);
        tree
    }

    fn grow(
        &mut self, data: &Prepared, residual: &[f64], rows: Vec<usize>,
        columns: &[usize], depth: usize, growth: &Growth,
    ) -> usize {
        let total: f64 = rows.iter().map(|&i| residual[i]).sum();
        let index = self.nodes.len();
        self.nodes.push(Node {
            feature: 0, threshold: 0, missing_left: false,
            left: usize::MAX, right: usize::MAX,
            value: growth.rate * total / (rows.len() as f64 + growth.penalty),
        });
        if depth >= growth.depth || rows.len() < 2 * growth.min_leaf { return index; }
        let parent = total * total / (rows.len() as f64 + growth.penalty);
        let mut best_gain = 1e-10;
        let mut best = None;
        for &feature in columns {
            let last_bin = data.features[feature].cuts.len() + 1;
            let mut counts = vec![0usize; last_bin + 1];
            let mut sums = vec![0.0; last_bin + 1];
            for &row in &rows {
                let bin = data.bins[feature * data.n + row] as usize;
                counts[bin] += 1;
                sums[bin] += residual[row];
            }
            let mut left_count = 0;
            let mut left_sum = 0.0;
            // The two endpoint thresholds allow a pure missing-versus-observed split.
            for threshold in 0..=last_bin {
                if threshold > 0 {
                    left_count += counts[threshold];
                    left_sum += sums[threshold];
                }
                for missing_left in [false, true] {
                    let count = left_count + if missing_left { counts[0] } else { 0 };
                    if count < growth.min_leaf || rows.len() - count < growth.min_leaf { continue; }
                    let sum = left_sum + if missing_left { sums[0] } else { 0.0 };
                    let other = total - sum;
                    let gain = sum * sum / (count as f64 + growth.penalty)
                        + other * other / ((rows.len() - count) as f64 + growth.penalty) - parent;
                    if gain > best_gain {
                        best_gain = gain;
                        best = Some((feature, threshold as u8, missing_left));
                    }
                }
            }
        }
        if let Some((feature, threshold, missing_left)) = best {
            let mut left = Vec::new();
            let mut right = Vec::new();
            for row in rows {
                let bin = data.bins[feature * data.n + row];
                let goes_left = if bin == 0 { missing_left } else { bin <= threshold };
                if goes_left { left.push(row); } else { right.push(row); }
            }
            let left_index = self.grow(data, residual, left, columns, depth + 1, growth);
            let right_index = self.grow(data, residual, right, columns, depth + 1, growth);
            self.nodes[index].feature = feature;
            self.nodes[index].threshold = threshold;
            self.nodes[index].missing_left = missing_left;
            self.nodes[index].left = left_index;
            self.nodes[index].right = right_index;
        }
        index
    }
}

fn tree_width(p: usize) -> usize {
    p.min(32 + p / 4)
}

fn growth(n: usize) -> Growth {
    Growth {
        depth: if n >= 2048 { 6 } else { 4 },
        min_leaf: 3usize.max(n / 2048),
        penalty: 5.0,
        rate: 0.06,
    }
}

fn tree_limit(data: &Prepared) -> usize {
    if data.features.is_empty() || data.scale.sd <= 1e-150 { return 0; }
    let config = growth(data.fit_n);
    let work = data.fit_n.saturating_mul(tree_width(data.features.len()))
        .saturating_mul(config.depth).max(1);
    (750_000_000usize / work).clamp(32, 640)
}

fn validation_error(data: &Prepared, predictions: &[f64]) -> f64 {
    let mut loss = 0.0;
    for i in data.fit_n..data.n {
        let delta = data.y[i] - predictions[i];
        loss += delta * delta;
    }
    loss / (data.n - data.fit_n) as f64
}

fn choose(data: &Prepared, seed: u64) -> Choice {
    let penalties = [0.0001, 0.01, 1.0, 100.0];
    let mut best_penalty = penalties[0];
    let mut best_loss = f64::INFINITY;
    let mut baseline = Vec::new();
    for penalty in penalties {
        let model = Linear::learn(data, penalty);
        let predictions = model.prepared_predictions(data);
        let loss = validation_error(data, &predictions);
        if loss < best_loss {
            best_loss = loss;
            best_penalty = penalty;
            baseline = predictions;
        }
    }
    let mut choice = Choice { penalty: best_penalty, trees: 0, mixture: 0.0 };
    if baseline.is_empty() { return choice; }
    let mut predictions = baseline.clone();
    let mut residual = vec![0.0; data.fit_n];
    let mut rng = Random::new(seed ^ 0x94d049bb133111eb);
    let config = growth(data.fit_n);
    let limit = tree_limit(data);
    let mut last_improvement = 0;
    for t in 0..limit {
        for i in 0..data.fit_n { residual[i] = data.y[i] - predictions[i]; }
        let tree = Tree::learn(data, &residual, &mut rng, &config);
        for (i, prediction) in predictions.iter_mut().enumerate() {
            *prediction += tree.predict_prepared(data, i);
        }
        if (t + 1) % 8 != 0 && t + 1 != limit { continue; }
        let mut dot = 0.0;
        let mut norm = 0.0;
        for i in data.fit_n..data.n {
            let delta = predictions[i] - baseline[i];
            dot += delta * (data.y[i] - baseline[i]);
            norm += delta * delta;
        }
        let mixture = (dot / (norm + 1e-24)).clamp(0.0, 1.0);
        let mut loss = 0.0;
        for i in data.fit_n..data.n {
            let prediction = baseline[i] + mixture * (predictions[i] - baseline[i]);
            let delta = data.y[i] - prediction;
            loss += delta * delta;
        }
        loss /= (data.n - data.fit_n) as f64;
        if loss < best_loss * (1.0 - 1e-5) {
            best_loss = loss;
            choice.trees = t + 1;
            choice.mixture = mixture;
            last_improvement = t + 1;
        }
        if t + 1 >= 96 && t + 1 >= last_improvement + 64 { break; }
    }
    choice
}

fn fit_final(data: Prepared, width: usize, choice: Choice, seed: u64) -> Fitted {
    let linear = Linear::learn(&data, choice.penalty);
    let mut predictions = linear.prepared_predictions(&data);
    let mut residual = vec![0.0; data.fit_n];
    let config = growth(data.fit_n);
    let mut rng = Random::new(seed ^ 0x94d049bb133111eb);
    let mut trees = Vec::new();
    for _ in 0..choice.trees {
        for i in 0..data.fit_n { residual[i] = data.y[i] - predictions[i]; }
        let tree = Tree::learn(&data, &residual, &mut rng, &config);
        for (i, prediction) in predictions.iter_mut().enumerate() {
            *prediction += tree.predict_prepared(&data, i);
        }
        trees.push(tree);
    }
    Fitted {
        width,
        features: data.features,
        scale: data.scale,
        linear,
        trees,
        mixture: choice.mixture,
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
            return Err(String::from("fitting requires at least one row and the target column"));
        }
        let length = n_rows.checked_mul(n_cols)
            .ok_or_else(|| String::from("training dimensions overflow"))?;
        if train.len() != length { return Err(String::from("training length does not match shape")); }
        for row in train.chunks_exact(n_cols) {
            for &x in &row[..n_cols - 1] {
                if x.is_infinite() { return Err(String::from("infinite training feature")); }
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
            Choice { penalty: choice.penalty, trees: 0, mixture: 0.0 }
        } else { choice };
        self.fitted = Some(fit_final(data, n_cols - 1, choice, self.seed));
        Ok(())
    }

    fn predict(&self, features: &[f64], n_rows: usize, n_features: usize) -> Result<Vec<f64>, String> {
        let fitted = self.fitted.as_ref()
            .ok_or_else(|| String::from("regressor is not fitted"))?;
        if n_features != fitted.width { return Err(String::from("prediction feature width changed")); }
        let length = n_rows.checked_mul(n_features)
            .ok_or_else(|| String::from("prediction dimensions overflow"))?;
        if features.len() != length { return Err(String::from("prediction length does not match shape")); }
        for &value in features {
            if value.is_infinite() { return Err(String::from("infinite prediction feature")); }
        }
        let mut result = Vec::new();
        result.try_reserve(n_rows).map_err(|_| String::from("prediction allocation failed"))?;
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
            for tree in &fitted.trees { correction += tree.predict_bins(&bins); }
            prediction += fitted.mixture * correction;
            let value = fitted.scale.decode(prediction);
            if !value.is_finite() { return Err(String::from("numerical prediction failure")); }
            result.push(value);
        }
        Ok(result)
    }
}
