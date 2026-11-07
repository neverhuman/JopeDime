#![forbid(unsafe_code)]

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

pub struct Regressor {
    seed: u64,
    model: Option<Model>,
}

const MAX_VALUE: f64 = 1.7976931348623157e308;

fn clamp_value(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

fn clean(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

fn to_z(x: f64, center: f64, scale: f64) -> f64 {
    if x.is_nan() {
        return 0.0;
    }
    if !x.is_finite() {
        return if x.is_sign_negative() { -1.0e6 } else { 1.0e6 };
    }
    let s = if scale > 0.0 { scale } else { 1.0 };
    clamp_value(clean(x / s - center / s), -1.0e6, 1.0e6)
}

fn from_z(z: f64, center: f64, scale: f64) -> f64 {
    let zz = clamp_value(clean(z), -1.0e6, 1.0e6);
    let delta = scale * zz;
    let answer = center + delta;
    if answer.is_finite() {
        answer
    } else if delta.is_sign_negative() {
        -MAX_VALUE
    } else {
        MAX_VALUE
    }
}

fn float_order(a: f64, b: f64) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
}

#[derive(Clone, Copy)]
struct Data<'a> {
    table: &'a [f64],
    ids: &'a [usize],
    stride: usize,
    p: usize,
}

impl<'a> Data<'a> {
    fn n(&self) -> usize {
        self.ids.len()
    }

    fn x(&self, row: usize, feature: usize) -> f64 {
        self.table[self.ids[row] * self.stride + feature]
    }

    fn y(&self, row: usize) -> f64 {
        self.table[self.ids[row] * self.stride + self.p]
    }
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15u64);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9u64);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111ebu64);
        z ^ (z >> 31)
    }

    fn shuffle(&mut self, values: &mut [usize]) {
        let mut i = values.len();
        while i > 1 {
            i -= 1;
            let j = (self.next_u64() % ((i + 1) as u64)) as usize;
            values.swap(i, j);
        }
    }
}

fn subsample_ids(n: usize, cap: usize, rng: &mut Rng) -> Vec<usize> {
    if cap == 0 {
        return Vec::new();
    }
    if n <= cap {
        return (0..n).collect();
    }
    let mut ids: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut ids);
    ids.truncate(cap);
    ids.sort_unstable();
    ids
}

fn center_scale(data: &Data, local: &[usize]) -> (f64, f64) {
    if local.is_empty() {
        return (0.0, 1.0);
    }

    let mut magnitude = 0.0;
    for &i in local {
        let a = data.y(i).abs();
        if a > magnitude {
            magnitude = a;
        }
    }

    if magnitude == 0.0 {
        return (0.0, 1.0);
    }

    let mut sum = 0.0;
    for &i in local {
        sum += data.y(i) / magnitude;
    }
    let center = clean(magnitude * (sum / local.len() as f64));
    (center, magnitude)
}

fn target_stats(data: &Data, local: &[usize]) -> (f64, f64, Vec<f64>) {
    let (center, scale) = center_scale(data, local);
    let center_scaled = center / scale;
    let mut z = Vec::with_capacity(local.len());
    for &i in local {
        z.push(clean(data.y(i) / scale - center_scaled));
    }
    (center, scale, z)
}

fn mean_target(data: &Data, local: &[usize]) -> f64 {
    center_scale(data, local).0
}

fn ranked_scores(data: &Data, local: &[usize], z: &[f64]) -> Vec<(usize, f64)> {
    if data.p == 0 || local.is_empty() {
        return Vec::new();
    }

    let n = local.len();
    let nf = n as f64;
    let mut y_sum = 0.0;
    let mut y_sq = 0.0;
    for i in 0..n {
        y_sum += z[i];
        y_sq += z[i] * z[i];
    }

    let y_mean = y_sum / nf;
    let mut y_var = y_sq / nf - y_mean * y_mean;
    if y_var < 1.0e-12 {
        y_var = 1.0e-12;
    }
    let y_sd = y_var.sqrt();

    let mut scores = Vec::with_capacity(data.p);

    for j in 0..data.p {
        let mut magnitude = 0.0;
        let mut missing = 0usize;
        let mut missing_y = 0.0;

        for i in 0..n {
            let x = data.x(local[i], j);
            if x.is_nan() {
                missing += 1;
                missing_y += z[i];
            } else {
                let a = x.abs();
                if a > magnitude {
                    magnitude = a;
                }
            }
        }

        let observed = n - missing;
        let mut score = 0.0;

        if magnitude > 0.0 && observed > 1 {
            let mut sx = 0.0;
            let mut sxx = 0.0;
            let mut sy = 0.0;
            let mut syy = 0.0;
            let mut sxy = 0.0;

            for i in 0..n {
                let x = data.x(local[i], j);
                if !x.is_nan() {
                    let u = x / magnitude;
                    let v = z[i];
                    sx += u;
                    sxx += u * u;
                    sy += v;
                    syy += v * v;
                    sxy += u * v;
                }
            }

            let count = observed as f64;
            let vx = sxx - sx * sx / count;
            let vy = syy - sy * sy / count;
            if vx > 1.0e-20 && vy > 1.0e-20 {
                let covariance = sxy - sx * sy / count;
                let denominator = (vx * vy).sqrt();
                if denominator > 0.0 {
                    score = covariance.abs() / denominator;
                }
            }
        }

        if missing > 0 && observed > 0 {
            let observed_y = y_sum - missing_y;
            let difference = missing_y / missing as f64
                - observed_y / observed as f64;
            let weight = ((missing as f64) * (observed as f64)).sqrt() / nf;
            let missing_score = difference.abs() * weight / y_sd;
            if missing_score > score {
                score = missing_score;
            }
        }

        scores.push((j, clean(score)));
    }

    scores.sort_by(|a, b| {
        let order = float_order(b.1, a.1);
        if order == std::cmp::Ordering::Equal {
            a.0.cmp(&b.0)
        } else {
            order
        }
    });
    scores
}

fn select_features(
    scores: &[(usize, f64)],
    cap: usize,
    rng: &mut Rng,
) -> Vec<usize> {
    if cap == 0 {
        return Vec::new();
    }
    if scores.len() <= cap {
        return scores.iter().map(|v| v.0).collect();
    }

    let random_count = cap / 4;
    let top_count = cap - random_count;
    let mut selected = Vec::with_capacity(cap);

    for k in 0..top_count {
        selected.push(scores[k].0);
    }

    let mut remaining = Vec::with_capacity(scores.len() - top_count);
    for k in top_count..scores.len() {
        remaining.push(scores[k].0);
    }
    rng.shuffle(&mut remaining);
    for k in 0..random_count {
        selected.push(remaining[k]);
    }
    selected
}

fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() == 1 {
        return values[0];
    }

    let position = q * (values.len() - 1) as f64;
    let lo = position.floor() as usize;
    let hi = if lo + 1 < values.len() { lo + 1 } else { lo };
    let fraction = position - lo as f64;
    values[lo] * (1.0 - fraction) + values[hi] * fraction
}

struct LinearFeature {
    original: usize,
    magnitude: f64,
    center: f64,
    scale: f64,
    knots: [f64; 3],
    has_missing: bool,
}

impl LinearFeature {
    fn fit(data: &Data, local: &[usize], original: usize) -> Self {
        let mut values = Vec::new();
        let mut magnitude = 0.0;
        let mut missing = 0usize;

        for &i in local {
            let x = data.x(i, original);
            if x.is_nan() {
                missing += 1;
            } else {
                let a = x.abs();
                if a > magnitude {
                    magnitude = a;
                }
                values.push(x);
            }
        }

        for value in values.iter_mut() {
            *value = if magnitude > 0.0 {
                *value / magnitude
            } else {
                0.0
            };
        }
        values.sort_by(|a, b| float_order(*a, *b));

        let q25 = quantile(&values, 0.25);
        let q50 = quantile(&values, 0.50);
        let q75 = quantile(&values, 0.75);
        let center = q50;

        let mut scale = (q75 - q25) / 1.349;
        if scale <= 1.0e-12 || !scale.is_finite() {
            if values.is_empty() {
                scale = 1.0;
            } else {
                let mut mean = 0.0;
                for &value in &values {
                    mean += value;
                }
                mean /= values.len() as f64;

                let mut variance = 0.0;
                for &value in &values {
                    let d = value - mean;
                    variance += d * d;
                }
                scale = (variance / values.len() as f64).sqrt();
            }
        }
        if scale <= 1.0e-12 || !scale.is_finite() {
            scale = 1.0;
        }

        let knot = |q: f64| clamp_value((q - center) / scale, -8.0, 8.0);

        Self {
            original,
            magnitude,
            center,
            scale,
            knots: [knot(q25), knot(q50), knot(q75)],
            has_missing: missing > 0,
        }
    }

    fn value(&self, x: f64) -> (f64, f64) {
        if x.is_nan() {
            return (0.0, 1.0);
        }

        let u = if self.magnitude > 0.0 {
            x / self.magnitude
        } else {
            0.0
        };
        let z = clamp_value(clean((u - self.center) / self.scale), -32.0, 32.0);
        (z, 0.0)
    }
}

#[derive(Clone, Copy)]
enum TermKind {
    Raw(usize),
    Missing(usize),
    Asinh(usize),
    Hinge(usize, usize),
    Product(usize, usize),
}

fn fill_linear_row(
    stats: &[LinearFeature],
    terms: &[TermKind],
    data: &Data,
    row: usize,
    values: &mut [f64],
    missing: &mut [f64],
    output: &mut [f64],
) {
    for j in 0..stats.len() {
        let (value, is_missing) = stats[j].value(data.x(row, stats[j].original));
        values[j] = value;
        missing[j] = is_missing;
    }

    for k in 0..terms.len() {
        output[k] = match terms[k] {
            TermKind::Raw(j) => values[j],
            TermKind::Missing(j) => missing[j],
            TermKind::Asinh(j) => values[j].asinh(),
            TermKind::Hinge(j, knot) => {
                let d = values[j] - stats[j].knots[knot];
                if d > 0.0 { d } else { 0.0 }
            }
            TermKind::Product(a, b) => values[a] * values[b],
        };
    }
}

struct LinearModel {
    stats: Vec<LinearFeature>,
    terms: Vec<TermKind>,
    means: Vec<f64>,
    scales: Vec<f64>,
    coefficients: Vec<f64>,
    intercept: f64,
    center: f64,
    scale: f64,
}

impl LinearModel {
    fn fit(data: &Data, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let raw_cap = if data.p > 256 { 256 } else { data.p };
        let nonlinear_cap = if raw_cap <= 16 { raw_cap } else { 24 };
        let interaction_cap = if raw_cap >= 8 {
            28
        } else {
            raw_cap * raw_cap.saturating_sub(1) / 2
        };
        let expected_q = 2 * raw_cap + 4 * nonlinear_cap + interaction_cap;

        let mut row_cap = 50_000usize;
        if expected_q > 0 {
            let by_cells = 8_000_000usize / expected_q;
            if by_cells < 1_000 {
                row_cap = 1_000;
            } else if by_cells < row_cap {
                row_cap = by_cells;
            }
        }

        let local = subsample_ids(data.n(), row_cap, &mut rng);
        let (center, scale, yz) = target_stats(data, &local);
        let scores = ranked_scores(data, &local, &yz);
        let raw = select_features(&scores, raw_cap, &mut rng);

        let mut stats = Vec::with_capacity(raw.len());
        for j in raw {
            stats.push(LinearFeature::fit(data, &local, j));
        }

        let mut terms: Vec<TermKind> = Vec::new();
        for j in 0..stats.len() {
            terms.push(TermKind::Raw(j));
            if stats[j].has_missing {
                terms.push(TermKind::Missing(j));
            }
        }

        let nonlinear = if stats.len() < nonlinear_cap {
            stats.len()
        } else {
            nonlinear_cap
        };
        for j in 0..nonlinear {
            terms.push(TermKind::Asinh(j));
            for knot in 0..3 {
                terms.push(TermKind::Hinge(j, knot));
            }
        }

        let interactions = if stats.len() < 8 { stats.len() } else { 8 };
        for a in 0..interactions {
            for b in (a + 1)..interactions {
                terms.push(TermKind::Product(a, b));
            }
        }

        let n = local.len();
        let q = terms.len();
        let mut design = vec![0.0; n * q];
        let mut values = vec![0.0; stats.len()];
        let mut missing = vec![0.0; stats.len()];
        let mut row_buffer = vec![0.0; q];

        for i in 0..n {
            fill_linear_row(
                &stats,
                &terms,
                data,
                local[i],
                &mut values,
                &mut missing,
                &mut row_buffer,
            );
            for k in 0..q {
                design[i * q + k] = row_buffer[k];
            }
        }

        let mut means = vec![0.0; q];
        let mut scales = vec![1.0; q];

        for k in 0..q {
            let mut mean = 0.0;
            for i in 0..n {
                mean += design[i * q + k] / n as f64;
            }
            means[k] = mean;

            let mut variance = 0.0;
            for i in 0..n {
                let d = design[i * q + k] - mean;
                variance += d * d;
            }
            let sd = (variance / n as f64).sqrt();

            if sd.is_finite() && sd > 1.0e-10 {
                scales[k] = sd;
                for i in 0..n {
                    design[i * q + k] = (design[i * q + k] - mean) / sd;
                }
            } else {
                scales[k] = 1.0;
                for i in 0..n {
                    design[i * q + k] = 0.0;
                }
            }
        }

        let mut intercept = 0.0;
        for &value in &yz {
            intercept += value;
        }
        intercept /= n as f64;

        let mut response = vec![0.0; n];
        for i in 0..n {
            response[i] = yz[i] - intercept;
        }

        let lambda = 0.01 * (n + q) as f64 + 1.0e-8;
        let mut coefficients = solve_ridge(&design, &response, n, q, lambda);
        for coefficient in coefficients.iter_mut() {
            *coefficient = clamp_value(clean(*coefficient), -1.0e6, 1.0e6);
        }

        Self {
            stats,
            terms,
            means,
            scales,
            coefficients,
            intercept,
            center,
            scale,
        }
    }

    fn predict(&self, data: &Data) -> Vec<f64> {
        let mut output = vec![0.0; data.n()];
        let mut values = vec![0.0; self.stats.len()];
        let mut missing = vec![0.0; self.stats.len()];
        let mut row_buffer = vec![0.0; self.terms.len()];

        for i in 0..data.n() {
            fill_linear_row(
                &self.stats,
                &self.terms,
                data,
                i,
                &mut values,
                &mut missing,
                &mut row_buffer,
            );

            let mut z = self.intercept;
            for k in 0..self.terms.len() {
                let value = (row_buffer[k] - self.means[k]) / self.scales[k];
                z += self.coefficients[k] * value;
            }
            output[i] = from_z(z, self.center, self.scale);
        }
        output
    }
}

fn spd_solve_lower(mut gram: Vec<f64>, mut rhs: Vec<f64>, n: usize) -> Vec<f64> {
    for i in 0..n {
        for j in 0..=i {
            let mut value = gram[i * n + j];
            for k in 0..j {
                value -= gram[i * n + k] * gram[j * n + k];
            }

            if i == j {
                let mut diagonal = value;
                if !diagonal.is_finite() || diagonal < 1.0e-12 {
                    diagonal = 1.0e-12;
                }
                gram[i * n + i] = diagonal.sqrt();
            } else {
                let diagonal = gram[j * n + j];
                gram[i * n + j] = if diagonal > 0.0 {
                    value / diagonal
                } else {
                    0.0
                };
            }
        }
    }

    for i in 0..n {
        let mut value = rhs[i];
        for k in 0..i {
            value -= gram[i * n + k] * rhs[k];
        }
        rhs[i] = value / gram[i * n + i];
    }

    for reverse in 0..n {
        let i = n - 1 - reverse;
        let mut value = rhs[i];
        for k in (i + 1)..n {
            value -= gram[k * n + i] * rhs[k];
        }
        rhs[i] = value / gram[i * n + i];
    }
    rhs
}

fn solve_ridge(
    design: &[f64],
    response: &[f64],
    n: usize,
    q: usize,
    lambda: f64,
) -> Vec<f64> {
    if q == 0 {
        return Vec::new();
    }

    let primal_work = (n as f64) * (q as f64) * (q as f64);
    if q <= 220 && q <= n && primal_work <= 40_000_000.0 {
        let mut gram = vec![0.0; q * q];
        let mut rhs = vec![0.0; q];

        for i in 0..n {
            for j in 0..q {
                let value = design[i * q + j];
                rhs[j] += value * response[i];
                for k in 0..=j {
                    gram[j * q + k] += value * design[i * q + k];
                }
            }
        }
        for j in 0..q {
            gram[j * q + j] += lambda;
        }
        return spd_solve_lower(gram, rhs, q);
    }

    let dual_work = (n as f64) * (n as f64) * (q as f64);
    if n <= 220 && n < q && dual_work <= 40_000_000.0 {
        let mut gram = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut value = 0.0;
                for k in 0..q {
                    value += design[i * q + k] * design[j * q + k];
                }
                gram[i * n + j] = value;
            }
            gram[i * n + i] += lambda;
        }

        let alpha = spd_solve_lower(gram, response.to_vec(), n);
        let mut coefficients = vec![0.0; q];
        for k in 0..q {
            let mut value = 0.0;
            for i in 0..n {
                value += design[i * q + k] * alpha[i];
            }
            coefficients[k] = value;
        }
        return coefficients;
    }

    let mut rhs = vec![0.0; q];
    let mut diagonal = vec![lambda; q];
    for i in 0..n {
        for k in 0..q {
            let value = design[i * q + k];
            rhs[k] += value * response[i];
            diagonal[k] += value * value;
        }
    }

    let mut coefficients = vec![0.0; q];
    let mut residual = rhs;
    let mut preconditioned = vec![0.0; q];
    let mut direction = vec![0.0; q];
    let mut rz = 0.0;

    for k in 0..q {
        preconditioned[k] = residual[k] / diagonal[k];
        direction[k] = preconditioned[k];
        rz += residual[k] * preconditioned[k];
    }

    let initial = rz;
    if !initial.is_finite() || initial <= 1.0e-30 {
        return coefficients;
    }

    let mut product = vec![0.0; q];

    for _ in 0..64 {
        for k in 0..q {
            product[k] = 0.0;
        }

        for i in 0..n {
            let mut value = 0.0;
            for k in 0..q {
                value += design[i * q + k] * direction[k];
            }
            for k in 0..q {
                product[k] += value * design[i * q + k];
            }
        }
        for k in 0..q {
            product[k] += lambda * direction[k];
        }

        let mut denominator = 0.0;
        for k in 0..q {
            denominator += direction[k] * product[k];
        }
        if !denominator.is_finite() || denominator <= 1.0e-30 {
            break;
        }

        let alpha = rz / denominator;
        if !alpha.is_finite() {
            break;
        }

        for k in 0..q {
            coefficients[k] += alpha * direction[k];
            residual[k] -= alpha * product[k];
        }

        let mut new_rz = 0.0;
        for k in 0..q {
            preconditioned[k] = residual[k] / diagonal[k];
            new_rz += residual[k] * preconditioned[k];
        }
        if !new_rz.is_finite() || new_rz <= initial * 1.0e-8 {
            break;
        }

        let beta = new_rz / rz;
        if !beta.is_finite() {
            break;
        }
        rz = new_rz;
        for k in 0..q {
            direction[k] = preconditioned[k] + beta * direction[k];
        }
    }

    coefficients
}

struct BinFeature {
    original: usize,
    thresholds: Vec<f64>,
}

fn add_threshold(thresholds: &mut Vec<f64>, value: f64) {
    if !value.is_finite() {
        return;
    }
    let mut acceptable = true;
    if let Some(last) = thresholds.last() {
        acceptable = value > *last;
    }
    if acceptable {
        thresholds.push(value);
    }
}

impl BinFeature {
    fn fit(data: &Data, local: &[usize], original: usize) -> Self {
        let mut values = Vec::new();
        for &i in local {
            let value = data.x(i, original);
            if !value.is_nan() {
                values.push(value);
            }
        }
        values.sort_by(|a, b| float_order(*a, *b));

        let mut unique = Vec::with_capacity(values.len());
        for value in values {
            if unique.is_empty() || unique[unique.len() - 1] != value {
                unique.push(value);
            }
        }

        let mut thresholds = Vec::new();
        if unique.len() <= 64 {
            for k in 1..unique.len() {
                let midpoint = unique[k - 1] / 2.0 + unique[k] / 2.0;
                add_threshold(&mut thresholds, midpoint);
            }
        } else {
            for k in 1..64 {
                let position = k * unique.len() / 64;
                if position > 0 && position < unique.len() {
                    let a = unique[position - 1];
                    let b = unique[position];
                    if a != b {
                        let midpoint = a / 2.0 + b / 2.0;
                        add_threshold(&mut thresholds, midpoint);
                    }
                }
            }
        }

        Self { original, thresholds }
    }
}

fn bin_index(value: f64, thresholds: &[f64]) -> usize {
    if value.is_nan() {
        return thresholds.len() + 1;
    }

    let mut lo = 0usize;
    let mut hi = thresholds.len();
    while lo < hi {
        let middle = lo + (hi - lo) / 2;
        if value > thresholds[middle] {
            lo = middle + 1;
        } else {
            hi = middle;
        }
    }
    lo
}

#[derive(Clone, Copy)]
struct Node {
    leaf: bool,
    feature: usize,
    split: u16,
    missing_bin: u16,
    missing_left: bool,
    left: usize,
    right: usize,
    value: f64,
}

fn leaf_node(value: f64) -> Node {
    Node {
        leaf: true,
        feature: 0,
        split: 0,
        missing_bin: 0,
        missing_left: false,
        left: 0,
        right: 0,
        value,
    }
}

struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn predict_binned(&self, bins: &[u16], row: usize, feature_count: usize) -> f64 {
        let mut at = 0usize;
        for _ in 0..self.nodes.len() {
            let node = self.nodes[at];
            if node.leaf {
                return node.value;
            }

            let bin = bins[row * feature_count + node.feature] as usize;
            let go_left = if bin == node.missing_bin as usize {
                node.missing_left
            } else {
                bin <= node.split as usize
            };
            at = if go_left { node.left } else { node.right };
        }
        0.0
    }
}

fn split_gain(
    left_sum: f64,
    left_count: usize,
    right_sum: f64,
    right_count: usize,
    parent: f64,
    lambda: f64,
) -> f64 {
    left_sum * left_sum / (left_count as f64 + lambda)
        + right_sum * right_sum / (right_count as f64 + lambda)
        - parent
}

fn build_node(
    bins: &[u16],
    target: &[f64],
    rows: &[usize],
    feature_count: usize,
    candidates: &[usize],
    bin_counts: &[usize],
    depth: usize,
    max_depth: usize,
    min_leaf: usize,
    lambda: f64,
    nodes: &mut Vec<Node>,
) -> usize {
    let index = nodes.len();

    if rows.is_empty() {
        nodes.push(leaf_node(0.0));
        return index;
    }

    let mut sum = 0.0;
    for &row in rows {
        sum += target[row];
    }
    let value = sum / (rows.len() as f64 + lambda);
    nodes.push(leaf_node(value));

    if depth >= max_depth || rows.len() < 2 * min_leaf {
        return index;
    }

    let parent = sum * sum / (rows.len() as f64 + lambda);
    let mut best_gain = 1.0e-12;
    let mut best_feature = 0usize;
    let mut best_split = 0usize;
    let mut best_missing_left = false;
    let mut found = false;

    for &feature in candidates {
        let bin_count = bin_counts[feature];
        if bin_count < 2 {
            continue;
        }

        let mut counts = vec![0usize; bin_count];
        let mut sums = vec![0.0; bin_count];

        for &row in rows {
            let bin = bins[row * feature_count + feature] as usize;
            counts[bin] += 1;
            sums[bin] += target[row];
        }

        let missing_bin = bin_count - 1;
        let missing_count = counts[missing_bin];
        let missing_sum = sums[missing_bin];
        let total_count = rows.len();
        let total_sum = sum;

        let mut left_count = 0usize;
        let mut left_sum = 0.0;

        for split in 0..(bin_count - 1) {
            left_count += counts[split];
            left_sum += sums[split];

            let right_finite_count = total_count - left_count - missing_count;
            let right_finite_sum = total_sum - left_sum - missing_sum;

            let nl = left_count + missing_count;
            let nr = right_finite_count;
            if nl >= min_leaf && nr >= min_leaf {
                let gain = split_gain(
                    left_sum + missing_sum,
                    nl,
                    right_finite_sum,
                    nr,
                    parent,
                    lambda,
                );
                if gain.is_finite() && gain > best_gain {
                    best_gain = gain;
                    best_feature = feature;
                    best_split = split;
                    best_missing_left = true;
                    found = true;
                }
            }

            let nl = left_count;
            let nr = right_finite_count + missing_count;
            if nl >= min_leaf && nr >= min_leaf {
                let gain = split_gain(
                    left_sum,
                    nl,
                    right_finite_sum + missing_sum,
                    nr,
                    parent,
                    lambda,
                );
                if gain.is_finite() && gain > best_gain {
                    best_gain = gain;
                    best_feature = feature;
                    best_split = split;
                    best_missing_left = false;
                    found = true;
                }
            }
        }
    }

    if !found {
        return index;
    }

    let missing_bin = bin_counts[best_feature] - 1;
    let mut left_rows = Vec::new();
    let mut right_rows = Vec::new();

    for &row in rows {
        let bin = bins[row * feature_count + best_feature] as usize;
        let go_left = if bin == missing_bin {
            best_missing_left
        } else {
            bin <= best_split
        };
        if go_left {
            left_rows.push(row);
        } else {
            right_rows.push(row);
        }
    }

    if left_rows.len() < min_leaf || right_rows.len() < min_leaf {
        return index;
    }

    let left = build_node(
        bins,
        target,
        &left_rows,
        feature_count,
        candidates,
        bin_counts,
        depth + 1,
        max_depth,
        min_leaf,
        lambda,
        nodes,
    );
    let right = build_node(
        bins,
        target,
        &right_rows,
        feature_count,
        candidates,
        bin_counts,
        depth + 1,
        max_depth,
        min_leaf,
        lambda,
        nodes,
    );

    nodes[index] = Node {
        leaf: false,
        feature: best_feature,
        split: best_split as u16,
        missing_bin: missing_bin as u16,
        missing_left: best_missing_left,
        left,
        right,
        value,
    };
    index
}

fn build_tree(
    bins: &[u16],
    target: &[f64],
    rows: &[usize],
    feature_count: usize,
    candidates: &[usize],
    bin_counts: &[usize],
    max_depth: usize,
    min_leaf: usize,
    lambda: f64,
) -> Tree {
    let mut nodes = Vec::new();
    build_node(
        bins,
        target,
        rows,
        feature_count,
        candidates,
        bin_counts,
        0,
        max_depth,
        min_leaf,
        lambda,
        &mut nodes,
    );
    Tree { nodes }
}

#[derive(Clone, Copy)]
enum TreeMode {
    Boost,
    Bag,
}

struct TreeModel {
    features: Vec<BinFeature>,
    center: f64,
    scale: f64,
    mode: TreeMode,
    learning_rate: f64,
    trees: Vec<Tree>,
}

impl TreeModel {
    fn empty(center: f64, scale: f64, mode: TreeMode) -> Self {
        Self {
            features: Vec::new(),
            center,
            scale,
            mode,
            learning_rate: 1.0,
            trees: Vec::new(),
        }
    }

    fn fit(data: &Data, seed: u64, mode: TreeMode) -> Self {
        let mut rng = Rng::new(seed);
        let local = subsample_ids(data.n(), 60_000, &mut rng);
        let (center, scale, z) = target_stats(data, &local);

        if local.is_empty() {
            return Self::empty(center, scale, mode);
        }

        let mut energy = 0.0;
        for &value in &z {
            energy += value * value;
        }
        if energy <= 1.0e-12 || data.p == 0 {
            return Self::empty(center, scale, mode);
        }

        let scores = ranked_scores(data, &local, &z);
        let feature_cap = if data.p > 64 { 64 } else { data.p };
        let selected = select_features(&scores, feature_cap, &mut rng);

        let mut features = Vec::with_capacity(selected.len());
        for j in selected {
            features.push(BinFeature::fit(data, &local, j));
        }

        let n = local.len();
        let f = features.len();
        if f == 0 || n < 4 {
            return Self::empty(center, scale, mode);
        }

        let mut bins = vec![0u16; n * f];
        for i in 0..n {
            for j in 0..f {
                bins[i * f + j] = bin_index(
                    data.x(local[i], features[j].original),
                    &features[j].thresholds,
                ) as u16;
            }
        }

        let bin_counts: Vec<usize> = features
            .iter()
            .map(|feature| feature.thresholds.len() + 2)
            .collect();

        let max_depth = match mode {
            TreeMode::Boost => 3,
            TreeMode::Bag => 7,
        };
        let max_rounds = match mode {
            TreeMode::Boost => 260usize,
            TreeMode::Bag => 140usize,
        };
        let work_cap = match mode {
            TreeMode::Boost => 90_000_000usize,
            TreeMode::Bag => 70_000_000usize,
        };
        let learning_rate = match mode {
            TreeMode::Boost => 0.04,
            TreeMode::Bag => 1.0,
        };
        let lambda = match mode {
            TreeMode::Boost => 2.0,
            TreeMode::Bag => 1.0,
        };

        let min_leaf = match mode {
            TreeMode::Boost => {
                let value = n / 200;
                if value < 3 {
                    3
                } else if value > 30 {
                    30
                } else {
                    value
                }
            }
            TreeMode::Bag => {
                let value = n / 500;
                if value < 2 {
                    2
                } else if value > 20 {
                    20
                } else {
                    value
                }
            }
        };

        let candidate_count = match mode {
            TreeMode::Boost => {
                if f <= 24 { f } else { 16 }
            }
            TreeMode::Bag => {
                if f <= 16 {
                    f
                } else {
                    let value = (f as f64).sqrt() as usize;
                    if value < 4 { 4 } else { value }
                }
            }
        };

        let work = n * candidate_count * max_depth;
        let by_work = work_cap / work;
        let rounds = if n < 16 {
            if max_rounds < 48 { max_rounds } else { 48 }
        } else {
            let value = if by_work < 1 { 1 } else { by_work };
            if value > max_rounds { max_rounds } else { value }
        };

        let mut trees = Vec::new();
        let mut fitted = vec![0.0; n];

        for _ in 0..rounds {
            let sample_size = if n < 32 { n } else { (n * 4) / 5 };
            let sample = subsample_ids(n, sample_size, &mut rng);

            let mut candidates: Vec<usize> = (0..f).collect();
            if candidate_count < f {
                rng.shuffle(&mut candidates);
                candidates.truncate(candidate_count);
            }

            let tree = match mode {
                TreeMode::Boost => {
                    let mut residual = vec![0.0; n];
                    for i in 0..n {
                        residual[i] = z[i] - fitted[i];
                    }
                    build_tree(
                        &bins,
                        &residual,
                        &sample,
                        f,
                        &candidates,
                        &bin_counts,
                        max_depth,
                        min_leaf,
                        lambda,
                    )
                }
                TreeMode::Bag => build_tree(
                    &bins,
                    &z,
                    &sample,
                    f,
                    &candidates,
                    &bin_counts,
                    max_depth,
                    min_leaf,
                    lambda,
                ),
            };

            if tree.nodes.len() <= 1 {
                break;
            }

            if let TreeMode::Boost = mode {
                for i in 0..n {
                    let update = tree.predict_binned(&bins, i, f);
                    fitted[i] = clean(fitted[i] + learning_rate * update);
                }
            }
            trees.push(tree);
        }

        Self {
            features,
            center,
            scale,
            mode,
            learning_rate,
            trees,
        }
    }

    fn predict(&self, data: &Data) -> Vec<f64> {
        let mut output = vec![0.0; data.n()];
        let f = self.features.len();
        let mut bins = vec![0u16; f];

        for i in 0..data.n() {
            for j in 0..f {
                bins[j] = bin_index(
                    data.x(i, self.features[j].original),
                    &self.features[j].thresholds,
                ) as u16;
            }

            let mut z = 0.0;
            match self.mode {
                TreeMode::Boost => {
                    for tree in &self.trees {
                        z += self.learning_rate * tree.predict_binned(&bins, 0, f);
                    }
                }
                TreeMode::Bag => {
                    if !self.trees.is_empty() {
                        for tree in &self.trees {
                            z += tree.predict_binned(&bins, 0, f);
                        }
                        z /= self.trees.len() as f64;
                    }
                }
            }
            output[i] = from_z(z, self.center, self.scale);
        }
        output
    }
}

#[derive(Clone, Copy)]
enum BaseKind {
    Linear,
    Boost,
    Bag,
}

enum BaseModel {
    Linear(LinearModel),
    Tree(TreeModel),
}

fn fit_base(data: &Data, seed: u64, kind: BaseKind) -> BaseModel {
    match kind {
        BaseKind::Linear => BaseModel::Linear(LinearModel::fit(data, seed)),
        BaseKind::Boost => {
            BaseModel::Tree(TreeModel::fit(data, seed, TreeMode::Boost))
        }
        BaseKind::Bag => BaseModel::Tree(TreeModel::fit(data, seed, TreeMode::Bag)),
    }
}

impl BaseModel {
    fn predict(&self, data: &Data) -> Vec<f64> {
        match self {
            BaseModel::Linear(model) => model.predict(data),
            BaseModel::Tree(model) => model.predict(data),
        }
    }
}

struct Model {
    p: usize,
    center: f64,
    scale: f64,
    constant: f64,
    weights: Vec<f64>,
    bases: Vec<BaseModel>,
}

impl Model {
    fn predict(&self, data: &Data) -> Vec<f64> {
        let mut blended = vec![0.0; data.n()];
        let constant_z = to_z(self.constant, self.center, self.scale);

        for i in 0..data.n() {
            blended[i] = self.weights[0] * constant_z;
        }

        for b in 0..self.bases.len() {
            let weight = self.weights[b + 1];
            if weight == 0.0 {
                continue;
            }
            let predictions = self.bases[b].predict(data);
            for i in 0..data.n() {
                blended[i] += weight * to_z(predictions[i], self.center, self.scale);
            }
        }

        for value in blended.iter_mut() {
            *value = from_z(*value, self.center, self.scale);
        }
        blended
    }
}

fn prior_weights(n: usize) -> Vec<f64> {
    if n < 40 {
        vec![0.20, 0.35, 0.30, 0.15]
    } else if n > 6_000 {
        vec![0.03, 0.22, 0.50, 0.25]
    } else {
        vec![0.05, 0.25, 0.45, 0.25]
    }
}

fn optimize_weights(oof: &[f64], yz: &[f64], prior: &[f64]) -> Vec<f64> {
    let n = yz.len();
    let mut weights = prior.to_vec();
    if n == 0 {
        return weights;
    }

    let mut residual = vec![0.0; n];
    let mut variance_sum = 0.0;
    for i in 0..n {
        let mut value = yz[i];
        for b in 0..4 {
            value -= weights[b] * oof[i * 4 + b];
        }
        residual[i] = value;
        variance_sum += yz[i] * yz[i];
    }

    if variance_sum <= 1.0e-12 {
        return weights;
    }

    let penalty = (0.03 + 1.0 / n as f64) * variance_sum;

    for _ in 0..18 {
        for a in 0..4 {
            for b in (a + 1)..4 {
                let mut residual_difference = 0.0;
                let mut difference_sq = 0.0;

                for i in 0..n {
                    let difference = oof[i * 4 + a] - oof[i * 4 + b];
                    residual_difference += residual[i] * difference;
                    difference_sq += difference * difference;
                }

                let numerator = residual_difference
                    - penalty
                        * (weights[a] - prior[a] - weights[b] + prior[b]);
                let denominator = difference_sq + 2.0 * penalty;
                if denominator <= 1.0e-20 || !denominator.is_finite() {
                    continue;
                }

                let mut change = numerator / denominator;
                if !change.is_finite() {
                    continue;
                }

                let lower = -weights[a];
                let upper = weights[b];
                if change < lower {
                    change = lower;
                }
                if change > upper {
                    change = upper;
                }
                if change == 0.0 {
                    continue;
                }

                weights[a] += change;
                weights[b] -= change;
                for i in 0..n {
                    let difference = oof[i * 4 + a] - oof[i * 4 + b];
                    residual[i] -= change * difference;
                }
            }
        }
    }

    let mut sum = 0.0;
    for b in 0..4 {
        if !weights[b].is_finite() || weights[b] < 0.0 {
            weights[b] = 0.0;
        }
        sum += weights[b];
    }
    if sum <= 1.0e-12 {
        return prior.to_vec();
    }
    for b in 0..4 {
        weights[b] /= sum;
    }
    weights
}

fn fit_ensemble(data: &Data, seed: u64) -> Model {
    let all: Vec<usize> = (0..data.n()).collect();
    let (center, scale, yz) = target_stats(data, &all);

    let mut energy = 0.0;
    for &value in &yz {
        energy += value * value;
    }

    if data.p == 0 || data.n() < 2 || energy <= 1.0e-12 {
        return Model {
            p: data.p,
            center,
            scale,
            constant: center,
            weights: vec![1.0],
            bases: Vec::new(),
        };
    }

    let prior = prior_weights(data.n());
    let mut weights = prior.clone();

    let folds = if data.n() >= 40 && data.n() <= 2_000 {
        3
    } else if data.n() > 2_000 && data.n() <= 6_000 {
        2
    } else {
        0
    };

    if folds > 1 {
        let n = data.n();
        let mut permutation: Vec<usize> = (0..n).collect();
        let mut rng = Rng::new(seed ^ 0x51f15eED12345679u64);
        rng.shuffle(&mut permutation);

        let mut fold_of = vec![0usize; n];
        for position in 0..n {
            fold_of[permutation[position]] = position % folds;
        }

        let mut oof = vec![0.0; n * 4];
        let kinds = [BaseKind::Linear, BaseKind::Boost, BaseKind::Bag];

        for fold in 0..folds {
            let mut train_ids = Vec::new();
            let mut validation_ids = Vec::new();
            let mut validation_positions = Vec::new();

            for i in 0..n {
                if fold_of[i] == fold {
                    validation_ids.push(data.ids[i]);
                    validation_positions.push(i);
                } else {
                    train_ids.push(data.ids[i]);
                }
            }

            let training = Data {
                table: data.table,
                ids: &train_ids,
                stride: data.stride,
                p: data.p,
            };
            let validation = Data {
                table: data.table,
                ids: &validation_ids,
                stride: data.stride,
                p: data.p,
            };

            let training_local: Vec<usize> = (0..training.n()).collect();
            let constant = mean_target(&training, &training_local);
            let constant_z = to_z(constant, center, scale);
            for &position in &validation_positions {
                oof[position * 4] = constant_z;
            }

            for k in 0..kinds.len() {
                let model_seed = seed
                    .wrapping_add(0xA0761D6478BD642Fu64)
                    .wrapping_add((fold as u64).wrapping_mul(0xE7037ED1A0B428DBu64))
                    .wrapping_add((k as u64).wrapping_mul(0x8EBC6AF09C88C6E3u64));
                let model = fit_base(&training, model_seed, kinds[k]);
                let predictions = model.predict(&validation);

                for v in 0..validation_positions.len() {
                    let position = validation_positions[v];
                    oof[position * 4 + k + 1] =
                        to_z(predictions[v], center, scale);
                }
            }
        }

        weights = optimize_weights(&oof, &yz, &prior);
    }

    let mut bases = Vec::with_capacity(3);
    bases.push(fit_base(
        data,
        seed ^ 0x165667B19E3779F9u64,
        BaseKind::Linear,
    ));
    bases.push(fit_base(
        data,
        seed ^ 0x27D4EB2F165667C5u64,
        BaseKind::Boost,
    ));
    bases.push(fit_base(
        data,
        seed ^ 0x9E3779B97F4A7C15u64,
        BaseKind::Bag,
    ));

    Model {
        p: data.p,
        center,
        scale,
        constant: center,
        weights,
        bases,
    }
}

impl RegressorApi for Regressor {
    fn new(seed: u64) -> Self {
        Self { seed, model: None }
    }

    fn fit(
        &mut self,
        train: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(), String> {
        self.model = None;

        if n_rows == 0 {
            return Err(String::from("fit requires at least one row"));
        }
        if n_cols == 0 {
            return Err(String::from("fit requires at least one column"));
        }

        let expected = match n_rows.checked_mul(n_cols) {
            Some(value) => value,
            None => return Err(String::from("training shape overflows usize")),
        };
        if train.len() != expected {
            return Err(String::from("training slice length does not match shape"));
        }

        for i in 0..n_rows {
            for j in 0..n_cols {
                let value = train[i * n_cols + j];
                if j + 1 == n_cols {
                    if !value.is_finite() {
                        return Err(String::from("targets must be finite"));
                    }
                } else if value.is_infinite() {
                    return Err(String::from("infinite features are invalid"));
                }
            }
        }

        let ids: Vec<usize> = (0..n_rows).collect();
        let data = Data {
            table: train,
            ids: &ids,
            stride: n_cols,
            p: n_cols - 1,
        };
        self.model = Some(fit_ensemble(&data, self.seed));
        Ok(())
    }

    fn predict(
        &self,
        features: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Result<Vec<f64>, String> {
        let model = match self.model.as_ref() {
            Some(model) => model,
            None => return Err(String::from("predict called before successful fit")),
        };

        if n_features != model.p {
            return Err(String::from("prediction feature count does not match fit"));
        }

        let expected = match n_rows.checked_mul(n_features) {
            Some(value) => value,
            None => return Err(String::from("prediction shape overflows usize")),
        };
        if features.len() != expected {
            return Err(String::from("prediction slice length does not match shape"));
        }

        for &value in features {
            if value.is_infinite() {
                return Err(String::from("infinite features are invalid"));
            }
        }

        let ids: Vec<usize> = (0..n_rows).collect();
        let data = Data {
            table: features,
            ids: &ids,
            stride: n_features,
            p: n_features,
        };
        Ok(model.predict(&data))
    }
}
