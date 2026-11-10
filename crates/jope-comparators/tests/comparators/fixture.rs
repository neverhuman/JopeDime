//! Seeded synthetic regression data shared by the comparator tests.

/// `Table` is a struct for the synthetic fixture contract in this test target: a row-major
/// feature buffer and its targets.
pub struct Table {
    pub values: Vec<f64>,
    pub targets: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

/// Small explicit generator so the fixture does not depend on crate internals.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }
}

/// `y = 3a + 2 sin(2πb) + 4·[c > 0.5] + 0.1·noise`, with an uninformative column `d` and a
/// column `e` that is missing (NaN) in about one row in eight.
pub fn nonlinear(rows: usize, seed: u64) -> Table {
    let mut rng = Lcg(seed);
    let cols = 5;
    let mut values = Vec::with_capacity(rows * cols);
    let mut targets = Vec::with_capacity(rows);
    for _ in 0..rows {
        let (a, b, c, d) = (rng.unit(), rng.unit(), rng.unit(), rng.unit());
        let e = if rng.unit() < 0.125 {
            f64::NAN
        } else {
            rng.unit()
        };
        let noise = rng.unit() - 0.5;
        values.extend_from_slice(&[a, b, c, d, e]);
        let step = if c > 0.5 { 4.0 } else { 0.0 };
        targets.push(3.0 * a + 2.0 * (std::f64::consts::TAU * b).sin() + step + 0.1 * noise);
    }
    Table {
        values,
        targets,
        rows,
        cols,
    }
}

/// `y = 2 + 1.5a − 0.5b + 0.01·noise` over three columns (the third is constant).
pub fn linear(rows: usize, seed: u64) -> Table {
    let mut rng = Lcg(seed);
    let cols = 3;
    let mut values = Vec::with_capacity(rows * cols);
    let mut targets = Vec::with_capacity(rows);
    for _ in 0..rows {
        let (a, b, noise) = (rng.unit(), rng.unit(), rng.unit() - 0.5);
        values.extend_from_slice(&[a, b, 1.0]);
        targets.push(2.0 + 1.5 * a - 0.5 * b + 0.01 * noise);
    }
    Table {
        values,
        targets,
        rows,
        cols,
    }
}

/// RMSE of predicting the training-target mean for every evaluation row.
pub fn mean_baseline_rmse(train_targets: &[f64], eval_targets: &[f64]) -> f64 {
    let mean = train_targets.iter().sum::<f64>() / train_targets.len() as f64;
    let squared: f64 = eval_targets
        .iter()
        .map(|target| (target - mean).powi(2))
        .sum();
    (squared / eval_targets.len() as f64).sqrt()
}
