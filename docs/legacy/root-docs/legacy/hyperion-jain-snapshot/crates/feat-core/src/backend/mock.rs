use super::{Backend, BackendKind, Model, TrainSpec};
#[cfg(feature = "ci-smoke")]
use std::path::Path;

/// A tiny standardized-linear model: coef[j] = corr(feature_j, target). Deterministic and fast.
pub struct MockBackend;

#[cfg_attr(feature = "ci-smoke", derive(serde::Serialize, serde::Deserialize))]
pub(super) struct MockModel {
    dim: usize,
    p: usize,
    mean: Vec<f32>,
    std: Vec<f32>,
    coef: Vec<f32>,
}

fn col_stats(x: &[f32], n: usize, p: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0f32; p];
    let mut std = vec![1f32; p];
    for j in 0..p {
        let c = &x[j * n..(j + 1) * n];
        let (mut s, mut cnt) = (0f64, 0usize);
        for &v in c {
            if v.is_finite() {
                s += v as f64;
                cnt += 1;
            }
        }
        let m = if cnt > 0 {
            (s / cnt as f64) as f32
        } else {
            0.0
        };
        let mut var = 0f64;
        for &v in c {
            if v.is_finite() {
                let d = (v - m) as f64;
                var += d * d;
            }
        }
        let sd = if cnt > 1 {
            (var / cnt as f64).sqrt() as f32
        } else {
            1.0
        };
        mean[j] = m;
        std[j] = if sd > 1e-9 { sd } else { 1.0 };
    }
    (mean, std)
}

impl MockModel {
    fn train(x: &[f32], n: usize, p: usize, targets: &[&[f32]]) -> Self {
        let dim = targets.len();
        let (mean, std) = col_stats(x, n, p);
        let mut coef = vec![0f32; dim * p];
        for (t, y) in targets.iter().enumerate() {
            let (ym, ys) = {
                let (mut s, mut cnt) = (0f64, 0usize);
                for &v in *y {
                    s += v as f64;
                    cnt += 1;
                }
                let m = (s / cnt.max(1) as f64) as f32;
                let mut var = 0f64;
                for &v in *y {
                    let d = (v - m) as f64;
                    var += d * d;
                }
                (m, ((var / cnt.max(1) as f64).sqrt() as f32).max(1e-9))
            };
            for j in 0..p {
                let c = &x[j * n..(j + 1) * n];
                let mut acc = 0f64;
                for i in 0..n {
                    let xv = if c[i].is_finite() {
                        (c[i] - mean[j]) / std[j]
                    } else {
                        0.0
                    };
                    let yv = (y[i] - ym) / ys;
                    acc += (xv * yv) as f64;
                }
                coef[t * p + j] = (acc / n as f64) as f32;
            }
        }
        MockModel {
            dim,
            p,
            mean,
            std,
            coef,
        }
    }
}

impl Model for MockModel {
    fn output_dim(&self) -> usize {
        self.dim
    }

    fn predict(&self, x: &[f32], rows: usize, p: usize, proba: bool) -> Vec<f32> {
        assert_eq!(p, self.p);
        let mut out = vec![0f32; rows * self.dim];
        for r in 0..rows {
            for t in 0..self.dim {
                let mut s = 0f32;
                for j in 0..p {
                    let v = x[j * rows + r];
                    let xv = if v.is_finite() {
                        (v - self.mean[j]) / self.std[j]
                    } else {
                        0.0
                    };
                    s += self.coef[t * p + j] * xv;
                }
                out[r * self.dim + t] = if proba { 1.0 / (1.0 + (-s).exp()) } else { s };
            }
        }
        out
    }

    fn importance(&self, p: usize) -> Vec<f32> {
        let mut imp = vec![0f32; p];
        for (j, slot) in imp.iter_mut().enumerate() {
            let mut a = 0f32;
            for t in 0..self.dim {
                a += self.coef[t * p + j].abs();
            }
            *slot = a / self.dim as f32;
        }
        let sum: f32 = imp.iter().sum::<f32>().max(1e-9);
        for v in &mut imp {
            *v = *v / sum * 100.0;
        }
        imp
    }

    #[cfg(feature = "ci-smoke")]
    fn save_native(&self, path: &Path) -> Result<(), String> {
        let body = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, body).map_err(|e| e.to_string())
    }
}

impl Backend for MockBackend {
    fn train_single(
        &self,
        x: &[f32],
        n: usize,
        p: usize,
        y: &[f32],
        _s: &TrainSpec,
    ) -> Box<dyn Model> {
        Box::new(MockModel::train(x, n, p, &[y]))
    }

    fn train_multi(
        &self,
        x: &[f32],
        n: usize,
        p: usize,
        y_multi: &[f32],
        k: usize,
        _s: &TrainSpec,
    ) -> Box<dyn Model> {
        let targets: Vec<&[f32]> = (0..k).map(|t| &y_multi[t * n..(t + 1) * n]).collect();
        Box::new(MockModel::train(x, n, p, &targets))
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Mock
    }
}
