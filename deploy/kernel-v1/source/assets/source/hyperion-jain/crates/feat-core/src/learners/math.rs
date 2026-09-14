//! Invented-model backend: executes a fixed pipeline genome through the feat-math Rust
//! interpreter, so evolved algorithms join the phase-5 OOF comparison (fold-pure CV, leaderboard,
//! CV heatmap) exactly like any other member. The genome's own preprocessing stages are refit
//! inside `train_single` on every fold, so it is leak-safe by construction.

use crate::backend::{Backend, BackendKind, Model, TrainSpec};
use feat_math::{model as fm, GenomeIr};
use std::path::Path;

pub struct MathBackend {
    genome: GenomeIr,
    dsl: String,
}

impl MathBackend {
    pub fn from_dsl(dsl: &str) -> Option<Self> {
        GenomeIr::parse(dsl).ok().map(|genome| MathBackend {
            genome,
            dsl: dsl.to_string(),
        })
    }
}

/// Column-major `[f32]` (jain's FFI layout) -> row-major `Vec<Vec<f64>>` (feat-math's layout).
fn to_rows(x: &[f32], n: usize, p: usize) -> feat_math::Rows {
    (0..n)
        .map(|i| (0..p).map(|j| x[j * n + i] as f64).collect())
        .collect()
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PersistedMathModel {
    version: u32,
    dsl: String,
    targets: Vec<Option<fm::Model>>,
}

pub struct MathModel {
    dsl: String,
    /// One fitted pipeline per target (empty slot = fit was infeasible; predicts 0).
    targets: Vec<Option<fm::Model>>,
}

impl Model for MathModel {
    fn output_dim(&self) -> usize {
        self.targets.len().max(1)
    }
    fn predict(&self, x: &[f32], rows: usize, p: usize, _proba: bool) -> Vec<f32> {
        let data = to_rows(x, rows, p);
        let k = self.output_dim();
        let mut out = vec![0f32; rows * k];
        for (t, fitted) in self.targets.iter().enumerate() {
            if let Some(m) = fitted {
                for (i, v) in m.predict(&data).into_iter().enumerate() {
                    out[i * k + t] = if v.is_finite() { v as f32 } else { 0.0 };
                }
            }
        }
        out
    }
    fn importance(&self, p: usize) -> Vec<f32> {
        vec![0.0; p]
    }

    fn save_native(&self, path: &Path) -> Result<(), String> {
        let persisted = PersistedMathModel {
            version: 1,
            dsl: self.dsl.clone(),
            targets: self.targets.clone(),
        };
        let body = serde_json::to_string_pretty(&persisted).map_err(|e| e.to_string())?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, body).map_err(|e| e.to_string())
    }
}

pub fn load_math_model(path: &Path) -> anyhow::Result<Box<dyn Model>> {
    let body = std::fs::read_to_string(path)?;
    let persisted: PersistedMathModel = serde_json::from_str(&body)?;
    if persisted.version != 1 {
        anyhow::bail!(
            "unsupported math model version {} (expected 1)",
            persisted.version
        );
    }
    Ok(Box::new(MathModel {
        dsl: persisted.dsl,
        targets: persisted.targets,
    }))
}

impl Backend for MathBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Invented
    }
    fn train_single(
        &self,
        x: &[f32],
        n: usize,
        p: usize,
        y: &[f32],
        _s: &TrainSpec,
    ) -> Box<dyn Model> {
        let rows = to_rows(x, n, p);
        let y64: Vec<f64> = y.iter().map(|v| *v as f64).collect();
        Box::new(MathModel {
            dsl: self.dsl.clone(),
            targets: vec![fm::fit(&self.genome, &rows, &y64).ok()],
        })
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
        let rows = to_rows(x, n, p);
        let targets = (0..k)
            .map(|t| {
                let y64: Vec<f64> = y_multi[t * n..(t + 1) * n]
                    .iter()
                    .map(|v| *v as f64)
                    .collect();
                fm::fit(&self.genome, &rows, &y64).ok()
            })
            .collect();
        Box::new(MathModel {
            dsl: self.dsl.clone(),
            targets,
        })
    }
}
