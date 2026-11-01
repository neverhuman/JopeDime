//! Real full/high CPU measurements from the immutable FIT closure only.
use crate::{
    dataset::jmat::{JmatCell, Matrix},
    discover::{r#loop::Plan, splits::rung0_subsample},
    ranker::landmarks,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

/// Private continuation evidence, not a learned-guidance qualification receipt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct FullFitLandmarks {
    pub recipe: String,
    pub fit_sha256: String,
    pub fit_rows: usize,
    pub sampled_fit_rows: Vec<usize>,
    pub features: Vec<String>,
    pub target: String,
    pub vector: Vec<f32>,
    pub columns: Vec<f32>,
}

impl FullFitLandmarks {
    pub fn extract(bytes: &[u8], plan: &Plan) -> Result<Self> {
        ensure!(
            jope_runtime::sha256_hex(bytes) == plan.matrix_sha256,
            "full landmark FIT identity mismatch"
        );
        let matrix = Matrix::open(bytes)?;
        ensure!(matrix.rows() > 0, "full landmarks require FIT rows");
        let column = |name: &str| {
            matrix
                .columns()
                .iter()
                .position(|c| c.name == name)
                .context("unknown full landmark FIT column")
        };
        let target = column(&plan.target)?;
        let features = plan
            .features
            .iter()
            .map(|name| {
                ensure!(name != &plan.target, "target entered landmark features");
                column(name)
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(!features.is_empty(), "full landmarks require features");
        let rows = rung0_subsample(
            &(0..matrix.rows()).collect::<Vec<_>>(),
            1024,
            plan.seed,
            "public-run/landmarks/full-fit-v1",
        );
        let mut x = Vec::with_capacity(rows.len() * features.len());
        let mut y = Vec::with_capacity(rows.len());
        for &row in &rows {
            for &col in &features {
                x.push(match matrix.cell(row, col) {
                    Some(JmatCell::Numeric(value)) if (value as f32).is_finite() => value as f32,
                    Some(JmatCell::Missing) => f32::NAN,
                    _ => anyhow::bail!("invalid full landmark FIT feature"),
                });
            }
            let Some(JmatCell::Numeric(value)) = matrix.cell(row, target) else {
                anyhow::bail!("invalid full landmark FIT target");
            };
            ensure!(
                (value as f32).is_finite(),
                "full landmark target overflows F32"
            );
            y.push(value as f32);
        }
        let output =
            landmarks::fit_transform_and_extract_bundle_high(&x, rows.len(), features.len(), &y)?;
        Ok(Self {
            recipe: "numeric-regression/full-high-cpu-v1".into(),
            fit_sha256: plan.matrix_sha256.clone(),
            fit_rows: matrix.rows(),
            sampled_fit_rows: rows,
            features: plan.features.clone(),
            target: plan.target.clone(),
            vector: output.vector().to_vec(),
            columns: output.columns().iter().flatten().copied().collect(),
        })
    }
}
