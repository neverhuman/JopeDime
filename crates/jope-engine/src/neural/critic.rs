//! Small independent structural critic candidate.
//!
//! This module exposes inference only. Its parameter store is caller-owned and separate from
//! `Policy`; there is deliberately no actor optimizer, uncertainty output, or training hook.

use anyhow::{ensure, Result};
use tch::{nn, nn::Module, Kind, Tensor};

pub const INPUT_DIM: i64 = 128;
pub const WIDTH: i64 = 128;

/// Two hidden layers and one scalar acquisition score per candidate.
pub struct StructuralCritic {
    first: nn::Linear,
    second: nn::Linear,
    output: nn::Linear,
}

impl StructuralCritic {
    pub fn new(path: &nn::Path<'_>) -> Self {
        Self {
            first: nn::linear(path / "first", INPUT_DIM, WIDTH, Default::default()),
            second: nn::linear(path / "second", WIDTH, WIDTH, Default::default()),
            output: nn::linear(path / "acquisition", WIDTH, 1, Default::default()),
        }
    }

    /// Score `[batch, 128]` structural features as `[batch]` acquisition values.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let shape = features.size();
        ensure!(
            shape.len() == 2 && shape[0] > 0 && shape[1] == INPUT_DIM,
            "structural critic expects a non-empty [batch, 128] tensor"
        );
        ensure!(
            features.kind() == Kind::Float,
            "structural critic expects f32 features"
        );
        ensure!(
            features.isfinite().all().int64_value(&[]) != 0,
            "structural critic refuses non-finite features"
        );
        let hidden = self.first.forward(features).gelu("none");
        let hidden = self.second.forward(&hidden).gelu("none");
        let score = self.output.forward(&hidden).squeeze_dim(-1);
        ensure!(
            score.isfinite().all().int64_value(&[]) != 0,
            "structural critic produced non-finite scores"
        );
        Ok(score)
    }
}
