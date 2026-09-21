//! Small candle ops (vendored verbatim from the veox reference).

use candle_core::{Result, Tensor};

pub struct LinearWeights {
    pub weight: Tensor, // (out_features, in_features)
    pub bias: Tensor,   // (out_features,)
}

pub fn linear3(x: &Tensor, w: &LinearWeights) -> Result<Tensor> {
    x.broadcast_matmul(&w.weight.transpose(0, 1)?)?
        .broadcast_add(&w.bias)
}

pub fn relu(x: &Tensor) -> Result<Tensor> {
    x.relu()
}

pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
    // candle_nn::LayerNorm applies LayerNorm on the last dimension.
    let ln = candle_nn::LayerNorm::new(weight.clone(), bias.clone(), eps as f64);
    candle_nn::Module::forward(&ln, x)
}

pub fn softmax_last(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::softmax(x, candle_core::D::Minus1)
}
