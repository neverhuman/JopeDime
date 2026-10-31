//! Linear `lm48` projection loaded from a fused-ranker safetensors file.
//!
//! The projection stores row-major output weights and one bias per output. A
//! safetensors weight matrix may be stored as either `[out, in]` or `[in, out]`;
//! the latter is transposed while loading. The RES-09 golden fixture will bind
//! this primitive to the complete T087 normalization recipe.

use std::path::Path;

use anyhow::{bail, Context, Result};
use safetensors::{Dtype, SafeTensors};

/// An immutable dense projection for the 48-dimensional landmark embedding.
#[derive(Clone, Debug)]
pub struct Lm48Projection {
    /// Row-major matrix with `out_dim` rows and `in_dim` columns.
    pub weight: Vec<f32>,
    /// One offset per output coordinate.
    pub bias: Vec<f32>,
    /// Number of input coordinates.
    pub in_dim: usize,
    /// Number of output coordinates.
    pub out_dim: usize,
}

impl Lm48Projection {
    /// Load `<prefix>.weight` and `<prefix>.bias` from a safetensors file.
    pub fn from_safetensors(path: &Path, prefix: &str) -> Result<Self> {
        if prefix.is_empty() {
            bail!("lm48 tensor prefix must not be empty");
        }

        let bytes = std::fs::read(path)
            .with_context(|| format!("read lm48 safetensors file {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&bytes).context("parse lm48 safetensors file")?;
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");
        let weight_tensor = tensors
            .tensor(&weight_name)
            .with_context(|| format!("missing lm48 tensor {weight_name}"))?;
        let bias_tensor = tensors
            .tensor(&bias_name)
            .with_context(|| format!("missing lm48 tensor {bias_name}"))?;

        if weight_tensor.dtype() != Dtype::F32 || bias_tensor.dtype() != Dtype::F32 {
            bail!("lm48 weights and bias must use F32 dtype");
        }

        let weight_shape = weight_tensor.shape();
        if weight_shape.len() != 2 {
            bail!("lm48 weight must be rank 2, got shape {weight_shape:?}");
        }
        let bias_shape = bias_tensor.shape();
        if bias_shape.len() != 1 {
            bail!("lm48 bias must be rank 1, got shape {bias_shape:?}");
        }
        let out_dim = bias_shape[0];
        if out_dim == 0 {
            bail!("lm48 output dimension must be nonzero");
        }

        let loaded_weight = read_f32(weight_tensor.data())?;
        let bias = read_f32(bias_tensor.data())?;
        let (in_dim, weight) = if weight_shape[0] == out_dim {
            (weight_shape[1], loaded_weight)
        } else if weight_shape[1] == out_dim {
            let in_dim = weight_shape[0];
            let mut transposed = vec![0.0; in_dim * out_dim];
            for input in 0..in_dim {
                for output in 0..out_dim {
                    transposed[output * in_dim + input] = loaded_weight[input * out_dim + output];
                }
            }
            (in_dim, transposed)
        } else {
            bail!("lm48 weight shape {weight_shape:?} is incompatible with bias length {out_dim}");
        };
        if in_dim == 0 {
            bail!("lm48 input dimension must be nonzero");
        }
        if weight.iter().chain(&bias).any(|value| !value.is_finite()) {
            bail!("lm48 weights and bias must be finite");
        }

        Ok(Self {
            weight,
            bias,
            in_dim,
            out_dim,
        })
    }

    /// Apply the row-major affine projection to one input vector.
    ///
    /// The input length must match `in_dim`; the T087 normalization/clamping
    /// stages are applied by the caller around this linear primitive.
    #[must_use]
    pub fn project(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(
            x.len(),
            self.in_dim,
            "lm48 input length must match the loaded projection"
        );
        (0..self.out_dim)
            .map(|output| {
                let row = &self.weight[output * self.in_dim..(output + 1) * self.in_dim];
                row.iter()
                    .zip(x)
                    .fold(self.bias[output], |sum, (weight, value)| {
                        sum + weight * value
                    })
            })
            .collect()
    }
}

fn read_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        bail!("lm48 F32 tensor data length is not divisible by four");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::Lm48Projection;

    struct Fixture(tempfile::NamedTempFile);

    impl Fixture {
        fn new(weight_shape: &[usize], weights: &[f32], bias: &[f32]) -> Self {
            let mut file =
                tempfile::NamedTempFile::new().expect("create synthetic safetensors fixture");
            let mut header = format!(
                "{{\"proj.weight\":{{\"dtype\":\"F32\",\"shape\":{weight_shape:?},\"data_offsets\":[0,{}]}},\"proj.bias\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}}}",
                weights.len() * 4,
                bias.len(),
                weights.len() * 4,
                (weights.len() + bias.len()) * 4
            );
            while header.len() % 8 != 0 {
                header.push(' ');
            }

            file.write_all(&(header.len() as u64).to_le_bytes())
                .expect("write safetensors header length");
            file.write_all(header.as_bytes())
                .expect("write safetensors header");
            for value in weights.iter().chain(bias) {
                file.write_all(&value.to_le_bytes())
                    .expect("write synthetic tensor value");
            }
            Self(file)
        }
    }

    #[test]
    fn loads_and_projects_synthetic_safetensors() {
        let fixture = Fixture::new(&[2, 3], &[1.0, 2.0, -1.0, 0.5, -3.0, 4.0], &[0.25, -2.0]);
        let projection = Lm48Projection::from_safetensors(fixture.0.path(), "proj").unwrap();

        assert_eq!(projection.in_dim, 3);
        assert_eq!(projection.out_dim, 2);
        assert_eq!(projection.project(&[2.0, -1.0, 0.5]), [-0.25, 4.0]);
    }

    #[test]
    fn transposes_input_major_synthetic_weight() {
        let fixture = Fixture::new(&[3, 2], &[1.0, 0.5, 2.0, -3.0, -1.0, 4.0], &[0.25, -2.0]);
        let projection = Lm48Projection::from_safetensors(fixture.0.path(), "proj").unwrap();

        assert_eq!(projection.project(&[2.0, -1.0, 0.5]), [-0.25, 4.0]);
    }

    #[test]
    #[ignore = "needs RES-09 real golden fixture; do not invent expected values"]
    fn matches_t087_lm48_golden() {
        panic!("RES-09 must provide the real T087 inputs and expected lm48 outputs");
    }
}
