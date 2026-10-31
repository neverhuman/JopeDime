//! Linear `lm48` projection loaded from a fused-ranker safetensors file.
//!
//! The affine projection supports stored dense layers; [`Lm48Transform`]
//! implements the complete T087 mean/std, SVD-basis and output-standardization
//! formula using the five named tensors in `jope.safetensors`.

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

pub const V256_DIM: usize = 256;
pub const LM48_DIM: usize = 48;

/// The raw and standardized landmark vectors produced by the frozen bundle.
#[derive(Clone, Debug, PartialEq)]
pub struct Lm48Features {
    pub lm48: [f32; LM48_DIM],
    pub lm48_z: [f32; LM48_DIM],
}

/// Frozen parameters for the T087 `v256` → `lm48` transform.
///
/// The SVD basis is row-major `[256, 48]`, matching the specified right-side
/// matrix multiplication. All five tensors come from `jope.safetensors`.
#[derive(Clone, Debug)]
pub struct Lm48Transform {
    step2_lm_mean: [f32; V256_DIM],
    step2_lm_std: [f32; V256_DIM],
    step2_svd_basis_48: Vec<f32>,
    mu_lm48: [f32; LM48_DIM],
    sd_lm48: [f32; LM48_DIM],
}

impl Lm48Transform {
    /// Load the five T087 tensors from `jope.safetensors`.
    pub fn from_safetensors(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("read lm48 parameters from {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&bytes).context("parse lm48 parameters")?;
        let step2_lm_mean: [f32; V256_DIM] =
            read_parameter(&tensors, "step2_lm_mean", &[V256_DIM])?
                .try_into()
                .expect("validated lm48 mean width");
        let step2_lm_std: [f32; V256_DIM] = read_parameter(&tensors, "step2_lm_std", &[V256_DIM])?
            .try_into()
            .expect("validated lm48 std width");
        let step2_svd_basis_48 =
            read_parameter(&tensors, "step2_svd_basis_48", &[V256_DIM, LM48_DIM])?;
        let mu_lm48: [f32; LM48_DIM] = read_parameter(&tensors, "mu_lm48", &[LM48_DIM])?
            .try_into()
            .expect("validated lm48 center width");
        let sd_lm48: [f32; LM48_DIM] = read_parameter(&tensors, "sd_lm48", &[LM48_DIM])?
            .try_into()
            .expect("validated lm48 scale width");

        if step2_lm_std.iter().any(|value| *value <= 0.0) {
            bail!("step2_lm_std values must be positive");
        }
        if sd_lm48.iter().any(|value| *value <= 0.0) {
            bail!("sd_lm48 values must be positive");
        }

        Ok(Self {
            step2_lm_mean,
            step2_lm_std,
            step2_svd_basis_48,
            mu_lm48,
            sd_lm48,
        })
    }

    /// Apply the frozen clamp, SVD projection and final standardization.
    pub fn project(&self, v256: &[f32]) -> Result<Lm48Features> {
        if v256.len() != V256_DIM {
            bail!("lm48 input must have {V256_DIM} values, got {}", v256.len());
        }
        if v256.iter().any(|value| !value.is_finite()) {
            bail!("lm48 input must contain only finite values");
        }

        let mut lm48 = [0.0; LM48_DIM];
        for (input, value) in v256.iter().enumerate() {
            let standardized =
                ((*value - self.step2_lm_mean[input]) / self.step2_lm_std[input]).clamp(-6.0, 6.0);
            let basis_row = &self.step2_svd_basis_48[input * LM48_DIM..(input + 1) * LM48_DIM];
            for (output, component) in lm48.iter_mut().zip(basis_row) {
                *output += standardized * component;
            }
        }
        if lm48.iter().any(|value| !value.is_finite()) {
            bail!("lm48 projection produced a non-finite value");
        }

        let lm48_z = std::array::from_fn(|index| {
            ((lm48[index] - self.mu_lm48[index]) / self.sd_lm48[index]).clamp(-6.0, 6.0)
        });
        if lm48_z.iter().any(|value| !value.is_finite()) {
            bail!("standardized lm48 projection produced a non-finite value");
        }

        Ok(Lm48Features { lm48, lm48_z })
    }
}

fn read_parameter(tensors: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("missing lm48 tensor {name}"))?;
    if tensor.dtype() != Dtype::F32 {
        bail!("lm48 tensor {name} must use F32 dtype");
    }
    if tensor.shape() != shape {
        bail!(
            "lm48 tensor {name} has shape {:?}; expected {shape:?}",
            tensor.shape()
        );
    }
    let values = read_f32(tensor.data())?;
    let expected_len = shape.iter().product::<usize>();
    if values.len() != expected_len {
        bail!("lm48 tensor {name} has an invalid data length");
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("lm48 tensor {name} must contain only finite values");
    }
    Ok(values)
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

    use super::{Lm48Projection, Lm48Transform, LM48_DIM, V256_DIM};

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

    fn transform_fixture(first_output_std: f32) -> tempfile::NamedTempFile {
        let mean = vec![0.0; V256_DIM];
        let input_std = vec![1.0; V256_DIM];
        let mut basis = vec![0.0; V256_DIM * LM48_DIM];
        for diagonal in 0..3 {
            basis[diagonal * LM48_DIM + diagonal] = 1.0;
        }
        let center = vec![1.0; LM48_DIM];
        let mut output_std = vec![2.0; LM48_DIM];
        output_std[0] = first_output_std;
        let tensors = [
            ("step2_lm_mean", vec![V256_DIM], mean),
            ("step2_lm_std", vec![V256_DIM], input_std),
            ("step2_svd_basis_48", vec![V256_DIM, LM48_DIM], basis),
            ("mu_lm48", vec![LM48_DIM], center),
            ("sd_lm48", vec![LM48_DIM], output_std),
        ];

        let mut entries = Vec::new();
        let mut offset = 0;
        for (name, shape, values) in &tensors {
            let shape = shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let end = offset + values.len() * std::mem::size_of::<f32>();
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape}],\"data_offsets\":[{offset},{end}]}}"
            ));
            offset = end;
        }
        let mut header = format!("{{{}}}", entries.join(","));
        while header.len() % 8 != 0 {
            header.push(' ');
        }

        let mut file = tempfile::NamedTempFile::new().expect("create lm48 transform fixture");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("write safetensors header length");
        file.write_all(header.as_bytes())
            .expect("write safetensors header");
        for (_, _, values) in &tensors {
            for value in values {
                file.write_all(&value.to_le_bytes())
                    .expect("write transform parameter");
            }
        }
        file
    }

    #[test]
    fn applies_t087_clamped_svd_projection_and_lm48_standardization() {
        let fixture = transform_fixture(2.0);
        let transform = Lm48Transform::from_safetensors(fixture.path()).unwrap();
        let mut v256 = vec![0.0; V256_DIM];
        v256[0] = 9.0;
        v256[1] = -9.0;
        v256[2] = 3.0;

        let features = transform.project(&v256).unwrap();

        assert_eq!(&features.lm48[..3], &[6.0, -6.0, 3.0]);
        assert_eq!(&features.lm48_z[..3], &[2.5, -3.5, 1.0]);
        assert!(features.lm48[3..].iter().all(|value| *value == 0.0));
        assert!(features.lm48_z[3..].iter().all(|value| *value == -0.5));
    }

    #[test]
    fn refuses_invalid_lm48_standard_deviation_and_input_shape() {
        let fixture = transform_fixture(0.0);
        assert!(Lm48Transform::from_safetensors(fixture.path()).is_err());

        let fixture = transform_fixture(2.0);
        let transform = Lm48Transform::from_safetensors(fixture.path()).unwrap();
        assert!(transform.project(&[0.0; V256_DIM - 1]).is_err());
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
    #[ignore = "needs RES-07 real golden fixtures; do not invent expected values"]
    fn matches_t087_lm48_golden() {
        panic!("RES-07 must publish real T087 inputs and expected lm48 outputs");
    }
}
