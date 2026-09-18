use super::support::{interleave_rows, load_safetensors, uncertainty_tensor};
use super::{
    RegressionExpertFeatures, CONF_DIM, GATE_META_DIM, HIDDEN_DIM, N_EXPERTS, QUANTILE_DIM,
    STACK_INPUT_DIM,
};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use std::collections::HashMap;
use std::path::Path;

pub(super) struct StackFusionReg {
    weights: HashMap<String, Tensor>,
    device: Device,
}

impl StackFusionReg {
    pub(super) fn load(path: &Path, device: Device) -> Result<Self> {
        let weights = load_safetensors(path, &device)?;
        Ok(Self { weights, device })
    }

    pub(super) fn forward(
        &self,
        a: &RegressionExpertFeatures,
        b: &RegressionExpertFeatures,
        gate_meta: &[f32; GATE_META_DIM],
    ) -> Result<Vec<f64>> {
        let m = a.n_rows;
        if m == 0 {
            return Ok(Vec::new());
        }
        let mean0 = Tensor::from_vec(a.mean.clone(), (m, 1), &self.device)?;
        let mean1 = Tensor::from_vec(b.mean.clone(), (m, 1), &self.device)?;
        let cal0 = self.calibrate(&mean0, 0)?;
        let cal1 = self.calibrate(&mean1, 1)?;
        let cal = Tensor::cat(&[&cal0, &cal1], 1)?;

        let h0 = Tensor::from_vec(a.hidden.clone(), (m, HIDDEN_DIM), &self.device)?;
        let h1 = Tensor::from_vec(b.hidden.clone(), (m, HIDDEN_DIM), &self.device)?;
        let z0 = self.adapter(&h0, 0)?;
        let z1 = self.adapter(&h1, 1)?;
        let z_flat = Tensor::cat(&[&z0, &z1], 1)?;

        let quant = interleave_rows(&a.quantiles, &b.quantiles, QUANTILE_DIM);
        let quant = Tensor::from_vec(quant, (m, N_EXPERTS * QUANTILE_DIM), &self.device)?;
        let conf = interleave_rows(&a.conf, &b.conf, CONF_DIM);
        let conf_t = Tensor::from_vec(conf, (m, N_EXPERTS, CONF_DIM), &self.device)?;
        let conf_flat = conf_t.reshape((m, N_EXPERTS * CONF_DIM))?;
        let unc = uncertainty_tensor(a, b, &self.device)?;
        let prec = ((&unc * &unc)? + 1e-8)?.recip()?;
        let weights = prec.broadcast_div(&prec.sum_keepdim(1)?)?;
        let base = weights.broadcast_mul(&cal)?.sum(1)?;

        let mut gate = Vec::with_capacity(m * GATE_META_DIM);
        for _ in 0..m {
            gate.extend_from_slice(gate_meta);
        }
        let gate = Tensor::from_vec(gate, (m, GATE_META_DIM), &self.device)?;
        let z_prod = z0.broadcast_mul(&z1)?;
        let z_diff = (&z0 - &z1)?;
        let cal_diff = (&cal0 - &cal1)?.abs()?;
        let feat = Tensor::cat(
            &[
                &z_flat, &cal, &quant, &conf_flat, &gate, &unc, &z_prod, &z_diff, &cal_diff,
            ],
            1,
        )?;
        if feat.dims() != [m, STACK_INPUT_DIM] {
            bail!("invalid StackFusionReg feature width: {:?}", feat.dims());
        }
        let delta = self.head(&feat)?.reshape((m,))?;
        let final_pred = (&base + &delta)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        Ok(final_pred.into_iter().map(f64::from).collect())
    }

    fn adapter(&self, h: &Tensor, expert: usize) -> Result<Tensor> {
        let prefix = format!("adapters.{expert}");
        let z = self.layer_norm(h, &format!("{prefix}.pre"))?;
        let z = self
            .linear(&z, &format!("{prefix}.down.weight"), None)?
            .gelu_erf()?;
        let z = self.linear(&z, &format!("{prefix}.up.weight"), None)?;
        let z = self.layer_norm(&z, &format!("{prefix}.post"))?;
        Ok(z.broadcast_mul(self.w(&format!("{prefix}.gain"))?)?)
    }

    fn calibrate(&self, mean: &Tensor, expert: usize) -> Result<Tensor> {
        Ok(mean
            .broadcast_mul(self.w(&format!("cal.{expert}.scale"))?)?
            .broadcast_add(self.w(&format!("cal.{expert}.bias"))?)?)
    }

    fn head(&self, feat: &Tensor) -> Result<Tensor> {
        let mut x = self
            .linear(feat, "head.0.weight", Some("head.0.bias"))?
            .gelu_erf()?;
        x = self.layer_norm(&x, "head.2")?;
        x = self
            .linear(&x, "head.4.weight", Some("head.4.bias"))?
            .gelu_erf()?;
        x = self.layer_norm(&x, "head.6")?;
        x = self
            .linear(&x, "head.8.weight", Some("head.8.bias"))?
            .gelu_erf()?;
        x = self.layer_norm(&x, "head.10")?;
        self.linear(&x, "head.12.weight", Some("head.12.bias"))
    }

    fn layer_norm(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let denom = (var + 1e-5)?.sqrt()?;
        let xn = xc.broadcast_div(&denom)?;
        Ok(xn
            .broadcast_mul(self.w(&format!("{prefix}.weight"))?)?
            .broadcast_add(self.w(&format!("{prefix}.bias"))?)?)
    }

    fn linear(&self, x: &Tensor, weight: &str, bias: Option<&str>) -> Result<Tensor> {
        let w = self.w(weight)?;
        let dims = x.dims().to_vec();
        let in_dim = *dims.last().context("linear input has no dims")?;
        let out_dim = w.dims()[0];
        let rows = x.elem_count() / in_dim;
        let flat = x.reshape((rows, in_dim))?;
        let mut y = flat.matmul(&w.t()?)?;
        if let Some(bias) = bias {
            y = y.broadcast_add(&self.w(bias)?.reshape((1, out_dim))?)?;
        }
        let mut out_shape = dims;
        *out_shape.last_mut().unwrap() = out_dim;
        Ok(y.reshape(out_shape)?)
    }

    fn w(&self, name: &str) -> Result<&Tensor> {
        self.weights
            .get(name)
            .with_context(|| format!("missing Chimera tensor {name}"))
    }
}
