//! Frozen high-level DOPE and Stage-1B graph on one explicit Candle device.
use super::super::{
    adapter::{CpuAdapter, PredictCostsInputs, PredictedCosts},
    bundle::{check_tensor_header, BundleLimits, VerifiedBundle},
};
use super::cpu;
use anyhow::{ensure, Context, Result};
use candle_core_0_11_0::{Device, Tensor};
use jope_api::{error::ErrorCode, Placement};
use jope_runtime::backend::DeviceRequest;
use safetensors::{Dtype, SafeTensors};
use std::collections::BTreeMap;

pub struct StageOutput {
    pub v256: Vec<f32>,
    pub col: Vec<f32>,
    pub dope629: Vec<f32>,
    pub lm48: Vec<f32>,
    pub lm48_z: Vec<f32>,
    pub tokens: Vec<f32>,
    pub mask: Vec<f32>,
}

pub struct FusedForward {
    device: Device,
    placement: Placement,
    weights: BTreeMap<String, Tensor>,
    adapter: CpuAdapter,
}

fn select(request: DeviceRequest) -> std::result::Result<Device, ErrorCode> {
    match request {
        DeviceRequest::Cpu | DeviceRequest::Auto => Ok(Device::Cpu),
        DeviceRequest::Cuda(index) => {
            #[cfg(feature = "fused-cuda")]
            {
                // Dynamic driver loading can panic before Candle returns an
                // error. An unavailable explicit backend never becomes CPU.
                std::panic::catch_unwind(|| Device::new_cuda(index as usize))
                    .ok()
                    .and_then(std::result::Result::ok)
                    .ok_or(ErrorCode::BackendUnavailable)
            }
            #[cfg(not(feature = "fused-cuda"))]
            {
                let _ = index;
                Err(ErrorCode::BackendUnavailable)
            }
        }
    }
}

fn selected_device(request: DeviceRequest) -> Result<Device> {
    select(request).map_err(|code| {
        jope_runtime::Failure::backend(format!(
            "{code}: {} requested for fused ranker, but that backend is unavailable",
            request.as_string()
        ))
    })
}

impl FusedForward {
    /// Only the existing authenticated release boundary supplies production bytes.
    /// CUDA callers hold the qualification/runtime lease for the entire operation.
    pub fn from_bundle(
        bundle: &VerifiedBundle,
        request: DeviceRequest,
        workers: u32,
    ) -> Result<Self> {
        ensure!(workers > 0, "workers must be positive");
        let device = selected_device(request)?;
        Self::from_bytes(
            bundle
                .asset("dope.safetensors")
                .context("missing authenticated DOPE")?,
            bundle
                .asset("jope.safetensors")
                .context("missing authenticated ranker")?,
            device,
            workers,
        )
    }

    pub(super) fn from_bytes(
        dope: &[u8],
        jope: &[u8],
        device: Device,
        workers: u32,
    ) -> Result<Self> {
        ensure!(workers > 0, "workers must be positive");
        let mut weights = BTreeMap::new();
        for bytes in [dope, jope] {
            check_tensor_header(bytes, BundleLimits::default())?;
            let tensors = SafeTensors::deserialize(bytes)?;
            for name in tensors.names() {
                // Only inference tensors used by this graph, not unused target heads/bank.
                if !(name.starts_with("dope_model.router.")
                    || name.starts_with("dope_model.target_net.col_gate_head.")
                    || !name.contains('.'))
                {
                    continue;
                }
                let t = tensors.tensor(name)?;
                if t.dtype() != Dtype::F32 {
                    continue;
                }
                let values: Vec<f32> = t
                    .data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().expect("F32")))
                    .collect();
                ensure!(
                    values.iter().all(|v| v.is_finite()),
                    "nonfinite parameter {name}"
                );
                ensure!(!weights.contains_key(name), "duplicate parameter {name}");
                weights.insert(
                    name.to_owned(),
                    Tensor::from_vec(values, t.shape(), &device)?,
                );
            }
        }
        let selected = match device.location() {
            candle_core_0_11_0::DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
            _ => "cpu".into(),
        };
        let adapter = CpuAdapter::from_bytes_on_device(jope, &device)?;
        Ok(Self {
            device,
            placement: Placement {
                dope: selected.clone(),
                ranker: selected,
                evaluation: "cpu".into(),
                workers,
            },
            weights,
            adapter,
        })
    }

    #[cfg(test)]
    pub(super) fn device_sync_for_test(&self) {
        self.device.synchronize().unwrap();
    }

    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    fn p(&self, name: &str) -> Result<&Tensor> {
        self.weights
            .get(name)
            .with_context(|| format!("missing frozen tensor {name}"))
    }
    fn tensor(
        &self,
        values: Vec<f32>,
        shape: impl Into<candle_core_0_11_0::Shape>,
    ) -> Result<Tensor> {
        Ok(Tensor::from_vec(values, shape, &self.device)?)
    }
    fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        cpu::linear(
            x,
            self.p(&format!("{name}.weight"))?,
            self.p(&format!("{name}.bias"))?,
        )
    }
    fn norm(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        cpu::norm(
            x,
            self.p(&format!("{name}.weight"))?,
            self.p(&format!("{name}.bias"))?,
        )
    }
    fn standard(&self, x: &Tensor, mean: &str, std: &str, bound: f64) -> Result<Tensor> {
        let scale = self.p(std)?;
        ensure!(
            flat(scale)?.iter().all(|v| *v > 0.),
            "invalid frozen standard deviation {std}"
        );
        Ok(x.broadcast_sub(self.p(mean)?)?
            .broadcast_div(scale)?
            .clamp(-bound, bound)?)
    }
    fn slot(&self, x: &Tensor, slot: &str) -> Result<Tensor> {
        Ok(self
            .standard(x, &format!("{slot}_mu"), &format!("{slot}_sd_safe"), 8.)?
            .broadcast_mul(self.p(&format!("{slot}_active_mask"))?)?
            .matmul(self.p(&format!("{slot}_effective_proj_mat"))?)?
            .clamp(-6., 6.)?)
    }

    /// One context: exact shapes reject implicit broadcasting across identities.
    pub fn stages(&self, v256: &[f32], col: &[f32]) -> Result<StageOutput> {
        ensure!(
            v256.len() == 256 && col.len() == 64 * 17,
            "exact v256/column dimensions required"
        );
        ensure!(
            v256.iter().chain(col).all(|v| v.is_finite()),
            "nonfinite stage input"
        );
        let v = self.tensor(v256.to_vec(), (1, 256))?;
        let c = self.tensor(col.to_vec(), (64, 17))?;
        let r = "dope_model.router";
        let x140 = self.norm(&v.narrow(1, 0, 140)?, &format!("{r}.input_norm"))?;
        let extra = self.norm(&v.narrow(1, 140, 116)?, &format!("{r}.v3_inter_norm"))?;
        let x = Tensor::cat(&[&x140, &extra], 1)?;
        let mut h = self.norm(
            &cpu::silu(&self.linear(&x140, &format!("{r}.backbone.0"))?)?,
            &format!("{r}.backbone.2"),
        )?;
        h = self.norm(
            &cpu::silu(&self.linear(&h, &format!("{r}.backbone.3"))?)?,
            &format!("{r}.backbone.5"),
        )?;
        // _LEVEL_INDEX['high'] ==1, not ultra==2; small_embed is absent at high.
        h = h.broadcast_add(
            &self
                .p(&format!("{r}.level_embed.weight"))?
                .narrow(0, 1, 1)?,
        )?;
        let a = format!("{r}.inter_attn");
        let tok = cpu::silu(&self.linear(&x.reshape((8, 32))?, &format!("{a}.token_proj"))?)?;
        let qkv = self.linear(&tok, &format!("{a}.qkv"))?;
        let logits = cpu::scale_attention(&cpu::matmul(
            &qkv.narrow(1, 0, 32)?,
            &qkv.narrow(1, 32, 32)?.t()?.contiguous()?,
        )?)?;
        let probs = cpu::softmax(&logits)?;
        let attended = (cpu::matmul(&probs, &qkv.narrow(1, 64, 32)?)? + &tok)?.reshape((1, 256))?;
        let bilin = (cpu::tanh(&self.linear(&x, &format!("{a}.bilinear_u"))?)?
            * cpu::silu(&self.linear(&x, &format!("{a}.bilinear_v"))?)?)?;
        let inter = self.norm(
            &self.linear(
                &Tensor::cat(&[&attended, &bilin], 1)?,
                &format!("{a}.out_proj"),
            )?,
            &format!("{a}.norm"),
        )?;
        h = (h + inter)?;
        let raw = self.linear(&h, &format!("{r}.gate_head"))?;
        let raw_v3 = self.linear(&h, &format!("{r}.v3_gate_head"))?;
        let sig = |x: Tensor| -> Result<Tensor> { Ok(x.neg()?.exp()?.affine(1., 1.)?.recip()?) };
        let g = |i: usize, scale: f64| -> Result<Tensor> {
            Ok(sig(raw.narrow(1, i, 1)?)?.affine(scale, 0.)?)
        };
        let v3 = |i: usize, scale: f64| -> Result<Tensor> {
            Ok(sig(raw_v3.narrow(1, i, 1)?)?.affine(scale, 0.)?)
        };
        // Frozen STEP3_GATE_KEYS_14 order (alphabetical, not gate-head order).
        let ridge_raw = raw.narrow(1, 0, 1)?;
        let ridge = (ridge_raw.relu()? + ridge_raw.abs()?.neg()?.exp()?.affine(1., 1.)?.log()?)?
            .affine(1., 0.15)?;
        let gates = Tensor::cat(
            &[
                v3(0, 0.55)?,
                g(7, 0.9)?,
                v3(3, 0.4)?,
                g(2, 0.55)?,
                g(1, 0.65)?,
                g(6, 0.65)?,
                g(9, 0.2)?,
                g(3, 0.6)?,
                v3(2, 0.45)?,
                g(8, 0.75)?,
                g(4, 0.45)?,
                ridge,
                v3(1, 0.5)?,
                g(5, 0.7)?,
            ],
            1,
        )?;
        let task = cpu::tanh(&self.linear(&h, &format!("{r}.task_latent_head"))?)?;
        let dist = cpu::tanh(&self.linear(&h, &format!("{r}.dist_latent_head"))?)?;
        let lm = cpu::tanh(&self.linear(&h, &format!("{r}.landmark_recon_head"))?)?;
        let bb = cpu::tanh(&self.linear(&h, &format!("{r}.bb_recon_head"))?)?;
        let col_gates = cpu::tanh(&self.linear(
            &Tensor::cat(&[&task.broadcast_as((64, 128))?, &c], 1)?,
            "dope_model.target_net.col_gate_head",
        )?)?
        .affine(0.25, 1.)?
        .reshape((1, 64))?;
        let dope = Tensor::cat(&[&v, &task, &dist, &gates, &lm, &bb, &col_gates], 1)?;
        let lm48 = self
            .standard(&v, "step2_lm_mean_1x256", "step2_lm_std_1x256", 6.)?
            .matmul(self.p("step2_svd_basis_256x48")?)?;
        let lm48_z = self.standard(&lm48, "mu_lm48", "sd_lm48", 6.)?;
        let lm80 = self.lm80(&lm48)?;
        let m0 = Tensor::cat(&[&self.tensor(vec![0.; 104], (1, 104))?, &lm80], 1)?;
        let m1 = Tensor::cat(
            &[
                &lm48_z.narrow(1, 36, 12)?,
                &self.tensor(vec![0.; 20], (1, 20))?,
                &lm80.narrow(1, 48, 16)?,
            ],
            1,
        )?;
        let t0 = self.slot(&m0, "svd_slot0_m0")?;
        let t2 = self.slot(&dope, "svd_slot2_m2")?;
        let kron = t2
            .narrow(1, 0, 8)?
            .unsqueeze(2)?
            .broadcast_mul(&t0.narrow(1, 0, 8)?.unsqueeze(1)?)?
            .reshape((1, 64))?;
        let zero = self.tensor(vec![0.; 32], (1, 32))?;
        let tokens = Tensor::stack(
            &[
                &t0,
                &self.slot(&m1, "svd_slot1_m1")?,
                &t2,
                &zero,
                &self.slot(&dope, "svd_slot4_d629_mid")?,
                &self.slot(&dope, "svd_slot5_d629_tail")?,
                &zero,
                &self.slot(&m0, "svd_slot7_lm_fine")?,
                &zero,
                &self.slot(&kron, "svd_slot9_kron")?,
            ],
            1,
        )?;
        let output = StageOutput {
            v256: flat(&v)?,
            col: flat(&c)?,
            dope629: flat(&dope)?,
            lm48: flat(&lm48)?,
            lm48_z: flat(&lm48_z)?,
            tokens: flat(&tokens)?,
            mask: flat(self.p("mask_10_base")?)?,
        };
        ensure!(
            output
                .dope629
                .iter()
                .chain(&output.lm48)
                .chain(&output.lm48_z)
                .chain(&output.tokens)
                .all(|v| v.is_finite()),
            "nonfinite frozen forward output"
        );
        ensure!(
            output.mask.len() == 10 && output.mask == [1., 1., 1., 0., 1., 1., 0., 1., 0., 1.],
            "frozen mask differs"
        );
        Ok(output)
    }

    fn lm80(&self, lm48: &Tensor) -> Result<Tensor> {
        // The 16-element order statistics are explicit scalar preprocessing.
        let raw = flat(lm48)?;
        let mut sorted: Vec<f32> = raw.chunks_exact(3).map(|r| r[0]).collect();
        sorted.sort_by(f32::total_cmp);
        let q25 = 0.25 * sorted[3] + 0.75 * sorted[4];
        let median = 0.5 * sorted[7] + 0.5 * sorted[8];
        let q75 = 0.75 * sorted[11] + 0.25 * sorted[12];
        let iqr = (q75 - q25).max(0.05);
        let gaps: Vec<f32> = raw
            .chunks_exact(3)
            .map(|r| ((r[2] - r[0]) / (r[0].abs() + r[1].abs() + 0.05)).clamp(-6., 6.))
            .collect();
        let mut values = raw.clone();
        values.extend(
            raw.chunks_exact(3)
                .map(|r| ((r[0] - median) / iqr).clamp(-6., 6.)),
        );
        values.extend(gaps.chunks_exact(2).map(|g| 0.5 * (g[0] + g[1])));
        let mean_std = raw.chunks_exact(3).map(|r| r[1].abs()).sum::<f32>() / 16.;
        values.extend([
            sorted[0],
            q25,
            median,
            q75,
            sorted[15],
            iqr,
            sorted[0] - median,
            sorted[2] - sorted[0] + 0.1 * mean_std,
        ]);
        self.standard(&self.tensor(values, (1, 80))?, "lm80_mu", "lm80_sd", 8.)
    }

    pub fn predict_costs(&self, inputs: PredictCostsInputs<'_>) -> Result<PredictedCosts> {
        self.adapter.predict_costs(inputs)
    }
}

fn flat(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.flatten_all()?.to_vec1()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "fused-cuda"))]
    #[test]
    fn explicit_cuda_request_preserves_backend_error_code() {
        let request = DeviceRequest::parse("cuda:0").unwrap();
        let error = selected_device(request).unwrap_err();
        assert_eq!(
            jope_runtime::error::kind_of(&error),
            jope_runtime::FailureKind::Backend
        );
        assert_eq!(jope_runtime::error::kind_of(&error).exit_code(), 4);
        assert!(error.to_string().starts_with("BACKEND_UNAVAILABLE:"));
        assert!(error.to_string().contains("cuda:0"));
    }

    #[test]
    fn default_backend_is_explicit_and_cuda_never_falls_back() {
        assert!(select(DeviceRequest::Auto).unwrap().is_cpu());
        assert!(select(DeviceRequest::Cpu).unwrap().is_cpu());
        #[cfg(not(feature = "fused-cuda"))]
        {
            assert!(matches!(
                select(DeviceRequest::Cuda(0)),
                Err(ErrorCode::BackendUnavailable)
            ));
            assert_eq!(
                jope_runtime::error::kind_of(&selected_device(DeviceRequest::Cuda(0)).unwrap_err()),
                jope_runtime::FailureKind::Backend
            );
        }
        if std::env::var_os("JOPE_ASSERT_CUDA_UNAVAILABLE").is_some() {
            assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok(""));
            assert!(matches!(
                select(DeviceRequest::Cuda(0)),
                Err(ErrorCode::BackendUnavailable)
            ));
            assert_eq!(
                jope_runtime::error::kind_of(&selected_device(DeviceRequest::Cuda(0)).unwrap_err()),
                jope_runtime::FailureKind::Backend
            );
            println!(
                "CUDA_UNAVAILABLE_CONTROL compiled_fused_cuda={}",
                cfg!(feature = "fused-cuda")
            );
        }
    }
}
