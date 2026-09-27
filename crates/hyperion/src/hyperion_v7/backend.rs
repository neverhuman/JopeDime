//! The candle-backed [`PatchModel`] — loads the full v7 checkpoint (embed trunk + confidence head)
//! and serves batched patch forwards on CPU or CUDA.

use std::path::Path;

#[cfg(feature = "hyperion-cuda")]
use crate::battle_gpu;
#[cfg(feature = "hyperion-cuda")]
use anyhow::anyhow;
use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};

use crate::hyperion::HyperionSweepSettings;

use super::mha::MhaWeights;
use super::model::{BlockWeights, HgeWeights, MguWeights};
use super::ops::LinearWeights;
use super::pooling::MpsWeights;
use super::sweep::{PatchModel, PatchOutputs};
use super::weights::{load_tensor_map, tensor_map_from_bytes};
use crate::hyperion::{hyperion_device, hyperion_device_candidates, hyperion_device_label_for};

/// Rough activation-memory cap per forward: the confidence head materializes ~22.5 KB/cell of
/// intermediates at dim 512, so ~24k cells ≈ 0.5 GB. Larger requests are sub-batched internally.
const MAX_FORWARD_CELLS: usize = 24_000;
const MAX_EMB_FORWARD_CELLS: usize = 900_000;

#[cfg(feature = "hyperion-cuda")]
type GpuQueueLease = battle_gpu::queue::GpuLease;

pub fn hyperion_v7_device() -> Device {
    hyperion_device()
}

pub struct V7Model {
    weights: MguWeights,
    device: Device,
    #[cfg(feature = "hyperion-cuda")]
    _gpu_queue_lease: Option<GpuQueueLease>,
    min_patch: usize,
    scale_rows: usize,
    /// Escape hatch (`JAIN_HYPERION_FULL_CANVAS=1`): pad every patch onto the reference's
    /// max_patch canvas (mask 0 outside the window) instead of the cropped p×p canvas.
    full_canvas: bool,
    pad_norm: f32,
}

impl V7Model {
    /// Explicit CPU execution for separately qualified context production.
    /// Does not inspect device environment or fall back from a GPU attempt.
    pub fn load_cpu(weights_path: &Path, settings: &HyperionSweepSettings) -> Result<Self> {
        Self::load_on_device(
            weights_path,
            settings,
            Device::Cpu,
            #[cfg(feature = "hyperion-cuda")]
            None,
        )
        .map(|model| model.with_full_canvas(false))
    }

    /// Explicit CUDA placement with the same canvas settings as CPU qualification.
    #[cfg(feature = "hyperion-cuda")]
    pub fn load_cuda(
        weights_path: &Path,
        settings: &HyperionSweepSettings,
        device: Device,
    ) -> Result<Self> {
        anyhow::ensure!(device.is_cuda(), "Hyperion requires CUDA");
        Self::load_on_device(weights_path, settings, device, None)
            .map(|model| model.with_full_canvas(false))
    }

    /// [`Self::load_cpu`] from weights bytes the caller has already authenticated. The model is
    /// built from exactly that buffer and no file is reopened, so a replacement of the weights
    /// file after authentication cannot reach it.
    pub fn load_cpu_from_bytes(weights: &[u8], settings: &HyperionSweepSettings) -> Result<Self> {
        Self::from_tensor_map(
            tensor_map_from_bytes(weights, &Device::Cpu)?,
            settings,
            Device::Cpu,
            #[cfg(feature = "hyperion-cuda")]
            None,
        )
        .map(|model| model.with_full_canvas(false))
    }

    /// [`Self::load_cuda`] from weights bytes the caller has already authenticated.
    #[cfg(feature = "hyperion-cuda")]
    pub fn load_cuda_from_bytes(
        weights: &[u8],
        settings: &HyperionSweepSettings,
        device: Device,
    ) -> Result<Self> {
        anyhow::ensure!(device.is_cuda(), "Hyperion requires CUDA");
        Self::from_tensor_map(
            tensor_map_from_bytes(weights, &device)?,
            settings,
            device,
            None,
        )
        .map(|model| model.with_full_canvas(false))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn load(weights_path: &Path, settings: &HyperionSweepSettings) -> Result<Self> {
        #[cfg(feature = "hyperion-cuda")]
        let mut cuda_error: Option<anyhow::Error> = None;
        // Held rather than returned, so a candidate that fails lets the next one be tried. The
        // `continue` below is behind `hyperion-cuda`, so without that feature every arm of the
        // match used to return and the loop ran exactly once: the fallback across candidates did
        // not exist on a CPU-only build, and the `bail!` after the loop was unreachable. Clippy's
        // `never_loop` caught it.
        let mut last_error: Option<anyhow::Error> = None;
        for device in hyperion_device_candidates() {
            #[cfg(feature = "hyperion-cuda")]
            let label = hyperion_device_label_for(&device);
            #[cfg(feature = "hyperion-cuda")]
            let gpu_queue_lease = if label == "cpu" {
                None
            } else {
                match battle_gpu::queue::acquire_default_gpu_lease() {
                    Ok(lease) => Some(lease),
                    Err(err) => {
                        let reason = battle_gpu::queue::gpu_cpu_handoff_reason(
                            "Hyperion CUDA queue admission",
                            format!("{err:#}"),
                        );
                        eprintln!("[hyperion] {reason}");
                        cuda_error = Some(anyhow!(reason));
                        continue;
                    }
                }
            };
            match Self::load_on_device(
                weights_path,
                settings,
                device,
                #[cfg(feature = "hyperion-cuda")]
                gpu_queue_lease,
            ) {
                Ok(model) => return Ok(model),
                #[cfg(feature = "hyperion-cuda")]
                Err(err) if label != "cpu" => {
                    let reason = battle_gpu::queue::gpu_cpu_handoff_reason(
                        "Hyperion CUDA model load",
                        format!("{err:#}"),
                    );
                    eprintln!("[hyperion] {reason}");
                    cuda_error = Some(anyhow!(reason));
                }
                Err(err) => {
                    last_error = Some(err);
                }
            }
        }
        if let Some(err) = last_error {
            #[cfg(feature = "hyperion-cuda")]
            if let Some(cuda_error) = cuda_error {
                return Err(err)
                    .with_context(|| format!("Hyperion CUDA load failed first: {cuda_error:#}"));
            }
            return Err(err);
        }
        bail!("no Hyperion device candidates available")
    }

    fn load_on_device(
        weights_path: &Path,
        settings: &HyperionSweepSettings,
        device: Device,
        #[cfg(feature = "hyperion-cuda")] gpu_queue_lease: Option<GpuQueueLease>,
    ) -> Result<Self> {
        Self::from_tensor_map(
            load_tensor_map(weights_path, &device)?,
            settings,
            device,
            #[cfg(feature = "hyperion-cuda")]
            gpu_queue_lease,
        )
    }

    fn from_tensor_map(
        tensors: std::collections::HashMap<String, Tensor>,
        settings: &HyperionSweepSettings,
        device: Device,
        #[cfg(feature = "hyperion-cuda")] gpu_queue_lease: Option<GpuQueueLease>,
    ) -> Result<Self> {
        let get = |k: &str| -> Result<Tensor> {
            tensors
                .get(k)
                .cloned()
                .with_context(|| format!("missing v7 tensor {k}"))
        };
        let lin = |prefix: &str| -> Result<LinearWeights> {
            Ok(LinearWeights {
                weight: get(&format!("{prefix}.weight"))?,
                bias: get(&format!("{prefix}.bias"))?,
            })
        };
        let mps = |prefix: &str| -> Result<MpsWeights> {
            Ok(MpsWeights {
                proj_0: lin(&format!("{prefix}.proj.0"))?,
                proj_2: lin(&format!("{prefix}.proj.2"))?,
            })
        };
        let mha = |prefix: &str| -> Result<MhaWeights> {
            Ok(MhaWeights {
                in_proj_weight: get(&format!("{prefix}.in_proj_weight"))?,
                in_proj_bias: get(&format!("{prefix}.in_proj_bias"))?,
                out_proj_weight: get(&format!("{prefix}.out_proj.weight"))?,
                out_proj_bias: get(&format!("{prefix}.out_proj.bias"))?,
            })
        };

        let mut blocks = Vec::with_capacity(settings.layers);
        for i in 0..settings.layers {
            let pfx = format!("blocks.{i}");
            blocks.push(BlockWeights {
                r2c: mha(&format!("{pfx}.r2c"))?,
                c2r: mha(&format!("{pfx}.c2r"))?,
                lnr_weight: get(&format!("{pfx}.lnr.weight"))?,
                lnr_bias: get(&format!("{pfx}.lnr.bias"))?,
                lnc_weight: get(&format!("{pfx}.lnc.weight"))?,
                lnc_bias: get(&format!("{pfx}.lnc.bias"))?,
                ffr_0: lin(&format!("{pfx}.ffr.0"))?,
                ffr_2: lin(&format!("{pfx}.ffr.2"))?,
                ffc_0: lin(&format!("{pfx}.ffc.0"))?,
                ffc_2: lin(&format!("{pfx}.ffc.2"))?,
            });
        }

        let mut bands_0 = Vec::with_capacity(settings.n_bands);
        let mut bands_2 = Vec::with_capacity(settings.n_bands);
        for b in 0..settings.n_bands {
            bands_0.push(lin(&format!("emb_head.bands.{b}.0"))?);
            bands_2.push(lin(&format!("emb_head.bands.{b}.2"))?);
        }

        let scale_emb = get("scale_emb.weight")?;
        let scale_rows = scale_emb.dim(0)?;

        let weights = MguWeights {
            val_0: lin("val.0")?,
            val_2: lin("val.2")?,
            scale_emb,
            row_pool: mps("row_pool")?,
            col_pool: mps("col_pool")?,
            blocks,
            emb_head: HgeWeights {
                shared_0: lin("emb_head.shared.0")?,
                shared_2: lin("emb_head.shared.2")?,
                bands_0,
                bands_2,
            },
            cell_mlp_0: lin("cell_mlp.0")?,
            cell_mlp_2: lin("cell_mlp.2")?,
            h_real: lin("h_real")?,
            dim: settings.dim,
            heads: settings.heads,
            max_patch: settings.max_patch,
            ws_list: settings.ws_list.clone(),
        };

        let full_canvas = std::env::var("JAIN_HYPERION_FULL_CANVAS").is_ok_and(|v| v == "1");
        let pad_norm = super::grid::normalize_u8_to_model(settings.pad_u8, settings.quant_scale);
        Ok(Self {
            weights,
            device,
            #[cfg(feature = "hyperion-cuda")]
            _gpu_queue_lease: gpu_queue_lease,
            min_patch: settings.min_patch,
            scale_rows,
            full_canvas,
            pad_norm,
        })
    }

    pub fn device_label(&self) -> &'static str {
        hyperion_device_label_for(&self.device)
    }

    /// Test hook: force the canvas mode regardless of `JAIN_HYPERION_FULL_CANVAS` (the parity test
    /// compares the cropped p×p canvas against the reference max_patch canvas).
    pub fn with_full_canvas(mut self, on: bool) -> Self {
        self.full_canvas = on;
        self
    }

    fn forward_sub(&self, x: &[f32], m: &[f32], batch: usize, p: usize) -> Result<PatchOutputs> {
        let (canvas, xs, ms) = if self.full_canvas {
            // Reference layout: window in the top-left of the max_patch canvas, mask 0 outside.
            let mp = self.weights.max_patch;
            let mut xf = vec![self.pad_norm; batch * mp * mp];
            let mut mf = vec![0.0f32; batch * mp * mp];
            for b in 0..batch {
                for i in 0..p {
                    for j in 0..p {
                        xf[b * mp * mp + i * mp + j] = x[b * p * p + i * p + j];
                        mf[b * mp * mp + i * mp + j] = m[b * p * p + i * p + j];
                    }
                }
            }
            (mp, xf, mf)
        } else {
            (p, x.to_vec(), m.to_vec())
        };

        let xt = Tensor::from_vec(xs, (batch, canvas, canvas), &self.device)?;
        let mt = Tensor::from_vec(ms, (batch, canvas, canvas), &self.device)?;
        let p_id = (p.saturating_sub(self.min_patch)).min(self.scale_rows - 1) as u32;
        let pid = Tensor::from_vec(vec![p_id; batch], (batch,), &self.device)?;

        let (emb, logits) = self.weights.forward_full(&xt, &mt, &pid, p)?;
        let emb_v = emb
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let logits_v = if self.full_canvas {
            // Extract the p×p window from the max_patch canvas.
            logits.narrow(1, 0, p)?.narrow(2, 0, p)?.contiguous()?
        } else {
            logits
        }
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
        Ok(PatchOutputs {
            emb: emb_v,
            logits: logits_v,
        })
    }

    fn forward_emb_sub(&self, x: &[f32], m: &[f32], batch: usize, p: usize) -> Result<Vec<f32>> {
        let (canvas, xs, ms) = if self.full_canvas {
            let mp = self.weights.max_patch;
            let mut xf = vec![self.pad_norm; batch * mp * mp];
            let mut mf = vec![0.0f32; batch * mp * mp];
            for b in 0..batch {
                for i in 0..p {
                    for j in 0..p {
                        xf[b * mp * mp + i * mp + j] = x[b * p * p + i * p + j];
                        mf[b * mp * mp + i * mp + j] = m[b * p * p + i * p + j];
                    }
                }
            }
            (mp, xf, mf)
        } else {
            (p, x.to_vec(), m.to_vec())
        };

        let xt = Tensor::from_vec(xs, (batch, canvas, canvas), &self.device)?;
        let mt = Tensor::from_vec(ms, (batch, canvas, canvas), &self.device)?;
        let p_id = (p.saturating_sub(self.min_patch)).min(self.scale_rows - 1) as u32;
        let pid = Tensor::from_vec(vec![p_id; batch], (batch,), &self.device)?;

        Ok(self
            .weights
            .forward_embed(&xt, &mt, &pid, p)?
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?)
    }
}

impl PatchModel for V7Model {
    fn forward_batch(&self, x: &[f32], m: &[f32], batch: usize, p: usize) -> Result<PatchOutputs> {
        if batch == 0 {
            bail!("empty Hyperion v7 batch");
        }
        let area = p * p;
        let canvas_area = if self.full_canvas {
            self.weights.max_patch * self.weights.max_patch
        } else {
            area
        };
        let sub = (MAX_EMB_FORWARD_CELLS / canvas_area.max(1)).max(1);
        if batch <= sub {
            return self.forward_sub(x, m, batch, p);
        }
        let mut emb = Vec::with_capacity(batch * self.emb_dim());
        let mut logits = Vec::with_capacity(batch * area);
        let mut off = 0usize;
        while off < batch {
            let take = sub.min(batch - off);
            let out = self.forward_sub(
                &x[off * area..(off + take) * area],
                &m[off * area..(off + take) * area],
                take,
                p,
            )?;
            emb.extend_from_slice(&out.emb);
            logits.extend_from_slice(&out.logits);
            off += take;
        }
        Ok(PatchOutputs { emb, logits })
    }

    fn forward_emb_batch(&self, x: &[f32], m: &[f32], batch: usize, p: usize) -> Result<Vec<f32>> {
        if batch == 0 {
            bail!("empty Hyperion v7 batch");
        }
        let area = p * p;
        let canvas_area = if self.full_canvas {
            self.weights.max_patch * self.weights.max_patch
        } else {
            area
        };
        let sub = (MAX_FORWARD_CELLS / canvas_area.max(1)).max(1);
        if batch <= sub {
            return self.forward_emb_sub(x, m, batch, p);
        }
        let mut emb = Vec::with_capacity(batch * self.emb_dim());
        let mut off = 0usize;
        while off < batch {
            let take = sub.min(batch - off);
            let out = self.forward_emb_sub(
                &x[off * area..(off + take) * area],
                &m[off * area..(off + take) * area],
                take,
                p,
            )?;
            emb.extend_from_slice(&out);
            off += take;
        }
        Ok(emb)
    }

    fn emb_dim(&self) -> usize {
        self.weights.dim
    }
}
