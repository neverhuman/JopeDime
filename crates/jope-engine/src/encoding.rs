use crate::manifest::*;
#[cfg(feature = "neural")]
use crate::prepare::Prepared;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderSpec {
    pub schema: String,
    pub dataset_dimensions: usize,
    pub target_dimensions: usize,
    pub module: PathBuf,
    pub module_sha256: String,
    pub source_manifest: PathBuf,
    pub source_manifest_sha256: String,
    pub configuration: PathBuf,
    pub configuration_sha256: String,
    pub parity_receipt: PathBuf,
    pub parity_receipt_sha256: String,
    pub numerical_backend: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Encoding {
    pub schema: String,
    pub dataset: String,
    pub family: String,
    pub partition: String,
    pub cut: usize,
    pub prepared_sha256: String,
    pub training_sha256: String,
    pub selected_training_rows_sha256: String,
    pub preprocessing_sha256: String,
    pub encoder_spec_sha256: String,
    pub numerical_backend: String,
    pub dataset_dimensions: usize,
    pub target_dimensions: usize,
    pub output_sha256: String,
}
impl EncoderSpec {
    pub fn verify(&self, root: &Path) -> Result<()> {
        ensure!(
            self.schema == "parallel.frozen_encoder.v1"
                && self.dataset_dimensions == 1536
                && self.target_dimensions == 512,
            "context must match dataset1536-target512 schema"
        );
        for (path, sha) in [
            (&self.module, &self.module_sha256),
            (&self.source_manifest, &self.source_manifest_sha256),
            (&self.configuration, &self.configuration_sha256),
            (&self.parity_receipt, &self.parity_receipt_sha256),
        ] {
            ensure!(
                file_hash(&resolve(root, path)?)? == *sha,
                "encoder dependency changed: {}",
                path.display()
            );
        }
        let parity: serde_json::Value = read_json(&resolve(root, &self.parity_receipt)?)?;
        ensure!(
            parity["passed"] == true
                && parity["module_sha256"] == self.module_sha256
                && parity["numerical_backend"] == self.numerical_backend,
            "encoder parity does not bind these bytes/backend"
        );
        Ok(())
    }
}
#[cfg(feature = "neural")]
pub fn encode(
    root: &Path,
    prepared: &Path,
    cut: usize,
    spec: &Path,
    output: &Path,
    gpu: usize,
) -> Result<Encoding> {
    crate::task_contract::require_original_gpu_grant("encode-cut")?;
    use tch::{CModule, Device, Kind, Tensor};
    let p: Prepared = read_json(&prepared.join("prepared.json"))?;
    ensure!(
        p.schema == "parallel.prepared.v1" && p.clean_evaluation,
        "cut encodings require clean training rows"
    );
    let c = p
        .cuts
        .iter()
        .find(|c| c.index == cut)
        .ok_or_else(|| anyhow::anyhow!("unknown cut"))?;
    let spec_data: EncoderSpec = read_json(spec)?;
    spec_data.verify(root)?;
    let rows = crate::prepare::load_view(prepared, cut, true)?;
    let n = rows.len();
    let width = rows[0].len() - 1;
    let x = rows
        .iter()
        .flat_map(|r| r[..width].iter().map(|v| *v as f32))
        .collect::<Vec<_>>();
    let y = rows.iter().map(|r| r[width] as f32).collect::<Vec<_>>();
    let device = Device::Cuda(gpu);
    let mut module = CModule::load_on_device(resolve(root, &spec_data.module)?, device)?;
    module.set_eval();
    let result = tch::no_grad(|| {
        module.forward_ts(&[
            Tensor::from_slice(&x)
                .view([n as i64, width as i64])
                .to_device(device),
            Tensor::from_slice(&y).to_device(device),
        ])
    })?
    .to_kind(Kind::Float)
    .to_device(Device::Cpu)
    .flatten(0, -1);
    ensure!(
        result.numel() == 2048 && result.isfinite().all().int64_value(&[]) != 0,
        "invalid encoder output"
    );
    let values: Vec<f32> = Vec::try_from(result)?;
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let e = Encoding {
        schema: "parallel.encoding.v1".into(),
        dataset: p.dataset,
        family: p.family,
        partition: p.meta_partition,
        cut,
        prepared_sha256: file_hash(&prepared.join("prepared.json"))?,
        training_sha256: c.training_sha256.clone(),
        selected_training_rows_sha256: hash(&serde_json::to_vec(&c.train_indices)?),
        preprocessing_sha256: c.preprocessing_sha256.clone(),
        encoder_spec_sha256: file_hash(spec)?,
        numerical_backend: spec_data.numerical_backend,
        dataset_dimensions: 1536,
        target_dimensions: 512,
        output_sha256: hash(&bytes),
    };
    if output.exists() {
        let old: Encoding = read_json(&output.with_extension("json"))?;
        ensure!(
            serde_json::to_value(&old)? == serde_json::to_value(&e)?
                && file_hash(output)? == e.output_sha256,
            "stale encoding cache"
        );
        return Ok(old);
    }
    atomic_write(output, &bytes)?;
    write_json(&output.with_extension("json"), &e)?;
    Ok(e)
}
pub fn verify_bytes(path: &Path, e: &Encoding) -> Result<()> {
    ensure!(
        e.schema == "parallel.encoding.v1"
            && e.dataset_dimensions == 1536
            && e.target_dimensions == 512
            && fs::metadata(path)?.len() == 8192
            && file_hash(path)? == e.output_sha256,
        "invalid context cache identity"
    );
    Ok(())
}
