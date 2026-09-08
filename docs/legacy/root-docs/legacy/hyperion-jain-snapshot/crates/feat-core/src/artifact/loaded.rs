use super::{
    materialize_selected, read_manifest, transform_scoring_csv, transform_scoring_csv_text,
    ModelManifest, SavedMember,
};
use crate::backend::{load_native_model, Model};
use crate::data::Matrix;
use crate::ensemble::FuserMode;
use crate::metrics::rank01;
#[cfg(feature = "starforge-cpu")]
use crate::starforge_integration::{
    read_context, StarforgeConfig, StarforgeContext, StarforgeContextTask,
    StarforgeRegressionEngine,
};
use crate::task::model_predict;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

pub fn predict_dir(dir: &Path, data: &Path) -> Result<Vec<f32>> {
    let artifact = LoadedArtifact::load(dir)?;
    let base = transform_scoring_csv(data, &artifact.manifest)?;
    artifact.predict_base_matrix(base)
}

pub fn predict_csv_text(dir: &Path, text: &str) -> Result<Vec<f32>> {
    let artifact = LoadedArtifact::load(dir)?;
    let base = transform_scoring_csv_text(text, &artifact.manifest)?;
    artifact.predict_base_matrix(base)
}

pub fn predict_base_matrix(dir: &Path, manifest: &ModelManifest, base: Matrix) -> Result<Vec<f32>> {
    let artifact = LoadedArtifact::load_with_manifest(dir, manifest.clone())?;
    artifact.predict_base_matrix(base)
}

pub struct LoadedArtifact {
    pub manifest: ModelManifest,
    members: Vec<LoadedMember>,
}

enum LoadedMember {
    Native {
        saved: SavedMember,
        model: Box<dyn Model>,
    },
    #[cfg(feature = "starforge-cpu")]
    Starforge {
        saved: SavedMember,
        session: crate::starforge_integration::LoadedSession,
        context: StarforgeContext,
    },
}

impl LoadedArtifact {
    pub fn load(dir: &Path) -> Result<Self> {
        let manifest = read_manifest(dir)?;
        Self::load_with_manifest(dir, manifest)
    }

    pub fn load_with_manifest(dir: &Path, manifest: ModelManifest) -> Result<Self> {
        if manifest.fuser.members.is_empty() {
            bail!("artifact fuser has no members");
        }
        let mut members = Vec::with_capacity(manifest.fuser.members.len());
        for member in &manifest.fuser.members {
            if member.backend == "starforge" {
                members.push(load_starforge_member(dir, member)?);
            } else {
                let model_path: PathBuf = dir.join(&member.model_file);
                let model = load_native_model(&member.backend, &model_path)
                    .with_context(|| format!("loading {}", model_path.display()))?;
                members.push(LoadedMember::Native {
                    saved: member.clone(),
                    model,
                });
            }
        }
        Ok(Self { manifest, members })
    }

    pub fn predict_csv_text(&self, text: &str) -> Result<Vec<f32>> {
        let base = transform_scoring_csv_text(text, &self.manifest)?;
        self.predict_base_matrix(base)
    }

    pub fn predict_base_matrix(&self, base: Matrix) -> Result<Vec<f32>> {
        let native_x =
            materialize_selected(&base, &self.manifest.features, &self.manifest.selected_cols)?;
        let xs = native_x.data.as_ref().as_slice();
        let member_preds: Vec<(&SavedMember, Vec<f32>)> = self
            .members
            .iter()
            .map(|member| self.predict_member(member, &base, &native_x, xs))
            .collect::<Result<_>>()?;
        let mut out = match self.manifest.fuser.mode {
            FuserMode::WeightedBlend => {
                let mut out = vec![0.0f32; base.n];
                for (saved, mut pred) in member_preds {
                    if self.manifest.fuser.rank_outputs {
                        pred = rank01(&pred);
                    }
                    let w = saved.weight as f32;
                    for i in 0..base.n {
                        out[i] += w * pred[i];
                    }
                }
                out
            }
            FuserMode::Residual => {
                let residual = self
                    .manifest
                    .fuser
                    .residual
                    .as_ref()
                    .context("residual fuser missing residual metadata")?;
                let base_pred = member_preds
                    .iter()
                    .find_map(|(saved, pred)| {
                        (saved.name == residual.base_member).then_some(pred.as_slice())
                    })
                    .with_context(|| {
                        format!("residual base member '{}' not found", residual.base_member)
                    })?;
                let residual_pred = member_preds
                    .iter()
                    .find_map(|(saved, pred)| {
                        (saved.name == residual.residual_member).then_some(pred.as_slice())
                    })
                    .with_context(|| {
                        format!(
                            "residual Starforge member '{}' not found",
                            residual.residual_member
                        )
                    })?;
                let lo = residual.clamp_min;
                let hi = residual.clamp_max;
                (0..base.n)
                    .map(|i| {
                        (base_pred[i] + (residual.alpha as f32) * residual_pred[i]).clamp(lo, hi)
                    })
                    .collect()
            }
        };
        if self.manifest.task.kind == crate::task::TaskKind::Regression {
            for v in &mut out {
                *v = self.manifest.encoders.target_scaler.inverse(*v);
            }
        }
        Ok(out)
    }

    fn predict_member<'a>(
        &'a self,
        member: &'a LoadedMember,
        base: &Matrix,
        native_x: &Matrix,
        xs: &[f32],
    ) -> Result<(&'a SavedMember, Vec<f32>)> {
        let (saved, pred) = match member {
            LoadedMember::Native { saved, model } => {
                let pred = model_predict(
                    &**model,
                    xs,
                    native_x.n,
                    native_x.c,
                    self.manifest.task.kind,
                    self.manifest.task.n_classes,
                );
                (saved, pred)
            }
            #[cfg(feature = "starforge-cpu")]
            LoadedMember::Starforge {
                saved,
                session,
                context,
            } => {
                let sf = saved
                    .starforge
                    .as_ref()
                    .context("Starforge member missing manifest metadata")?;
                let x = materialize_selected(base, &self.manifest.features, &sf.selected_cols)?;
                let pred = session.predict_context(context, &x)?;
                (saved, pred)
            }
        };
        if pred.len() != base.n {
            bail!("multi-output artifact prediction is not supported by the CSV writer yet");
        }
        Ok((saved, pred))
    }
}

#[cfg(feature = "starforge-cpu")]
fn load_starforge_member(dir: &Path, member: &SavedMember) -> Result<LoadedMember> {
    let sf = member
        .starforge
        .as_ref()
        .context("Starforge member missing manifest metadata")?;
    let context_path = dir.join(&sf.context_file);
    let context = read_context(&context_path)?;
    let dtype = crate::starforge_integration::StarforgeDType::parse(&sf.dtype)?;
    let regression_engine = match sf.task {
        StarforgeContextTask::Regressor => sf
            .regression_engine
            .as_deref()
            .map(StarforgeRegressionEngine::parse)
            .transpose()?
            .unwrap_or(StarforgeRegressionEngine::Starlight),
        StarforgeContextTask::Classifier => StarforgeRegressionEngine::Auto,
    };
    let cfg = StarforgeConfig {
        mode: crate::starforge_integration::StarforgeMode::On,
        device: crate::starforge_integration::StarforgeDevice::Auto,
        dtype,
        require_gpu: false,
        max_context: sf.context_rows,
        top_features: sf.selected_cols.len(),
        seed: 20260616,
        weights_path: resolve_starforge_weights(dir, &sf.weights_path, sf.task),
        regressor_weights_path: resolve_starforge_weights(dir, &sf.weights_path, sf.task),
        regression_engine,
        chimera_weights_path: resolve_optional_weights(
            dir,
            sf.chimera_weights_path.as_deref(),
            &[
                "/opt/jain/starforge/chimera_stackfusionreg.safetensors.zst",
                "/opt/jain/starforge/chimera_stackfusionreg.safetensors",
                "artifacts/starforge/chimera_stackfusionreg.safetensors.zst",
                "artifacts/starforge/chimera_stackfusionreg.safetensors",
            ],
        ),
        tabicl_regressor_weights_path: resolve_optional_weights(
            dir,
            sf.tabicl_regressor_weights_path.as_deref(),
            &[
                "/opt/jain/starforge/tabicl-regressor-v2-20260212.safetensors.zst",
                "/opt/jain/starforge/tabicl-regressor-v2-20260212.safetensors",
                "artifacts/foundation/tabicl-regressor-v2-20260212.safetensors.zst",
                "artifacts/foundation/tabicl-regressor-v2-20260212.safetensors",
            ],
        ),
    };
    let task = match sf.task {
        StarforgeContextTask::Regressor => crate::task::TaskKind::Regression,
        StarforgeContextTask::Classifier => crate::task::TaskKind::Binary,
    };
    let session = crate::starforge_integration::load_session_for_task(&cfg, task)?
        .context("Starforge session unexpectedly disabled")?;
    Ok(LoadedMember::Starforge {
        saved: member.clone(),
        session,
        context,
    })
}

#[cfg(not(feature = "starforge-cpu"))]
fn load_starforge_member(_dir: &Path, _member: &SavedMember) -> Result<LoadedMember> {
    bail!("artifact contains a Starforge member, but this binary was not compiled with Starforge")
}

#[cfg(feature = "starforge-cpu")]
fn resolve_starforge_weights(dir: &Path, path: &str, task: StarforgeContextTask) -> PathBuf {
    let p = PathBuf::from(path);
    if p.is_file() {
        return p;
    }
    let under_model = dir.join(path);
    if under_model.is_file() {
        return under_model;
    }
    // Runtime image ships the weights zstd-compressed to shrink the image; prefer
    // the `.zst` and fall back to a plain `.safetensors`.
    let candidates: &[&str] = match task {
        StarforgeContextTask::Regressor => &[
            "/opt/jain/starforge/starlight_core_regressor.safetensors.zst",
            "/opt/jain/starforge/starlight_core_regressor.safetensors",
        ],
        StarforgeContextTask::Classifier => &[
            "/opt/jain/starforge/starlight_core.safetensors.zst",
            "/opt/jain/starforge/starlight_core.safetensors",
        ],
    };
    for candidate in candidates {
        let runtime = PathBuf::from(candidate);
        if runtime.is_file() {
            return runtime;
        }
    }
    p
}

#[cfg(feature = "starforge-cpu")]
fn resolve_optional_weights(dir: &Path, saved_path: Option<&str>, candidates: &[&str]) -> PathBuf {
    if let Some(path) = saved_path {
        let p = PathBuf::from(path);
        if p.is_file() {
            return p;
        }
        let under_model = dir.join(path);
        if under_model.is_file() {
            return under_model;
        }
    }
    for candidate in candidates {
        let p = PathBuf::from(candidate);
        if p.is_file() {
            return p;
        }
    }
    PathBuf::from(saved_path.unwrap_or_else(|| candidates.last().copied().unwrap_or("")))
}

pub fn artifact_healthy_loaded(artifact: &LoadedArtifact) -> Result<()> {
    if artifact.members.is_empty() {
        bail!("artifact fuser has no members");
    }
    Ok(())
}
