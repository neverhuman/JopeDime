use crate::buildmatrix::FeatureInfo;
use crate::encode::Encoders;
use crate::ensemble::{FuserMode, FuserPlan, ResidualFuserPlan};
use crate::hyperion::HyperionState;
use crate::starforge_integration::SavedStarforgeMember;
use crate::task::TaskSpec;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const ARTIFACT_VERSION: u32 = 1;
pub const MANIFEST_FILE: &str = "manifest.json";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedMember {
    pub name: String,
    pub backend: String,
    pub weight: f64,
    pub model_file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub starforge: Option<SavedStarforgeMember>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedFuser {
    pub name: String,
    #[serde(default)]
    pub mode: FuserMode,
    pub rank_outputs: bool,
    pub members: Vec<SavedMember>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual: Option<ResidualFuserPlan>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelManifest {
    pub version: u32,
    pub target: String,
    pub targets: Vec<String>,
    pub task: TaskSpec,
    pub base_names: Vec<String>,
    pub encoders: Encoders,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hyperion: Option<HyperionState>,
    pub selected_cols: Vec<u32>,
    pub features: Vec<FeatureInfo>,
    pub fuser: SavedFuser,
}

impl ModelManifest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target: String,
        targets: Vec<String>,
        task: TaskSpec,
        base_names: Vec<String>,
        encoders: Encoders,
        selected_cols: Vec<u32>,
        features: Vec<FeatureInfo>,
        fuser: SavedFuser,
    ) -> Self {
        Self {
            version: ARTIFACT_VERSION,
            target,
            targets,
            task,
            base_names,
            encoders,
            hyperion: None,
            selected_cols,
            features,
            fuser,
        }
    }

    pub fn with_hyperion(mut self, hyperion: Option<HyperionState>) -> Self {
        self.hyperion = hyperion;
        self
    }
}

pub fn write_manifest(dir: &Path, manifest: &ModelManifest) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(MANIFEST_FILE);
    let body = serde_json::to_string_pretty(manifest)?;
    std::fs::write(&path, body).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

pub fn read_manifest(dir: &Path) -> Result<ModelManifest> {
    let path = dir.join(MANIFEST_FILE);
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let manifest: ModelManifest = serde_json::from_str(&text)?;
    if manifest.version != ARTIFACT_VERSION {
        bail!(
            "unsupported artifact version {} (expected {})",
            manifest.version,
            ARTIFACT_VERSION
        );
    }
    Ok(manifest)
}

#[derive(Clone, Debug)]
pub struct SavedModelEntry {
    pub model_file: String,
    pub starforge: Option<SavedStarforgeMember>,
}

pub fn saved_fuser_from_plan(
    plan: &FuserPlan,
    model_files: Vec<SavedModelEntry>,
) -> Result<SavedFuser> {
    if plan.members.len() != model_files.len() {
        bail!("model file count does not match fuser member count");
    }
    // The downloaded manifest carries USER-FACING names (`catboost*` → `jboost*`); the load-time
    // dispatch uses `backend` (unchanged), and the residual join is on these display names, so map the
    // member/fuser names AND the residual base/residual members together to keep the join consistent.
    let members = plan
        .members
        .iter()
        .zip(model_files)
        .map(|(m, entry)| SavedMember {
            name: crate::naming::display_model_name(&m.name),
            backend: m.backend.clone(),
            weight: m.weight,
            model_file: entry.model_file,
            starforge: entry.starforge,
        })
        .collect();
    let residual = plan.residual.clone().map(|mut r| {
        r.base_member = crate::naming::display_model_name(&r.base_member);
        r.residual_member = crate::naming::display_model_name(&r.residual_member);
        r
    });
    Ok(SavedFuser {
        name: crate::naming::display_model_name(&plan.name),
        mode: plan.mode,
        rank_outputs: plan.rank_outputs,
        members,
        residual,
    })
}
