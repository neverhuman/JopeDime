//! Pure-Rust v301 fair router.
//!
//! The production path consumes compact JSON artifacts and never shells out to Python. The math mirrors
//! `round_fair.py::run_v301`: adaptive retrieval scores, source-family shrinkage, family smoothing of
//! lifts, then Starlight Core-anchor gating.

mod routing;
mod scoring;

use crate::data::Dataset;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub use routing::{algorithm_family_name, route_algorithm, stable_router_hash};
pub use scoring::{
    adaptive_retrieval_scores, argmax, certify_margin, elo_delta, family_smooth_lifts,
    source_family_mean_predict, starlight_anchor_index, tiny_elo,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouterConfig {
    pub enabled: bool,
    pub require: bool,
    pub assets_path: Option<PathBuf>,
    pub tau2: f64,
    pub smooth: f64,
    pub z: f64,
    pub k: usize,
    pub bandwidth_scale: f64,
    pub family_shrink: f64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            require: false,
            assets_path: None,
            tau2: 0.002,
            smooth: 0.12,
            z: 0.0,
            k: 16,
            bandwidth_scale: 1.0,
            family_shrink: 8.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouterArtifacts {
    pub algorithms: Vec<String>,
    pub train_features: Vec<Vec<f64>>,
    pub train_utilities: Vec<Vec<f64>>,
    #[serde(default)]
    pub train_source_families: Vec<String>,
    #[serde(default)]
    pub family_names: Vec<String>,
    #[serde(default)]
    pub family_ids: Vec<usize>,
    #[serde(default)]
    pub standardize: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetSketch {
    pub features: Vec<f64>,
    pub source_family: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouteFamily {
    CatBoost,
    XGBoost,
    LightGbm,
    StarlightCore,
    Unsupported,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlgorithmRoute {
    pub requested: String,
    pub selected: String,
    pub family: RouteFamily,
    pub audit: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouterDecision {
    pub selected_label: String,
    pub selected_algorithm: String,
    pub selected_index: usize,
    pub anchor_algorithm: String,
    pub anchor_index: usize,
    pub route: AlgorithmRoute,
    pub scores: Vec<f64>,
    pub audit: Vec<String>,
}

impl RouterArtifacts {
    pub fn load(path: &Path) -> Result<Self> {
        let path = if path.is_dir() {
            let direct = path.join("router_v301.json");
            if direct.is_file() {
                direct
            } else {
                path.join("manifest.json")
            }
        } else {
            path.to_path_buf()
        };
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let mut art: RouterArtifacts = serde_json::from_str(&text)
            .with_context(|| format!("parsing router artifact {}", path.display()))?;
        art.validate()?;
        art.fill_families();
        Ok(art)
    }

    fn validate(&self) -> Result<()> {
        if self.algorithms.is_empty() {
            bail!("router artifact has no algorithms");
        }
        if self.train_features.is_empty() || self.train_utilities.is_empty() {
            bail!("router artifact is missing train features/utilities");
        }
        if self.train_features.len() != self.train_utilities.len() {
            bail!("router artifact train feature/utility row count mismatch");
        }
        let n_alg = self.algorithms.len();
        if self.train_utilities.iter().any(|r| r.len() != n_alg) {
            bail!("router utility rows do not match algorithm count");
        }
        Ok(())
    }

    fn fill_families(&mut self) {
        if self.train_source_families.len() != self.train_features.len() {
            self.train_source_families = vec!["global".into(); self.train_features.len()];
        }
        if self.family_ids.len() == self.algorithms.len() && !self.family_names.is_empty() {
            return;
        }
        let mut map = BTreeMap::<String, usize>::new();
        self.family_ids.clear();
        for alg in &self.algorithms {
            let fam = algorithm_family_name(alg);
            let next = map.len();
            let id = *map.entry(fam).or_insert(next);
            self.family_ids.push(id);
        }
        self.family_names = map.keys().cloned().collect();
    }

    pub fn prior(&self) -> Vec<f64> {
        let n_alg = self.algorithms.len();
        let mut out = vec![0.0; n_alg];
        for row in &self.train_utilities {
            for (j, v) in row.iter().enumerate() {
                if v.is_finite() {
                    out[j] += *v;
                }
            }
        }
        let denom = self.train_utilities.len().max(1) as f64;
        for v in &mut out {
            *v /= denom;
        }
        out
    }
}

pub fn sketch_dataset(train: &Dataset, hold: Option<&Dataset>) -> DatasetSketch {
    let n = train.n().max(1);
    let p = train.p().max(1);
    let mut finite = 0usize;
    let mut missing = 0usize;
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut constant = 0usize;
    for j in 0..train.p() {
        let col = train.x.col(j);
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in col {
            if v.is_finite() {
                finite += 1;
                sum += v as f64;
                sum2 += (v as f64) * (v as f64);
                min = min.min(v);
                max = max.max(v);
            } else {
                missing += 1;
            }
        }
        if !min.is_finite() || (max - min).abs() <= 1e-12 {
            constant += 1;
        }
    }
    let total = (n * p).max(1) as f64;
    let mean = sum / finite.max(1) as f64;
    let var = (sum2 / finite.max(1) as f64 - mean * mean).max(0.0);
    let prevalence = train.y.iter().filter(|&&v| v > 0.5).count() as f64 / n as f64;
    let hold_n = hold.map(|h| h.n()).unwrap_or(0).max(1);
    DatasetSketch {
        features: vec![
            (n as f64 + 1.0).ln(),
            (p as f64 + 1.0).ln(),
            prevalence,
            1.0 - prevalence,
            missing as f64 / total,
            constant as f64 / p as f64,
            mean,
            var.sqrt(),
            (hold_n as f64 + 1.0).ln(),
        ],
        source_family: "jain_rust".into(),
    }
}

pub fn route_dataset(
    artifacts: &RouterArtifacts,
    sketch: &DatasetSketch,
    cfg: &RouterConfig,
) -> Result<RouterDecision> {
    let prior = artifacts.prior();
    let anchor = starlight_anchor_index(&artifacts.algorithms, &prior);
    let mut scores = adaptive_retrieval_scores(artifacts, &sketch.features, cfg);
    let fam = source_family_mean_predict(artifacts, &sketch.source_family, cfg.family_shrink);
    let se = vec![0.03; artifacts.algorithms.len()];
    for j in 0..scores.len() {
        let w = se[j] * se[j] / (se[j] * se[j] + cfg.tau2);
        scores[j] = (1.0 - w) * scores[j] + w * fam[j];
    }
    let anchor_score = scores[anchor];
    let mut lifts: Vec<f64> = scores.iter().map(|s| s - anchor_score).collect();
    family_smooth_lifts(&mut lifts, &artifacts.family_ids, cfg.smooth);
    for j in 0..scores.len() {
        scores[j] = anchor_score + lifts[j];
    }
    let top = argmax(&scores);
    let certified = top == anchor || (scores[top] - scores[anchor]) >= cfg.z;
    let selected = if certified { top } else { anchor };
    let selected_algorithm = artifacts.algorithms[selected].clone();
    let mut audit = Vec::new();
    if !certified {
        audit.push(format!(
            "router selected Starlight Core anchor '{}' because margin {:.6} < z {:.6}",
            artifacts.algorithms[anchor],
            scores[top] - scores[anchor],
            cfg.z
        ));
    }
    let route = route_algorithm(&selected_algorithm, Some(&artifacts.algorithms[anchor]));
    audit.extend(route.audit.iter().cloned());
    Ok(RouterDecision {
        selected_label: format!("CVBEST_t{:.3}_s{:.2}_z{:.1}", cfg.tau2, cfg.smooth, cfg.z),
        selected_algorithm,
        selected_index: selected,
        anchor_algorithm: artifacts.algorithms[anchor].clone(),
        anchor_index: anchor,
        route,
        scores,
        audit,
    })
}
