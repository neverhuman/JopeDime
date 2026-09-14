//! Fast MVP feature reduction: accept novel GP features by importance percentile, then greedily
//! reduce with a light correlation penalty.
use crate::buildmatrix::FeatureInfo;
use crate::data::Matrix;
use crate::metrics::{corr_std, standardize};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SelectionMode {
    Pbil,
    MvpFast,
}

impl SelectionMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "pbil" => Ok(Self::Pbil),
            "mvp-fast" | "mvp_fast" => Ok(Self::MvpFast),
            other => Err(format!(
                "unsupported selection mode {other:?}; use pbil or mvp-fast"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pbil => "pbil",
            Self::MvpFast => "mvp-fast",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MvpFastConfig {
    pub importance_pct: f32,
    pub target_new: usize,
    pub top_frac: f32,
    pub max_features: usize,
    pub corr_penalty: f32,
}

impl Default for MvpFastConfig {
    fn default() -> Self {
        Self {
            importance_pct: 0.30,
            target_new: 1000,
            top_frac: 0.10,
            max_features: 500,
            corr_penalty: 0.25,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MvpFastReport {
    pub accepted_gp: usize,
    pub selected_gp: usize,
    pub final_features: usize,
    pub shortfall: usize,
    pub base_features: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MvpFastSelection {
    pub cols: Vec<u32>,
    pub report: MvpFastReport,
}

pub fn reduce_features(
    x: &Matrix,
    features: &[FeatureInfo],
    importance_percentile: &[f32],
    cfg: &MvpFastConfig,
) -> MvpFastSelection {
    assert_eq!(
        features.len(),
        x.c,
        "feature metadata/matrix shape mismatch"
    );
    assert_eq!(
        importance_percentile.len(),
        x.c,
        "importance percentile/matrix shape mismatch"
    );
    if x.c == 0 {
        return MvpFastSelection {
            cols: Vec::new(),
            report: MvpFastReport::default(),
        };
    }

    let max_features = cfg.max_features.max(1).min(x.c);
    let base: Vec<u32> = features
        .iter()
        .enumerate()
        .filter_map(|(j, f)| (f.source == "base").then_some(j as u32))
        .collect();
    let accepted_gp: Vec<u32> = features
        .iter()
        .enumerate()
        .filter_map(|(j, f)| {
            (f.source == "gp" && importance_percentile[j] >= cfg.importance_pct).then_some(j as u32)
        })
        .collect();

    let shortfall = cfg.target_new.saturating_sub(accepted_gp.len());
    if base.len() <= max_features {
        let mut kept = base.clone();
        let accepted_gp_len = accepted_gp.len();
        let gp_limit = ((accepted_gp_len as f32) * cfg.top_frac.clamp(0.0, 1.0)).ceil() as usize;
        let gp_limit = gp_limit.min(max_features.saturating_sub(kept.len()));
        let selected_gp = greedy_add(
            x,
            importance_percentile,
            &mut kept,
            accepted_gp,
            gp_limit,
            cfg.corr_penalty,
        );
        return MvpFastSelection {
            report: MvpFastReport {
                accepted_gp: accepted_gp_len,
                selected_gp,
                final_features: kept.len(),
                shortfall,
                base_features: base.len(),
            },
            cols: kept,
        };
    }

    let mut kept = Vec::with_capacity(max_features);
    let all: Vec<u32> = (0..x.c as u32).collect();
    greedy_add(
        x,
        importance_percentile,
        &mut kept,
        all,
        max_features,
        cfg.corr_penalty,
    );
    let selected_gp = kept
        .iter()
        .filter(|&&c| features[c as usize].source == "gp")
        .count();
    MvpFastSelection {
        report: MvpFastReport {
            accepted_gp: accepted_gp.len(),
            selected_gp,
            final_features: kept.len(),
            shortfall,
            base_features: base.len(),
        },
        cols: kept,
    }
}

fn greedy_add(
    x: &Matrix,
    importance_percentile: &[f32],
    kept: &mut Vec<u32>,
    candidates: Vec<u32>,
    limit: usize,
    corr_penalty: f32,
) -> usize {
    if limit == 0 || candidates.is_empty() {
        return 0;
    }
    let std_cols: Vec<Vec<f32>> = (0..x.c).map(|j| standardize(x.col(j)).0).collect();
    let mut selected = 0usize;
    let mut remaining = candidates;
    remaining.retain(|c| !kept.contains(c));
    while selected < limit && !remaining.is_empty() {
        let mut best_pos = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (pos, &c) in remaining.iter().enumerate() {
            let max_corr = kept
                .iter()
                .map(|&k| corr_std(&std_cols[c as usize], &std_cols[k as usize]))
                .fold(0.0f32, f32::max);
            let score = importance_percentile[c as usize] * (1.0 - corr_penalty * max_corr);
            if score > best_score + 1e-12
                || ((score - best_score).abs() <= 1e-12 && c < remaining[best_pos])
            {
                best_score = score;
                best_pos = pos;
            }
        }
        let c = remaining.swap_remove(best_pos);
        kept.push(c);
        selected += 1;
    }
    selected
}
