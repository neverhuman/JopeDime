use crate::data::Matrix;
use crate::rng::{seeded, Rng};
use crate::task::TaskKind;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StarforgeContextTask {
    #[default]
    Classifier,
    Regressor,
}

impl StarforgeContextTask {
    pub fn from_task(task: TaskKind) -> Self {
        match task {
            TaskKind::Regression => Self::Regressor,
            _ => Self::Classifier,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Classifier => "classifier",
            Self::Regressor => "regressor",
        }
    }
}

fn is_classifier_task(task: &StarforgeContextTask) -> bool {
    *task == StarforgeContextTask::Classifier
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StarforgeMemberPlan {
    pub selected_cols: Vec<u32>,
    pub weights_path: String,
    pub dtype: String,
    pub max_context: usize,
    pub context_seed: u64,
    pub task: StarforgeContextTask,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regression_engine: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chimera_weights_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tabicl_regressor_weights_path: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedStarforgeMember {
    pub selected_cols: Vec<u32>,
    pub weights_path: String,
    pub context_file: String,
    pub dtype: String,
    pub context_rows: usize,
    pub n_features: usize,
    #[serde(default, skip_serializing_if = "is_classifier_task")]
    pub task: StarforgeContextTask,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regression_engine: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chimera_weights_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tabicl_regressor_weights_path: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StarforgeContext {
    #[serde(default, skip_serializing_if = "is_classifier_task")]
    pub task: StarforgeContextTask,
    pub selected_cols: Vec<u32>,
    pub n_rows: usize,
    pub n_features: usize,
    /// Row-major context matrix, `n_rows * n_features`.
    pub x: Vec<f32>,
    pub y: Vec<f32>,
}

pub fn top_importance_cols(mv: &[f32], limit: usize) -> Vec<u32> {
    let mut idx: Vec<usize> = (0..mv.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        mv[b]
            .partial_cmp(&mv[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.into_iter()
        .take(limit.min(mv.len()))
        .map(|i| i as u32)
        .collect()
}

pub fn stratified_context_rows(y: &[f32], max_rows: usize, seed: u64) -> Vec<usize> {
    if y.len() <= max_rows {
        return (0..y.len()).collect();
    }
    let mut zeros: Vec<usize> = y
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (v <= 0.5).then_some(i))
        .collect();
    let mut ones: Vec<usize> = y
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (v > 0.5).then_some(i))
        .collect();
    let mut rng0 = seeded(seed ^ 0x4d59_5df4_d0f3_3173);
    let mut rng1 = seeded(seed ^ 0x91e1_0da5_c79e_7b1d);
    shuffle(&mut zeros, &mut rng0);
    shuffle(&mut ones, &mut rng1);
    let n1 = ((max_rows as f64) * (ones.len() as f64) / (y.len() as f64)).round() as usize;
    let n1 = n1.min(ones.len());
    let n0 = max_rows.saturating_sub(n1).min(zeros.len());
    let mut out = Vec::with_capacity(n0 + n1);
    out.extend_from_slice(&zeros[..n0]);
    out.extend_from_slice(&ones[..n1]);
    if out.is_empty() {
        out.extend(0..max_rows.min(y.len()));
    }
    out.sort_unstable();
    out
}

pub fn context_rows_for_task(y: &[f32], max_rows: usize, seed: u64, task: TaskKind) -> Vec<usize> {
    if task == TaskKind::Regression {
        random_context_rows(y.len(), max_rows, seed)
    } else {
        stratified_context_rows(y, max_rows, seed)
    }
}

fn random_context_rows(n: usize, max_rows: usize, seed: u64) -> Vec<usize> {
    if n <= max_rows {
        return (0..n).collect();
    }
    let mut rows: Vec<usize> = (0..n).collect();
    let mut rng = seeded(seed ^ 0x52bd_9f47_6a73_0af1);
    shuffle(&mut rows, &mut rng);
    rows.truncate(max_rows);
    rows.sort_unstable();
    rows
}

fn shuffle<T>(items: &mut [T], rng: &mut crate::rng::Rng64) {
    for i in (1..items.len()).rev() {
        let j = rng.gen_range(0..=i);
        items.swap(i, j);
    }
}

pub fn matrix_rows(x: &Matrix, rows: &[usize], cols: &[u32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * cols.len());
    for &r in rows {
        for &c in cols {
            out.push(x.at(r, c as usize));
        }
    }
    out
}

pub fn labels_u8(y: &[f32], rows: &[usize]) -> Vec<u8> {
    rows.iter()
        .map(|&i| if y[i] > 0.5 { 1 } else { 0 })
        .collect()
}

pub fn labels_f32(y: &[f32], rows: &[usize]) -> Vec<f32> {
    rows.iter().map(|&i| y[i]).collect()
}

pub fn write_context(
    path: &Path,
    x: &Matrix,
    y: &[f32],
    selected_cols: &[u32],
    max_rows: usize,
    seed: u64,
    task: TaskKind,
) -> Result<StarforgeContext> {
    let rows = context_rows_for_task(y, max_rows, seed, task);
    let context = StarforgeContext {
        task: StarforgeContextTask::from_task(task),
        selected_cols: selected_cols.to_vec(),
        n_rows: rows.len(),
        n_features: selected_cols.len(),
        x: matrix_rows(x, &rows, selected_cols),
        y: match task {
            TaskKind::Regression => labels_f32(y, &rows),
            _ => labels_u8(y, &rows).into_iter().map(f32::from).collect(),
        },
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(path, serde_json::to_string(&context)? + "\n")
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(context)
}

pub fn read_context(path: &Path) -> Result<StarforgeContext> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let ctx: StarforgeContext =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    if ctx.x.len() != ctx.n_rows * ctx.n_features || ctx.y.len() != ctx.n_rows {
        bail!("invalid Starforge context shape in {}", path.display());
    }
    Ok(ctx)
}
