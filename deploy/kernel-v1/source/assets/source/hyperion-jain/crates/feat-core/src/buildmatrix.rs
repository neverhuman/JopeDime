//! Assemble the augmented feature matrix = base columns + GP-survivor-evaluated columns, for both
//! train and holdout, with train-median NaN/inf fill (applied identically to both). Output is the
//! canonical column-major matrix all later phases index into.
use crate::canary::{self, CanarySpec};
use crate::data::{Dataset, Matrix};
use crate::dsl::Expr;
use crate::gp::SurvivorRec;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureInfo {
    pub source: String, // "base" | "gp"
    pub formula: String,
    pub expr: Expr,
    pub fill: f32,
    #[serde(default)]
    pub scale_min: f32,
    #[serde(default = "default_scale_span")]
    pub scale_span: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canary: Option<CanarySpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub canary_refs: Vec<CanarySpec>,
}

fn default_scale_span() -> f32 {
    1.0
}

pub struct Augmented {
    pub train: Matrix,
    pub hold: Matrix,
    pub features: Vec<FeatureInfo>,
}

fn nanmedian(v: &[f32]) -> f32 {
    let mut f: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if f.is_empty() {
        return 0.0;
    }
    f.sort_by(|a, b| a.partial_cmp(b).unwrap());
    f[f.len() / 2]
}

/// Build the augmented train/holdout matrices. `max_gp` caps how many survivors to include.
pub fn build_augmented(
    train: &Dataset,
    hold: &Dataset,
    survivors: &[SurvivorRec],
    max_gp: usize,
) -> Augmented {
    build_augmented_with_base_features(train, hold, survivors, max_gp, None)
}

/// Build the augmented matrix, preserving caller-supplied metadata for the base coordinate frame.
pub fn build_augmented_with_base_features(
    train: &Dataset,
    hold: &Dataset,
    survivors: &[SurvivorRec],
    max_gp: usize,
    base_features: Option<&[FeatureInfo]>,
) -> Augmented {
    let n = train.n();
    let nh = hold.n();
    let base_p = train.p();
    let take = survivors.len().min(max_gp);
    let p = base_p + take;
    if let Some(base_features) = base_features {
        assert_eq!(
            base_features.len(),
            base_p,
            "base feature metadata length mismatch"
        );
    }

    let mut features = Vec::with_capacity(p);
    let mut tr_cols: Vec<Vec<f32>> = Vec::with_capacity(p);
    let mut ho_cols: Vec<Vec<f32>> = Vec::with_capacity(p);

    for j in 0..base_p {
        tr_cols.push(train.x.col(j).to_vec());
        ho_cols.push(hold.x.col(j).to_vec());
        if let Some(base_features) = base_features {
            features.push(base_features[j].clone());
        } else {
            features.push(FeatureInfo {
                source: "base".into(),
                formula: train.names[j].clone(),
                expr: Expr::Col(j as u16),
                fill: 0.0,
                scale_min: 0.0,
                scale_span: 1.0,
                canary: None,
                canary_refs: Vec::new(),
            });
        }
    }

    // evaluate GP survivor exprs on train + hold (parallel; each is independent)
    let evald: Vec<(Vec<f32>, Vec<f32>)> = survivors[..take]
        .par_iter()
        .map(|s| (s.expr.eval(&train.x), s.expr.eval(&hold.x)))
        .collect();
    for (k, (tv, hv)) in evald.into_iter().enumerate() {
        tr_cols.push(tv);
        ho_cols.push(hv);
        features.push(FeatureInfo {
            source: "gp".into(),
            formula: survivors[k].formula.clone(),
            expr: survivors[k].expr.clone(),
            fill: 0.0,
            scale_min: 0.0,
            scale_span: 1.0,
            canary: None,
            canary_refs: Vec::new(),
        });
    }

    // train-median fill for NaN/inf, then train-fit min-max scale, applied to both train and holdout.
    for j in 0..p {
        let med = nanmedian(&tr_cols[j]);
        let fill = if med.is_finite() { med } else { 0.0 };
        features[j].fill = fill;
        for v in tr_cols[j].iter_mut() {
            if !v.is_finite() {
                *v = fill;
            }
        }
        for v in ho_cols[j].iter_mut() {
            if !v.is_finite() {
                *v = fill;
            }
        }
        let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in &tr_cols[j] {
            if v.is_finite() {
                min = min.min(v);
                max = max.max(v);
            }
        }
        if !min.is_finite() || !max.is_finite() || (max - min).abs() <= 1e-12 {
            features[j].scale_min = if min.is_finite() { min } else { 0.0 };
            features[j].scale_span = 1.0;
            for v in tr_cols[j].iter_mut().chain(ho_cols[j].iter_mut()) {
                *v = 0.0;
            }
        } else {
            let span = max - min;
            features[j].scale_min = min;
            features[j].scale_span = span;
            for v in tr_cols[j].iter_mut().chain(ho_cols[j].iter_mut()) {
                *v = ((*v - min) / span).clamp(0.0, 1.0);
            }
        }
    }

    let mut tflat = Vec::with_capacity(n * p);
    for c in &tr_cols {
        tflat.extend_from_slice(c);
    }
    let mut hflat = Vec::with_capacity(nh * p);
    for c in &ho_cols {
        hflat.extend_from_slice(c);
    }
    Augmented {
        train: Matrix::new(tflat, n, p),
        hold: Matrix::new(hflat, nh, p),
        features,
    }
}

/// Build the model matrix from a compact visible base while letting GP survivors evaluate against a
/// wider search matrix (for example original + Hyperion + CANARY_FEATURES terminals). Search-only
/// columns are not added as standalone trainable features.
pub fn build_augmented_from_search(
    visible_train: &Dataset,
    visible_hold: &Dataset,
    search_train: &Dataset,
    search_hold: &Dataset,
    survivors: &[SurvivorRec],
    max_gp: usize,
    base_features: &[FeatureInfo],
    canary_specs: &[CanarySpec],
) -> Augmented {
    let n = visible_train.n();
    let nh = visible_hold.n();
    let base_p = visible_train.p();
    assert_eq!(
        base_features.len(),
        base_p,
        "base feature metadata length mismatch"
    );
    let take = survivors.len().min(max_gp);
    let cap = base_p + take;

    let mut features = Vec::with_capacity(cap);
    let mut tr_cols: Vec<Vec<f32>> = Vec::with_capacity(cap);
    let mut ho_cols: Vec<Vec<f32>> = Vec::with_capacity(cap);

    for (j, feature) in base_features.iter().enumerate().take(base_p) {
        tr_cols.push(visible_train.x.col(j).to_vec());
        ho_cols.push(visible_hold.x.col(j).to_vec());
        features.push(feature.clone());
    }

    let evald: Vec<(Vec<f32>, Vec<f32>)> = survivors[..take]
        .par_iter()
        .map(|s| (s.expr.eval(&search_train.x), s.expr.eval(&search_hold.x)))
        .collect();
    for (k, (tv, hv)) in evald.into_iter().enumerate() {
        let mut refs = Vec::new();
        survivors[k].expr.col_refs(&mut refs);
        // CANARY quarantine: a survivor that references a negative-control sentinel column is dropped
        // here — before feature reduction — so it can never become a selectable feature. This is the
        // hard guarantee that 0 canary-derived features reach the final selection.
        if canary::expr_refs_canary(&refs, canary_specs) {
            continue;
        }
        tr_cols.push(tv);
        ho_cols.push(hv);
        features.push(FeatureInfo {
            source: "gp".into(),
            formula: survivors[k].formula.clone(),
            expr: survivors[k].expr.clone(),
            fill: 0.0,
            scale_min: 0.0,
            scale_span: 1.0,
            canary: None,
            // Quarantined above, so a retained GP feature never references a canary.
            canary_refs: Vec::new(),
        });
    }

    // Actual width after quarantine (may be < base_p + take when canary survivors were dropped).
    let p = features.len();
    finalize_columns(n, nh, p, tr_cols, ho_cols, features)
}

fn finalize_columns(
    n: usize,
    nh: usize,
    p: usize,
    mut tr_cols: Vec<Vec<f32>>,
    mut ho_cols: Vec<Vec<f32>>,
    mut features: Vec<FeatureInfo>,
) -> Augmented {
    // train-median fill for NaN/inf, then train-fit min-max scale, applied to both train and holdout.
    for j in 0..p {
        let med = nanmedian(&tr_cols[j]);
        let fill = if med.is_finite() { med } else { 0.0 };
        features[j].fill = fill;
        for v in tr_cols[j].iter_mut() {
            if !v.is_finite() {
                *v = fill;
            }
        }
        for v in ho_cols[j].iter_mut() {
            if !v.is_finite() {
                *v = fill;
            }
        }
        let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in &tr_cols[j] {
            if v.is_finite() {
                min = min.min(v);
                max = max.max(v);
            }
        }
        if !min.is_finite() || !max.is_finite() || (max - min).abs() <= 1e-12 {
            features[j].scale_min = if min.is_finite() { min } else { 0.0 };
            features[j].scale_span = 1.0;
            for v in tr_cols[j].iter_mut().chain(ho_cols[j].iter_mut()) {
                *v = 0.0;
            }
        } else {
            let span = max - min;
            features[j].scale_min = min;
            features[j].scale_span = span;
            for v in tr_cols[j].iter_mut().chain(ho_cols[j].iter_mut()) {
                *v = ((*v - min) / span).clamp(0.0, 1.0);
            }
        }
    }

    let mut tflat = Vec::with_capacity(n * p);
    for c in &tr_cols {
        tflat.extend_from_slice(c);
    }
    let mut hflat = Vec::with_capacity(nh * p);
    for c in &ho_cols {
        hflat.extend_from_slice(c);
    }
    Augmented {
        train: Matrix::new(tflat, n, p),
        hold: Matrix::new(hflat, nh, p),
        features,
    }
}
