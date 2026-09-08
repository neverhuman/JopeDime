//! Versioned structural operators for compatibility pipelines. Classification
//! is shared by construction and validation; old evolve::* streams stay frozen.
//! This is structural admission, not an algorithm resource/capability receipt.
use super::{Error, Result};
use crate::{
    dsl::{DslExpr, ParamValue, Stage},
    math_expr::{ExpressionScope, MathExpr},
    Pcg32,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageClass {
    Target,
    Input,
    Transform,
    Learner,
}

/// Stage classification for every currently executable legacy/v3 operation.
/// Stable names include aliases because their historical spelling is retained.
pub fn stage_class(name: &str) -> Option<StageClass> {
    Some(match name {
        "target" => StageClass::Target,
        "input" => StageClass::Input,
        "ridge"
        | "logistic"
        | "ridge_gcv"
        | "elasticnet"
        | "histgbm"
        | "gbm"
        | "ensemble"
        | "ensemble_ridge_gbm"
        | "histgbm2"
        | "krr"
        | "gpr"
        | "ksmooth"
        | "hurdle2"
        | "stack"
        | "portfolio"
        | "logit_ridge"
        | "tab_boost"
        | "extra_trees"
        | "gam_boost"
        | "pls"
        | "multiclass_ridge"
        | "multiclass_ovr_logistic"
        | "multiclass_diagonal_lda"
        | "multiclass_nearest_centroid" => StageClass::Learner,
        "standardize" | "norm_features" | "robust_scale" | "rank" | "blom" | "quantile_bins"
        | "bin_onehot" | "hinge" | "winsor_clip" | "winsor_moments" | "cheb" | "poly_interact"
        | "rff_cos" | "rff_sin" | "rand_proj" | "landmark_dist" | "oblique_hinge" | "row_stats"
        | "pair_prod" | "target_bin_encode" | "sigmoid" | "step" | "norm_row" | "polar" | "lag"
        | "roll" | "fft_mag" | "row_spectral" | "row_diff_stats" | "svd_project" | "nystrom"
        | "map" | "derive" | "symlog" | "select_corr" | "impute" | "pair_ratio" | "spline"
        | "maxabs_scale" | "power_transform" | "piecewise_linear" | "corr_screen"
        | "pls_project" => StageClass::Transform,
        _ => return None,
    })
}

#[derive(Clone, Debug)]
pub struct PipelineSlots {
    pub target: Option<Stage>,
    pub input: Option<Stage>,
    pub transforms: Vec<Stage>,
    pub learner: Stage,
}

impl PipelineSlots {
    pub fn from_pipeline(pipe: &DslExpr) -> Result<Self> {
        let mut result = Self {
            target: None,
            input: None,
            transforms: Vec::new(),
            learner: Stage::new("ridge"),
        };
        for (i, stage) in pipe.stages.iter().enumerate() {
            match stage_class(&stage.op).ok_or_else(|| Error::Unsupported(stage.op.clone()))? {
                StageClass::Target if i == 0 => result.target = Some(stage.clone()),
                StageClass::Input if i == usize::from(result.target.is_some()) => {
                    result.input = Some(stage.clone())
                }
                StageClass::Transform => result.transforms.push(stage.clone()),
                StageClass::Learner if i + 1 == pipe.stages.len() => result.learner = stage.clone(),
                _ => return Err(Error::Shape(format!("misplaced {} stage at {i}", stage.op))),
            }
        }
        Ok(result)
    }

    pub fn to_pipeline(&self) -> Result<DslExpr> {
        let mut stages = Vec::new();
        if let Some(stage) = &self.target {
            stages.push(stage.clone());
        }
        if let Some(stage) = &self.input {
            stages.push(stage.clone());
        }
        stages.extend(self.transforms.iter().cloned());
        stages.push(self.learner.clone());
        let pipe = DslExpr { stages };
        // Public fields never bypass the constructor's slot contracts.
        let checked = Self::from_pipeline(&pipe)?;
        if checked.transforms.len() != self.transforms.len()
            || self
                .target
                .as_ref()
                .is_some_and(|s| stage_class(&s.op) != Some(StageClass::Target))
            || self
                .input
                .as_ref()
                .is_some_and(|s| stage_class(&s.op) != Some(StageClass::Input))
            || stage_class(&self.learner.op) != Some(StageClass::Learner)
        {
            return Err(Error::Shape("invalid pipeline slot class".into()));
        }
        Ok(pipe)
    }

    /// Legacy row-local width semantics, including append/truncate and the
    /// one-zero-column fallback. Data-dependent widths use a conservative
    /// interval; every referenced column must exist at its lower bound.
    pub fn validate_legacy_widths(
        &self,
        rows: usize,
        columns: usize,
        max_stages: usize,
    ) -> Result<()> {
        if rows == 0 || columns == 0 || columns > crate::MAX_WIDTH {
            return Err(Error::Shape(
                "legacy pipeline needs nonempty bounded matrix dimensions".into(),
            ));
        }
        let pipe = self.to_pipeline()?;
        if pipe.stages.len() > max_stages {
            return Err(Error::Resource("pipeline stage count".into()));
        }
        if self.input.is_some() {
            return Err(Error::Unsupported(
                "input encoding needs a table schema binding".into(),
            ));
        }
        let mut width = (columns, columns);
        for stage in &self.transforms {
            if matches!(stage.op.as_str(), "map" | "derive") {
                let Some(ParamValue::Str(source)) = stage.params.get("expr") else {
                    return Err(Error::Syntax(format!(
                        "{} needs a string expression",
                        stage.op
                    )));
                };
                MathExpr::parse(source)
                    .and_then(|e| e.validate_columns(width.0, stage.op == "map"))
                    .map_err(|e| Error::Shape(e.to_string()))?;
            }
            width = next_width(stage, rows, width)?;
        }
        Ok(())
    }

    /// A new stream and fixed generation profile. Target/input slots are never
    /// moved into feature transforms, and modern learners remain terminal.
    pub fn mutate_v1(
        &self,
        seed: u64,
        rows: usize,
        columns: usize,
        max_stages: usize,
    ) -> Result<Self> {
        self.validate_legacy_widths(rows, columns, max_stages)?;
        let mut rng = Pcg32::new(seed, 0x706970657631);
        let mut candidates = Vec::new();
        let position = rng.next_u32() as usize % (self.transforms.len() + 1);
        for name in [
            "standardize",
            "robust_scale",
            "sigmoid",
            "norm_row",
            "symlog",
            "step",
            "row_stats",
        ] {
            let mut child = self.clone();
            let mut stage = Stage::new(name);
            if name == "step" {
                stage
                    .params
                    .insert("t".into(), ParamValue::Float(rng.range_f64(-1.0, 1.0)));
            }
            child.transforms.insert(position, stage);
            candidates.push(child);
        }
        if position < self.transforms.len() {
            let mut child = self.clone();
            child.transforms.remove(position);
            candidates.push(child);
            let stage = &self.transforms[position];
            for (key, value) in &stage.params {
                if let ParamValue::Float(v) = value {
                    let mut child = self.clone();
                    let next = *v * rng.range_f64(0.5, 1.5);
                    if next.is_finite() {
                        child.transforms[position]
                            .params
                            .insert(key.clone(), ParamValue::Float(next));
                        candidates.push(child);
                    }
                }
            }
            if matches!(stage.op.as_str(), "map" | "derive") {
                let mut width = (columns, columns);
                for previous in &self.transforms[..position] {
                    width = next_width(previous, rows, width)?;
                }
                if let Some(ParamValue::Str(source)) = stage.params.get("expr") {
                    let expr = MathExpr::parse(source).map_err(|e| Error::Syntax(e.to_string()))?;
                    let scope = ExpressionScope {
                        columns: width.0,
                        elementwise: stage.op == "map",
                        max_nodes: 64,
                        max_depth: 16,
                    };
                    let edited = expr
                        .mutate_v1(seed, scope)
                        .map_err(|e| Error::Syntax(e.to_string()))?;
                    let mut child = self.clone();
                    child.transforms[position].params.insert(
                        "expr".into(),
                        ParamValue::Str(edited.expression.to_source()?),
                    );
                    candidates.push(child);
                }
            }
        }
        let valid: Vec<_> = candidates
            .into_iter()
            .filter(|child| {
                child
                    .validate_legacy_widths(rows, columns, max_stages)
                    .is_ok()
            })
            .collect();
        if valid.is_empty() {
            return Ok(self.clone());
        }
        Ok(valid[rng.next_u32() as usize % valid.len()].clone())
    }

    pub fn crossover_v1(
        &self,
        donor: &Self,
        seed: u64,
        rows: usize,
        columns: usize,
        max_stages: usize,
    ) -> Result<Self> {
        self.validate_legacy_widths(rows, columns, max_stages)?;
        let mut rng = Pcg32::new(seed, 0x7069706563726f73);
        let cut = rng.next_u32() as usize % (self.transforms.len() + 1);
        let mut valid = Vec::new();
        for other in 0..=donor.transforms.len() {
            let mut child = self.clone();
            child.transforms.truncate(cut);
            child
                .transforms
                .extend(donor.transforms[other..].iter().cloned());
            if child
                .validate_legacy_widths(rows, columns, max_stages)
                .is_ok()
            {
                valid.push(child);
            }
        }
        if valid.is_empty() {
            return Ok(self.clone());
        }
        Ok(valid[rng.next_u32() as usize % valid.len()].clone())
    }
}

fn next_width(stage: &Stage, rows: usize, (lo, hi): (usize, usize)) -> Result<(usize, usize)> {
    let mul = |a: usize, b: usize| {
        a.checked_mul(b)
            .ok_or_else(|| Error::Resource("width multiplication".into()))
    };
    let generated = match stage.op.as_str() {
        "standardize" | "norm_features" | "robust_scale" | "rank" | "blom" | "quantile_bins"
        | "winsor_clip" | "winsor_moments" | "target_bin_encode" | "sigmoid" | "step"
        | "norm_row" | "polar" | "lag" | "roll" | "map" | "symlog" => (lo, hi),
        "bin_onehot" | "hinge" | "cheb" => {
            let k = match stage.op.as_str() {
                "bin_onehot" => stage.get_usize("bins", 6).clamp(2, 32),
                "hinge" => stage.get_usize("knots", 4).clamp(1, 16),
                _ => stage.get_usize("degree", 4).clamp(1, 8),
            };
            (mul(lo, k)?, mul(hi, k)?)
        }
        "poly_interact" => (mul(lo, lo + 1)? / 2, mul(hi, hi + 1)? / 2),
        "rff_cos" | "rff_sin" | "oblique_hinge" => {
            let n = stage.get_usize("n", 32).clamp(1, 256);
            (n, n)
        }
        "rand_proj" => {
            let n = stage.get_usize("k", 16).clamp(1, 256);
            (n, n)
        }
        "landmark_dist" => {
            let n = stage.get_usize("k", 16).clamp(1, 128);
            (n, n)
        }
        "nystrom" => {
            let n = stage.get_usize("k", 16).clamp(1, 128).min(rows);
            (n, n)
        }
        "row_stats" => (10, 10),
        "row_spectral" => (5, 5),
        "row_diff_stats" => (8, 8),
        "derive" => (1, 1),
        "fft_mag" => {
            let n = stage.get_usize("k", 8).max(1);
            (n.min((lo / 2).max(1)), n.min((hi / 2).max(1)))
        }
        "select_corr" | "svd_project" => {
            let n = if stage.op == "select_corr" {
                stage.get_usize("k", 32).clamp(1, 512)
            } else {
                stage.get_usize("k", 8).clamp(1, 64)
            };
            (lo.min(n), hi.min(n))
        }
        "pair_prod" => {
            let k = stage.get_usize("topk", 8).clamp(2, 24);
            let a = k.min(lo);
            let b = k.min(hi);
            (
                mul(a, a.saturating_sub(1))? / 2,
                mul(b, b.saturating_sub(1))? / 2,
            )
        }
        "pair_ratio" => (0, stage.get_usize("topk", 8).clamp(1, 64)),
        "spline" => (lo, mul(hi, stage.get_usize("knots", 5).clamp(3, 12) - 1)?),
        "impute" => {
            let indicator = matches!(
                stage.params.get("add_indicator"),
                Some(ParamValue::Bool(true))
            ) || stage.get_usize("add_indicator", 0) == 1;
            (lo, if indicator { mul(hi, 2)? } else { hi })
        }
        _ => {
            return Err(Error::Unsupported(format!(
                "{} needs table-v3 width binding",
                stage.op
            )))
        }
    };
    let append = matches!(stage.params.get("append"), Some(ParamValue::Bool(true)))
        || stage.get_usize("append", 0) == 1;
    let bounded = |width: usize, input: usize| -> Result<usize> {
        Ok(if append {
            width
                .checked_add(input)
                .ok_or_else(|| Error::Resource("append width".into()))?
        } else {
            width
        }
        .clamp(1, crate::MAX_WIDTH))
    };
    Ok((bounded(generated.0, lo)?, bounded(generated.1, hi)?))
}
