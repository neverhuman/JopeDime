//! Opt-in full Rust-emittable grammar for test-adaptive campaigns.
//!
//! The historical GP in [`crate::evolve`] deliberately searches a small,
//! inexpensive subset.  This module is additive: it does not alter that RNG
//! stream or any legacy constants, but exposes every canonical pipeline
//! operation that both the interpreter and Rust emitter support.

use crate::dsl::{DslExpr, ParamValue, Stage};
use crate::rng::Pcg32;
use crate::transpile::{emit, EmitOptions};
use crate::GenomeIr;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const MAX_TRANSFORMS: usize = 6;
pub const MAX_EXPRESSION_DEPTH: usize = 3;
pub const MAX_EXPRESSION_NODES: usize = 15;

pub const TRANSFORMS: [&str; 31] = [
    "standardize",
    "robust_scale",
    "rank",
    "blom",
    "quantile_bins",
    "bin_onehot",
    "hinge",
    "winsor_clip",
    "cheb",
    "poly_interact",
    "rff_cos",
    "rff_sin",
    "rand_proj",
    "landmark_dist",
    "oblique_hinge",
    "row_stats",
    "pair_prod",
    "target_bin_encode",
    "sigmoid",
    "step",
    "norm_row",
    "polar",
    "lag",
    "roll",
    "fft_mag",
    "row_spectral",
    "row_diff_stats",
    "svd_project",
    "nystrom",
    "map",
    "derive",
];

/// Expansion vocabulary: canonicalizable and schema-described like
/// [`TRANSFORMS`], but deliberately kept OUT of the legacy draw arrays so the
/// historical random-generation RNG stream (and every recorded test-adaptive
/// replay) is untouched. Search-side generation opts in separately.
pub const EXPANSION_TRANSFORMS: [&str; 5] =
    ["symlog", "select_corr", "impute", "pair_ratio", "spline"];

pub const HEADS: [&str; 11] = [
    "ridge",
    "ridge_gcv",
    "elasticnet",
    "histgbm",
    "ensemble",
    "histgbm2",
    "krr",
    "gpr",
    "ksmooth",
    "stack",
    "portfolio",
];

pub const TARGET_KINDS: [&str; 3] = ["rank_gauss", "log1p", "winsor"];

/// Kinds accepted by canonicalization/validation. Superset of the seeded
/// `TARGET_KINDS` generation array, which must never grow (extending it would
/// remap every recorded random-generation stream); expansion kinds enter
/// search through starters/seeds instead.
pub const VALID_TARGET_KINDS: [&str; 4] = ["rank_gauss", "log1p", "winsor", "zscore"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationKind {
    Transform,
    Head,
    Target,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ParamDomain {
    Numeric(Vec<ParamValue>),
    Categorical(Vec<&'static str>),
    Boolean,
    Expression,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParamSchema {
    pub name: &'static str,
    pub domain: ParamDomain,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OperationSchema {
    pub op: &'static str,
    pub kind: OperationKind,
    pub params: Vec<ParamSchema>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalGenome {
    pub canonical: String,
    pub semantic_sha256: String,
    pub complexity: usize,
    pub rust_source_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvolvedGenome {
    pub canonical: String,
    pub semantic_sha256: String,
    pub complexity: usize,
    pub rust_source_len: usize,
}

fn ints(values: &[i64]) -> ParamDomain {
    ParamDomain::Numeric(values.iter().copied().map(ParamValue::Int).collect())
}

fn floats(values: &[f64]) -> ParamDomain {
    ParamDomain::Numeric(values.iter().copied().map(ParamValue::Float).collect())
}

fn numeric(name: &'static str, domain: ParamDomain) -> ParamSchema {
    ParamSchema { name, domain }
}

fn boolean(name: &'static str) -> ParamSchema {
    ParamSchema {
        name,
        domain: ParamDomain::Boolean,
    }
}

fn categorical(name: &'static str, values: Vec<&'static str>) -> ParamSchema {
    ParamSchema {
        name,
        domain: ParamDomain::Categorical(values),
    }
}

/// Explicit parameter domains. Existing search grids are retained where one
/// exists; otherwise numeric domains contain minimum, default, midpoint and
/// maximum values from the interpreter/emitter clamps.
pub fn operation_schemas() -> Vec<OperationSchema> {
    let mut out = Vec::new();
    for op in TRANSFORMS {
        let mut params = match op {
            "quantile_bins" => vec![numeric("bins", ints(&[2, 8, 33, 64]))],
            "bin_onehot" => vec![numeric("bins", ints(&[2, 6, 17, 32]))],
            "hinge" => vec![numeric("knots", ints(&[1, 4, 8, 16]))],
            "winsor_clip" => vec![numeric("q", floats(&[0.0, 0.05, 0.125, 0.25]))],
            "cheb" => vec![numeric("degree", ints(&[1, 4, 5, 8]))],
            "rff_cos" | "rff_sin" => vec![
                numeric("n", ints(&[1, 32, 128, 256])),
                numeric("gamma", floats(&[0.1, 0.5, 1.0, 2.0])),
                numeric("seed", ints(&[1, 7, 5000, 9999])),
            ],
            "rand_proj" => vec![
                numeric("k", ints(&[1, 16, 128, 256])),
                numeric("seed", ints(&[1, 7, 5000, 9999])),
            ],
            "landmark_dist" | "nystrom" => vec![
                numeric("k", ints(&[1, 16, 64, 128])),
                numeric("gamma", floats(&[0.1, 0.5, 1.0, 2.0])),
                numeric("seed", ints(&[1, 7, 5000, 9999])),
            ],
            "oblique_hinge" => vec![
                numeric("n", ints(&[1, 32, 128, 256])),
                numeric("seed", ints(&[1, 7, 5000, 9999])),
            ],
            "pair_prod" => vec![numeric("topk", ints(&[2, 8, 13, 24]))],
            "target_bin_encode" => vec![
                numeric("bins", ints(&[2, 8, 17, 32])),
                numeric("prior", floats(&[0.0, 10.0, 50.0, 100.0])),
            ],
            "step" => vec![numeric("t", floats(&[-1.0, 0.0, 0.5, 1.0]))],
            "lag" | "roll" => vec![numeric("k", ints(&[1, 2, 25, 50]))],
            "fft_mag" => vec![numeric("k", ints(&[1, 8, 16, 25]))],
            "svd_project" => vec![
                numeric("k", ints(&[1, 8, 32, 64])),
                numeric("iters", ints(&[1, 20, 100, 200])),
                numeric("seed", ints(&[1, 7, 5000, 9999])),
            ],
            "map" | "derive" => vec![ParamSchema {
                name: "expr",
                domain: ParamDomain::Expression,
            }],
            _ => Vec::new(),
        };
        params.push(boolean("append"));
        out.push(OperationSchema {
            op,
            kind: OperationKind::Transform,
            params,
        });
    }
    for op in EXPANSION_TRANSFORMS {
        let mut params = match op {
            "symlog" => vec![numeric("c", floats(&[0.1, 1.0, 10.0, 100.0]))],
            "select_corr" => vec![
                numeric("k", ints(&[1, 16, 64, 512])),
                categorical("by", vec!["corr", "var"]),
            ],
            "impute" => vec![
                categorical("strategy", vec!["mean", "median", "const"]),
                numeric("value", floats(&[-1.0, 0.0, 0.5, 1.0])),
                boolean("add_indicator"),
            ],
            "pair_ratio" => vec![numeric("topk", ints(&[1, 4, 8, 64]))],
            "spline" => vec![numeric("knots", ints(&[3, 4, 5, 12]))],
            _ => Vec::new(),
        };
        params.push(boolean("append"));
        out.push(OperationSchema {
            op,
            kind: OperationKind::Transform,
            params,
        });
    }
    out.push(OperationSchema {
        op: "target",
        kind: OperationKind::Target,
        params: vec![
            categorical("kind", TARGET_KINDS.to_vec()),
            numeric("q", floats(&[0.0, 0.05, 0.125, 0.25])),
        ],
    });
    for op in HEADS {
        let params = match op {
            "ridge" => vec![numeric("lambda_frac", floats(&[0.0, 0.02, 0.25, 0.5]))],
            "elasticnet" => vec![
                numeric("alpha", floats(&[0.0001, 0.001, 0.025, 0.05])),
                numeric("l1_ratio", floats(&[0.0, 0.5, 0.75, 1.0])),
                numeric("sweeps", ints(&[1, 60, 100, 200])),
            ],
            "histgbm" | "ensemble" => vec![
                numeric("rounds", ints(&[1, 150, 250, 500])),
                numeric("depth", ints(&[1, 3, 4, 8])),
                numeric("lr", floats(&[0.01, 0.1, 0.25, 0.5])),
                numeric("lambda", floats(&[0.0, 1.0, 5.0, 10.0])),
                numeric("bins", ints(&[2, 64, 128, 255])),
                numeric("min_leaf", ints(&[1, 10, 25, 50])),
                numeric("max_features", ints(&[1, 50, 256, 512])),
                numeric("lambda_frac", floats(&[0.0, 0.02, 0.25, 0.5])),
            ],
            "histgbm2" => vec![
                numeric("leaves", ints(&[2, 31, 127, 255])),
                numeric("lr", floats(&[0.01, 0.05, 0.08, 0.2])),
                numeric("bags", ints(&[1, 3, 5, 10])),
                numeric("rounds", ints(&[1, 500, 750, 1000])),
                numeric("row_subsample", floats(&[0.5, 0.7, 0.85, 1.0])),
                numeric("col_subsample", floats(&[0.5, 0.8, 0.9, 1.0])),
                numeric("min_leaf", ints(&[1, 3, 10, 20])),
                numeric("val_pct", ints(&[5, 20, 30, 40])),
                boolean("val_shuffle"),
                categorical("loss", vec!["squared", "huber"]),
                numeric("delta", floats(&[0.000001, 0.1, 1.0, 1_000_000.0])),
                categorical("stop", vec!["rmse", "mae"]),
            ],
            "krr" => vec![numeric("m", ints(&[50, 800, 1025, 2000]))],
            "gpr" => vec![
                numeric("m", ints(&[50, 500, 1025, 2000])),
                numeric("steps", ints(&[1, 40, 100, 200])),
                categorical(
                    "kernel",
                    vec![
                        "rbf",
                        "linear",
                        "rq",
                        "periodic",
                        "rbf+linear",
                        "rq+linear",
                        "rbf*periodic",
                        "periodic+linear",
                    ],
                ),
            ],
            "ksmooth" => vec![
                numeric("k", ints(&[1, 32, 256, 512])),
                numeric("gamma", floats(&[0.000001, 1.0, 500_000.0, 1_000_000.0])),
            ],
            "portfolio" => vec![
                categorical("preset", vec!["core", "none"]),
                numeric("folds", ints(&[5])),
                categorical("repeats", vec!["auto", "1"]),
                categorical("blend", vec!["nnls"]),
            ],
            _ => Vec::new(),
        };
        out.push(OperationSchema {
            op,
            kind: OperationKind::Head,
            params,
        });
    }
    out
}

fn canonical_op(op: &str) -> &str {
    match op {
        "norm_features" => "standardize",
        "winsor_moments" => "winsor_clip",
        "logistic" => "ridge",
        "gbm" => "histgbm",
        "ensemble_ridge_gbm" => "ensemble",
        other => other,
    }
}

fn is_head(op: &str) -> bool {
    HEADS.contains(&canonical_op(op))
}

fn finite(stage: &Stage, key: &str, default: f64) -> Result<f64, String> {
    let value = stage.get_f64(key, default);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{}: {key} must be finite", stage.op))
    }
}

fn set_int(params: &mut BTreeMap<String, ParamValue>, key: &str, value: usize) {
    params.insert(key.to_string(), ParamValue::Int(value as i64));
}

fn set_float(params: &mut BTreeMap<String, ParamValue>, key: &str, value: f64) {
    params.insert(key.to_string(), ParamValue::Float(value));
}

fn append_true(stage: &Stage, params: &mut BTreeMap<String, ParamValue>) {
    if matches!(stage.params.get("append"), Some(ParamValue::Bool(true)))
        || stage.get_usize("append", 0) == 1
    {
        params.insert("append".to_string(), ParamValue::Bool(true));
    }
}

fn canonical_target(stage: &Stage) -> Result<Stage, String> {
    let kind = match stage.params.get("kind") {
        Some(ParamValue::Str(kind)) if VALID_TARGET_KINDS.contains(&kind.as_str()) => kind.clone(),
        _ => return Err("target requires kind=rank_gauss|log1p|winsor|zscore".to_string()),
    };
    let mut out = Stage::new("target");
    out.params
        .insert("kind".to_string(), ParamValue::Str(kind.clone()));
    if kind == "winsor" {
        set_float(
            &mut out.params,
            "q",
            finite(stage, "q", 0.05)?.clamp(0.0, 0.25),
        );
    }
    Ok(out)
}

fn canonical_transform(stage: &Stage) -> Result<Stage, String> {
    let op = canonical_op(&stage.op);
    if !TRANSFORMS.contains(&op) && !EXPANSION_TRANSFORMS.contains(&op) {
        return Err(format!("unsupported FullRust transform '{}'", stage.op));
    }
    let mut out = Stage::new(op);
    match op {
        "quantile_bins" => set_int(
            &mut out.params,
            "bins",
            stage.get_usize("bins", 8).clamp(2, 64),
        ),
        "bin_onehot" => set_int(
            &mut out.params,
            "bins",
            stage.get_usize("bins", 6).clamp(2, 32),
        ),
        "hinge" => set_int(
            &mut out.params,
            "knots",
            stage.get_usize("knots", 4).clamp(1, 16),
        ),
        "winsor_clip" => set_float(
            &mut out.params,
            "q",
            finite(stage, "q", 0.05)?.clamp(0.0, 0.25),
        ),
        "cheb" => set_int(
            &mut out.params,
            "degree",
            stage.get_usize("degree", 4).clamp(1, 8),
        ),
        "rff_cos" | "rff_sin" => {
            set_int(&mut out.params, "n", stage.get_usize("n", 32).clamp(1, 256));
            set_float(
                &mut out.params,
                "gamma",
                finite(stage, "gamma", 1.0)?.max(0.000001),
            );
            set_int(&mut out.params, "seed", stage.get_u64("seed", 7) as usize);
        }
        "rand_proj" => {
            set_int(&mut out.params, "k", stage.get_usize("k", 16).clamp(1, 256));
            set_int(&mut out.params, "seed", stage.get_u64("seed", 7) as usize);
        }
        "landmark_dist" | "nystrom" => {
            set_int(&mut out.params, "k", stage.get_usize("k", 16).clamp(1, 128));
            set_float(
                &mut out.params,
                "gamma",
                finite(stage, "gamma", 1.0)?.max(0.000001),
            );
            set_int(&mut out.params, "seed", stage.get_u64("seed", 7) as usize);
        }
        "oblique_hinge" => {
            set_int(&mut out.params, "n", stage.get_usize("n", 32).clamp(1, 256));
            set_int(&mut out.params, "seed", stage.get_u64("seed", 7) as usize);
        }
        "pair_prod" => set_int(
            &mut out.params,
            "topk",
            stage.get_usize("topk", 8).clamp(2, 24),
        ),
        "target_bin_encode" => {
            set_int(
                &mut out.params,
                "bins",
                stage.get_usize("bins", 8).clamp(2, 32),
            );
            set_float(
                &mut out.params,
                "prior",
                finite(stage, "prior", 10.0)?.max(0.0),
            );
        }
        "symlog" => set_float(
            &mut out.params,
            "c",
            finite(stage, "c", 1.0)?.clamp(1e-9, 1e9),
        ),
        "select_corr" => {
            set_int(&mut out.params, "k", stage.get_usize("k", 32).clamp(1, 512));
            let by = match stage.params.get("by") {
                Some(ParamValue::Str(s)) if s == "var" => "var",
                _ => "corr",
            };
            out.params
                .insert("by".to_string(), ParamValue::Str(by.to_string()));
        }
        "impute" => {
            let strategy = match stage.params.get("strategy") {
                Some(ParamValue::Str(s)) if s == "mean" => "mean",
                Some(ParamValue::Str(s)) if s == "const" => "const",
                _ => "median",
            };
            out.params.insert(
                "strategy".to_string(),
                ParamValue::Str(strategy.to_string()),
            );
            set_float(
                &mut out.params,
                "value",
                finite(stage, "value", 0.0)?.clamp(-1e12, 1e12),
            );
            let indicator = matches!(
                stage.params.get("add_indicator"),
                Some(ParamValue::Bool(true))
            ) || stage.get_usize("add_indicator", 0) == 1;
            if indicator {
                out.params
                    .insert("add_indicator".to_string(), ParamValue::Bool(true));
            }
        }
        "pair_ratio" => set_int(
            &mut out.params,
            "topk",
            stage.get_usize("topk", 8).clamp(1, 64),
        ),
        "spline" => set_int(
            &mut out.params,
            "knots",
            stage.get_usize("knots", 5).clamp(3, 12),
        ),
        "step" => set_float(&mut out.params, "t", finite(stage, "t", 0.5)?),
        "lag" | "roll" => set_int(&mut out.params, "k", stage.get_usize("k", 1).max(1)),
        "fft_mag" => set_int(&mut out.params, "k", stage.get_usize("k", 8)),
        "svd_project" => {
            set_int(&mut out.params, "k", stage.get_usize("k", 8).clamp(1, 64));
            set_int(
                &mut out.params,
                "iters",
                stage.get_usize("iters", 20).clamp(1, 200),
            );
            set_int(&mut out.params, "seed", stage.get_u64("seed", 7) as usize);
        }
        "map" | "derive" => match stage.params.get("expr") {
            Some(ParamValue::Str(expr)) => {
                crate::math_expr::MathExpr::parse(expr)
                    .map_err(|error| format!("{op}: {error}"))?;
                out.params
                    .insert("expr".to_string(), ParamValue::Str(expr.clone()));
            }
            _ => return Err(format!("{op} requires expr=\"...\"")),
        },
        _ => {}
    }
    append_true(stage, &mut out.params);
    Ok(out)
}

fn canonical_head(stage: &Stage) -> Result<Stage, String> {
    let op = canonical_op(&stage.op);
    if !HEADS.contains(&op) {
        return Err(format!("unsupported FullRust head '{}'", stage.op));
    }
    let mut out = Stage::new(op);
    match op {
        "ridge" => set_float(
            &mut out.params,
            "lambda_frac",
            finite(stage, "lambda_frac", 0.02)?,
        ),
        "elasticnet" => {
            set_float(&mut out.params, "alpha", finite(stage, "alpha", 0.001)?);
            set_float(&mut out.params, "l1_ratio", finite(stage, "l1_ratio", 0.5)?);
            set_int(&mut out.params, "sweeps", stage.get_usize("sweeps", 60));
        }
        "histgbm" | "ensemble" => {
            set_int(
                &mut out.params,
                "rounds",
                stage.get_usize("rounds", 150).clamp(1, 500),
            );
            set_int(
                &mut out.params,
                "depth",
                stage.get_usize("depth", 3).clamp(1, 8),
            );
            set_float(&mut out.params, "lr", finite(stage, "lr", 0.1)?);
            set_float(&mut out.params, "lambda", finite(stage, "lambda", 1.0)?);
            set_int(
                &mut out.params,
                "bins",
                stage.get_usize("bins", 64).clamp(2, 255),
            );
            set_int(
                &mut out.params,
                "min_leaf",
                stage.get_usize("min_leaf", 10).max(1),
            );
            set_int(
                &mut out.params,
                "max_features",
                stage.get_usize("max_features", 256).max(1),
            );
            if op == "ensemble" {
                set_float(
                    &mut out.params,
                    "lambda_frac",
                    finite(stage, "lambda_frac", 0.02)?,
                );
            }
        }
        "histgbm2" => {
            set_int(
                &mut out.params,
                "leaves",
                stage.get_usize("leaves", 31).clamp(2, 255),
            );
            set_float(&mut out.params, "lr", finite(stage, "lr", 0.05)?);
            set_int(
                &mut out.params,
                "bags",
                stage.get_usize("bags", 3).clamp(1, 10),
            );
            set_int(
                &mut out.params,
                "rounds",
                stage.get_usize("rounds", 500).clamp(1, 1000),
            );
            set_float(
                &mut out.params,
                "row_subsample",
                finite(stage, "row_subsample", 0.7)?.clamp(0.5, 1.0),
            );
            set_float(
                &mut out.params,
                "col_subsample",
                finite(stage, "col_subsample", 1.0)?.clamp(0.5, 1.0),
            );
            set_int(
                &mut out.params,
                "min_leaf",
                stage.get_usize("min_leaf", 3).clamp(1, 20),
            );
            set_int(
                &mut out.params,
                "val_pct",
                stage.get_usize("val_pct", 20).clamp(5, 40),
            );
            if matches!(
                stage.params.get("val_shuffle"),
                Some(ParamValue::Bool(true))
            ) {
                out.params
                    .insert("val_shuffle".to_string(), ParamValue::Bool(true));
            }
            if matches!(stage.params.get("loss"), Some(ParamValue::Str(v)) if v == "huber") {
                out.params
                    .insert("loss".to_string(), ParamValue::Str("huber".to_string()));
                set_float(
                    &mut out.params,
                    "delta",
                    finite(stage, "delta", 0.1)?.clamp(0.000001, 1_000_000.0),
                );
            }
            if matches!(stage.params.get("stop"), Some(ParamValue::Str(v)) if v == "mae") {
                out.params
                    .insert("stop".to_string(), ParamValue::Str("mae".to_string()));
            }
        }
        "krr" => set_int(
            &mut out.params,
            "m",
            stage.get_usize("m", 800).clamp(50, 2000),
        ),
        "gpr" => {
            set_int(
                &mut out.params,
                "m",
                stage.get_usize("m", 500).clamp(50, 2000),
            );
            set_int(
                &mut out.params,
                "steps",
                stage.get_usize("steps", 40).clamp(1, 200),
            );
            let kernel = match stage.params.get("kernel") {
                Some(ParamValue::Str(value)) => value.clone(),
                _ => "rbf".to_string(),
            };
            crate::gpr::parse_kernel(&kernel).map_err(|error| format!("gpr kernel: {error}"))?;
            out.params
                .insert("kernel".to_string(), ParamValue::Str(kernel));
        }
        "ksmooth" => {
            set_int(&mut out.params, "k", stage.get_usize("k", 32).clamp(1, 512));
            set_float(
                &mut out.params,
                "gamma",
                finite(stage, "gamma", 1.0)?.clamp(0.000001, 1_000_000.0),
            );
        }
        "portfolio" => {
            let (preset, adds, drop) = crate::members::compact_params(stage)
                .map_err(|error| format!("portfolio: {error}"))?;
            out.params
                .insert("preset".to_string(), ParamValue::Str(preset));
            out.params.insert("folds".to_string(), ParamValue::Int(5));
            out.params.insert(
                "repeats".to_string(),
                if crate::members::compact_repeats_auto(stage) {
                    ParamValue::Str("auto".to_string())
                } else {
                    ParamValue::Int(1)
                },
            );
            out.params
                .insert("blend".to_string(), ParamValue::Str("nnls".to_string()));
            for (index, add) in adds.into_iter().enumerate() {
                out.params
                    .insert(format!("add{}", index + 1), ParamValue::Str(add));
            }
            if let Some(drop) = drop {
                out.params
                    .insert("drop1".to_string(), ParamValue::Str(drop));
            }
            crate::members::resolve_compact(&out).map_err(|error| format!("portfolio: {error}"))?;
        }
        _ => {}
    }
    Ok(out)
}

pub fn canonicalize(source: &str) -> Result<CanonicalGenome, String> {
    let genome = GenomeIr::parse(source).map_err(|error| error.to_string())?;
    let GenomeIr::Pipeline(dsl) = genome else {
        return Err("FullRust accepts pipeline genomes only".to_string());
    };
    let mut index = 0;
    let mut stages = Vec::new();
    if dsl.stages.first().is_some_and(|stage| stage.op == "target") {
        stages.push(canonical_target(&dsl.stages[0])?);
        index = 1;
    }
    if dsl.stages[index..].iter().any(|stage| stage.op == "target") {
        return Err("target transform must be the first stage".to_string());
    }
    let rest = &dsl.stages[index..];
    let (transforms, head) = match rest.last() {
        Some(stage) if is_head(&stage.op) => (&rest[..rest.len() - 1], stage.clone()),
        _ => (rest, Stage::new("ridge")),
    };
    if transforms.len() > MAX_TRANSFORMS {
        return Err(format!(
            "FullRust allows at most {MAX_TRANSFORMS} feature transforms"
        ));
    }
    for transform in transforms {
        if is_head(&transform.op) {
            return Err("model heads are only valid as the final stage".to_string());
        }
        stages.push(canonical_transform(transform)?);
    }
    stages.push(canonical_head(&head)?);
    let canonical_ir = GenomeIr::Pipeline(DslExpr { stages });
    let canonical = canonical_ir.canonical();
    let emitted =
        emit(&canonical_ir, &EmitOptions::default()).map_err(|error| error.to_string())?;
    let semantic_sha256 = hex_sha256(canonical.as_bytes());
    let complexity = match &canonical_ir {
        GenomeIr::Pipeline(dsl) => dsl
            .stages
            .iter()
            .filter(|stage| stage.op != "target")
            .count(),
        _ => unreachable!(),
    };
    Ok(CanonicalGenome {
        canonical,
        semantic_sha256,
        complexity,
        rust_source_len: emitted.rust.len(),
    })
}

fn hex_sha256(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn choice<'a, T>(rng: &mut Pcg32, values: &'a [T]) -> &'a T {
    &values[(rng.next_u32() as usize) % values.len()]
}

fn schema_for(op: &str) -> OperationSchema {
    operation_schemas()
        .into_iter()
        .find(|schema| schema.op == op)
        .expect("canonical operation has a schema")
}

fn random_param(rng: &mut Pcg32, domain: &ParamDomain, feature_count: usize) -> ParamValue {
    match domain {
        ParamDomain::Numeric(values) => choice(rng, values).clone(),
        ParamDomain::Categorical(values) => ParamValue::Str((*choice(rng, values)).to_string()),
        ParamDomain::Boolean => ParamValue::Bool(rng.next_u32() & 1 == 1),
        ParamDomain::Expression => ParamValue::Str(random_expression(rng, feature_count)),
    }
}

fn random_param_expanded(
    rng: &mut Pcg32,
    domain: &ParamDomain,
    feature_count: usize,
    op: &str,
    param: &str,
) -> ParamValue {
    if op == "target" && param == "kind" {
        return ParamValue::Str((*choice(rng, &VALID_TARGET_KINDS)).to_string());
    }
    match domain {
        ParamDomain::Expression => {
            ParamValue::Str(random_expression_expanded(rng, feature_count, op == "map"))
        }
        _ => random_param(rng, domain, feature_count),
    }
}

fn random_stage(rng: &mut Pcg32, op: &str, feature_count: usize) -> Stage {
    let schema = schema_for(op);
    let mut stage = Stage::new(op);
    for param in schema.params {
        if param.name == "append" && rng.next_u32() & 1 == 0 {
            continue;
        }
        stage.params.insert(
            param.name.to_string(),
            random_param(rng, &param.domain, feature_count),
        );
    }
    if op == "portfolio"
        && matches!(stage.params.get("preset"), Some(ParamValue::Str(v)) if v == "none")
    {
        stage
            .params
            .insert("preset".to_string(), ParamValue::Str("core".to_string()));
    }
    stage
}

fn random_stage_expanded(rng: &mut Pcg32, op: &str, feature_count: usize) -> Stage {
    let schema = schema_for(op);
    let mut stage = Stage::new(op);
    for param in schema.params {
        if param.name == "append" && rng.next_u32() & 1 == 0 {
            continue;
        }
        stage.params.insert(
            param.name.to_string(),
            random_param_expanded(rng, &param.domain, feature_count, op, param.name),
        );
    }
    if op == "portfolio"
        && matches!(stage.params.get("preset"), Some(ParamValue::Str(v)) if v == "none")
    {
        stage
            .params
            .insert("preset".to_string(), ParamValue::Str("core".to_string()));
    }
    stage
}

fn expanded_transform(rng: &mut Pcg32) -> &'static str {
    let index = rng.next_u32() as usize % (TRANSFORMS.len() + EXPANSION_TRANSFORMS.len());
    if index < TRANSFORMS.len() {
        TRANSFORMS[index]
    } else {
        EXPANSION_TRANSFORMS[index - TRANSFORMS.len()]
    }
}

fn normalize_expanded_head(head: &mut Stage) {
    if head.op == "portfolio"
        && matches!(head.params.get("preset"), Some(ParamValue::Str(value)) if value == "none")
    {
        head.params
            .insert("preset".to_string(), ParamValue::Str("core".to_string()));
    }
}

fn split_pipeline(source: &str) -> Result<(Option<Stage>, Vec<Stage>, Stage), String> {
    let canonical = canonicalize(source)?;
    let GenomeIr::Pipeline(dsl) =
        GenomeIr::parse(&canonical.canonical).map_err(|error| error.to_string())?
    else {
        unreachable!()
    };
    let mut stages = dsl.stages;
    let target = if stages.first().is_some_and(|stage| stage.op == "target") {
        Some(stages.remove(0))
    } else {
        None
    };
    let head = stages.pop().expect("canonical pipeline has a head");
    Ok((target, stages, head))
}

fn assemble(target: Option<Stage>, mut transforms: Vec<Stage>, head: Stage) -> String {
    let mut stages = Vec::with_capacity(transforms.len() + 2);
    if let Some(target) = target {
        stages.push(target);
    }
    stages.append(&mut transforms);
    stages.push(head);
    GenomeIr::Pipeline(DslExpr { stages }).canonical()
}

pub fn random_pipeline(seed: u64, feature_count: usize) -> Result<EvolvedGenome, String> {
    let mut rng = Pcg32::new(seed, 0x4655_4c4c_5255_5354);
    let target = if rng.next_f64() < 0.2 {
        Some(random_stage(&mut rng, "target", feature_count))
    } else {
        None
    };
    let count = 1 + (rng.next_u32() as usize % MAX_TRANSFORMS);
    let transforms = (0..count)
        .map(|_| {
            let op = *choice(&mut rng, &TRANSFORMS);
            random_stage(&mut rng, op, feature_count)
        })
        .collect();
    let head_op = *choice(&mut rng, &HEADS);
    let head = random_stage(&mut rng, head_op, feature_count);
    canonicalize(&assemble(target, transforms, head)).map(Into::into)
}

pub fn mutate_pipeline(
    source: &str,
    seed: u64,
    feature_count: usize,
) -> Result<EvolvedGenome, String> {
    let (mut target, mut transforms, mut head) = split_pipeline(source)?;
    let mut rng = Pcg32::new(seed, 0x4d55_5441_5445);
    match rng.next_u32() % 7 {
        0 if transforms.len() < MAX_TRANSFORMS => {
            let op = *choice(&mut rng, &TRANSFORMS);
            let at = rng.next_u32() as usize % (transforms.len() + 1);
            transforms.insert(at, random_stage(&mut rng, op, feature_count));
        }
        1 if transforms.len() > 1 => {
            let at = rng.next_u32() as usize % transforms.len();
            transforms.remove(at);
        }
        2 if transforms.len() > 1 => {
            let a = rng.next_u32() as usize % transforms.len();
            let b = rng.next_u32() as usize % transforms.len();
            transforms.swap(a, b);
        }
        3 => {
            let op = *choice(&mut rng, &HEADS);
            head = random_stage(&mut rng, op, feature_count);
        }
        4 => {
            target = match target {
                None => Some(random_stage(&mut rng, "target", feature_count)),
                Some(_) if rng.next_u32() & 1 == 0 => None,
                Some(_) => Some(random_stage(&mut rng, "target", feature_count)),
            };
        }
        _ => {
            let mutate_head = transforms.is_empty() || (rng.next_u32() as usize).is_multiple_of(5);
            let stage = if mutate_head {
                &mut head
            } else {
                let at = rng.next_u32() as usize % transforms.len();
                &mut transforms[at]
            };
            let schema = schema_for(&stage.op);
            if schema.params.is_empty() {
                if !mutate_head {
                    let op = *choice(&mut rng, &TRANSFORMS);
                    *stage = random_stage(&mut rng, op, feature_count);
                } else {
                    let op = *choice(&mut rng, &HEADS);
                    *stage = random_stage(&mut rng, op, feature_count);
                }
            } else {
                let param = choice(&mut rng, &schema.params);
                stage.params.insert(
                    param.name.to_string(),
                    random_param(&mut rng, &param.domain, feature_count),
                );
            }
        }
    }
    canonicalize(&assemble(target, transforms, head)).map(Into::into)
}

/// Golden46-only mutation over the complete Rust-emittable vocabulary.
///
/// This is additive so every historical `mutate_pipeline` RNG stream remains
/// byte-identical. It can introduce the expansion transforms, the `zscore`
/// target, `x`-aware map expressions, and every accepted scalar math function.
pub fn mutate_pipeline_expanded(
    source: &str,
    seed: u64,
    feature_count: usize,
) -> Result<EvolvedGenome, String> {
    let (mut target, mut transforms, mut head) = split_pipeline(source)?;
    let mut rng = Pcg32::new(seed, 0x4d55_5441_5445_5632);
    match rng.next_u32() % 7 {
        0 if transforms.len() < MAX_TRANSFORMS => {
            let op = expanded_transform(&mut rng);
            let at = rng.next_u32() as usize % (transforms.len() + 1);
            transforms.insert(at, random_stage_expanded(&mut rng, op, feature_count));
        }
        1 if transforms.len() > 1 => {
            let at = rng.next_u32() as usize % transforms.len();
            transforms.remove(at);
        }
        2 if transforms.len() > 1 => {
            let a = rng.next_u32() as usize % transforms.len();
            let b = rng.next_u32() as usize % transforms.len();
            transforms.swap(a, b);
        }
        3 => {
            let op = *choice(&mut rng, &HEADS);
            head = random_stage_expanded(&mut rng, op, feature_count);
        }
        4 => {
            target = match target {
                None => Some(random_stage_expanded(&mut rng, "target", feature_count)),
                Some(_) if rng.next_u32() & 1 == 0 => None,
                Some(_) => Some(random_stage_expanded(&mut rng, "target", feature_count)),
            };
        }
        _ => {
            let mutate_head = transforms.is_empty() || (rng.next_u32() as usize).is_multiple_of(5);
            let stage = if mutate_head {
                &mut head
            } else {
                let at = rng.next_u32() as usize % transforms.len();
                &mut transforms[at]
            };
            let schema = schema_for(&stage.op);
            if schema.params.is_empty() {
                if !mutate_head {
                    let op = expanded_transform(&mut rng);
                    *stage = random_stage_expanded(&mut rng, op, feature_count);
                } else {
                    let op = *choice(&mut rng, &HEADS);
                    *stage = random_stage_expanded(&mut rng, op, feature_count);
                }
            } else {
                let param = choice(&mut rng, &schema.params);
                stage.params.insert(
                    param.name.to_string(),
                    random_param_expanded(
                        &mut rng,
                        &param.domain,
                        feature_count,
                        &stage.op,
                        param.name,
                    ),
                );
            }
        }
    }
    normalize_expanded_head(&mut head);
    canonicalize(&assemble(target, transforms, head)).map(Into::into)
}

pub fn crossover_pipelines(left: &str, right: &str, seed: u64) -> Result<EvolvedGenome, String> {
    let (left_target, left_transforms, left_head) = split_pipeline(left)?;
    let (right_target, right_transforms, right_head) = split_pipeline(right)?;
    let mut rng = Pcg32::new(seed, 0x4352_4f53_534f_5645);
    let a = rng.next_u32() as usize % (left_transforms.len() + 1);
    let b = rng.next_u32() as usize % (right_transforms.len() + 1);
    let mut transforms = left_transforms[..a].to_vec();
    transforms.extend_from_slice(&right_transforms[b..]);
    transforms.truncate(MAX_TRANSFORMS);
    if transforms.is_empty() {
        transforms.push(Stage::new("standardize"));
    }
    let target = if rng.next_u32() & 1 == 0 {
        left_target
    } else {
        right_target
    };
    let head = if rng.next_u32() & 1 == 0 {
        left_head
    } else {
        right_head
    };
    canonicalize(&assemble(target, transforms, head)).map(Into::into)
}

impl From<CanonicalGenome> for EvolvedGenome {
    fn from(value: CanonicalGenome) -> Self {
        Self {
            canonical: value.canonical,
            semantic_sha256: value.semantic_sha256,
            complexity: value.complexity,
            rust_source_len: value.rust_source_len,
        }
    }
}

#[derive(Clone, Debug)]
enum Expr {
    Terminal(String),
    Unary(&'static str, Box<Expr>),
    Binary(&'static str, Box<Expr>, Box<Expr>),
    Ternary(&'static str, Box<Expr>, Box<Expr>, Box<Expr>),
}

impl Expr {
    fn nodes(&self) -> usize {
        match self {
            Expr::Terminal(_) => 1,
            Expr::Unary(_, value) => 1 + value.nodes(),
            Expr::Binary(_, left, right) => 1 + left.nodes() + right.nodes(),
            Expr::Ternary(_, a, b, c) => 1 + a.nodes() + b.nodes() + c.nodes(),
        }
    }

    fn depth(&self) -> usize {
        match self {
            Expr::Terminal(_) => 0,
            Expr::Unary(_, value) => 1 + value.depth(),
            Expr::Binary(_, left, right) => 1 + left.depth().max(right.depth()),
            Expr::Ternary(_, a, b, c) => 1 + a.depth().max(b.depth()).max(c.depth()),
        }
    }

    fn source(&self) -> String {
        match self {
            Expr::Terminal(value) => value.clone(),
            Expr::Unary(op, value) => {
                if *op == "-" {
                    format!("(-{})", value.source())
                } else {
                    format!("{op}({})", value.source())
                }
            }
            Expr::Binary(op, left, right) if matches!(*op, "+" | "-" | "*" | "/" | "^") => {
                format!("({} {op} {})", left.source(), right.source())
            }
            Expr::Binary(op, left, right) => {
                format!("{op}({}, {})", left.source(), right.source())
            }
            Expr::Ternary(op, a, b, c) => {
                format!("{op}({}, {}, {})", a.source(), b.source(), c.source())
            }
        }
    }
}

fn expression_terminal(rng: &mut Pcg32, feature_count: usize) -> Expr {
    const ROWS: [&str; 7] = [
        "row_mean", "row_sum", "row_l1", "row_l2", "row_min", "row_max", "row_std",
    ];
    const CONSTANTS: [f64; 7] = [-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0];
    match rng.next_u32() % 3 {
        0 => Expr::Terminal(format!(
            "col({})",
            rng.next_u32() as usize % feature_count.max(1)
        )),
        1 => Expr::Terminal((*choice(rng, &ROWS)).to_string()),
        _ => Expr::Terminal(choice(rng, &CONSTANTS).to_string()),
    }
}

fn build_expression(rng: &mut Pcg32, feature_count: usize, depth: usize, budget: usize) -> Expr {
    if depth == 0 || budget < 2 || rng.next_f64() < 0.25 {
        return expression_terminal(rng, feature_count);
    }
    const UNARY: [&str; 8] = ["-", "abs", "sqrt", "log1p", "sin", "cos", "tanh", "sigmoid"];
    const BINARY: [&str; 8] = ["+", "-", "*", "/", "^", "min", "max", "step"];
    match rng.next_u32() % 5 {
        0 if budget >= 4 => {
            let each = (budget - 1) / 3;
            Expr::Ternary(
                "clip",
                Box::new(build_expression(rng, feature_count, depth - 1, each)),
                Box::new(build_expression(rng, feature_count, depth - 1, each)),
                Box::new(build_expression(rng, feature_count, depth - 1, each)),
            )
        }
        1 | 2 if budget >= 3 => {
            let left_budget = (budget - 1) / 2;
            let right_budget = budget - 1 - left_budget;
            Expr::Binary(
                choice(rng, &BINARY),
                Box::new(build_expression(rng, feature_count, depth - 1, left_budget)),
                Box::new(build_expression(
                    rng,
                    feature_count,
                    depth - 1,
                    right_budget,
                )),
            )
        }
        _ => Expr::Unary(
            choice(rng, &UNARY),
            Box::new(build_expression(rng, feature_count, depth - 1, budget - 1)),
        ),
    }
}

fn expression_terminal_expanded(rng: &mut Pcg32, feature_count: usize, allow_x: bool) -> Expr {
    const ROWS: [&str; 7] = [
        "row_mean", "row_sum", "row_l1", "row_l2", "row_min", "row_max", "row_std",
    ];
    const CONSTANTS: [f64; 7] = [-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0];
    let draw = rng.next_u32() % if allow_x { 4 } else { 3 };
    match (allow_x, draw) {
        (true, 0) => Expr::Terminal("x".to_string()),
        (true, 1) | (false, 0) => Expr::Terminal(format!(
            "col({})",
            rng.next_u32() as usize % feature_count.max(1)
        )),
        (true, 2) | (false, 1) => Expr::Terminal((*choice(rng, &ROWS)).to_string()),
        _ => Expr::Terminal(choice(rng, &CONSTANTS).to_string()),
    }
}

fn build_expression_expanded(
    rng: &mut Pcg32,
    feature_count: usize,
    allow_x: bool,
    depth: usize,
    budget: usize,
) -> Expr {
    if depth == 0 || budget < 2 || rng.next_f64() < 0.25 {
        return expression_terminal_expanded(rng, feature_count, allow_x);
    }
    const UNARY: [&str; 16] = [
        "-", "abs", "sqrt", "log1p", "sin", "cos", "tanh", "sigmoid", "exp", "log", "asinh",
        "atan", "erf", "sign", "floor", "round",
    ];
    const BINARY: [&str; 8] = ["+", "-", "*", "/", "^", "min", "max", "step"];
    match rng.next_u32() % 5 {
        0 if budget >= 4 => {
            let each = (budget - 1) / 3;
            Expr::Ternary(
                "clip",
                Box::new(build_expression_expanded(
                    rng,
                    feature_count,
                    allow_x,
                    depth - 1,
                    each,
                )),
                Box::new(build_expression_expanded(
                    rng,
                    feature_count,
                    allow_x,
                    depth - 1,
                    each,
                )),
                Box::new(build_expression_expanded(
                    rng,
                    feature_count,
                    allow_x,
                    depth - 1,
                    each,
                )),
            )
        }
        1 | 2 if budget >= 3 => {
            let left_budget = (budget - 1) / 2;
            let right_budget = budget - 1 - left_budget;
            Expr::Binary(
                choice(rng, &BINARY),
                Box::new(build_expression_expanded(
                    rng,
                    feature_count,
                    allow_x,
                    depth - 1,
                    left_budget,
                )),
                Box::new(build_expression_expanded(
                    rng,
                    feature_count,
                    allow_x,
                    depth - 1,
                    right_budget,
                )),
            )
        }
        _ => Expr::Unary(
            choice(rng, &UNARY),
            Box::new(build_expression_expanded(
                rng,
                feature_count,
                allow_x,
                depth - 1,
                budget - 1,
            )),
        ),
    }
}

pub fn random_expression_expanded(rng: &mut Pcg32, feature_count: usize, allow_x: bool) -> String {
    let expression = build_expression_expanded(
        rng,
        feature_count,
        allow_x,
        MAX_EXPRESSION_DEPTH,
        MAX_EXPRESSION_NODES,
    );
    debug_assert!(expression.depth() <= MAX_EXPRESSION_DEPTH);
    debug_assert!(expression.nodes() <= MAX_EXPRESSION_NODES);
    let source = expression.source();
    // A bare numeric expression would be serialized as an integer/float DSL
    // parameter and then rejected because map/derive require a string. Keep
    // the same value while forcing expression syntax (and therefore quotes).
    if source.parse::<f64>().is_ok() {
        format!("({source} + 0.0)")
    } else {
        source
    }
}

pub fn random_expression(rng: &mut Pcg32, feature_count: usize) -> String {
    let expression = build_expression(
        rng,
        feature_count,
        MAX_EXPRESSION_DEPTH,
        MAX_EXPRESSION_NODES,
    );
    debug_assert!(expression.depth() <= MAX_EXPRESSION_DEPTH);
    debug_assert!(expression.nodes() <= MAX_EXPRESSION_NODES);
    expression.source()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn all_full_rust_operations_are_reachable_and_emittable() {
        for schema in operation_schemas() {
            if schema.kind == OperationKind::Target {
                continue;
            }
            let source = if schema.kind == OperationKind::Head {
                let head = random_stage(&mut Pcg32::new(7, 9), schema.op, 50);
                assemble(None, vec![Stage::new("standardize")], head)
            } else {
                let transform = random_stage(&mut Pcg32::new(7, 9), schema.op, 50);
                assemble(None, vec![transform], Stage::new("ridge"))
            };
            canonicalize(&source).unwrap_or_else(|error| panic!("{}: {error}", schema.op));
        }
    }

    #[test]
    fn aliases_have_one_semantic_hash() {
        let aliases = [
            ("norm_features", "standardize"),
            ("winsor_moments", "winsor_clip"),
        ];
        for (alias, canonical) in aliases {
            let a = canonicalize(&format!("(pipe X ({alias}) (ridge))")).unwrap();
            let b = canonicalize(&format!("(pipe X ({canonical}) (ridge))")).unwrap();
            assert_eq!(a.semantic_sha256, b.semantic_sha256);
        }
    }

    #[test]
    fn expression_generator_respects_campaign_bounds() {
        let mut rng = Pcg32::new(42, 11);
        for _ in 0..10_000 {
            let expr = build_expression(&mut rng, 50, MAX_EXPRESSION_DEPTH, MAX_EXPRESSION_NODES);
            assert!(expr.depth() <= MAX_EXPRESSION_DEPTH);
            assert!(expr.nodes() <= MAX_EXPRESSION_NODES);
            crate::math_expr::MathExpr::parse(&expr.source()).unwrap();
        }
    }

    #[test]
    fn expanded_transform_draw_reaches_the_complete_vocabulary() {
        let expected = TRANSFORMS
            .iter()
            .chain(EXPANSION_TRANSFORMS.iter())
            .copied()
            .collect::<HashSet<_>>();
        let mut seen = HashSet::new();
        for seed in 0..10_000 {
            let mut rng = Pcg32::new(seed, 0x4655_4c4c_4745_4e45);
            seen.insert(expanded_transform(&mut rng));
            if seen == expected {
                break;
            }
        }
        assert_eq!(seen, expected);
    }

    #[test]
    fn expanded_expression_draw_reaches_new_math_and_x() {
        const NEW_UNARY: [&str; 8] = [
            "exp(", "log(", "asinh(", "atan(", "erf(", "sign(", "floor(", "round(",
        ];
        let mut rng = Pcg32::new(42, 0x4558_5052_5f56_3200);
        let mut seen = HashSet::new();
        let mut saw_x = false;
        for _ in 0..100_000 {
            let source = random_expression_expanded(&mut rng, 50, true);
            crate::math_expr::MathExpr::parse(&source).unwrap();
            saw_x |= source
                .split(|value: char| !value.is_ascii_alphanumeric() && value != '_')
                .any(|token| token == "x");
            for op in NEW_UNARY {
                if source.contains(op) {
                    seen.insert(op);
                }
            }
            if saw_x && seen.len() == NEW_UNARY.len() {
                break;
            }
        }
        assert!(saw_x);
        assert_eq!(seen.len(), NEW_UNARY.len());
    }

    #[test]
    fn expanded_mutation_is_stable_and_introduces_expansion_transforms() {
        let source = "(pipe X (standardize) (ridge))";
        assert_eq!(
            mutate_pipeline_expanded(source, 42, 50).unwrap(),
            mutate_pipeline_expanded(source, 42, 50).unwrap()
        );
        let mut seen = HashSet::new();
        for seed in 0..10_000 {
            let evolved = mutate_pipeline_expanded(source, seed, 50)
                .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
            canonicalize(&evolved.canonical).unwrap();
            for op in EXPANSION_TRANSFORMS {
                if evolved.canonical.contains(&format!("({op}")) {
                    seen.insert(op);
                }
            }
            if seen.len() == EXPANSION_TRANSFORMS.len() {
                break;
            }
        }
        assert_eq!(seen.len(), EXPANSION_TRANSFORMS.len());
    }

    #[test]
    fn expanded_mutation_keeps_portfolio_heads_canonical() {
        let mut head = Stage::new("portfolio");
        head.params
            .insert("preset".to_string(), ParamValue::Str("none".to_string()));
        normalize_expanded_head(&mut head);
        assert_eq!(
            head.params.get("preset"),
            Some(&ParamValue::Str("core".to_string()))
        );
    }

    #[test]
    fn categorical_boolean_and_expression_params_mutate_deterministically() {
        let source = "(pipe X (map append=true expr=\"col(0)\") (histgbm2 val_shuffle=true) (portfolio preset=core))";
        assert!(canonicalize(source).is_err());
        let source = "(pipe X (map append=true expr=\"col(0)\") (portfolio preset=core))";
        let a = mutate_pipeline(source, 42, 50).unwrap();
        let b = mutate_pipeline(source, 42, 50).unwrap();
        assert_eq!(a, b);
        canonicalize(&a.canonical).unwrap();
    }

    #[test]
    fn target_transform_is_only_accepted_first() {
        canonicalize("(pipe X (target kind=winsor q=0.1) (ridge))").unwrap();
        assert!(canonicalize("(pipe X (rank) (target kind=winsor) (ridge))").is_err());
    }

    #[test]
    fn random_mutation_and_crossover_are_stable() {
        let a = random_pipeline(42, 50).unwrap();
        let b = random_pipeline(43, 50).unwrap();
        assert_eq!(a, random_pipeline(42, 50).unwrap());
        assert_eq!(
            mutate_pipeline(&a.canonical, 99, 50).unwrap(),
            mutate_pipeline(&a.canonical, 99, 50).unwrap()
        );
        assert_eq!(
            crossover_pipelines(&a.canonical, &b.canonical, 100).unwrap(),
            crossover_pipelines(&a.canonical, &b.canonical, 100).unwrap()
        );
    }
}
