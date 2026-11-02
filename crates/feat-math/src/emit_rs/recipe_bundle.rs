//! Structured-IR recipe glue against a single versioned operator runtime.
//!
//! The runtime is assembled from the emitter's operation/head table, independently
//! of any program bank. DSL literals live only in Model wiring; every numeric
//! kernel and the CSV transport are concrete functions compiled in the runtime.

use super::*;
use crate::spec::GbmPlan;
use std::collections::BTreeSet;
use std::sync::OnceLock;

pub const OPS_ABI_VERSION: u32 = 2;
pub const EMITTER_VERSION: &str = "jope-recipe-bundle-ir-v2";

struct RuntimeSource {
    source: String,
    exports: BTreeSet<String>,
}

fn operations() -> Vec<PlannedOp> {
    use PlannedOp::*;
    vec![
        Standardize,
        RobustScale,
        Rank,
        Blom,
        QuantileBins { bins: 8 },
        BinOnehot { bins: 8 },
        Hinge { knots: 5 },
        WinsorClip { q: 0.05 },
        Cheb { degree: 3 },
        PolyInteract,
        RffCos {
            n: 32,
            gamma: 1.0,
            seed: 7,
        },
        RffSin {
            n: 32,
            gamma: 1.0,
            seed: 7,
        },
        RandProj { k: 16, seed: 7 },
        LandmarkDist {
            k: 16,
            gamma: 1.0,
            seed: 7,
        },
        ObliqueHinge { n: 16, seed: 7 },
        RowStats,
        PairProd { topk: 8 },
        TargetBinEncode {
            bins: 8,
            prior: 1.0,
        },
        Sigmoid,
        Step { t: 0.5 },
        NormRow,
        Polar,
        Lag { k: 1 },
        Roll { k: 1 },
        FftMag { k: 4 },
        RowSpectral,
        RowDiffStats,
        SvdProject {
            k: 8,
            iters: 20,
            seed: 7,
        },
        Nystrom {
            k: 16,
            gamma: 1.0,
            seed: 7,
        },
        Symlog { c: 1.0 },
        SelectCorr {
            k: 16,
            by_var: false,
        },
        Impute {
            strategy: 0,
            value: 0.0,
            add_indicator: false,
        },
        PairRatio { topk: 8 },
        Spline { knots: 5 },
    ]
}

fn heads() -> Vec<PlannedHead> {
    use PlannedHead::*;
    let gbm = GbmPlan {
        rounds: 64,
        depth: 3,
        lr: 0.1,
        lambda: 1.0,
        bins: 32,
        min_leaf: 5,
        max_features: 0,
    };
    vec![
        Ridge { lambda_frac: 0.01 },
        RidgeGcv,
        ElasticNet {
            alpha: 0.01,
            l1_ratio: 0.5,
            sweeps: 40,
        },
        Gbm(gbm.clone()),
        Blend {
            lambda_frac: 0.01,
            gbm,
        },
        HistGbm2 {
            leaves: 16,
            lr: 0.1,
            bags: 1,
            rounds: 64,
            row_subsample: 1.0,
            col_subsample: 1.0,
            min_leaf: 5,
            val_pct: 20,
            val_shuffle: false,
            huber_delta: 0.0,
            stop_mae: false,
        },
        Krr { m: 128 },
        Gpr { m: 128, steps: 20 },
        GprKernel {
            m: 128,
            steps: 20,
            source: "rbf".into(),
            spec: vec![vec![crate::gpr::KernelPrim::Rbf]],
        },
        Ksmooth { k: 16, gamma: 1.0 },
        Stack,
    ]
}

fn public_kernel_source(source: &str, exports: &mut BTreeSet<String>) -> String {
    let mut lines = Vec::new();
    let mut in_struct = false;
    let mut in_impl = false;
    for line in source.lines() {
        // Descriptions containing representative planner parameters are not part
        // of the operator ABI. Bodies come directly from the shared IR emitters.
        if line.trim_start().starts_with("///") || line.trim_start().starts_with("//!") {
            continue;
        }
        let declaration = ["fn ", "struct ", "enum ", "type ", "const ", "trait "]
            .iter()
            .find_map(|prefix| line.strip_prefix(prefix).map(|tail| (*prefix, tail)));
        if let Some((_, tail)) = declaration {
            let name: String = tail
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            exports.insert(name);
            in_struct = line.starts_with("struct ") && line.ends_with('{');
            lines.push(format!("pub {line}"));
        } else if line.starts_with("impl ") {
            in_impl = true;
            lines.push(line.to_owned());
        } else if in_impl && line.starts_with("    fn ") {
            lines.push(line.replacen("    fn ", "    pub fn ", 1));
        } else if in_struct
            && line.starts_with("    ")
            && line.contains(':')
            && !line.trim_start().starts_with('#')
        {
            lines.push(format!("    pub {}", line.trim_start()));
        } else {
            lines.push(line.to_owned());
        }
        if line == "}" {
            in_struct = false;
            in_impl = false;
        }
    }
    lines.join("\n")
}

fn runtime() -> &'static RuntimeSource {
    static RUNTIME: OnceLock<RuntimeSource> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        let ops = operations();
        let heads = heads();
        let targets = [TargetKind::RankGauss, TargetKind::Log1p, TargetKind::Winsor, TargetKind::ZScore];
        let stages: Vec<PlannedStage> = ops.iter().cloned().map(|op| PlannedStage { op, append: false }).collect();
        let mut helpers = BTreeSet::new();
        for head in &heads {
            for kind in targets {
                helpers.extend(helper_set(&Plan::Pipeline {
                    target: Some(PlannedTarget { kind, q: 0.05 }), stages: stages.clone(), head: head.clone()
                }));
            }
        }
        let mut snippets: Vec<String> = helpers.into_iter()
            .map(|helper| helper_src(helper, true, true, true).to_owned()).collect();
        for kind in targets {
            snippets.push(target_src(&PlannedTarget { kind, q: 0.05 }).to_owned());
        }
        for op in &ops {
            snippets.push(op_src(op, true, true));
        }
        let mut seen_aux = Vec::new();
        for head in &heads {
            let parts = head_srcs(head, true, true);
            // Blend and stack compose the already registered concrete kernels.
            let parts = if matches!(head, PlannedHead::Blend { .. } | PlannedHead::Stack) {
                parts.into_iter().rev().take(1).collect::<Vec<_>>()
            } else { parts };
            snippets.extend(parts.into_iter().map(|src| rs_strip_dup_aux(src, &mut seen_aux)));
        }
        snippets.push(BOUNDED_PREDICT.to_owned());
        snippets.push(IO_BLOCK_HPC_V2
            .replace("fn main() {", "fn run_recipe(fit: fn(&[Vec<f64>], &[f64]) -> Box<dyn RecipeModel>) {")
            .replace("Model::fit(&x_train, &y_train)", "fit(&x_train, &y_train)")
            .replace("model.predict(&x_test)", "bounded_predict(model.as_ref(), &x_test)"));
        let mut exports = BTreeSet::new();
        let mut source = format!("//! {EMITTER_VERSION}; operator ABI {OPS_ABI_VERSION}.\n#![forbid(unsafe_code)]\n#![allow(dead_code)]\n\n");
        for snippet in snippets {
            source.push_str(&public_kernel_source(&snippet, &mut exports));
            source.push_str("\n\n");
        }
        RuntimeSource { source, exports }
    })
}

pub fn operator_runtime_source() -> &'static str {
    &runtime().source
}

/// Unsupported IR is an explicit monolithic hpc_v2 fallback. No bound predictor
/// is substituted: the glue and the standalone oracle both refit the same Plan.
pub fn emit_glue(plan: &Plan, opts: &EmitOptions) -> Result<String, String> {
    crate::transpile::validate_math_kernel(opts).map_err(|e| e.to_string())?;
    if !opts.hpc_v2 {
        return Err("hpc_bundle_v2 requires the hpc_v2 numerical recipe".into());
    }
    if !opts.partition_point_quantile_bounds {
        return Err("hpc_bundle_v2 requires partition_point_quantile_bounds = true".into());
    }
    let Plan::Pipeline {
        target,
        stages,
        head,
    } = plan
    else {
        return Err("symbolic expressions are not in operator ABI 2".into());
    };
    if matches!(head, PlannedHead::Portfolio(_)) {
        return Err("portfolio roster is not in operator ABI 2".into());
    }
    if stages.iter().any(|stage| {
        matches!(
            stage.op,
            PlannedOp::MapExpr { .. }
                | PlannedOp::DeriveExpr { .. }
                | PlannedOp::ScalarV2 { .. }
                | PlannedOp::RowVectorV2 { .. }
                | PlannedOp::RowVectorInputV2 { .. }
        )
    }) {
        return Err("formula/scalar/row-vector stages are not in operator ABI 2".into());
    }
    let imports = runtime()
        .exports
        .iter()
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    let model = model_pipeline_impl(
        target.as_ref(),
        stages,
        head,
        true,
        true,
        true,
        opts.guard_zero_width_roll
            && stages
                .iter()
                .any(|s| matches!(s.op, PlannedOp::Roll { .. })),
        true,
    );
    Ok(format!("//! {EMITTER_VERSION}; operator ABI {OPS_ABI_VERSION}.\n#![forbid(unsafe_code)]\n#![allow(dead_code, unused_imports)]\nuse jope_recipe_ops::{{{imports}}};\n\n{model}\n\nimpl jope_recipe_ops::RecipeModel for Model {{\n    fn predict_rows(&self, x: &[Vec<f64>]) -> Vec<f64> {{ Model::predict_rows(self, x) }}\n}}\n\nfn main() {{ jope_recipe_ops::run_recipe(|x, y| Box::new(Model::fit(x, y))); }}\n"))
}

const BOUNDED_PREDICT: &str = r#"trait RecipeModel: Sync {
    fn predict_rows(&self, x: &[Vec<f64>]) -> Vec<f64>;
}

fn bounded_predict(model: &dyn RecipeModel, x: &[Vec<f64>]) -> Vec<f64> {
    let workers = std::env::var("JOPE_CPU_THREADS").ok()
        .and_then(|s| s.parse::<usize>().ok()).unwrap_or(1).clamp(1, 6);
    if workers <= 1 || x.len() <= 64 { return model.predict_rows(x); }
    let chunk = (x.len() + workers - 1) / workers;
    let mut out = Vec::with_capacity(x.len());
    let chunks: Vec<&[Vec<f64>]> = x.chunks(chunk).collect();
    if chunks.len() <= 1 { return model.predict_rows(x); }
    let (spawn_chunks, local_chunks) = chunks.split_at(chunks.len() - 1);
    std::thread::scope(|scope| {
        let mut parts = Vec::with_capacity(spawn_chunks.len());
        for rows in spawn_chunks { parts.push(scope.spawn(move || model.predict_rows(rows))); }
        let local_part = model.predict_rows(local_chunks[0]);
        for part in parts { out.extend(part.join().expect("prediction thread")); }
        out.extend(local_part);
    });
    out
}"#;

#[cfg(test)]
mod tests;
