//! CUDA C++ consumers share the full native Rust recipe and an explicit device path.
//! Device inference offloads only a linear head; feature transforms remain on the host.
use crate::artifact_manifest::{sha256, FileBinding, OptionsIdentity};
use crate::spec::{Plan, PlannedHead};
use crate::transpile::{self, EmitOptions};
use crate::GenomeIr;
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

pub const EMITTER: &str = "feat-math-cuda/1";
pub const BUILD: &str =
    "cmake -S cuda -B cuda/build -DCMAKE_BUILD_TYPE=Release && cmake --build cuda/build";
pub const DEVICE_FLAGS: &[&str] = &[
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "-Xcompiler=-ffp-contract=off",
    "-Xcompiler=-fno-fast-math",
];

#[derive(Clone, Debug)]
pub struct Emitted {
    /// Complete source closure, relative to the output directory.
    pub files: BTreeMap<String, String>,
}

/// Full Rust recipe coverage. Emission never executes a compiler or opens a device.
pub fn emit(genome: &GenomeIr, options: &EmitOptions) -> Result<Emitted> {
    let mut framed = options.clone();
    framed.title = options
        .title
        .as_ref()
        .map(|s| s.split_whitespace().collect::<Vec<_>>().join(" "));
    let pair = transpile::emit(genome, &framed)?;
    let plan = crate::spec::build_plan(genome).map_err(|e| anyhow::anyhow!(e.join(", ")))?;
    let linear = match &plan {
        Plan::Pipeline {
            target: None, head, ..
        } => linear_facade(&pair.rust, head)?,
        _ => None,
    };
    assemble(
        pair.rust,
        recipe_facade(linear.as_deref()),
        options,
        "legacy_recipe",
        &sha256(genome.canonical().as_bytes()),
        linear.is_some(),
        None,
    )
}

pub fn emit_scalar(
    program: &crate::kernel::scalar_program::ScalarExpressionProgram,
    carrier: crate::kernel::scalar_program::ScalarCarrier,
    width: usize,
    lambda: f64,
) -> Result<Emitted> {
    let rust = program.emit_recipe(
        crate::kernel::scalar_program::ScalarLanguage::Rust,
        carrier,
        width,
        lambda,
    )?;
    let linear = linear_facade(
        &rust,
        &PlannedHead::Ridge {
            lambda_frac: lambda,
        },
    )?
    .context("scalar Rust recipe has no linear head")?;
    assemble(
        rust,
        recipe_facade(Some(&linear)),
        &EmitOptions::default(),
        "scalar_v2",
        &sha256(&program.to_bytes()?),
        true,
        None,
    )
}

pub fn emit_row_vector(program: &crate::kernel::row_vector::Program) -> Result<Emitted> {
    let rust = program.emit_recipe(crate::kernel::row_vector::emit::RowRecipeLanguage::Rust)?;
    let linear = linear_facade(&rust, &PlannedHead::Ridge { lambda_frac: 0.0 })?
        .context("row Rust recipe has no linear head")?;
    assemble(
        rust,
        recipe_facade(Some(&linear)),
        &EmitOptions::default(),
        "row_vector_v2",
        &sha256(&program.to_bytes()?),
        true,
        None,
    )
}

/// The raw-string predictor has exactly the bound Rust emitter's admission.
/// Fitting and CUDA head inference are explicitly unavailable for bound artifacts.
pub fn emit_bound(
    model: &crate::advanced_v3::BoundModel,
    options: &EmitOptions,
) -> Result<Emitted> {
    let pair = crate::advanced_v3::emit_bound(model, options)?;
    let headers: Vec<_> = model.schema.columns.iter().map(|c| &c.name).collect();
    let quoted = headers
        .iter()
        .map(|h| format!("{h:?}.to_string()"))
        .collect::<Vec<_>>()
        .join(",");
    let facade = include_str!("cuda/bound.rs.txt")
        .replace("__BOUND_WIDTH__", &headers.len().to_string())
        .replace("__BOUND_HEADERS__", &format!("vec![{quoted}]"));
    assemble(
        pair.rust,
        facade,
        options,
        "advanced_v3",
        &model.genome_sha256,
        false,
        Some((
            headers.into_iter().cloned().collect(),
            crate::advanced_v3::emit::user_text(model),
        )),
    )
}

fn recipe_facade(linear: Option<&str>) -> String {
    include_str!("cuda/recipe.rs.txt").replace(
        "__LINEAR__",
        linear.unwrap_or(
            "pub(crate) fn linear_width(&self)->Result<usize,i32>{Err(5)}\n\
         pub(crate) fn linear(&self,_x:&[Vec<f64>])->Result<(Vec<f64>,Vec<f64>),i32>{Err(5)}",
        ),
    )
}

/// Reuse the *emitted* prediction prefix verbatim. A template change fails closed
/// here instead of maintaining a second table of transform calls and field names.
fn linear_facade(rust: &str, head: &PlannedHead) -> Result<Option<String>> {
    let (call, bias, weights, standardize) = match head {
        PlannedHead::Ridge { .. } => ("ridge_regression_predict", "self.model.head.w[0]",
            "self.model.head.w[1..].to_vec()",
            "for row in &x { for j in 0..p { values.push(((row[j]-self.model.head.means[j])/self.model.head.scales[j]).clamp(-20.0,20.0)); } }"),
        PlannedHead::ElasticNet { .. } => ("elastic_net_predict", "self.model.head.w0",
            "self.model.head.w.clone()",
            "for row in &x { for j in 0..p { values.push(((row[j]-self.model.head.means[j])/self.model.head.scales[j]).clamp(-20.0,20.0)); } }"),
        PlannedHead::RidgeGcv => ("gcv_ridge_regression_predict", "self.model.head.w[0]",
            "self.model.head.w[1..].to_vec()",
            "for row in &x { for (k,&j) in self.model.head.cols.iter().enumerate() { if j>=row.len(){return Err(4);} values.push((row[j]-self.model.head.means[k])/self.model.head.scales[k]); } }"),
        _ => return Ok(None),
    };
    let model = rust
        .split_once("impl Model {")
        .context("Rust model template missing")?
        .1;
    let signature = if model.contains("    fn predict_rows(&self, x: &[Vec<f64>]) -> Vec<f64> {") {
        "    fn predict_rows(&self, x: &[Vec<f64>]) -> Vec<f64> {"
    } else {
        "    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {"
    };
    let body = model
        .split_once(signature)
        .context("Rust prediction template missing")?
        .1;
    let tail = format!("        {call}(&self.head, &x)\n    }}");
    let prefix = body
        .split_once(&tail)
        .context("Rust linear prediction template changed")?
        .0;
    // Access the original Model fields from this same safe module.
    let prefix = prefix.replace("self.stage", "self.model.stage");
    Ok(Some(format!(
        r#"
    pub(crate) fn linear_width(&self)->Result<usize,i32>{{Ok(self.model.head.means.len())}}
    pub(crate) fn linear(&self,x:&[Vec<f64>])->Result<(Vec<f64>,Vec<f64>),i32>{{
        {prefix}
        let p=self.model.head.means.len();
        if x.iter().any(|r|r.len()<p){{return Err(4);}}
        let mut values=Vec::new();
        {standardize}
        let mut weights=vec![{bias}];weights.extend({weights});
        if values.iter().chain(&weights).any(|v|!v.is_finite()){{return Err(1);}}
        Ok((values,weights))
    }}
"#
    )))
}

fn assemble(
    rust: String,
    facade: String,
    options: &EmitOptions,
    dialect: &str,
    program_sha256: &str,
    device_linear: bool,
    bound_text: Option<(Vec<String>, Vec<String>)>,
) -> Result<Emitted> {
    // The numerical module retains forbid(unsafe_code); raw pointers exist only
    // in the independently audited native boundary, inherited from C17.
    let bridge = include_str!("c17/bridge.rs")
        .replace("C17Model", "CudaModel")
        .replace("jope_model_", "jope_cuda_native_")
        .replace(
            "CudaModel::fit(&x, &y, cols)",
            "CudaModel::fit(&x, &y, cols)?",
        )
        .replace(
            "let predictions = model.predict(&x);",
            "let predictions = model.predict(&x)?;",
        )
        .replace(
            "let score = model.score(&x, &y);",
            "let score = model.score(&x, &y)?;",
        );
    let core_hash = sha256(rust.as_bytes());
    let (headers, user) = bound_text.unwrap_or_default();
    // Fixed-width octal bytes preserve arbitrary UTF-8, quotes, backslashes and
    // embedded NULs without source injection or greedy C++ hex escapes.
    let headers = headers
        .iter()
        .map(|h| {
            format!(
                "std::string(\"{}\",{})",
                h.as_bytes()
                    .iter()
                    .map(|b| format!("\\{b:03o}"))
                    .collect::<String>(),
                h.len()
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let mut files: BTreeMap<String, String> = BTreeMap::from([
        (
            "cuda/model.hpp".into(),
            include_str!("cuda/model.hpp").into(),
        ),
        (
            "cuda/model.cu".into(),
            include_str!("cuda/model.cu").replace("__BOUND_HEADERS__", &format!("{{{headers}}}")),
        ),
        (
            "cuda/model_native.rs".into(),
            format!("{bridge}\n{}", include_str!("cuda/bridge.rs.txt")),
        ),
        ("cuda/model_core.rs".into(), format!("{rust}\n{facade}")),
        (
            "cuda/CMakeLists.txt".into(),
            include_str!("cuda/CMakeLists.txt").into(),
        ),
    ]);
    // Distinct generated models can coexist in one executable. Bind the C and
    // C++ symbol namespaces to the entire numerical/boundary source closure.
    let mut identity = Vec::new();
    for (path, source) in &files {
        identity.extend_from_slice(&(path.len() as u64).to_be_bytes());
        identity.extend_from_slice(path.as_bytes());
        identity.extend_from_slice(&(source.len() as u64).to_be_bytes());
        identity.extend_from_slice(source.as_bytes());
    }
    let symbol_identity = sha256(&identity);
    let namespace = format!("jope_cuda_{symbol_identity}");
    let native_prefix = format!("{namespace}_native_");
    for (path, source) in &mut files {
        // Rename only fixed boundary templates. Caller strings in the numerical
        // module must remain byte-exact, even when they match a symbol template.
        if matches!(path.as_str(), "cuda/model.cu" | "cuda/model_native.rs") {
            *source = source.replace("jope_cuda_native_", &native_prefix);
        }
        if matches!(path.as_str(), "cuda/model.cu" | "cuda/model.hpp") {
            *source = source.replace(
                "namespace jope_cuda {",
                &format!("namespace {namespace} {{"),
            );
        }
        if path == "cuda/model_native.rs" {
            *source = format!("#![crate_name=\"model_native_{symbol_identity}\"]\n{source}");
        }
        if path == "cuda/CMakeLists.txt" {
            *source = source.replace("__CMAKE_TARGET__", &namespace);
        }
        if path == "cuda/model.hpp" {
            *source = source.replace("JOPE_CUDA_MODEL_HPP", &format!("JOPE_CUDA_MODEL_HPP_{}",symbol_identity.to_ascii_uppercase()))
                .replace("} // namespace jope_cuda", &format!("}} // namespace {namespace}\n#ifndef JOPE_CUDA_NO_DEFAULT_NAMESPACE\nnamespace jope_cuda = {namespace};\n#endif"));
        }
    }
    for (path, source) in &files {
        let hits = crate::advanced_v3::emit::lint_shipped_masked(source, &user, false);
        ensure!(
            hits.is_empty(),
            "banned vocabulary in {path}: {}",
            hits.join(", ")
        );
    }
    let bindings: Vec<_> = files
        .iter()
        .map(|(p, s)| FileBinding::new(p, s.as_bytes()))
        .collect();
    let tokens: BTreeMap<_, _> = files
        .iter()
        .filter_map(|(path, source)| {
            let lang = if path.ends_with(".rs") {
                crate::tokens::Lang::Rust
            } else if path.ends_with(".cu") || path.ends_with(".hpp") {
                crate::tokens::Lang::CudaCpp
            } else {
                return None;
            };
            Some((path, crate::tokens::count_tokens(source, lang)))
        })
        .collect();
    let manifest = serde_json::json!({
        "schema":"jope.cuda-model/v1", "emitter":EMITTER, "abi_version":1,
        "dialect":dialect, "program_sha256":program_sha256, "rust_source_sha256":core_hash,
        "options":OptionsIdentity::from(options), "files":bindings, "build":BUILD,
        "host":"rust-native-full", "device":if device_linear{"linear-head-f64-v1"}else{"unsupported"},
        "device_preprocessing":"host", "device_flags":DEVICE_FLAGS, "execution":"unknown", "tokens":tokens
        ,"cpp_namespace":namespace, "cmake_target":namespace,
        "native_prefix":native_prefix, "native_crate":format!("model_native_{symbol_identity}")
    });
    files.insert(
        "cuda/manifest.json".into(),
        format!("{}\n", serde_json::to_string_pretty(&manifest)?),
    );
    Ok(Emitted { files })
}
