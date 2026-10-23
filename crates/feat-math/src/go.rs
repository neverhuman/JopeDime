//! Go artifacts backed by the full standalone Rust numerical implementation.
//! The emitted bridge owns its buffers; no raw pointers are accepted from the caller.
use crate::transpile::{self, EmitOptions, TokenCount};
use crate::GenomeIr;
use anyhow::Result;

pub const EMITTER_VERSION: &str = "go-wasm/1";
pub const MODULE: &str = include_str!("go/go.mod.txt");
pub const SUM: &str = include_str!("go/go.sum.txt");
pub const BUILD_COMMAND: &str = "rustc --edition=2021 --target wasm32-unknown-unknown --crate-type cdylib --crate-name model_kernel -C opt-level=3 -C panic=abort model.go.kernel.rs -o model.wasm && CGO_ENABLED=0 go build -mod=readonly ./...";

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct GoEmitted {
    pub source: String,
    pub kernel: String,
    pub source_sha256: String,
    pub kernel_sha256: String,
    pub tokens: TokenCount,
    pub math_policy: &'static str,
}

fn assemble(rust: String, body: &str, source: String) -> GoEmitted {
    // Export attributes cannot live under forbid(unsafe_code). Only this lint
    // changes; the numerical Rust source is otherwise retained byte for byte.
    let rust = rust.replacen(
        "\n#![forbid(unsafe_code)]\n",
        "\n#![deny(unsafe_op_in_unsafe_fn)]\n",
        1,
    );
    let kernel = format!("{rust}\n{body}");
    let source_sha256 = crate::artifact_manifest::sha256(source.as_bytes());
    let kernel_sha256 = crate::artifact_manifest::sha256(kernel.as_bytes());
    let tokens = crate::tokens::count_tokens(&source, crate::tokens::Lang::Go);
    GoEmitted {
        source,
        kernel,
        source_sha256,
        kernel_sha256,
        tokens,
        math_policy: "wasm-rust-std-v1",
    }
}

/// Full recipe coverage follows the Rust emitter, including symbolic programs,
/// target mappings, nonlinear heads and scalar/row-vector operations.
pub fn emit(genome: &GenomeIr, options: &EmitOptions) -> Result<GoEmitted> {
    let mut sequential = options.clone();
    sequential.fast_runtime = false;
    // The inherited Rust emitter places the title in a single doc-comment
    // line. Keep caller metadata on that line, including CR/LF and Unicode
    // separators, without changing the frozen numerical templates.
    sequential.title = options
        .title
        .as_ref()
        .map(|title| title.split_whitespace().collect::<Vec<_>>().join(" "));
    let pair = transpile::emit(genome, &sequential)?;
    Ok(recipe_from_rust(pair.rust))
}

fn recipe_from_rust(rust: String) -> GoEmitted {
    let source = include_str!("go/model.go").replace("__BOUND_HEADERS__", "[]string(nil)");
    let body = format!(
        "{}\n{}",
        include_str!("go/buffers.rs.txt"),
        include_str!("go/recipe.rs.txt")
    );
    assemble(rust, &body, source)
}

/// Explicit scalar-v2 programs use their existing Rust recipe and validation.
pub fn emit_scalar(
    program: &crate::kernel::scalar_program::ScalarExpressionProgram,
    carrier: crate::kernel::scalar_program::ScalarCarrier,
    width: usize,
    lambda: f64,
) -> Result<GoEmitted> {
    Ok(recipe_from_rust(program.emit_recipe(
        crate::kernel::scalar_program::ScalarLanguage::Rust,
        carrier,
        width,
        lambda,
    )?))
}

/// Explicit feature-position Savitzky–Golay/Goertzel programs retain their
/// geometry-only Rust constants and independent per-row numerical contract.
pub fn emit_row_vector(program: &crate::kernel::row_vector::Program) -> Result<GoEmitted> {
    Ok(recipe_from_rust(program.emit_recipe(
        crate::kernel::row_vector::emit::RowRecipeLanguage::Rust,
    )?))
}

/// Full fitted coverage follows the existing bound Rust emitter. Unsupported
/// model forms fail there, before any artifact is published.
pub fn emit_bound(
    model: &crate::advanced_v3::BoundModel,
    options: &EmitOptions,
) -> Result<GoEmitted> {
    let pair = crate::advanced_v3::emit_bound(model, options)?;
    let headers: Vec<_> = model
        .schema
        .columns
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    let json = format!(
        "[]string{{{}}}",
        headers
            .iter()
            .map(|s| quote(s))
            .collect::<Vec<_>>()
            .join(",")
    );
    let source = include_str!("go/model.go").replace("__BOUND_HEADERS__", &json);
    let rust_headers = format!(
        "[{}]",
        headers
            .iter()
            .map(|h| format!("{h:?}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    let body = format!(
        "{}\n{}",
        include_str!("go/buffers.rs.txt"),
        include_str!("go/bound.rs.txt").replace("__BOUND_HEADERS__", &rust_headers)
    );
    Ok(assemble(pair.rust, &body, source))
}

pub fn sources(e: &GoEmitted) -> [(&'static str, &str); 4] {
    [
        ("model.go", &e.source),
        ("model.go.kernel.rs", &e.kernel),
        ("go.mod", MODULE),
        ("go.sum", SUM),
    ]
}
fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for b in s.as_bytes() {
        out.push_str(&format!("\\x{b:02x}"));
    }
    out.push('"');
    out
}
