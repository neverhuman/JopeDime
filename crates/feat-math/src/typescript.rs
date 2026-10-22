//! TypeScript artifacts backed by the full standalone Rust numerical implementation.
//! The emitted bridge owns its buffers; no raw pointers are accepted from the caller.
use crate::transpile::{self, EmitOptions, TokenCount};
use crate::GenomeIr;
use anyhow::{ensure, Result};

pub const EMITTER_VERSION: &str = "typescript-wasm/1";
pub const BUILD_COMMAND: &str = "rustc --edition=2021 --target wasm32-unknown-unknown --crate-type cdylib --crate-name model_kernel -C opt-level=3 -C panic=abort model.kernel.rs -o model.wasm && tsc --strict --target ES2020 --module commonjs --lib ES2020,DOM --declaration model.ts";

#[derive(Clone, Debug)]
pub struct TypeScriptEmitted {
    pub source: String,
    pub kernel: String,
    pub source_sha256: String,
    pub kernel_sha256: String,
    pub tokens: TokenCount,
    pub math_policy: &'static str,
}

fn assemble(rust: String, body: &str, source: String) -> Result<TypeScriptEmitted> {
    // Export attributes cannot live under forbid(unsafe_code). Only this lint
    // changes; the numerical Rust source is otherwise retained byte for byte.
    let rust = rust.replace("#![forbid(unsafe_code)]", "#![deny(unsafe_op_in_unsafe_fn)]");
    let kernel = format!("{rust}\n{body}");
    ensure!(!kernel.contains("unsafe {"), "unexpected unsafe bridge code");
    let source_sha256 = crate::artifact_manifest::sha256(source.as_bytes());
    let kernel_sha256 = crate::artifact_manifest::sha256(kernel.as_bytes());
    let tokens = crate::tokens::count_tokens(&source, crate::tokens::Lang::TypeScript);
    Ok(TypeScriptEmitted {
        source,
        kernel,
        source_sha256,
        kernel_sha256,
        tokens,
        math_policy: "wasm-rust-std-v1",
    })
}

/// Full recipe coverage follows the Rust emitter, including symbolic programs,
/// target mappings, nonlinear heads and scalar/row-vector operations.
pub fn emit(genome: &GenomeIr, options: &EmitOptions) -> Result<TypeScriptEmitted> {
    let mut sequential = options.clone();
    sequential.fast_runtime = false;
    let pair = transpile::emit(genome, &sequential)?;
    let source = include_str!("typescript/model.ts").replace(
        "__BOUND_HEADERS__",
        "null",
    );
    let body = format!(
        "{}\n{}",
        include_str!("typescript/buffers.rs.txt"),
        include_str!("typescript/recipe.rs.txt")
    );
    assemble(pair.rust, &body, source)
}

/// Full fitted coverage follows the existing bound Rust emitter. Unsupported
/// model forms fail there, before any artifact is published.
pub fn emit_bound(
    model: &crate::advanced_v3::BoundModel,
    options: &EmitOptions,
) -> Result<TypeScriptEmitted> {
    let pair = crate::advanced_v3::emit_bound(model, options)?;
    let headers: Vec<_> = model.schema.columns.iter().map(|c| c.name.as_str()).collect();
    let json = serde_json::to_string(&headers)?;
    let source = include_str!("typescript/model.ts").replace("__BOUND_HEADERS__", &json);
    let body = format!(
        "{}\n{}",
        include_str!("typescript/buffers.rs.txt"),
        include_str!("typescript/bound.rs.txt").replace(
            "__BOUND_HEADERS__",
            &format!("[{}]", headers.iter().map(|h| format!("{h:?}")).collect::<Vec<_>>().join(",")),
        )
    );
    assemble(pair.rust, &body, source)
}
