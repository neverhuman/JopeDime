//! C17 consumers share the complete, safe Rust training recipe. No arithmetic fork.
use crate::artifact_manifest::{sha256, FileBinding, OptionsIdentity};
use crate::transpile::{self, EmitOptions};
use crate::GenomeIr;
use anyhow::{ensure, Result};
use std::collections::BTreeMap;

pub const EMITTER: &str = "feat-math-c17/1";
pub const BUILD: &str =
    "rustc --edition=2021 -O -C panic=unwind --crate-type=staticlib model_ffi.rs -o libmodel.a";

/// A complete source bundle, including a manifest binding every compilation input.
/// Paths are relative to the output directory; `c17/` avoids legacy source collisions.
#[derive(Clone, Debug)]
pub struct Emitted {
    pub files: BTreeMap<String, String>,
}

/// Same coverage and option admission as the Rust/Python pair. No tool executes here.
pub fn emit(genome: &GenomeIr, options: &EmitOptions) -> Result<Emitted> {
    let pair = transpile::emit(genome, options)?;
    let rust_source_sha256 = sha256(pair.rust.as_bytes());
    let mut files: BTreeMap<String, String> = BTreeMap::from([
        ("c17/model.h".into(), include_str!("c17/model.h").into()),
        (
            "c17/model_ffi.rs".into(),
            include_str!("c17/bridge.rs").into(),
        ),
        (
            "c17/model_core.rs".into(),
            format!("{}\n{}", pair.rust, include_str!("c17/facade.rs")),
        ),
    ]);
    for (path, source) in &files {
        let hits = transpile::lint_shipped_text(source);
        ensure!(
            hits.is_empty(),
            "banned vocabulary in {path}: {}",
            hits.join(", ")
        );
    }
    let bindings: Vec<_> = files
        .iter()
        .map(|(path, source)| FileBinding::new(path, source.as_bytes()))
        .collect();
    let manifest = serde_json::json!({
        "schema": "jope.c17-model/v1", "emitter": EMITTER, "abi_version": 1,
        "program_sha256": sha256(genome.canonical().as_bytes()),
        "rust_source_sha256": rust_source_sha256,
        "options": OptionsIdentity::from(options),
        "files": bindings, "build": BUILD,
        "execution": "unknown"
    });
    files.insert(
        "c17/manifest.json".into(),
        format!("{}\n", serde_json::to_string_pretty(&manifest)?),
    );
    Ok(Emitted { files })
}
