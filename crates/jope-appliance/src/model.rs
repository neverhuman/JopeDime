//! Model object CLI and deliberately root-scoped MCP tools. No training occurs here.
use clap::{Args, Subcommand};
use jope_runtime::limits::Limits;
use jope_runtime::model_object::{self as objects, Manifest, ModelObject};
use jope_runtime::{model_session as sessions, Failure, FailureKind};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Args)]
pub struct ModelArgs {
    #[command(subcommand)]
    command: ModelCommand,
}
#[derive(Subcommand)]
enum ModelCommand {
    /// Wrap a legacy model.jope, or ingest a foreign language manifest and its adjacent parts.
    Import {
        #[arg(
            long,
            required_unless_present = "manifest",
            conflicts_with = "manifest"
        )]
        input: Option<PathBuf>,
        #[arg(long)]
        manifest: Option<PathBuf>,
        #[arg(long, default_value = "models")]
        root: PathBuf,
    },
    Inspect {
        #[arg(long)]
        model: PathBuf,
    },
    Verify {
        #[arg(long)]
        model: PathBuf,
        /// Also require a supported, semantically valid decoder in this build.
        #[arg(long)]
        require_executable: bool,
    },
    /// Export fitted bytes, bound JSON state, or an exact object copy. ONNX is capability-gated.
    Export {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "object")]
        format: String,
        #[arg(long)]
        output: PathBuf,
    },
    List {
        #[arg(long, default_value = "models")]
        root: PathBuf,
    },
    /// Version history and reversible active selection, independent of immutable model bytes.
    Session {
        #[command(subcommand)]
        command: SessionCommand,
    },
    /// MCP 2025-11-25 stdio service. Filesystem access is limited to this explicitly chosen store.
    Mcp {
        #[arg(long, default_value = "models")]
        root: PathBuf,
    },
}

#[derive(Subcommand)]
enum SessionCommand {
    Show {
        #[arg(long)]
        name: String,
        #[arg(long, default_value = "models")]
        root: PathBuf,
    },
    Record {
        #[arg(long)]
        name: String,
        #[arg(long)]
        id: String,
        /// Previously observed history head, or "empty" for the first version.
        #[arg(long)]
        expected_head: String,
        #[arg(long, default_value = "")]
        label: String,
        #[arg(long, default_value = "models")]
        root: PathBuf,
    },
    Rollback {
        #[arg(long)]
        name: String,
        #[arg(long)]
        version: u64,
        #[arg(long)]
        expected_head: String,
        #[arg(long, default_value = "")]
        label: String,
        #[arg(long, default_value = "models")]
        root: PathBuf,
    },
}

const MAX_FRAME: usize = 8 << 20;
fn max_bytes() -> u64 {
    Limits::default().max_model_bytes
}

fn open(path: &Path) -> anyhow::Result<ModelObject> {
    if path.is_dir() {
        ModelObject::read(path, max_bytes())
    } else {
        ModelObject::from_fitted_bytes(objects::read_bounded(path, max_bytes())?)
    }
}

fn import_manifest(path: &Path) -> anyhow::Result<ModelObject> {
    let bytes = objects::read_bounded(path, objects::MAX_MANIFEST_BYTES)?;
    let manifest: Manifest =
        jope_runtime::strict_json::decode(&bytes).map_err(|e| Failure::integrity(e.to_string()))?;
    manifest.validate(max_bytes())?;
    let parent = path.parent().unwrap_or(Path::new("."));
    let mut parts = BTreeMap::new();
    for part in manifest.files {
        parts.insert(
            part.path.clone(),
            objects::read_bounded(&parent.join(&part.path), part.bytes)?,
        );
    }
    ModelObject::decode(bytes, parts, max_bytes())
}

fn list(root: &Path) -> anyhow::Result<Value> {
    if !root.exists() {
        return Ok(json!({"objects": []}));
    }
    let entries = std::fs::read_dir(root).map_err(|e| Failure::input(e.to_string()))?;
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| Failure::input(e.to_string()))?;
        if entry.file_name().to_string_lossy().starts_with("sha256-") {
            paths.push(entry.path());
        }
    }
    paths.sort();
    let objects: Vec<Value> = paths
        .into_iter()
        .map(|path| match open(&path).and_then(|o| o.inspect()) {
            Ok(info) => json!({"path": path, "inspection": info}),
            Err(e) => json!({"path": path, "integrity_verified": false, "error": format!("{e:#}")}),
        })
        .collect();
    Ok(json!({"objects": objects}))
}

fn export_bytes(object: &ModelObject, format: &str) -> anyhow::Result<Vec<u8>> {
    match format {
        "fitted" => {
            object.fitted()?;
            Ok(object.parts()[&object.manifest().primary].clone())
        },
        "bound-json" => serde_json::to_vec(&object.bound()?).map_err(Into::into),
        "python" | "rust" | "javascript" => object.export_source(format),
        "onnx" => Err(Failure::backend("ONNX export is unavailable in this build; retain the complete model object. A custom-domain ONNX graph also requires a compatible operator runtime.")),
        _ => Err(Failure::input("format must be object, fitted, bound-json, python, rust, javascript or onnx")),
    }
}

pub fn run(args: ModelArgs) -> anyhow::Result<()> {
    // Authenticate packaged executables before any store access, including MCP initialization.
    crate::bootstrap::open()?;
    let value = match args.command {
        ModelCommand::Import {
            input,
            manifest,
            root,
        } => {
            let object = match (input, manifest) {
                (Some(path), None) => open(&path)?,
                (None, Some(path)) => import_manifest(&path)?,
                _ => return Err(Failure::input("choose exactly one import source")),
            };
            let path = object.save(&root, max_bytes())?;
            json!({"path": path, "inspection": object.inspect()?})
        }
        ModelCommand::Inspect { model } => json!(open(&model)?.inspect()?),
        ModelCommand::Verify {
            model,
            require_executable,
        } => {
            let object = open(&model)?;
            if require_executable {
                object.verify_executable()?;
            }
            json!(object.inspect()?)
        }
        ModelCommand::Export {
            model,
            format,
            output,
        } => {
            let object = open(&model)?;
            if format == "object" {
                json!({"path": object.save(&output, max_bytes())?, "id": object.id()})
            } else {
                let bytes = export_bytes(&object, &format)?;
                objects::write_new(&output, &bytes)?;
                json!({"path": output, "bytes": bytes.len(), "sha256": jope_runtime::sha256_hex(&bytes), "format": format})
            }
        }
        ModelCommand::List { root } => list(&root)?,
        ModelCommand::Session { command } => match command {
            SessionCommand::Show { name, root } => json!(sessions::read(&root, &name)?),
            SessionCommand::Record {
                name,
                id,
                expected_head,
                label,
                root,
            } => json!(sessions::append(
                &root,
                &name,
                &expected_head,
                Some(&id),
                None,
                &label,
                max_bytes()
            )?),
            SessionCommand::Rollback {
                name,
                version,
                expected_head,
                label,
                root,
            } => json!(sessions::append(
                &root,
                &name,
                &expected_head,
                None,
                Some(version),
                &label,
                max_bytes()
            )?),
        },
        ModelCommand::Mcp { root } => return serve(&root),
    };
    crate::print_json(&value);
    Ok(())
}

fn tool(
    name: &str,
    description: &str,
    properties: Value,
    required: &[&str],
    readonly: bool,
) -> Value {
    json!({"name": name, "description": description,
        "inputSchema": {"type": "object", "properties": properties, "required": required, "additionalProperties": false},
        "annotations": {"readOnlyHint": readonly, "destructiveHint": false, "openWorldHint": false}})
}
fn tools_list() -> Value {
    let id = json!({"type": "string", "pattern": "^sha256-[0-9a-f]{64}$"});
    json!({"tools": [
        tool("model_list", "List committed and invalid objects in the configured store.", json!({}), &[], true),
        tool("model_inspect", "Verify hashes and report execution capability separately.", json!({"id": id}), &["id"], true),
        tool("model_verify", "Require both byte integrity and a supported state decoder.", json!({"id": id}), &["id"], true),
        tool("model_export", "Return exact portable object manifest and hexadecimal parts, without filesystem writes.", json!({"id": id}), &["id"], true),
        tool("model_import", "Commit a portable object to the configured store. No executable code is loaded.", json!({"manifest": {"type":"string"}, "parts_hex": {"type":"object", "additionalProperties":{"type":"string"}}}), &["manifest", "parts_hex"], false),
        tool("model_session_show", "Read version history and the active model path.", json!({"name":{"type":"string"}}), &["name"], true),
        tool("model_session_record", "Append a retained model version, comparing the observed head.", json!({"name":{"type":"string"}, "id": id, "expected_head":{"type":"string"}, "label":{"type":"string"}}), &["name","id","expected_head"], false),
        tool("model_session_rollback", "Append a selection of an earlier version; preserve every candidate.", json!({"name":{"type":"string"}, "version":{"type":"integer","minimum":1}, "expected_head":{"type":"string"}, "label":{"type":"string"}}), &["name","version","expected_head"], false),
        tool("model_predict", "Load an appliance object and infer from inline CSV; never refits.", json!({"id": id, "csv": {"type": "string"}}), &["id", "csv"], true)
    ]})
}

fn object_path(root: &Path, id: &str) -> anyhow::Result<PathBuf> {
    sessions::object_path(root, id)
}
fn encode_hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{b:02x}")).collect()
}
fn decode_hex(text: &str) -> anyhow::Result<Vec<u8>> {
    if !text.len().is_multiple_of(2) || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(Failure::input("invalid hexadecimal part"));
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).map_err(|e| Failure::input(e.to_string())))
        .collect()
}

mod mcp;
#[cfg(feature = "mcp")]
pub(crate) use mcp::call_mcp_legacy_tool;
use mcp::serve;
#[cfg(all(test, feature = "mcp"))]
mod tests;
