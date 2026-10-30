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
        "onnx" => Err(Failure::backend("ONNX conversion is not implemented; retain the complete model object. A custom-domain ONNX graph also requires a compatible operator runtime.")),
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IdArgs {
    id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PredictArgs {
    id: String,
    csv: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportArgs {
    manifest: String,
    parts_hex: BTreeMap<String, String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionName {
    name: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionRecord {
    name: String,
    id: String,
    expected_head: String,
    #[serde(default)]
    label: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionRollback {
    name: String,
    version: u64,
    expected_head: String,
    #[serde(default)]
    label: String,
}
fn arguments<T: serde::de::DeserializeOwned>(value: Value) -> anyhow::Result<T> {
    serde_json::from_value(value).map_err(|e| Failure::input(e.to_string()))
}
fn call_tool(root: &Path, name: &str, args: Value) -> anyhow::Result<Value> {
    match name {
        "model_session_show" => {
            let args: SessionName = arguments(args)?;
            Ok(json!(sessions::read(root, &args.name)?))
        }
        "model_session_record" => {
            let args: SessionRecord = arguments(args)?;
            Ok(json!(sessions::append(
                root,
                &args.name,
                &args.expected_head,
                Some(&args.id),
                None,
                &args.label,
                MAX_FRAME as u64
            )?))
        }
        "model_session_rollback" => {
            let args: SessionRollback = arguments(args)?;
            Ok(json!(sessions::append(
                root,
                &args.name,
                &args.expected_head,
                None,
                Some(args.version),
                &args.label,
                MAX_FRAME as u64
            )?))
        }
        "model_list" => {
            if args != json!({}) {
                return Err(Failure::input("model_list accepts no arguments"));
            }
            list(root)
        }
        "model_import" => {
            let args: ImportArgs = arguments(args)?;
            let parts = args
                .parts_hex
                .into_iter()
                .map(|(k, v)| Ok((k, decode_hex(&v)?)))
                .collect::<anyhow::Result<_>>()?;
            let object = ModelObject::decode(args.manifest.into_bytes(), parts, MAX_FRAME as u64)?;
            let path = object.save(root, MAX_FRAME as u64)?;
            Ok(json!({"path":path,"inspection":object.inspect()?}))
        }
        "model_predict" => {
            let args: PredictArgs = arguments(args)?;
            if args.csv.len() > 1 << 20 {
                return Err(Failure::input("prediction CSV exceeds 1 MiB"));
            }
            let object = ModelObject::read(&object_path(root, &args.id)?, MAX_FRAME as u64)?;
            let limits = Limits {
                max_input_bytes: 1 << 20,
                max_rows: 10_000,
                ..Limits::default()
            };
            let budget = jope_runtime::budget::Budget::new(None, 1);
            // A Platt-calibrated object predicts whole; `fitted()` refuses to drop its map.
            let predictions = if let Some(contract) = object.calibrated_input()? {
                let csv = jope_runtime::table::parse_csv(args.csv.as_bytes(), &limits)?;
                let (rows, _) = contract.table(&csv)?;
                jope_runtime::api::predict_object(&object, &rows, &budget)?
            } else {
                let fitted = object.fitted()?;
                let csv = jope_runtime::table::parse_csv(args.csv.as_bytes(), &limits)?;
                let (rows, _) = jope_runtime::table::prediction_ignoring(
                    &csv,
                    fitted.schema(),
                    &fitted.header.target,
                    &fitted.header.auxiliary_columns,
                )?;
                jope_runtime::api::predict(&fitted, &rows, &budget)?
            };
            Ok(
                json!({"id":args.id,"predictions_csv":String::from_utf8(jope_runtime::api::predictions_csv(&predictions)?)?}),
            )
        }
        "model_inspect" | "model_verify" | "model_export" => {
            let args: IdArgs = arguments(args)?;
            let object = ModelObject::read(&object_path(root, &args.id)?, MAX_FRAME as u64)?;
            if name == "model_verify" {
                object.verify_executable()?;
            }
            if name == "model_export" {
                let parts: BTreeMap<_, _> = object
                    .parts()
                    .iter()
                    .map(|(k, v)| (k, encode_hex(v)))
                    .collect();
                Ok(
                    json!({"id":object.id(),"manifest":String::from_utf8(object.manifest_bytes().to_vec())?,"parts_hex":parts}),
                )
            } else {
                Ok(json!(object.inspect()?))
            }
        }
        _ => Err(Failure::input("unknown model tool")),
    }
}

/// Call the existing model-session handler from the shared MCP adapter.
#[cfg(feature = "mcp")]
pub(crate) fn call_mcp_legacy_tool(root: &Path, name: &str, args: Value) -> anyhow::Result<Value> {
    call_tool(root, name, args)
}

fn serve(root: &Path) -> anyhow::Result<()> {
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    loop {
        let mut line = Vec::new();
        let count = (&mut input)
            .take(MAX_FRAME as u64 + 1)
            .read_until(b'\n', &mut line)
            .map_err(|e| Failure::input(e.to_string()))?;
        if count == 0 {
            break;
        }
        if line.len() > MAX_FRAME {
            return Err(Failure::input("MCP frame exceeds 8 MiB"));
        }
        let request: Value = match jope_runtime::strict_json::decode(&line) {
            Ok(request) => request,
            Err(_) => {
                writeln!(
                    output,
                    "{}",
                    json!({"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"parse error"}})
                )?;
                output.flush()?;
                continue;
            }
        };
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        if request.get("jsonrpc") != Some(&json!("2.0"))
            || !request.get("method").is_some_and(Value::is_string)
            || !(id.is_null() || id.is_string() || id.is_number())
        {
            writeln!(
                output,
                "{}",
                json!({"jsonrpc":"2.0","id":null,"error":{"code":-32600,"message":"invalid request"}})
            )?;
            output.flush()?;
            continue;
        }
        let method = request["method"].as_str().unwrap_or_default();
        // Notifications never execute tools or write objects.
        if request.get("id").is_none() {
            continue;
        }
        let result = match method {
            "initialize" => Ok(
                json!({"protocolVersion":"2025-11-25","capabilities":{"tools":{"listChanged":false}},"serverInfo":{"name":"jopedime-models","version":env!("CARGO_PKG_VERSION")}}),
            ),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(tools_list()),
            "tools/call" => {
                let params = &request["params"];
                let name = params["name"].as_str().unwrap_or_default();
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                let response = call_tool(root, name, args);
                Ok(match response {
                    Ok(value) => {
                        json!({"content":[{"type":"text","text":value.to_string()}],"structuredContent":value,"isError":false})
                    }
                    Err(e) => {
                        json!({"content":[{"type":"text","text":format!("{e:#}")}],"isError":true,
                        "structuredContent":{"exit_code":jope_runtime::error::kind_of(&e).exit_code()}})
                    }
                })
            }
            _ => Err((-32601, "method not found")),
        };
        let response = match result {
            Ok(result) => json!({"jsonrpc":"2.0","id":id,"result":result}),
            Err((code, message)) => {
                json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}})
            }
        };
        if response.to_string().len() > MAX_FRAME {
            writeln!(
                output,
                "{}",
                json!({"jsonrpc":"2.0","id":id,"error":{"code":-32000,"message":"response exceeds 8 MiB; use the CLI for large objects"}})
            )?;
        } else {
            writeln!(output, "{response}")?;
        }
        output
            .flush()
            .map_err(|e| Failure::new(FailureKind::Output, e.to_string()))?;
    }
    Ok(())
}

#[cfg(all(test, feature = "mcp"))]
mod tests {
    use super::*;

    #[test]
    fn mcp_wrapper_preserves_session_show_result() {
        let root = tempfile::tempdir().unwrap();
        let result = call_mcp_legacy_tool(
            root.path(),
            "model_session_show",
            json!({"name":"baseline"}),
        )
        .unwrap();
        assert_eq!(result["schema"], sessions::SCHEMA);
        assert_eq!(result["session"], "baseline");
        assert_eq!(result["head"], "empty");
        assert_eq!(result["versions"], json!([]));
    }
}
