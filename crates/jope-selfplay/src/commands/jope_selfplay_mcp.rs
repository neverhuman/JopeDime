//! Stdio MCP adapter for the Rust self-play freshness authority.

use anyhow::Result;
use clap::Parser;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use crate::freshness_protocol;

use freshness_protocol::{socket_request, tools, MCP_PROTOCOL_VERSION, SERVER_VERSION, TOOL_NAMES};
use jope_campaigns::paths;

jope_paths::once_str!(
    fn default_root_str => paths::flywheel_root()
);

#[derive(Parser)]
#[command(name = "jope-selfplay-mcp")]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long)]
    socket: Option<PathBuf>,
}

fn response(identifier: Value, result: Result<Value, Value>) -> Value {
    match result {
        Ok(result) => json!({"jsonrpc": "2.0", "id": identifier, "result": result}),
        Err(error) => json!({"jsonrpc": "2.0", "id": identifier, "error": error}),
    }
}

fn handle(message: &Value, socket_path: &std::path::Path) -> Option<Value> {
    let identifier = message.get("id").cloned().unwrap_or(Value::Null);
    let method = message.get("method").and_then(Value::as_str).unwrap_or("");
    match method {
        "notifications/initialized" => None,
        "initialize" => Some(response(
            identifier,
            Ok(json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": false}},
                "serverInfo": {
                    "name": "jope-selfplay-freshness",
                    "version": SERVER_VERSION
                },
                "instructions": concat!(
                    "Read-only provenance and receipt-gated freshness mutations. ",
                    "The server cannot execute arbitrary SCQ commands or cancel running work."
                )
            })),
        )),
        "tools/list" => Some(response(identifier, Ok(json!({"tools": tools()})))),
        "tools/call" => {
            let params = message.get("params").unwrap_or(&Value::Null);
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            if !TOOL_NAMES.contains(&name) {
                return Some(response(
                    identifier,
                    Err(json!({"code": -32601, "message": format!("unknown tool {name}")})),
                ));
            }
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            let result = match socket_request(socket_path, name, &arguments) {
                Ok(value) => json!({
                    "content": [{
                        "type": "text",
                        "text": serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_owned())
                    }],
                    "structuredContent": value,
                    "isError": false
                }),
                Err(error) => json!({
                    "content": [{"type": "text", "text": error.to_string()}],
                    "isError": true
                }),
            };
            Some(response(identifier, Ok(result)))
        }
        _ => Some(response(
            identifier,
            Err(json!({
                "code": -32601,
                "message": format!("unsupported method {method}")
            })),
        )),
    }
}

pub fn main(args: Vec<String>) -> Result<()> {
    let args = Args::parse_from(&args);
    let socket_path = args
        .socket
        .unwrap_or_else(|| args.root.join("ops/freshness.sock"));
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let outgoing = match serde_json::from_str::<Value>(&line) {
            Ok(message) => handle(&message, &socket_path),
            Err(error) => Some(response(
                Value::Null,
                Err(json!({
                    "code": -32603,
                    "message": format!("JSON parse error: {error}")
                })),
            )),
        };
        if let Some(outgoing) = outgoing {
            serde_json::to_writer(&mut stdout, &outgoing)?;
            stdout.write_all(b"\n")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialize_describes_no_arbitrary_scheduler_access() {
        let response = handle(
            &json!({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
            std::path::Path::new("/unused"),
        )
        .unwrap();
        assert!(response["result"]["instructions"]
            .as_str()
            .unwrap()
            .contains("cannot execute arbitrary SCQ commands"));
    }

    #[test]
    fn unknown_tool_is_rejected_before_socket_access() {
        let response = handle(
            &json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "cancel_job", "arguments": {"job_id": "danger"}}
            }),
            std::path::Path::new("/unused"),
        )
        .unwrap();
        assert_eq!(response["error"]["code"], -32601);
    }
}
