//! Explicit MCP 2025-11-25 stdio compatibility profile; finite read-only tools.
use super::index::{sequence, Index, MAX_PAGE_BYTES, MAX_PAGE_RECORDS};
use anyhow::{ensure, Result};
use jope_narrator::Cursor;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
const MAX_REQUEST_BYTES: usize = 64 * 1024;
// Structured data plus its text form can expand to three times the page bytes.
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

fn tools() -> Value {
    json!({"tools":[
        {"name":"narration_read","description":"Read one bounded page of retained search commentary. Cursor is exclusive and run-scoped. No search control.","inputSchema":{"type":"object","properties":{"run_id":{"type":"string"},"after":{"type":"string","pattern":"^(0|[1-9][0-9]*)$"},"limit":{"type":"integer","minimum":1,"maximum":256}},"required":["run_id"],"additionalProperties":false},"annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}},
        {"name":"narration_inspect","description":"Inspect journal identity, validated cursor and partial tail. This does not establish producer liveness.","inputSchema":{"type":"object","properties":{"run_id":{"type":"string"}},"required":["run_id"],"additionalProperties":false},"annotations":{"readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false}}
    ]})
}
#[derive(Default)]
struct State {
    initialized: bool,
    ready: bool,
}
fn error(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}})
}
fn tool_result(id: Value, value: Value) -> Value {
    let text = value.to_string();
    json!({"jsonrpc":"2.0","id":id,"result":{"content":[{"type":"text","text":text}],"structuredContent":value,"isError":false}})
}
fn dispatch(index: &mut Index, state: &mut State, request: Value) -> Option<Value> {
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    if request.get("jsonrpc") != Some(&Value::String("2.0".into()))
        || !request.is_object()
        || request.get("method").and_then(Value::as_str).is_none()
        || request.get("id").is_some_and(|id| {
            !id.is_string() && !id.is_null() && id.as_i64().is_none() && id.as_u64().is_none()
        })
    {
        return Some(error(id, -32600, "Invalid Request"));
    }
    let method = request["method"].as_str().unwrap_or_default();
    if request.get("id").is_none() {
        if method == "notifications/initialized" && state.initialized {
            state.ready = true;
        }
        return None;
    }
    if method == "ping" {
        return Some(json!({"jsonrpc":"2.0","id":id,"result":{}}));
    }
    if method == "initialize" {
        if state.initialized {
            return Some(error(id, -32600, "Already initialized"));
        }
        if request
            .pointer("/params/protocolVersion")
            .and_then(Value::as_str)
            .is_none()
            || !request
                .pointer("/params/capabilities")
                .is_some_and(Value::is_object)
            || request
                .pointer("/params/clientInfo/name")
                .and_then(Value::as_str)
                .is_none()
            || request
                .pointer("/params/clientInfo/version")
                .and_then(Value::as_str)
                .is_none()
        {
            return Some(error(id, -32602, "Invalid initialize parameters"));
        }
        state.initialized = true;
        return Some(
            json!({"jsonrpc":"2.0","id":id,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{}},"serverInfo":{"name":"jope-search-commentary","version":env!("CARGO_PKG_VERSION")},"instructions":"Read-only retained search commentary derived from edits and measurements. Decimal cursors are run-scoped and exclusive. Narration does not expose hidden model thoughts."}}),
        );
    }
    if !state.ready {
        return Some(error(
            id,
            -32002,
            "Initialize and send notifications/initialized first",
        ));
    }
    match method {
        "tools/list" => Some(json!({"jsonrpc":"2.0","id":id,"result":tools()})),
        "tools/call" => {
            let Some(name) = request.pointer("/params/name").and_then(Value::as_str) else {
                return Some(error(id, -32602, "Missing tool name"));
            };
            let args = request
                .pointer("/params/arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            if !args.is_object() || args["run_id"].as_str() != Some(index.run_id.as_str()) {
                return Some(error(id, -32602, "run_id must match selected journal"));
            }
            let result = match name {
                "narration_read" => {
                    if args.as_object().is_some_and(|m| {
                        m.keys()
                            .any(|k| !matches!(k.as_str(), "run_id" | "after" | "limit"))
                    }) {
                        return Some(error(id, -32602, "Unknown arguments"));
                    }
                    let after = match args.get("after") {
                        None => "0",
                        Some(Value::String(s)) => s,
                        _ => return Some(error(id, -32602, "after must be a decimal string")),
                    };
                    let limit = match args.get("limit") {
                        None => 64,
                        Some(v) => match v.as_u64().and_then(|n| usize::try_from(n).ok()) {
                            Some(n) if (1..=MAX_PAGE_RECORDS).contains(&n) => n,
                            _ => return Some(error(id, -32602, "limit out of range")),
                        },
                    };
                    if sequence(after).is_err() {
                        return Some(error(id, -32602, "Invalid cursor sequence"));
                    }
                    index
                        .page(
                            &Cursor {
                                run_id: index.run_id.clone(),
                                sequence: after.into(),
                            },
                            limit,
                            MAX_PAGE_BYTES,
                        )
                        .and_then(|page| Ok(serde_json::to_value(page)?))
                }
                "narration_inspect" => {
                    if args
                        .as_object()
                        .is_some_and(|m| m.keys().any(|k| k != "run_id"))
                    {
                        return Some(error(id, -32602, "Unknown arguments"));
                    }
                    index
                        .status()
                        .and_then(|status| Ok(serde_json::to_value(status)?))
                }
                _ => return Some(error(id, -32602, "Unknown tool")),
            };
            Some(match result {
                Ok(value) => tool_result(id, value),
                Err(_) => {
                    json!({"jsonrpc":"2.0","id":id,"result":{"content":[{"type":"text","text":"Journal unavailable or cursor does not resolve to a retained boundary"}],"isError":true}})
                }
            })
        }
        _ => Some(error(id, -32601, "Method not found")),
    }
}
fn request_line(reader: &mut impl BufRead) -> Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    loop {
        let bytes = reader.fill_buf()?;
        if bytes.is_empty() {
            ensure!(line.is_empty(), "incomplete MCP request at EOF");
            return Ok(None);
        }
        let n = bytes
            .iter()
            .position(|b| *b == b'\n')
            .map_or(bytes.len(), |n| n + 1);
        ensure!(
            line.len().saturating_add(n) <= MAX_REQUEST_BYTES,
            "MCP request exceeds limit"
        );
        let complete = bytes[n - 1] == b'\n';
        line.extend_from_slice(&bytes[..n]);
        reader.consume(n);
        if complete {
            return Ok(Some(line));
        }
    }
}
/// Newline-delimited JSON-RPC only. Tool responses are finite; logs belong on stderr.
pub fn mcp(index: &mut Index, input: &mut impl BufRead, output: &mut impl Write) -> Result<()> {
    let mut state = State::default();
    while let Some(line) = request_line(input)? {
        let response = match serde_json::from_slice::<Value>(&line) {
            Ok(value) => dispatch(index, &mut state, value),
            Err(_) => Some(error(Value::Null, -32700, "Parse error")),
        };
        if let Some(response) = response {
            let bytes = serde_json::to_vec(&response)?;
            ensure!(
                bytes.len() <= MAX_RESPONSE_BYTES,
                "MCP response exceeds limit"
            );
            output.write_all(&bytes)?;
            output.write_all(b"\n")?;
            output.flush()?;
        }
    }
    Ok(())
}
