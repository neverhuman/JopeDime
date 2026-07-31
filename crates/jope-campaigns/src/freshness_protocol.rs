use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

pub const MCP_PROTOCOL_VERSION: &str = "2025-06-18";
pub const SERVER_VERSION: &str = "2.0.0";

pub const TOOL_NAMES: [&str; 9] = [
    "get_freshness_status",
    "get_active_checkpoint",
    "list_queue_epochs",
    "explain_job_provenance",
    "preview_stale_retirement",
    "apply_stale_retirement",
    "pause_admission",
    "resume_admission",
    "request_micro_wave",
];

fn empty_schema() -> Value {
    json!({"type": "object", "properties": {}, "additionalProperties": false})
}

fn mutation_schema(with_reason: bool) -> Value {
    let mut properties = serde_json::Map::from_iter([
        (
            "preview_receipt_sha256".to_owned(),
            json!({"type": "string", "pattern": "^[0-9a-f]{64}$"}),
        ),
        (
            "idempotency_key".to_owned(),
            json!({"type": "string", "minLength": 1}),
        ),
    ]);
    if with_reason {
        properties.insert("reason".to_owned(), json!({"type": "string"}));
    }
    json!({
        "type": "object",
        "properties": properties,
        "required": ["preview_receipt_sha256", "idempotency_key"],
        "additionalProperties": false
    })
}

pub fn tools() -> Vec<Value> {
    vec![
        json!({
            "name": "get_freshness_status",
            "description": "Return checkpoint freshness, queue epochs, and latency health.",
            "inputSchema": empty_schema()
        }),
        json!({
            "name": "get_active_checkpoint",
            "description": "Return the admitted checkpoint and immutable epoch.",
            "inputSchema": empty_schema()
        }),
        json!({
            "name": "list_queue_epochs",
            "description": "List useful, held, running, and dispatching work by epoch.",
            "inputSchema": empty_schema()
        }),
        json!({
            "name": "explain_job_provenance",
            "description": "Join one job to its manifest, claim, epoch, and pair identities.",
            "inputSchema": {
                "type": "object",
                "properties": {"job_id": {"type": "string", "minLength": 1}},
                "required": ["job_id"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "preview_stale_retirement",
            "description": "Preview stale holds/retirements without changing scheduler state.",
            "inputSchema": empty_schema()
        }),
        json!({
            "name": "apply_stale_retirement",
            "description": "Apply a preview after rechecking that every target is held and unstarted.",
            "inputSchema": mutation_schema(false)
        }),
        json!({
            "name": "pause_admission",
            "description": "Pause new pair admission without interrupting active attempts.",
            "inputSchema": mutation_schema(false)
        }),
        json!({
            "name": "resume_admission",
            "description": "Resume new pair admission after a reviewed preview.",
            "inputSchema": mutation_schema(false)
        }),
        json!({
            "name": "request_micro_wave",
            "description": "Write an idempotent current-checkpoint 128-dataset micro-wave request.",
            "inputSchema": mutation_schema(true)
        }),
    ]
}

pub fn socket_request(socket_path: &Path, method: &str, arguments: &Value) -> Result<Value> {
    if !TOOL_NAMES.contains(&method) {
        bail!("unsupported freshness method {method}");
    }
    let mut stream = UnixStream::connect(socket_path)
        .with_context(|| format!("connect {}", socket_path.display()))?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    serde_json::to_writer(
        &mut stream,
        &json!({"method": method, "arguments": arguments}),
    )?;
    stream.write_all(b"\n")?;
    stream.flush()?;

    let mut line = String::new();
    BufReader::new(stream)
        .take(4 << 20)
        .read_line(&mut line)
        .context("read freshness socket response")?;
    if line.is_empty() {
        bail!("freshness socket closed without a response");
    }
    let response: Value = serde_json::from_str(&line).context("parse freshness response")?;
    if response.get("ok").and_then(Value::as_bool) != Some(true) {
        bail!(
            "{}",
            response
                .get("detail")
                .or_else(|| response.get("error"))
                .and_then(Value::as_str)
                .unwrap_or("freshness request failed")
        );
    }
    response
        .get("result")
        .cloned()
        .context("freshness response is missing result")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_surface_is_exact_and_unique() {
        let listed = tools();
        let names = listed
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(names, TOOL_NAMES);
        assert_eq!(
            names
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            9
        );
    }

    #[test]
    fn mutations_require_receipt_and_idempotency() {
        for tool in tools().into_iter().skip(5) {
            assert_eq!(
                tool["inputSchema"]["required"],
                json!(["preview_receipt_sha256", "idempotency_key"])
            );
        }
    }
}
