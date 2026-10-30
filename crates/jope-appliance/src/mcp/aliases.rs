//! Deprecated MCP names retained by delegating to their existing handlers.
use rmcp::model::CallToolResult;
use serde_json::{json, Value};

const FITNESS: &[&str] = &[
    "fitness_list",
    "fitness_validate",
    "fitness_evaluate",
    "regression_fit_predict",
];
const MODEL_SESSIONS: &[&str] = &[
    "model_session_show",
    "model_session_record",
    "model_session_rollback",
];
const REMOVED: &[&str] = &["narration_read", "narration_inspect"];

pub(super) fn contains(name: &str) -> bool {
    FITNESS.contains(&name) || MODEL_SESSIONS.contains(&name) || REMOVED.contains(&name)
}

pub(super) fn tools() -> Vec<Value> {
    FITNESS
        .iter()
        .chain(MODEL_SESSIONS)
        .chain(REMOVED)
        .map(|name| {
            let replacement = if REMOVED.contains(name) {
                "none"
            } else {
                "none (legacy compatibility only)"
            };
            json!({
                "name": name,
                "description": format!("Deprecated compatibility alias. Replacement: {replacement}."),
                "inputSchema": {"type":"object", "additionalProperties":true},
                "annotations": {"readOnlyHint": *name == "fitness_list" || *name == "fitness_validate" || *name == "fitness_evaluate" || *name == "model_session_show", "openWorldHint":false},
                "_meta": {"deprecated": true, "replacement": replacement}
            })
        })
        .collect()
}

pub(super) fn call(name: &str, args: Value) -> CallToolResult {
    if REMOVED.contains(&name) {
        return CallToolResult::structured_error(json!({
            "code":"tool_removed",
            "message":format!("{name} was removed; replacement: none"),
            "replacement":"none"
        }));
    }

    let result = if FITNESS.contains(&name) {
        let mut server =
            jope_runtime::service::Server::new(jope_runtime::fitted::CreatedBy::default());
        let initialize = server.dispatch(
            json!({"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"jopedime-mcp-adapter","version":env!("CARGO_PKG_VERSION")}}}),
            true,
        );
        if initialize.is_some_and(|response| response.get("error").is_some()) {
            return CallToolResult::structured_error(json!({
                "code":"legacy_handler_failed",
                "message":"fitness handler initialization failed"
            }));
        }
        server.dispatch(
            json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
            true,
        );
        server
            .dispatch(
                json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":name,"arguments":args}}),
                true,
            )
            .and_then(|response| response.get("result").cloned())
    } else if MODEL_SESSIONS.contains(&name) {
        match crate::model::call_mcp_legacy_tool(std::path::Path::new("models"), name, args) {
            Ok(value) => Some(json!({
                "content":[{"type":"text","text":value.to_string()}],
                "structuredContent":value,
                "isError":false
            })),
            Err(error) => Some(json!({
                "content":[{"type":"text","text":format!("{error:#}")}],
                "structuredContent":{"exit_code":jope_runtime::error::kind_of(&error).exit_code()},
                "isError":true
            })),
        }
    } else {
        return CallToolResult::structured_error(json!({
            "code":"unknown_tool",
            "message":format!("unknown legacy tool {name}")
        }));
    };

    match result {
        Some(value) => match serde_json::from_value::<CallToolResult>(value) {
            Ok(result) => result,
            Err(_) => CallToolResult::structured_error(json!({
                "code":"legacy_handler_failed",
                "message":"legacy handler returned an invalid tool result"
            })),
        },
        None => CallToolResult::structured_error(json!({
            "code":"legacy_handler_failed",
            "message":"legacy handler returned no response"
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_every_legacy_name_as_deprecated_with_replacement() {
        let tools = tools();
        assert_eq!(tools.len(), 9);
        for tool in tools {
            assert_eq!(tool["_meta"]["deprecated"], true);
            assert!(tool["_meta"]["replacement"].is_string());
        }
    }

    #[test]
    fn removed_narration_tools_return_structured_removal_error() {
        for name in REMOVED {
            let result = call(name, json!({}));
            assert_eq!(result.is_error, Some(true));
            assert_eq!(
                result.structured_content.as_ref().unwrap()["code"],
                "tool_removed"
            );
            assert_eq!(
                result.structured_content.as_ref().unwrap()["replacement"],
                "none"
            );
        }
    }

    #[test]
    fn fitness_legacy_handler_keeps_its_result_shape() {
        let result = call("fitness_list", json!({}));
        assert_eq!(result.is_error, Some(false));
        assert!(result.structured_content.unwrap()["fitness"].is_array());
    }
}
