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

fn input_schema(name: &str) -> Value {
    let definitions = jope_runtime::fitness::schema()["$defs"].clone();
    let fitness = json!({"oneOf":[{"$ref":"#/$defs/fitness"},
        {"type":"string","description":"A catalog name or name:key=value,... shorthand"}]});
    match name {
        "fitness_list" => json!({
            "type":"object","properties":{},"additionalProperties":false
        }),
        "fitness_validate" => json!({
            "$defs":definitions,"type":"object","required":["fitness"],
            "properties":{"fitness":fitness},"additionalProperties":false
        }),
        "fitness_evaluate" => json!({
            "$defs":definitions,"type":"object","required":["fitness","targets","predictions"],
            "properties":{"fitness":fitness,
                "targets":{"type":"array","items":{"type":"number"},"minItems":1},
                "predictions":{"type":"array","items":{"type":"number"},"minItems":1},
                "columns":{"type":"object","additionalProperties":{"type":"array","items":{"type":"number"}}}},
            "additionalProperties":false
        }),
        "regression_fit_predict" => json!({
            "$defs":definitions,"type":"object","required":["train_csv","input_csv","target"],
            "properties":{"train_csv":{"type":"string"},"input_csv":{"type":"string"},"target":{"type":"string"},
                "options":{"type":"object","additionalProperties":false,"properties":{
                    "method":{"type":"string","default":"auto"},
                    "candidates":{"type":"array","items":{"type":"string"},"maxItems":32},
                    "folds":{"type":"integer","minimum":2,"maximum":20,"default":5},"fitness":fitness,
                    "validation":{"oneOf":[
                        {"type":"object","required":["kind"],"additionalProperties":false,"properties":{"kind":{"const":"iid"}}},
                        {"type":"object","required":["kind","gap"],"additionalProperties":false,"properties":{"kind":{"const":"forward"},"gap":{"type":"integer","minimum":0}}}]},
                    "calibration":{"enum":["auto","none","shift","affine"],"default":"auto"}}}},
            "additionalProperties":false
        }),
        "model_session_show" => json!({
            "type":"object","required":["name"],"properties":{"name":{"type":"string"}},"additionalProperties":false
        }),
        "model_session_record" => json!({
            "type":"object","required":["name","id","expected_head"],
            "properties":{"name":{"type":"string"},"id":{"type":"string","pattern":"^sha256-[0-9a-f]{64}$"},
                "expected_head":{"type":"string"},"label":{"type":"string"}},"additionalProperties":false
        }),
        "model_session_rollback" => json!({
            "type":"object","required":["name","version","expected_head"],
            "properties":{"name":{"type":"string"},"version":{"type":"integer","minimum":1},
                "expected_head":{"type":"string"},"label":{"type":"string"}},"additionalProperties":false
        }),
        "narration_read" => json!({
            "type":"object","properties":{"run_id":{"type":"string"},
                "after":{"type":"string","pattern":"^(0|[1-9][0-9]*)$"},
                "limit":{"type":"integer","minimum":1,"maximum":256}},
            "required":["run_id"],"additionalProperties":false
        }),
        "narration_inspect" => json!({
            "type":"object","properties":{"run_id":{"type":"string"}},
            "required":["run_id"],"additionalProperties":false
        }),
        _ => json!({"type":"object","properties":{},"additionalProperties":false}),
    }
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
                "inputSchema": input_schema(name),
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
    fn alias_schemas_preserve_legacy_arguments() {
        let tools = tools();
        let find = |name: &str| tools.iter().find(|tool| tool["name"] == name).unwrap();
        assert_eq!(
            find("fitness_validate")["inputSchema"]["required"][0],
            "fitness"
        );
        assert_eq!(
            find("regression_fit_predict")["inputSchema"]["required"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            find("model_session_record")["inputSchema"]["required"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            find("narration_read")["inputSchema"]["required"][0],
            "run_id"
        );
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
