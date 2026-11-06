//! `send_feedback`: an agent working through this MCP server tells the VEOX team what worked well,
//! what failed, or what it wishes existed.
//!
//! Not an engine operation, so it is not in `OPERATIONS` and has no `jope/apiSchemas` entry: it
//! posts one note to the VEOX workspace through `crate::cloud::feedback`, the same path as the
//! terminal's `/feedback` and `jain cloud feedback`. That module documents what leaves the
//! machine (only the text supplied here, the binary's name and version, and tool names).
use rmcp::model::CallToolResult;
use serde_json::{json, Value};

pub(super) const NAME: &str = "send_feedback";

const DESCRIPTION: &str = "Send the VEOX team feedback about JopeDime/jAIn: what worked well \
(kind working_well), a problem (issue), something you or your user wish existed \
(feature_request), or anything else (other). Use it when a task finishes, when something fails \
or confuses you, or when your user asks for a feature; a short note on what went well is just as \
useful. Set source to agent for your own experience of the tools, or user when relaying your \
user's words, and ask your user before quoting them. Never include dataset rows, file contents, \
paths that identify people, or credentials. Works signed in or not; signed in, the team can \
follow up on the account.";

pub(super) fn tool() -> Value {
    json!({
        "name": NAME,
        "title": "Send feedback to the VEOX team",
        "description": DESCRIPTION,
        "inputSchema": {
            "type": "object",
            "required": ["kind", "summary"],
            "properties": {
                "kind": {"enum": ["working_well", "issue", "feature_request", "other"]},
                "summary": {"type": "string", "minLength": 1, "maxLength": 200,
                    "description": "One line: what happened, or what you want."},
                "details": {"type": "string", "maxLength": 8000,
                    "description": "Steps, what you expected and what happened, or why the feature matters."},
                "source": {"enum": ["agent", "user"], "default": "agent"},
                "tools_used": {"type": "array", "maxItems": 50,
                    "items": {"type": "string", "pattern": "^[a-z0-9_]{1,64}$"},
                    "description": "Names of the tools you called in this task, names only."}
            },
            "additionalProperties": false
        },
        "annotations": {"readOnlyHint": false, "destructiveHint": false,
            "idempotentHint": false, "openWorldHint": true}
    })
}

/// Validate, send, and answer with the line a person would see. Runs on a blocking worker: the
/// workspace client is synchronous.
pub(super) fn call(args: &Value) -> CallToolResult {
    let text = |key: &str| args.get(key).and_then(Value::as_str);
    let built = crate::cloud::feedback::Feedback::new(
        text("kind").unwrap_or(""),
        text("source").unwrap_or("agent"),
        text("summary").unwrap_or(""),
        text("details"),
    )
    .map(|mut feedback| {
        feedback.tools_used = args
            .get("tools_used")
            .and_then(Value::as_array)
            .map(|names| {
                names
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|n| {
                        !n.is_empty()
                            && n.len() <= 64
                            && n.bytes()
                                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
                    })
                    .take(50)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        feedback
    });
    match built.and_then(|feedback| crate::cloud::feedback::send(&feedback)) {
        Ok(message) => CallToolResult::structured(json!({"sent": true, "message": message})),
        Err(error) => CallToolResult::structured_error(json!({
            "code": "feedback_not_sent",
            "message": format!("{error:#}")
        })),
    }
}

#[cfg(test)]
mod feedback_tool_tests {
    use super::*;

    #[test]
    fn invalid_arguments_are_refused_without_a_network_call() {
        let result = call(&json!({"kind": "nonsense", "summary": "x"}));
        assert_eq!(result.is_error, Some(true));
        let result = call(&json!({"kind": "issue", "summary": ""}));
        assert_eq!(result.is_error, Some(true));
    }

    #[test]
    fn the_tool_is_described_for_agents_and_takes_no_extra_fields() {
        let tool = tool();
        assert_eq!(tool["name"], NAME);
        assert_eq!(tool["inputSchema"]["additionalProperties"], false);
        let description = tool["description"].as_str().unwrap();
        assert!(description.contains("ask your user before quoting them"));
        assert!(description.contains("Never include dataset rows"));
    }
}
