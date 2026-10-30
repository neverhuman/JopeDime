use serde_json::{json, Value};

pub const INSTRUCTIONS: &str = "MCP conformance fixture: echo text or delay a bounded echo.";

pub fn tools() -> Value {
    let schema = json!({"type":"object", "required":["text"],
        "properties":{"text":{"type":"string","maxLength":4096},
        "delay_ms":{"type":"integer","minimum":0,"maximum":2000}},
        "additionalProperties":false});
    let definitions = ["echo", "slow_echo"].map(|name| {
        json!({
        "name":name,"description":"Echo a bounded string; slow_echo accepts delay_ms.",
        "inputSchema":schema,"outputSchema":{"type":"object","required":["text"],
            "properties":{"text":{"type":"string"}},"additionalProperties":false},
        "annotations":{"readOnlyHint":true,"idempotentHint":true,"openWorldHint":false}})
    });
    json!({"tools":definitions})
}

pub fn arguments(value: &Value) -> Result<(&str, u64), String> {
    let fields = value.as_object().ok_or("arguments must be an object")?;
    if fields.keys().any(|key| key != "text" && key != "delay_ms") {
        return Err("unknown argument".into());
    }
    let text = value["text"].as_str().ok_or("text must be a string")?;
    if text.chars().count() > 4096 {
        return Err("text exceeds 4096 characters".into());
    }
    let delay = match fields.get("delay_ms") {
        Some(value) => value
            .as_u64()
            .filter(|delay| *delay <= 2000)
            .ok_or("delay_ms must be an integer in 0..=2000")?,
        None => 0,
    };
    Ok((text, delay))
}

pub fn result(text: &str) -> Value {
    let value = json!({"text":text});
    json!({"content":[{"type":"text","text":value.to_string()}],
        "structuredContent":value,"isError":false})
}

pub fn tool_error(message: &str) -> Value {
    json!({"content":[{"type":"text","text":message}],"isError":true})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_input_matches_the_published_character_limit() {
        let limit = tools()["tools"][0]["inputSchema"]["properties"]["text"]["maxLength"]
            .as_u64()
            .unwrap() as usize;
        assert!(arguments(&json!({"text":"☃".repeat(limit)})).is_ok());
        assert!(arguments(&json!({"text":"☃".repeat(limit + 1)})).is_err());
    }
}
