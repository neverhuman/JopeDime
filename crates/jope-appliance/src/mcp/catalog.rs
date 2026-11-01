//! MCP envelopes over the canonical operation table. Missing API schemas remain explicit.
use jope_api::registry::{MediaType, Operation, OPERATIONS};
use serde_json::{json, Map, Value};

pub const INSTRUCTIONS: &str = "JopeDime discovers and exports predictive algorithms for tabular data. Typical flow: dataset_register → run_start with dry_run:true to review the plan → run_start → run_status with wait_seconds and after_event to follow progress (read run_events for details) → model_export → model_predict. Runs continue in the background; poll rather than wait inside one call. Always pass idempotency_key on mutating calls and reuse it when retrying. Results describe algorithms and measured scores; search internals are not exposed.";

pub fn tools() -> Value {
    json!({"tools": OPERATIONS.iter().filter(|op| op.mcp_tool.is_some()).map(tool).collect::<Vec<_>>()})
}

pub fn schema_gaps(op: &Operation) -> Vec<&'static str> {
    let mut gaps: Vec<_> = op
        .request_types
        .iter()
        .chain(op.response_types)
        .filter(|ty| ty.media_type == MediaType::Json && ty.json_schema.is_none())
        .map(|ty| ty.name)
        .collect();
    gaps.sort_unstable();
    gaps.dedup();
    gaps
}

fn string() -> Value {
    json!({"type":"string","minLength":1})
}
fn strings() -> Value {
    json!({"type":"array","items":string()})
}
fn object() -> Value {
    json!({"type":"object"})
}
fn integer(max: u64) -> Value {
    json!({"type":"integer","minimum":0,"maximum":max})
}
fn add(schema: &mut Value, name: &str, value: Value, required: bool) {
    schema["properties"]
        .as_object_mut()
        .expect("object properties")
        .insert(name.into(), value);
    if required {
        schema["required"]
            .as_array_mut()
            .expect("required array")
            .push(json!(name));
    }
}

pub fn input_schema(op: &Operation) -> Value {
    let mut s = if op.id == 6 {
        serde_json::to_value(
            op.request_types[0]
                .resolve_schema()
                .expect("RunSpec schema"),
        )
        .expect("schema JSON")
    } else {
        json!({"type":"object","properties":{},"required":[]})
    };
    s.as_object_mut().expect("object schema").remove("$schema");
    s["additionalProperties"] = json!(false);
    match op.id {
        1 => {}
        2 => {
            add(&mut s, "path", string(), false);
            add(
                &mut s,
                "csv_text",
                json!({"type":"string","maxLength":1048576}),
                false,
            );
            add(&mut s, "name", string(), false);
            s["oneOf"] = json!([{"required":["path"],"not":{"required":["csv_text"]}},{"required":["csv_text"],"not":{"required":["path"]}}]);
        }
        3 => {
            add(&mut s, "dataset_id", string(), false);
            add(&mut s, "columns", strings(), false);
        }
        5 => {
            add(&mut s, "spec", json!({"type":["string","object"]}), true);
            add(
                &mut s,
                "task",
                json!({"enum":["regression","binary","auto"]}),
                false,
            );
        }
        6 => {}
        7 => {
            add(&mut s, "run_id", string(), false);
            add(&mut s, "wait_seconds", integer(60), false);
            add(
                &mut s,
                "after_event",
                json!({"type":"string","pattern":"^(0|[1-9][0-9]*)$"}),
                false,
            );
        }
        8 => {
            add(&mut s, "run_id", string(), true);
            add(
                &mut s,
                "action",
                json!({"enum":["pause","continue","cancel","set_effort"]}),
                true,
            );
            add(
                &mut s,
                "effort",
                json!({"enum":["fast","medium","high","ultra"]}),
                false,
            );
            add(&mut s, "budget", object(), false);
            add(
                &mut s,
                "expected_revision",
                json!({"type":"string","pattern":"^(0|[1-9][0-9]*)$"}),
                false,
            );
        }
        9 => {
            add(&mut s, "run_id", string(), true);
            add(&mut s, "seed", json!({"enum":["same","new"]}), true);
            add(&mut s, "warm_start", json!({"type":"boolean"}), false);
            add(&mut s, "overrides", object(), false);
        }
        11 => {
            add(&mut s, "run_id", string(), true);
            add(
                &mut s,
                "after",
                json!({"type":"string","pattern":"^(0|[1-9][0-9]*)$"}),
                false,
            );
            add(&mut s, "kinds", strings(), false);
            add(
                &mut s,
                "limit",
                json!({"type":"integer","minimum":1,"maximum":256}),
                false,
            );
        }
        12 => {
            add(&mut s, "run_id", string(), true);
            add(&mut s, "algorithm_id", string(), false);
            add(&mut s, "language", language(), false);
            add(&mut s, "diff_from", string(), false);
            add(
                &mut s,
                "max_lines",
                json!({"type":"integer","minimum":1,"maximum":10000}),
                false,
            );
        }
        13 => {
            add(&mut s, "run_id", string(), true);
            add(&mut s, "algorithm_id", string(), false);
            add(
                &mut s,
                "kind",
                json!({"enum":["model","source","bundle"]}),
                true,
            );
            add(
                &mut s,
                "languages",
                json!({"type":"array","minItems":1,"items":language()}),
                true,
            );
            add(
                &mut s,
                "out_dir",
                json!({"type":"string","minLength":1}),
                false,
            );
        }
        15 => {
            add(&mut s, "model", object(), true);
            add(&mut s, "input", object(), true);
            add(&mut s, "out_path", string(), false);
        }
        _ => unreachable!("MCP operation inventory"),
    }
    if op.requires_idempotency_key {
        add(
            &mut s,
            "idempotency_key",
            json!({"type":"string","minLength":1,"maxLength":128,"pattern":"^[ -~]+$"}),
            true,
        );
    }
    s
}
fn language() -> Value {
    json!({"enum":["rust","python","javascript","typescript"]})
}

fn tool(op: &Operation) -> Value {
    let gaps = schema_gaps(op);
    let mut value = json!({"name":op.mcp_tool,"description":op.name,"inputSchema":input_schema(op),
        "annotations":{"readOnlyHint":!op.mutating,"idempotentHint":!op.mutating || op.requires_idempotency_key,"destructiveHint":op.id==8,"openWorldHint":false},
        "_meta":{"jope/schemaQualified":gaps.is_empty() && op.id!=15,"jope/missingCanonicalSchemas":gaps}});
    // An unresolved canonical exchange has no outputSchema. Never pass an empty/permissive
    // substitute off as a jope-api schema. Qualification stays open until its owner materializes it.
    if gaps.is_empty() {
        let mut schemas: Vec<_> = op
            .response_types
            .iter()
            .filter_map(|ty| ty.resolve_schema().ok())
            .map(|s| serde_json::to_value(s).expect("schema JSON"))
            .collect();
        if !schemas.is_empty() {
            schemas.push(
                serde_json::to_value(rmcp::schemars::schema_for!(jope_api::error::Problem))
                    .expect("Problem schema"),
            );
        }
        if schemas.len() > 1 {
            let mut defs = Map::new();
            let alternatives: Vec<_> = schemas
                .iter()
                .enumerate()
                .map(|(i, schema)| {
                    let name = format!("Response{i}");
                    let mut schema = schema.clone();
                    namespace_refs(&mut schema, &name);
                    defs.insert(name.clone(), schema);
                    json!({"$ref":format!("#/$defs/{name}")})
                })
                .collect();
            value["outputSchema"] = json!({"type":"object","anyOf":alternatives,"$defs":defs});
        }
        if schemas.len() == 1 {
            value["outputSchema"] = schemas[0].clone();
        }
    }
    value
}

/// Validate adapter fields without pretending to implement arbitrary canonical JSON Schema.
/// Full RunSpec/ExportRequest/PredictRequest validation also uses their real serde DTOs.
pub fn validate_arguments(op: &Operation, args: &Value) -> Result<(), String> {
    let map = args.as_object().ok_or("arguments must be an object")?;
    let schema = input_schema(op);
    let props = schema["properties"].as_object().expect("properties");
    for name in map.keys() {
        if !props.contains_key(name) {
            return Err(format!("unknown argument {name}"));
        }
    }
    for name in schema["required"].as_array().expect("required") {
        let name = name.as_str().expect("field name");
        if !map.contains_key(name) {
            return Err(format!("missing argument {name}"));
        }
    }
    for (name, value) in map {
        let rule = &props[name];
        if let Some(kind) = rule["type"].as_str() {
            let valid = match kind {
                "string" => value.is_string(),
                "integer" => value.as_u64().is_some(),
                "array" => value.is_array(),
                "object" => value.is_object(),
                "boolean" => value.is_boolean(),
                _ => true,
            };
            if !valid {
                return Err(format!("invalid type for {name}"));
            }
        }
        if let Some(allowed) = rule["enum"].as_array() {
            if !allowed.contains(value) {
                return Err(format!("invalid {name}"));
            }
        }
        if let Some(number) = value.as_u64() {
            if rule["maximum"].as_u64().is_some_and(|max| number > max)
                || rule["minimum"].as_u64().is_some_and(|min| number < min)
            {
                return Err(format!("{name} outside permitted range"));
            }
        }
        if value
            .as_array()
            .is_some_and(|a| a.iter().any(|v| !v.is_string()))
            && matches!(name.as_str(), "columns" | "kinds" | "languages")
        {
            return Err(format!("{name} must contain strings"));
        }
    }
    for (key, value) in map {
        let valid = match key.as_str() {
            "run_id" => value
                .as_str()
                .is_some_and(|s| s.parse::<jope_api::ids::RunId>().is_ok()),
            "dataset_id" => value
                .as_str()
                .is_some_and(|s| s.parse::<jope_api::ids::DatasetId>().is_ok()),
            "algorithm_id" | "diff_from" => value
                .as_str()
                .is_some_and(|s| s.parse::<jope_api::ids::AlgorithmId>().is_ok()),
            "after" | "after_event" | "expected_revision" => {
                serde_json::from_value::<jope_api::decimal::DecimalU64>(value.clone()).is_ok()
            }
            _ => true,
        };
        if !valid {
            return Err(format!("invalid {key}"));
        }
    }
    if op.id == 2 {
        if map.contains_key("path") == map.contains_key("csv_text") {
            return Err("provide exactly one of path or csv_text".into());
        }
        if args["csv_text"]
            .as_str()
            .is_some_and(|s| s.len() > 1024 * 1024)
        {
            return Err("csv_text exceeds 1 MiB UTF-8 bytes".into());
        }
    }
    if op.id == 8 {
        let action = map.get("action").and_then(Value::as_str);
        if action == Some("set_effort") && !map.contains_key("effort") {
            return Err("set_effort requires effort".into());
        }
        if matches!(action, Some("pause" | "continue" | "cancel"))
            && (map.contains_key("effort") || map.contains_key("budget"))
        {
            return Err("control arguments disagree with action".into());
        }
        if let Some(budget) = map.get("budget") {
            serde_json::from_value::<jope_api::Budget>(budget.clone())
                .map_err(|_| "invalid control budget")?;
        }
    }
    if op.id == 7
        && args.get("run_id").is_none()
        && (args.get("wait_seconds").is_some() || args.get("after_event").is_some())
    {
        return Err("long-poll requires run_id".into());
    }
    Ok(())
}

pub fn schema_inventory() -> Map<String, Value> {
    OPERATIONS
        .iter()
        .filter_map(|op| {
            op.mcp_tool
                .map(|name| (name.into(), json!(schema_gaps(op))))
        })
        .collect()
}

fn namespace_refs(value: &mut Value, name: &str) {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(reference)) = map.get_mut("$ref") {
                if let Some(tail) = reference.strip_prefix("#/") {
                    *reference = format!("#/$defs/{name}/{tail}");
                }
            }
            for value in map.values_mut() {
                namespace_refs(value, name);
            }
        }
        Value::Array(values) => {
            for value in values {
                namespace_refs(value, name);
            }
        }
        _ => {}
    }
}

/// Global canonical pending exchange inventory, including non-MCP operations, for API-05 parity.
pub fn pending_exchanges() -> Vec<&'static str> {
    let mut pending = std::collections::BTreeSet::new();
    for op in OPERATIONS {
        pending.extend(schema_gaps(op));
    }
    pending.into_iter().collect()
}
