use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::path::PathBuf;

use jope_api::registry::{MediaType, Operation, PayloadTypeRef, OPERATIONS};
use serde_json::{json, Map, Value};

const PENDING_EXCHANGES: &[&str] = &[
    "AlgorithmPage",
    "DatasetPage",
    "DatasetRegistration",
    "DeleteReceipt",
    "EventPage",
    "GcReport",
    "GcRequest",
    "ObjectiveValidation",
    "ObjectiveValidationRequest",
    "RunControlRequest",
    "RunForkRequest",
    "RunPage",
    "TrainingFeedback",
];

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("schemas/jope.api.v1.openapi.json"));
    let document = generate()?;
    std::fs::write(output, serde_json::to_vec_pretty(&document)?)?;
    Ok(())
}

fn generate() -> Result<Value, Box<dyn Error>> {
    let mut pending = BTreeSet::new();
    let mut emitted = BTreeSet::new();
    let mut schemas = Map::new();
    let mut paths: BTreeMap<String, Map<String, Value>> = BTreeMap::new();
    collect_named_schema(
        "Problem",
        schemars::schema_for!(jope_api::error::Problem),
        &mut schemas,
    )?;

    for operation in OPERATIONS {
        let unresolved: BTreeSet<_> = operation
            .request_types
            .iter()
            .chain(operation.response_types)
            .filter(|exchange| {
                exchange.json_schema.is_none() && exchange.media_type == MediaType::Json
            })
            .map(|exchange| exchange.name)
            .collect();
        if !unresolved.is_empty() {
            pending.extend(unresolved);
            continue;
        }

        emitted.insert(operation.name);
        for exchange in operation
            .request_types
            .iter()
            .chain(operation.response_types)
        {
            collect_exchange_schema(exchange, &mut schemas)?;
        }
        for path in operation.paths {
            let path_item = paths.entry((*path).to_owned()).or_default();
            path_item.insert(
                operation.method.as_str().to_ascii_lowercase(),
                operation_document(operation)?,
            );
        }
    }

    let expected: BTreeSet<_> = PENDING_EXCHANGES.iter().copied().collect();
    if pending != expected {
        return Err(format!(
            "pending exchange allowlist drift: registry has {pending:?}, configured allowlist has {expected:?}"
        )
        .into());
    }
    for operation in OPERATIONS {
        let listed = pending.iter().any(|name| {
            operation
                .request_types
                .iter()
                .chain(operation.response_types)
                .any(|exchange| exchange.name == *name && exchange.json_schema.is_none())
        });
        if emitted.contains(operation.name) == listed {
            return Err(format!(
                "registry operation {} is missing from both OpenAPI and pending coverage, or is emitted while pending",
                operation.name
            )
            .into());
        }
    }

    let mut path_values = Map::new();
    for (path, item) in paths {
        path_values.insert(path, Value::Object(item));
    }
    Ok(json!({
        "openapi": "3.1.0",
        "info": {"title": "JopeDime API", "version": "1"},
        "security": [{"bearerAuth": []}],
        "paths": path_values,
        "components": {
            "schemas": schemas,
            "securitySchemes": {
                "bearerAuth": {"type": "http", "scheme": "bearer"}
            }
        },
        "x-jope-pending-exchanges": pending,
    }))
}

fn operation_document(operation: &Operation) -> Result<Value, Box<dyn Error>> {
    let mut parameters = Vec::new();
    for path in operation.paths {
        for segment in path.split('/') {
            if let Some(parameter) = segment.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
                let parameter = parameter.to_owned();
                if !parameters
                    .iter()
                    .any(|item: &Value| item["name"] == parameter)
                {
                    parameters.push(json!({
                        "name": parameter,
                        "in": "path",
                        "required": true,
                        "schema": {"type": "string"}
                    }));
                }
            }
        }
    }

    let mut value = json!({
        "operationId": operation.name,
        "parameters": parameters,
        "responses": {
            "200": {
                "description": "Successful response",
                "headers": api_version_header(),
                "content": response_content(operation.response_types)?
            },
            "default": {"description": "RFC 9457 problem", "content": {
                "application/problem+json": {"schema": {"$ref": "#/components/schemas/Problem"}}
            }, "headers": api_version_header()}
        },
        "x-jope-operation-id": operation.id,
        "x-jope-mutating": operation.mutating,
        "x-jope-requires-idempotency-key": operation.requires_idempotency_key
    });
    if !operation.request_types.is_empty() {
        value["requestBody"] = json!({
            "required": true,
            "content": response_content(operation.request_types)?
        });
    }
    if operation.requires_idempotency_key {
        value["parameters"]
            .as_array_mut()
            .ok_or("operation parameters must be an array")?
            .push(json!({
                "name": "Idempotency-Key",
                "in": "header",
                "required": true,
                "schema": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 128,
                    "pattern": "^[ -~]{1,128}$"
                }
            }));
    }
    Ok(value)
}

fn api_version_header() -> Value {
    json!({
        "Jope-Api-Version": {
            "required": true,
            "schema": {"type": "string", "pattern": "^1\\.[0-9]+$"}
        }
    })
}

fn response_content(exchanges: &[PayloadTypeRef]) -> Result<Value, Box<dyn Error>> {
    let mut content: BTreeMap<&str, Vec<Value>> = BTreeMap::new();
    for exchange in exchanges {
        content
            .entry(exchange.media_type.as_str())
            .or_default()
            .push(exchange_schema(exchange)?);
    }
    let mut result = Map::new();
    for (media_type, schemas) in content {
        let schema = if schemas.len() == 1 {
            schemas
                .into_iter()
                .next()
                .ok_or("missing single content schema")?
        } else {
            json!({"oneOf": schemas})
        };
        result.insert(media_type.to_owned(), json!({"schema": schema}));
    }
    Ok(Value::Object(result))
}

fn exchange_schema(exchange: &PayloadTypeRef) -> Result<Value, Box<dyn Error>> {
    match exchange.media_type {
        MediaType::Json | MediaType::EventStream => Ok(json!({
            "$ref": format!("#/components/schemas/{}", exchange.name)
        })),
        MediaType::Csv => Ok(json!({"type": "string", "format": "csv"})),
        MediaType::Artifact => Ok(json!({"type": "string", "format": "binary"})),
    }
}

fn collect_exchange_schema(
    exchange: &PayloadTypeRef,
    schemas: &mut Map<String, Value>,
) -> Result<(), Box<dyn Error>> {
    if exchange.media_type != MediaType::Json && exchange.media_type != MediaType::EventStream {
        return Ok(());
    }
    let schema = exchange
        .resolve_schema()
        .map_err(|name| format!("exchange {name} has no materialized JSON schema"))?;
    collect_named_schema(exchange.name, schema, schemas)
}

fn collect_named_schema(
    name: &str,
    schema: schemars::Schema,
    schemas: &mut Map<String, Value>,
) -> Result<(), Box<dyn Error>> {
    let mut value = serde_json::to_value(schema)?;
    let definitions = value
        .as_object_mut()
        .and_then(|root| root.remove("$defs"))
        .unwrap_or_else(|| json!({}));
    let mut additions = definitions.as_object().cloned().unwrap_or_default();
    additions.insert(name.to_owned(), value);
    for schema in additions.values_mut() {
        rewrite_definition_refs(schema);
    }
    for (name, schema) in additions {
        if let Some(existing) = schemas.get(&name) {
            if existing != &schema {
                return Err(format!("conflicting generated schema definition {name}").into());
            }
        } else {
            schemas.insert(name, schema);
        }
    }
    Ok(())
}

fn rewrite_definition_refs(value: &mut Value) {
    match value {
        Value::Object(object) => {
            if let Some(Value::String(reference)) = object.get_mut("$ref") {
                if let Some(name) = reference.strip_prefix("#/$defs/") {
                    *reference = format!("#/components/schemas/{name}");
                }
            }
            for child in object.values_mut() {
                rewrite_definition_refs(child);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(rewrite_definition_refs),
        _ => {}
    }
}
