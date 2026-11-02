use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::ffi::OsString;
use std::io;
use std::path::PathBuf;

use jope_api::registry::{MediaType, Operation, PayloadTypeRef, OPERATIONS};
use serde_json::{json, Map, Value};

fn main() -> Result<(), Box<dyn Error>> {
    let output = match parse_output_destination(std::env::args_os().skip(1))? {
        OutputDestination::DefaultSchema => PathBuf::from("schemas/jope.api.v1.openapi.json"),
        OutputDestination::Explicit(path) => path,
    };
    let document = generate()?;
    std::fs::write(output, serde_json::to_vec_pretty(&document)?)?;
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum OutputDestination {
    DefaultSchema,
    Explicit(PathBuf),
}

fn parse_output_destination(
    args: impl IntoIterator<Item = OsString>,
) -> Result<OutputDestination, io::Error> {
    let mut args = args.into_iter();
    match (args.next(), args.next()) {
        (None, None) => Ok(OutputDestination::DefaultSchema),
        (Some(path), None) => Ok(OutputDestination::Explicit(PathBuf::from(path))),
        (_, Some(_)) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: jope-api-openapi [OUTPUT_PATH]",
        )),
    }
}

fn generate() -> Result<Value, Box<dyn Error>> {
    let mut emitted = BTreeSet::new();
    let mut schemas = Map::new();
    let mut paths: BTreeMap<String, Map<String, Value>> = BTreeMap::new();
    collect_named_schema(
        "Problem",
        schemars::schema_for!(jope_api::error::Problem),
        &mut schemas,
    )?;

    for operation in OPERATIONS {
        if operation.paths.is_empty() {
            return Err(format!("registry operation {} has no paths", operation.name).into());
        }
        for exchange in operation
            .request_types
            .iter()
            .chain(operation.response_types)
        {
            collect_exchange_schema(exchange, &mut schemas)?;
        }
        for path in operation.paths {
            let path_item = paths.entry((*path).to_owned()).or_default();
            let method = operation.method.as_str().to_ascii_lowercase();
            if path_item
                .insert(method.clone(), operation_document(operation)?)
                .is_some()
            {
                return Err(format!("multiple registry operations define {method} {path}").into());
            }
        }
        emitted.insert(operation.name);
    }

    let expected: BTreeSet<_> = OPERATIONS.iter().map(|operation| operation.name).collect();
    if emitted != expected {
        return Err(format!(
            "OpenAPI operation coverage mismatch: emitted {emitted:?}, registry has {expected:?}"
        )
        .into());
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
        }
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
    let mut additions = take_schema_definitions(&mut value, name)?;
    additions.insert(name.to_owned(), value);
    for schema in additions.values_mut() {
        rewrite_definition_refs(schema);
    }
    for (name, schema) in additions {
        if let Some(existing) = schemas.get(&name) {
            if existing != &schema
                && without_schema_metadata(existing) != without_schema_metadata(&schema)
            {
                return Err(format!("conflicting generated schema definition {name}").into());
            }
        } else {
            schemas.insert(name, schema);
        }
    }
    Ok(())
}

fn take_schema_definitions(
    schema: &mut Value,
    name: &str,
) -> Result<Map<String, Value>, io::Error> {
    let root = schema.as_object_mut().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("generated schema {name} must be an object"),
        )
    })?;
    match root.remove("$defs") {
        None => Ok(Map::new()),
        Some(Value::Object(definitions)) => Ok(definitions),
        Some(_) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("generated schema {name} $defs must be an object"),
        )),
    }
}

fn without_schema_metadata(value: &Value) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .filter(|(key, _)| key.as_str() != "$schema" && key.as_str() != "title")
                .map(|(key, value)| (key.clone(), without_schema_metadata(value)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(without_schema_metadata).collect()),
        _ => value.clone(),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_destination_defaults_to_committed_schema() {
        let destination = parse_output_destination(std::iter::empty()).unwrap();
        assert_eq!(destination, OutputDestination::DefaultSchema);
    }

    #[test]
    fn output_destination_accepts_one_explicit_path() {
        let destination =
            parse_output_destination([OsString::from("target/openapi.json")]).unwrap();
        assert_eq!(
            destination,
            OutputDestination::Explicit(PathBuf::from("target/openapi.json"))
        );
    }

    #[test]
    fn output_destination_rejects_extra_arguments() {
        let error = parse_output_destination([
            OsString::from("target/openapi.json"),
            OsString::from("unexpected"),
        ])
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(error.to_string(), "usage: jope-api-openapi [OUTPUT_PATH]");
    }

    #[test]
    fn schema_definitions_reject_malformed_shape() {
        let mut schema = json!({"$defs": []});
        let error = take_schema_definitions(&mut schema, "Problem").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(
            error.to_string(),
            "generated schema Problem $defs must be an object"
        );
    }

    #[test]
    fn generated_document_matches_committed_openapi() {
        let generated = serde_json::to_vec_pretty(&generate().unwrap()).unwrap();
        let committed = include_bytes!("../../../../schemas/jope.api.v1.openapi.json");
        assert_eq!(generated.as_slice(), committed.as_slice());
    }

    #[test]
    fn generated_document_covers_registry_without_pending_exchanges() {
        let document = generate().expect("all registered exchange schemas are materialized");
        assert!(document.get("x-jope-pending-exchanges").is_none());

        let emitted: BTreeSet<_> = document["paths"]
            .as_object()
            .expect("generated paths")
            .values()
            .flat_map(|path| {
                path.as_object()
                    .expect("generated path item")
                    .values()
                    .filter_map(|operation| operation["operationId"].as_str())
            })
            .collect();
        let expected: BTreeSet<_> = OPERATIONS.iter().map(|operation| operation.name).collect();
        assert_eq!(emitted, expected);
    }
}
