use super::*;

/// The committed v3 schema, parsed.
pub fn schema() -> serde_json::Value {
    serde_json::from_str(RECEIPT_SCHEMA_JSON).expect("committed receipt schema parses")
}

/// The committed objective receipt schema, parsed.
pub fn objective_schema() -> serde_json::Value {
    serde_json::from_str(OBJECTIVE_SCHEMA_JSON).expect("committed objective receipt schema parses")
}

/// The v3 schema schemars derives from [`Receipt`]: the type without what an objective adds.
pub fn derived_schema() -> serde_json::Value {
    let mut value = derive(
        RECEIPT_SCHEMA_ID,
        "https://github.com/veox-ai/JopeDime/schemas/jope.pipeline-receipt.v3.schema.json",
    );
    for pointer in OBJECTIVE_ONLY {
        let (parent, key) = pointer.rsplit_once('/').expect("a pointer below the root");
        let removed = value
            .pointer_mut(parent)
            .and_then(serde_json::Value::as_object_mut)
            .and_then(|map| map.remove(key));
        assert!(removed.is_some(), "the derived schema has no {pointer}");
    }
    value
}

/// The objective receipt schema schemars derives from [`Receipt`].
pub fn derived_objective_schema() -> serde_json::Value {
    derive(
        OBJECTIVE_SCHEMA_ID,
        "https://github.com/neverhuman/JopeDime/schemas/jope.pipeline-objective-receipt.v1.schema.json",
    )
}

fn derive(id: &str, url: &str) -> serde_json::Value {
    let mut value = schemars::schema_for!(Receipt).to_value();
    value["title"] = id.into();
    value["$id"] = url.into();
    value["properties"]["schema"]["const"] = id.into();
    value["properties"]["policy"]["enum"] = serde_json::json!(["auto", "skip", "require"]);
    value["$defs"]["Stage"]["properties"]["status"]["enum"] = serde_json::json!(STATUSES);
    let stops = serde_json::json!([
        "evaluations",
        "seconds",
        "search_space_exhausted",
        "comparison_pool_exhausted"
    ]);
    value["$defs"]["CutSearch"]["properties"]["stopped_by"]["enum"] = stops.clone();
    let mut phase_stops = stops.as_array().unwrap().clone();
    phase_stops.push(serde_json::json!("not_requested"));
    value["$defs"]["Phase"]["properties"]["stopped_by"]["enum"] = phase_stops.into();
    // serde(default) keeps v1 deserialization compatible. Schemars' `required`
    // annotation removes null from these Options but does not undo that default.
    for (definition, names) in [
        ("CutSearch", &["accounting"][..]),
        (
            "Search",
            &[
                "budget_policy",
                "compared_cuts",
                "coverage",
                "seeding",
                "islands",
            ][..],
        ),
    ] {
        let required = value["$defs"][definition]["required"]
            .as_array_mut()
            .unwrap();
        for name in names {
            let name = serde_json::json!(name);
            if !required.contains(&name) {
                required.push(name);
            }
        }
    }
    // Close objects consistently, including optional nested definitions.
    fn close(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(m) => {
                if m.contains_key("properties") {
                    m.insert("additionalProperties".into(), false.into());
                }
                for child in m.values_mut() {
                    close(child);
                }
            }
            serde_json::Value::Array(a) => {
                for child in a {
                    close(child);
                }
            }
            _ => {}
        }
    }
    close(&mut value);
    value
}

/// Validate one receipt document: the committed schema, then the Rust type, then
/// the invariants above.
pub fn validate_value(value: &serde_json::Value) -> Result<Receipt, Vec<String>> {
    let schema = if value["schema"] == LEGACY_SCHEMA_ID {
        serde_json::from_str(LEGACY_SCHEMA_JSON).expect("legacy schema parses")
    } else if value["schema"] == V2_SCHEMA_ID {
        serde_json::from_str(V2_SCHEMA_JSON).expect("v2 schema parses")
    } else if value["schema"] == OBJECTIVE_SCHEMA_ID {
        objective_schema()
    } else {
        schema()
    };
    let errors = jope_datasets::schema::validate(&schema, value);
    if !errors.is_empty() {
        return Err(errors);
    }
    let receipt: Receipt =
        serde_json::from_value(value.clone()).map_err(|error| vec![error.to_string()])?;
    let errors = receipt.invariant_errors();
    if errors.is_empty() {
        Ok(receipt)
    } else {
        Err(errors)
    }
}

/// Read and validate a receipt file.
pub fn read(path: &std::path::Path) -> anyhow::Result<Receipt> {
    let bytes =
        std::fs::read(path).map_err(|error| anyhow::anyhow!("read {}: {error}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| anyhow::anyhow!("parse {}: {error}", path.display()))?;
    validate_value(&value)
        .map_err(|errors| anyhow::anyhow!("{}: {}", path.display(), errors.join("; ")))
}
