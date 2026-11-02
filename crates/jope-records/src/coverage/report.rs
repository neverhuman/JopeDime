use super::*;

/// `coverage.py validate`: the error lines, then the JSON summary; exit 1 on any error. A map the
/// Python could not read (it raised) is an error, not a report.
pub fn validate_report(root: &Path, mapping_path: &Path) -> Result<(String, bool)> {
    let discovered = discover(root)?;
    let mapping = py::load_json(mapping_path)?;
    validate_text(&mapping, &discovered)
}

/// The `validate` report for a map and a discovered inventory, and whether it passes; an error
/// where the Python raised instead of reporting.
pub fn validate_text(mapping: &Value, discovered: &Dict) -> Result<(String, bool)> {
    let errors = validate(mapping, discovered)?;
    let mut out = String::new();
    for error in &errors {
        out.push_str(error);
        out.push('\n');
    }
    let summary: Dict = [
        ("stage", Value::from("static")),
        ("counts", Value::Dict(counts(discovered)?)),
        ("unassigned_or_invalid", Value::from(errors.len())),
    ]
    .into_iter()
    .collect();
    out.push_str(&pyjson::dumps(&Value::Dict(summary), true, None));
    out.push('\n');
    Ok((out, errors.is_empty()))
}

/// `coverage.py discover`: `{"stage": "static", "items": ...}` with a two-space indent.
pub fn discover_report(root: &Path) -> Result<String> {
    Ok(discover_document(discover(root)?))
}

/// The `discover` document for a discovered inventory.
pub fn discover_document(discovered: Dict) -> String {
    let document: Dict = [
        ("stage", Value::from("static")),
        ("items", Value::Dict(discovered)),
    ]
    .into_iter()
    .collect();
    pyjson::dumps(&Value::Dict(document), false, Some(2)) + "\n"
}
