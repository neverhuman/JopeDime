use super::rust_head::*;
use super::rust_input::*;
use super::rust_ops::*;
use super::*;

pub(crate) fn emit_bound_rust(model: &BoundModel) -> Result<String> {
    let BoundPredictor::Pipeline {
        target,
        ops,
        head,
        calibration,
        missing_aware,
    } = &model.predictor
    else {
        bail!("bound emission currently requires a v3 pipeline model");
    };
    let headers = model
        .schema
        .columns
        .iter()
        .map(|column| rust_string(&column.name))
        .collect::<Vec<_>>()
        .join(",");
    let input_body = rust_input_body(&model.input, *missing_aware);
    let (ops_helpers, ops_body) = rust_batch_ops_source(ops)?;
    let head_body = rust_head_body(head)?;
    let calibration_body = rust_calibration_body(calibration, model.task.kind == TaskKind::Binary);
    let target_body = rust_target_body(target, "calibrated");
    let mut source = String::from(
        r#"//! Schema-bound, standard-library-only prediction artifact.
use std::env;
use std::fs;

fn parse_csv(text: &str) -> Result<Vec<Vec<String>>, String> {
    let mut records = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut chars = text.chars().peekable();
    let mut quoted = false;
    let mut at_start = true;
    while let Some(ch) = chars.next() {
        if quoted {
            if ch == '"' {
                if chars.peek() == Some(&'"') { chars.next(); field.push('"'); }
                else { quoted = false; }
            } else { field.push(ch); }
            continue;
        }
        match ch {
            '"' if at_start => { quoted = true; at_start = false; }
            ',' => { row.push(std::mem::take(&mut field)); at_start = true; }
            '\n' => { row.push(std::mem::take(&mut field)); records.push(std::mem::take(&mut row)); at_start = true; }
            '\r' if chars.peek() == Some(&'\n') => {}
            '\r' => { row.push(std::mem::take(&mut field)); records.push(std::mem::take(&mut row)); at_start = true; }
            '"' => return Err("quote in unquoted CSV field".to_string()),
            other => { field.push(other); at_start = false; }
        }
    }
    if quoted { return Err("unterminated quoted CSV field".to_string()); }
    if !field.is_empty() || !row.is_empty() { row.push(field); records.push(row); }
    Ok(records)
}

fn is_missing(text: &str) -> bool {
    let value = text.trim().to_ascii_lowercase();
    matches!(value.as_str(), "" | "na" | "nan" | "null")
}

fn is_category_missing(text: &str) -> bool {
    let value = text.trim().to_ascii_lowercase();
    matches!(value.as_str(), "" | "na" | "null")
}

fn category(text: &str) -> String {
    if is_category_missing(text) { "\0MISSING".to_string() } else { text.to_string() }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FiniteKey(u64);

impl FiniteKey { fn value(self) -> f64 { f64::from_bits(self.0) } }
impl PartialOrd for FiniteKey { fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) } }
impl Ord for FiniteKey { fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.value().total_cmp(&other.value()) } }

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SemanticKey { Missing, Numeric(FiniteKey), Categorical(String) }

fn semantic_key(text: &str, numeric: bool) -> Result<SemanticKey, String> {
    if numeric {
        if is_missing(text) { return Ok(SemanticKey::Missing); }
        let value = text.trim().parse::<f64>().map_err(|_| format!("invalid numeric value {text}"))?;
        if !value.is_finite() { return Ok(SemanticKey::Missing); }
        return Ok(SemanticKey::Numeric(FiniteKey(if value == 0.0 { 0 } else { value.to_bits() })));
    }
    Ok(if is_category_missing(text) { SemanticKey::Missing } else { SemanticKey::Categorical(text.to_string()) })
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 { 1.0 / (1.0 + (-value.min(40.0)).exp()) }
    else { let z = value.max(-40.0).exp(); z / (1.0 + z) }
}

fn finite(value: f64) -> f64 { if value.is_finite() { value } else { 0.0 } }

fn yj(value: f64, lambda: f64) -> f64 {
    if value >= 0.0 {
        if lambda.abs() < 1e-12 { value.ln_1p() } else { ((value+1.0).powf(lambda)-1.0)/lambda }
    } else if (lambda-2.0).abs() < 1e-12 { -(-value).ln_1p() }
    else { -((1.0-value).powf(2.0-lambda)-1.0)/(2.0-lambda) }
}

fn piece(value: f64, lo: f64, hi: f64) -> f64 {
    if (hi-lo).abs() <= 1e-15 { if value > lo { 1.0 } else { 0.0 } }
    else { ((value-lo)/(hi-lo)).clamp(0.0, 1.0) }
}

fn dot_centered(x: &[f64], means: &[f64], weights: &[f64]) -> f64 {
    x.iter().zip(means).zip(weights).map(|((value, mean), weight)| weight*(value-mean)).sum()
}

fn rank_inverse(value: f64, scores: &[f64], values: &[f64]) -> f64 {
    // NaN routes to the low end (mirrors the reference inverse); `<=` would
    // fall through and underflow `hi - 1`.
    if !(value > scores[0]) { return values[0]; }
    if value >= scores[scores.len()-1] { return values[values.len()-1]; }
    let hi = scores.partition_point(|score| *score < value);
    let lo = hi - 1;
    values[lo] + (value-scores[lo])/(scores[hi]-scores[lo])*(values[hi]-values[lo])
}

fn encode(row: &[String]) -> Result<(Vec<f64>, Vec<bool>), String> {
"#,
    );
    source.push_str(&input_body);
    source.push_str("}\n\n");
    source.push_str(&ops_helpers);
    source.push_str("fn apply_ops(mut rows: Vec<Vec<f64>>, mut masks: Vec<Vec<bool>>, raw: &[Vec<String>]) -> Result<Vec<Vec<f64>>, String> {\n");
    source.push_str(&ops_body);
    source.push_str("}\n\nfn predict_numeric(x: &[f64]) -> f64 {\n    let value = ");
    source.push_str(&head_body);
    source.push_str(";\n    let calibrated = ");
    source.push_str(&calibration_body);
    source.push_str(";\n    let mapped = ");
    source.push_str(&target_body);
    // The final task mapping mirrors BoundModel::predict_table exactly:
    // binary NaN -> 0.5 chance, regression NaN -> 0.0.
    if model.task.kind == TaskKind::Binary {
        source.push_str(";\n    if mapped.is_finite() { mapped.clamp(0.0, 1.0) } else { 0.5 }");
    } else {
        source.push_str(";\n    if mapped.is_finite() { mapped } else { 0.0 }");
    }
    source.push_str("\n}\n\nfn run() -> Result<(), String> {\n");
    source.push_str(&format!(
        "    let expected: Vec<String> = vec![{headers}].into_iter().map(str::to_string).collect();\n"
    ));
    source.push_str(
        r#"    let args: Vec<String> = env::args().collect();
    if args.len() != 2 && args.len() != 3 { return Err("usage: model INPUT.csv [OUTPUT.csv]".to_string()); }
    let text = fs::read_to_string(&args[1]).map_err(|error| error.to_string())?;
    let records = parse_csv(&text)?;
    if records.first() != Some(&expected) { return Err("CSV header does not match bound schema".to_string()); }
    let raw = &records[1..];
    let mut features = Vec::with_capacity(raw.len());
    let mut masks = Vec::with_capacity(raw.len());
    for row in raw {
        if row.len() != expected.len() { return Err("CSV row width mismatch".to_string()); }
        let (values, missing) = encode(row)?;
        features.push(values);
        masks.push(missing);
    }
    let transformed = apply_ops(features, masks, raw)?;
    let mut output = String::new();
    for row in &transformed {
        let prediction = predict_numeric(row);
        // Debug formatting is shortest-round-trip: parses back to the exact
        // f64, unlike fixed-point {:.17} which zeroes small magnitudes.
        output.push_str(&format!("{prediction:?}\n"));
    }
    if args.len() == 3 { fs::write(&args[2], output).map_err(|error| error.to_string())?; }
    else { print!("{output}"); }
    Ok(())
}

fn main() {
    if let Err(error) = run() { eprintln!("{error}"); std::process::exit(2); }
}
"#,
    );
    Ok(source)
}
