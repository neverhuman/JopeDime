use super::rows::{row_genome, GenomeStage};
use super::truncate;
use serde_json::{Map, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

pub const CODE_PREVIEW_CHARS: usize = 10_000;
pub const HEAD_OPS: &[&str] = &[
    "elasticnet",
    "ensemble",
    "ensemble_ridge_gbm",
    "gbm",
    "gpr",
    "histgbm",
    "histgbm2",
    "huber_ridge",
    "knn",
    "krr",
    "ksmooth",
    "logistic",
    "random_forest",
    "ridge",
    "stack",
];

pub fn code_for_row(row: &Map<String, Value>, lang: &str, roots: &[PathBuf]) -> (String, String) {
    if let Some(code) = inline_code(row, lang) {
        return (clip_code(&code), "inline".into());
    }
    if let Some(path) = code_path(row, lang, roots) {
        if let Some(code) = read_code(&path).filter(|code| !code.is_empty()) {
            return (clip_code(&code), truncate(&path.to_string_lossy(), 34));
        }
    }
    let preview = if lang == "rust" {
        rust_algorithm_preview(row)
    } else {
        python_algorithm_preview(row)
    };
    (preview, "preview".into())
}

pub fn inline_code(row: &Map<String, Value>, lang: &str) -> Option<String> {
    let generated = row.get("generated_code").and_then(Value::as_object);
    if let Some(value) =
        generated.and_then(|map| map.get(lang).or_else(|| map.get(&format!("{lang}_code"))))
    {
        if nonblank(value) {
            return value.as_str().map(str::to_owned);
        }
    }
    for key in [
        format!("{lang}_code"),
        format!("{lang}_source"),
        format!("generated_{lang}"),
    ] {
        if let Some(value) = row.get(&key) {
            if nonblank(value) {
                return value.as_str().map(str::to_owned);
            }
        }
    }
    let code = row.get("code").and_then(Value::as_object);
    let value = code.and_then(|map| map.get(lang).or_else(|| map.get(&format!("{lang}_code"))))?;
    nonblank(value)
        .then(|| value.as_str().map(str::to_owned))
        .flatten()
}

pub fn code_path(row: &Map<String, Value>, lang: &str, roots: &[PathBuf]) -> Option<PathBuf> {
    let mut values = vec![
        row.get(&format!("{lang}_path")),
        row.get(&format!("{lang}_source_path")),
        row.get(&format!("{lang}_code_path")),
    ];
    for key in ["code_paths", "source_paths", "artifact_paths", "artifacts"] {
        if let Some(container) = row.get(key).and_then(Value::as_object) {
            values.extend([
                container.get(lang),
                container.get(&format!("{lang}_path")),
                container.get(&format!("{lang}_source_path")),
            ]);
        }
    }
    for value in values.into_iter().flatten() {
        let Some(raw) = value.as_str().filter(|text| !text.is_empty()) else {
            continue;
        };
        let path = expand_home(raw);
        let candidates: Vec<PathBuf> = if path.is_absolute() {
            vec![path]
        } else {
            roots
                .iter()
                .map(|root| root.join(&path))
                .chain(std::iter::once(path.clone()))
                .collect()
        };
        if let Some(found) = candidates
            .into_iter()
            .find(|candidate| candidate.exists() && candidate.is_file())
        {
            return Some(found);
        }
    }
    None
}

pub fn read_code(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    Some(take_chars(
        &String::from_utf8_lossy(&bytes),
        CODE_PREVIEW_CHARS,
    ))
}

pub fn clip_code(code: &str) -> String {
    let code = code.trim_matches('\n');
    if code.chars().count() <= CODE_PREVIEW_CHARS {
        return code.to_owned();
    }
    format!(
        "{}\n# ... clipped ...",
        take_chars(code, CODE_PREVIEW_CHARS)
    )
}

pub fn genome_stages(genome: &str) -> Vec<GenomeStage> {
    let bytes = genome.as_bytes();
    let mut stages = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'(' {
            index += 1;
            continue;
        }
        let start = index + 1;
        let mut end = start;
        while end < bytes.len() && bytes[end] != b')' && bytes[end] != b'(' {
            end += 1;
        }
        if end < bytes.len() && bytes[end] == b')' {
            let contents = &genome[start..end];
            let split = contents.find(char::is_whitespace).unwrap_or(contents.len());
            let op = &contents[..split];
            if valid_op(op) && op != "pipe" && op != "sym" {
                stages.push((op.to_owned(), stage_params(&contents[split..])));
            }
            index = end + 1;
        } else {
            index = start;
        }
    }
    stages
}

pub fn stage_params(raw: &str) -> Vec<(String, String)> {
    let mut params = Vec::<(String, String)>::new();
    for (index, token) in raw.split_whitespace().enumerate() {
        if let Some((key, value)) = token.split_once('=') {
            insert_ordered(&mut params, key.trim().to_owned(), value.trim().to_owned());
        } else {
            insert_ordered(
                &mut params,
                format!("arg{index}"),
                token.trim_matches('"').to_owned(),
            );
        }
    }
    params
}

pub fn split_transforms_head(
    row: &Map<String, Value>,
) -> (Vec<GenomeStage>, GenomeStage) {
    let genome = row_genome(row);
    let stages = genome_stages(&genome);
    if stages.is_empty() {
        return (
            Vec::new(),
            (
                row.get("head_type")
                    .filter(|v| truthy(v))
                    .map(python_string)
                    .unwrap_or_else(|| "ridge".into()),
                Vec::new(),
            ),
        );
    }
    let (last_op, last_params) = stages.last().cloned().unwrap_or_default();
    if HEAD_OPS.contains(&last_op.as_str()) {
        return (stages[..stages.len() - 1].to_vec(), (last_op, last_params));
    }
    let head = row
        .get("head_type")
        .filter(|v| truthy(v))
        .map(python_string)
        .or_else(|| head_from_genome(&genome))
        .unwrap_or_else(|| "ridge".into());
    (stages, (head, Vec::new()))
}

pub fn python_algorithm_preview(row: &Map<String, Value>) -> String {
    let (transforms, head) = split_transforms_head(row);
    let mut lines = vec![
        "class Model:".into(),
        "    def fit(self, X, y):".into(),
        "        Z = X".into(),
    ];
    if transforms.is_empty() {
        lines.push("        # raw features".into());
    }
    for (op, params) in transforms.iter().take(5) {
        lines.push(format!(
            "        Z = apply_stage(Z, {}, {}, y=y)",
            py_repr(op),
            py_repr_params(params)
        ));
    }
    if transforms.len() > 5 {
        lines.push(format!(
            "        # ... {} more transform stages ...",
            transforms.len() - 5
        ));
    }
    lines.extend([
        format!(
            "        self.head = fit_head({}, Z, y, {})",
            py_repr(&head.0),
            py_repr_params(&head.1)
        ),
        "        return self".into(),
        "".into(),
        "    def predict(self, X):".into(),
        "        Z = X".into(),
    ]);
    for (index, (op, _)) in transforms.iter().take(5).enumerate() {
        lines.push(format!(
            "        Z = self.transforms[{index}].transform(Z)  # {op}"
        ));
    }
    if transforms.len() > 5 {
        lines.push("        # remaining fitted transforms replay here".into());
    }
    lines.push("        return self.head.predict(Z)".into());
    lines.join("\n")
}

pub fn rust_algorithm_preview(row: &Map<String, Value>) -> String {
    let (transforms, head) = split_transforms_head(row);
    let mut lines = vec![
        "pub struct Model {".into(),
        "    transforms: Vec<FittedStage>,".into(),
        "    head: FittedHead,".into(),
        "}".into(),
        "".into(),
        "impl Model {".into(),
        "    pub fn fit(x: &Matrix, y: &[f64]) -> Self {".into(),
        "        let mut z = x.clone();".into(),
        "        let mut transforms = Vec::new();".into(),
    ];
    for (op, params) in transforms.iter().take(4) {
        lines.push(format!(
            "        z = fit_stage(&mut transforms, z, {}, {}, Some(y));",
            json_string(op),
            json_params(params)
        ));
    }
    if transforms.len() > 4 {
        lines.push(format!(
            "        // ... {} more transform stages ...",
            transforms.len() - 4
        ));
    }
    lines.extend([
        format!(
            "        let head = fit_head({}, &z, y, {});",
            json_string(&head.0),
            json_params(&head.1)
        ),
        "        Self { transforms, head }".into(),
        "    }".into(),
        "".into(),
        "    pub fn predict(&self, x: &Matrix) -> Vec<f64> {".into(),
        "        let mut z = x.clone();".into(),
    ]);
    for (index, (op, _)) in transforms.iter().take(4).enumerate() {
        lines.push(format!(
            "        z = self.transforms[{index}].transform(z); // {op}"
        ));
    }
    if transforms.len() > 4 {
        lines.push("        // remaining fitted transforms replay here".into());
    }
    lines.extend([
        "        self.head.predict(&z)".into(),
        "    }".into(),
        "}".into(),
    ]);
    lines.join("\n")
}

pub fn head_from_genome(genome: &str) -> Option<String> {
    let mut heads = HEAD_OPS.to_vec();
    heads.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    heads
        .into_iter()
        .find(|head| genome.contains(&format!("({head}")))
        .map(str::to_owned)
}

fn nonblank(value: &Value) -> bool {
    value.as_str().is_some_and(|text| !text.trim().is_empty())
}

fn expand_home(raw: &str) -> PathBuf {
    if raw == "~" {
        return std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(raw));
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        return std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_default()
            .join(rest);
    }
    PathBuf::from(raw)
}

fn valid_op(op: &str) -> bool {
    let mut chars = op.chars();
    chars
        .next()
        .is_some_and(|ch| ch == '_' || ch.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch == '-' || ch.is_ascii_alphanumeric())
}

fn insert_ordered(values: &mut Vec<(String, String)>, key: String, value: String) {
    if let Some(existing) = values.iter_mut().find(|(found, _)| *found == key) {
        existing.1 = value;
    } else {
        values.push((key, value));
    }
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|v| v != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        _ => value.to_string(),
    }
}

fn take_chars(text: &str, count: usize) -> String {
    text.chars().take(count).collect()
}

fn py_repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut result = String::from(quote);
    for ch in value.chars() {
        match ch {
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c == quote => {
                result.push('\\');
                result.push(c);
            }
            c => result.push(c),
        }
    }
    result.push(quote);
    result
}

fn py_repr_params(params: &[(String, String)]) -> String {
    let body = params
        .iter()
        .map(|(key, value)| format!("{}: {}", py_repr(key), py_repr(value)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{body}}}")
}

fn json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if c as u32 > 0x7f => {
                let scalar = c as u32;
                if scalar <= 0xffff {
                    out.push_str(&format!("\\u{scalar:04x}"));
                } else {
                    let n = scalar - 0x10000;
                    out.push_str(&format!(
                        "\\u{:04x}\\u{:04x}",
                        0xd800 + (n >> 10),
                        0xdc00 + (n & 0x3ff)
                    ));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_params(params: &[(String, String)]) -> String {
    let sorted: BTreeMap<&str, &str> = params
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect();
    let body = sorted
        .iter()
        .map(|(key, value)| format!("{}: {}", json_string(key), json_string(value)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{body}}}")
}
