use anyhow::{bail, Result};
use serde_json::{Number, Value};
use std::collections::BTreeMap;

use crate::grammar::{Expr, Op};

pub const SPECIALS: [&str; 3] = ["PAD", "BOS", "EOS"];
pub const ABSENT: &str = "<ABSENT>";
pub const CANONICAL_TASKS: [&str; 4] = ["regression", "binary", "timeseries", "multiclass"];

/// Parameter metadata in the source `GrammarSpec`.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamSpec {
    pub name: String,
    pub kind: String,
    pub bins: Vec<Value>,
    pub default: Value,
    pub required: bool,
}

/// Operation metadata, kept in source grammar order.
#[derive(Clone, Debug, PartialEq)]
pub struct OpSpec {
    pub name: String,
    pub role: String,
    pub params: Vec<ParamSpec>,
    pub allow_append: bool,
    pub generate: bool,
    pub takes_expr: bool,
    pub tasks: Vec<String>,
}

/// Bounded expression sub-grammar used by action-token construction.
#[derive(Clone, Debug, PartialEq)]
pub struct ExprSpec {
    pub vars_map: Vec<String>,
    pub vars_common: Vec<String>,
    pub funcs1: Vec<String>,
    pub funcs2: Vec<String>,
    pub funcs3: Vec<String>,
    pub binops: Vec<String>,
    pub consts: Vec<String>,
    pub max_col: usize,
    pub max_depth: usize,
    pub max_nodes: usize,
}

impl Default for ExprSpec {
    fn default() -> Self {
        Self {
            vars_map: vec!["x".into()],
            vars_common: ["row_mean", "row_std", "row_min", "row_max", "row_l2"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            funcs1: ["abs", "sqrt", "log1p", "sin", "cos", "tanh", "sigmoid"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            funcs2: ["min", "max", "step"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            funcs3: vec!["clip".into()],
            binops: ["+", "-", "*", "/"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            consts: [
                "-2", "-1", "-0.5", "-0.25", "-0.1", "0", "0.05", "0.1", "0.25", "0.5", "1", "2",
                "3", "5", "10", "20",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            max_col: 8,
            max_depth: 4,
            max_nodes: 15,
        }
    }
}

/// Ordered grammar input. `transforms` and `heads` retain Python dict insertion order.
#[derive(Clone, Debug, PartialEq)]
pub struct GrammarSpec {
    pub engine: String,
    pub version: String,
    pub max_transforms: usize,
    pub transforms: Vec<(String, OpSpec)>,
    pub heads: Vec<(String, OpSpec)>,
    pub expr: Option<ExprSpec>,
    pub max_expr_stages: usize,
}

impl GrammarSpec {
    pub(super) fn op(&self, name: &str) -> Option<&OpSpec> {
        self.transforms
            .iter()
            .chain(self.heads.iter())
            .find_map(|(key, op)| (key == name).then_some(op))
    }
}

/// Built action vocabulary, mirroring the Python `ActionVocab` fields.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ActionVocab {
    pub tokens: Vec<String>,
    pub index: BTreeMap<String, usize>,
    pub engines: Vec<String>,
    pub engine_ids: BTreeMap<String, usize>,
    pub ops: BTreeMap<usize, Op>,
    /// Lookup-only tuple-key map; Python removes this field before JSON serialization too.
    #[serde(skip)]
    pub op_id_by_name: BTreeMap<(String, String), usize>,
    pub engine_transforms: BTreeMap<String, Vec<usize>>,
    pub engine_heads: BTreeMap<String, Vec<usize>>,
    pub append_ids: [usize; 3],
    pub expr: Expr,
    pub max_transforms: BTreeMap<String, usize>,
    pub max_expr_stages: BTreeMap<String, usize>,
    pub version: String,
    pub tasks: Vec<String>,
    pub task_ids: BTreeMap<String, usize>,
    pub head_tasks: BTreeMap<usize, Vec<String>>,
    pub counters: BTreeMap<String, u64>,
}

pub(super) fn append_token(
    tokens: &mut Vec<String>,
    index: &mut BTreeMap<String, usize>,
    token: String,
) -> usize {
    let token_id = tokens.len();
    tokens.push(token.clone());
    index.insert(token, token_id);
    token_id
}

pub(super) fn append_unique(
    tokens: &mut Vec<String>,
    index: &mut BTreeMap<String, usize>,
    token: String,
) -> Result<usize> {
    if index.contains_key(&token) {
        bail!("token collision while extending vocab: {token}");
    }
    Ok(append_token(tokens, index, token))
}

/// Match `pipeline_text.format_param_token` for JSON scalar grammar values.
pub fn format_param_token(value: &Value) -> String {
    match value {
        Value::Bool(value) => return value.to_string(),
        Value::Number(number) => return format_number(number),
        Value::String(value) => value.clone(),
        Value::Null => "None".to_owned(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
    .pipe_quote_if_needed()
}

trait QuoteIfNeeded {
    fn pipe_quote_if_needed(self) -> String;
}

impl QuoteIfNeeded for String {
    fn pipe_quote_if_needed(self) -> String {
        if !self.is_empty()
            && self
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_+*.-".contains(&b))
        {
            self
        } else {
            format!("\"{}\"", self.replace('\\', "\\\\").replace('"', "\\\""))
        }
    }
}

fn format_number(number: &Number) -> String {
    if number.is_i64() || number.is_u64() {
        return number.to_string();
    }
    let Some(value) = number.as_f64() else {
        return number.to_string();
    };
    if (value - value.round()).abs() < 1.0e-12 {
        format!("{value:.1}")
    } else {
        python_float_repr(value)
    }
}

fn python_float_repr(value: f64) -> String {
    let shortest = format!("{value:?}");
    let Some((mantissa, exponent)) = shortest.split_once('e') else {
        return shortest;
    };
    let exponent = exponent.parse::<i32>().expect("Rust float exponent");
    if (-4..16).contains(&exponent) {
        let negative = mantissa.starts_with('-');
        let unsigned = mantissa.trim_start_matches('-');
        let (whole, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
        let digits = format!("{whole}{fraction}");
        let decimal = whole.len() as i32 + exponent;
        let expanded = if decimal <= 0 {
            format!("0.{}{}", "0".repeat((-decimal) as usize), digits)
        } else if decimal as usize >= digits.len() {
            format!("{}{}", digits, "0".repeat(decimal as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..decimal as usize],
                &digits[decimal as usize..]
            )
        };
        return if negative {
            format!("-{expanded}")
        } else {
            expanded
        };
    }
    let sign = if exponent >= 0 { "+" } else { "-" };
    format!("{mantissa}e{sign}{:02}", exponent.unsigned_abs())
}
