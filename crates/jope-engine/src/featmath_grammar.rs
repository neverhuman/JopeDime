//! Rust-owned copy of the static `jope.grammars.featmath` grammar specification.
//!
//! The Python module remains in place because its source digest is embedded in
//! the frozen featmath vocabularies. Keep operation order and task order stable:
//! policy token construction depends on first-seen order.

use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const ENGINE: &str = "featmath";
pub const VERSION: &str = "featmath.v3";
pub const EVALUATOR_VERSION: &str = "jope-validate.eval-regression.v2";
pub const MAX_TRANSFORMS: usize = 6;
pub const MAX_EXPR_STAGES: usize = 3;
pub const FALLBACK_GENOME: &str = "(pipe X (robust_scale) (ridge))";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ParamSpec {
    pub name: String,
    pub kind: String,
    pub bins: Vec<Value>,
    pub default: Option<Value>,
    pub required: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpSpec {
    pub name: String,
    pub role: String,
    pub params: Vec<ParamSpec>,
    pub families: Vec<String>,
    pub cost_class: String,
    pub allow_append: bool,
    pub generate: bool,
    pub takes_expr: bool,
    pub tasks: Vec<String>,
}

impl OpSpec {
    pub fn param(&self, name: &str) -> Option<&ParamSpec> {
        self.params.iter().find(|param| param.name == name)
    }

    pub fn supports_task(&self, task: &str) -> bool {
        self.tasks.iter().any(|candidate| candidate == task)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
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

#[derive(Clone, Debug, PartialEq)]
pub struct GrammarSpec {
    pub engine: String,
    pub version: String,
    pub evaluator_version: String,
    pub max_transforms: usize,
    pub transforms: Vec<OpSpec>,
    pub heads: Vec<OpSpec>,
    pub expr: Option<ExprSpec>,
    pub fallback_genome: String,
    pub max_expr_stages: usize,
}

struct OrderedOps<'a>(&'a [OpSpec]);

impl Serialize for OrderedOps<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for op in self.0 {
            map.serialize_entry(&op.name, op)?;
        }
        map.end()
    }
}

impl Serialize for GrammarSpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(9))?;
        map.serialize_entry("engine", &self.engine)?;
        map.serialize_entry("version", &self.version)?;
        map.serialize_entry("evaluator_version", &self.evaluator_version)?;
        map.serialize_entry("max_transforms", &self.max_transforms)?;
        map.serialize_entry("transforms", &OrderedOps(&self.transforms))?;
        map.serialize_entry("heads", &OrderedOps(&self.heads))?;
        map.serialize_entry("expr", &self.expr)?;
        map.serialize_entry("fallback_genome", &self.fallback_genome)?;
        map.serialize_entry("max_expr_stages", &self.max_expr_stages)?;
        map.end()
    }
}

impl GrammarSpec {
    pub fn op(&self, name: &str) -> Option<&OpSpec> {
        self.transforms
            .iter()
            .chain(&self.heads)
            .find(|op| op.name == name)
    }
}

pub(super) fn param(name: &str, kind: &str, bins: Vec<Value>) -> ParamSpec {
    ParamSpec {
        name: name.into(),
        kind: kind.into(),
        bins,
        default: None,
        required: false,
    }
}

pub(super) fn required_expr(name: &str, default: &str) -> ParamSpec {
    ParamSpec {
        name: name.into(),
        kind: "expr".into(),
        bins: Vec::new(),
        default: Some(json!(default)),
        required: true,
    }
}

pub(super) fn transform(name: &str, families: &[&str], params: Vec<ParamSpec>) -> OpSpec {
    OpSpec {
        name: name.into(),
        role: "transform".into(),
        params,
        families: families.iter().map(|value| (*value).into()).collect(),
        cost_class: "cheap".into(),
        allow_append: true,
        generate: true,
        takes_expr: false,
        tasks: default_tasks(),
    }
}

pub(super) fn head(
    name: &str,
    families: &[&str],
    cost_class: &str,
    params: Vec<ParamSpec>,
) -> OpSpec {
    OpSpec {
        name: name.into(),
        role: "head".into(),
        params,
        families: families.iter().map(|value| (*value).into()).collect(),
        cost_class: cost_class.into(),
        allow_append: false,
        generate: true,
        takes_expr: false,
        tasks: default_tasks(),
    }
}

pub(super) fn default_tasks() -> Vec<String> {
    ["regression", "binary", "timeseries", "multiclass"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

pub(super) fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).into()).collect()
}

pub(super) fn tasks(values: &[&str]) -> Vec<String> {
    strings(values)
}

mod expr;
mod heads;
#[cfg(test)]
mod tests;
mod transforms;

/// Build the static featmath grammar in Python insertion order.
pub fn grammar() -> GrammarSpec {
    GrammarSpec {
        engine: ENGINE.into(),
        version: VERSION.into(),
        evaluator_version: EVALUATOR_VERSION.into(),
        max_transforms: MAX_TRANSFORMS,
        transforms: transforms::definitions(),
        heads: heads::definitions(),
        expr: Some(expr::definition()),
        fallback_genome: FALLBACK_GENOME.into(),
        max_expr_stages: MAX_EXPR_STAGES,
    }
}
