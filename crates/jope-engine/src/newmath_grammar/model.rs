use super::strings;
use serde_json::{json, Value};

#[derive(Clone, Debug, PartialEq)]
pub struct ParamSpec {
    pub name: String,
    pub kind: String,
    pub bins: Vec<Value>,
    pub default: Option<Value>,
    pub required: bool,
}

impl ParamSpec {
    pub(super) fn new(name: &str, kind: &str, bins: Vec<Value>) -> Self {
        Self {
            name: name.into(),
            kind: kind.into(),
            bins,
            default: None,
            required: false,
        }
    }

    pub(super) fn required_default(name: &str, default: &str) -> Self {
        Self {
            name: name.into(),
            kind: "expr".into(),
            bins: Vec::new(),
            default: Some(json!(default)),
            required: true,
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "kind": self.kind,
            "bins": self.bins,
            "default": self.default,
            "required": self.required,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
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
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "role": self.role,
            "params": self.params.iter().map(ParamSpec::to_json).collect::<Vec<_>>(),
            "families": self.families,
            "cost_class": self.cost_class,
            "allow_append": self.allow_append,
            "generate": self.generate,
            "takes_expr": self.takes_expr,
            "tasks": self.tasks,
        })
    }
}

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
            vars_map: strings(&["x"]),
            vars_common: strings(&["row_mean", "row_std", "row_min", "row_max", "row_l2"]),
            funcs1: strings(&["abs", "sqrt", "log1p", "sin", "cos", "tanh", "sigmoid"]),
            funcs2: strings(&["min", "max", "step"]),
            funcs3: strings(&["clip"]),
            binops: strings(&["+", "-", "*", "/"]),
            consts: strings(&[
                "-2", "-1", "-0.5", "-0.25", "-0.1", "0", "0.05", "0.1", "0.25", "0.5", "1", "2",
                "3", "5", "10", "20",
            ]),
            max_col: 8,
            max_depth: 4,
            max_nodes: 15,
        }
    }
}

impl ExprSpec {
    pub fn to_json(&self) -> Value {
        json!({
            "vars_map": self.vars_map,
            "vars_common": self.vars_common,
            "funcs1": self.funcs1,
            "funcs2": self.funcs2,
            "funcs3": self.funcs3,
            "binops": self.binops,
            "consts": self.consts,
            "max_col": self.max_col,
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GrammarSpec {
    pub engine: String,
    pub version: String,
    pub evaluator_version: String,
    pub max_transforms: usize,
    pub transforms: Vec<OpSpec>,
    pub heads: Vec<OpSpec>,
    pub expr: ExprSpec,
    pub fallback_genome: String,
    pub max_expr_stages: usize,
}

impl GrammarSpec {
    pub fn op(&self, name: &str) -> Option<&OpSpec> {
        self.transforms
            .iter()
            .chain(&self.heads)
            .find(|op| op.name == name)
    }

    pub fn to_json(&self) -> Value {
        let transforms: serde_json::Map<String, Value> = self
            .transforms
            .iter()
            .map(|op| (op.name.clone(), op.to_json()))
            .collect();
        let heads: serde_json::Map<String, Value> = self
            .heads
            .iter()
            .map(|op| (op.name.clone(), op.to_json()))
            .collect();
        json!({
            "engine": self.engine,
            "version": self.version,
            "evaluator_version": self.evaluator_version,
            "max_transforms": self.max_transforms,
            "transforms": transforms,
            "heads": heads,
            "expr": self.expr.to_json(),
            "fallback_genome": self.fallback_genome,
            "max_expr_stages": self.max_expr_stages,
        })
    }
}

impl ParamSpec {
    pub(super) fn tap_bins(mut self, bins: Vec<Value>) -> Self {
        self.bins = bins;
        self
    }
}
