use super::{canon, Effects, Result, Value};
use crate::dsl::{DslExpr, Stage};
use crate::kernel::registry_v2;
use crate::math_expr::{BinOp, Expr, Func, MathExpr, UnOp, Var};
use crate::registry::{self, ShapeRule};
use crate::GenomeIr;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const GRAPH_SCHEMA: &str = "jope.selfplay.graph-effects/v1";
/// Domain-separated schema for the opt-in pure feature DAG. Its bytes never parse as v1 graphs.
pub const PURE_FEATURE_DAG_SCHEMA: &str = "jope.selfplay.pure-feature-dag/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphRowSemantics {
    Iid,
    Ordered,
}

/// Graph inputs record only whether row order matters; they do not identify the ordering columns.
pub type RowSemantics = GraphRowSemantics;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputSchema {
    columns: Vec<String>,
    rows: RowSemantics,
}
impl InputSchema {
    pub fn new(columns: Vec<String>, rows: RowSemantics) -> Result<Self> {
        let schema = Self { columns, rows };
        schema.validate()?;
        Ok(schema)
    }
    pub fn columns(&self) -> &[String] {
        &self.columns
    }
    pub fn rows(&self) -> RowSemantics {
        self.rows
    }
    pub fn validate(&self) -> Result<()> {
        if self.columns.is_empty()
            || self.columns.len() > 4096
            || self.columns.iter().any(|s| s.is_empty() || s.len() > 128)
        {
            return Err("input schema width/key bound".into());
        }
        let unique: std::collections::BTreeSet<_> = self.columns.iter().collect();
        if unique.len() != self.columns.len() {
            return Err("duplicate schema column key".into());
        }
        Ok(())
    }
    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        let bytes = canon::encode(
            &serde_json::json!({"schema":"jope.selfplay.input-schema/v1","columns":self.columns,"rows":self.rows}),
        )?;
        Ok(canon::ProgramIdV2::compute("jope.selfplay.input-schema/v1", &bytes)?.sha256)
    }
    pub fn column(&self, index: usize) -> Result<ColumnRef> {
        let key = self.columns.get(index).ok_or("column outside schema")?;
        Ok(ColumnRef {
            schema_id: self.identity()?,
            index,
            key: key.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnRef {
    pub schema_id: String,
    pub index: usize,
    pub key: String,
}
impl ColumnRef {
    pub fn validate(&self, schema: &InputSchema) -> Result<()> {
        if self.schema_id != schema.identity()? || schema.columns.get(self.index) != Some(&self.key)
        {
            return Err("column/schema identity, index or key mismatch".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorRef {
    pub dialect: String,
    pub name: String,
    pub stable_identity: String,
    pub semantic_version: String,
    pub registry_sha256: String,
}
impl OperatorRef {
    pub fn new(dialect: &str, name: &str) -> Result<Self> {
        let entry = registry_v2::operator(dialect, name).ok_or("unknown registry operator")?;
        entry.check_inheritance()?;
        Ok(Self {
            dialect: dialect.into(),
            name: name.into(),
            stable_identity: entry.identity.into(),
            semantic_version: entry.facts().semantic_version,
            registry_sha256: registry_v2::source_sha256(),
        })
    }
    pub fn entry(&self) -> Result<&'static registry_v2::Entry> {
        if self.dialect != "legacy_recipe" {
            return Err("graph adapter requires legacy_recipe dialect; advanced-v3 layout/receipt adapter unavailable".into());
        }
        let entry =
            registry_v2::operator(&self.dialect, &self.name).ok_or("unknown registry operator")?;
        entry.check_inheritance()?;
        if self.stable_identity != entry.identity
            || self.semantic_version != entry.facts().semantic_version
            || self.registry_sha256 != registry_v2::source_sha256()
        {
            return Err("operator version/registry identity mismatch".into());
        }
        if matches!(entry.admission, "quarantined" | "disabled" | "receipt") {
            return Err("operator admission unavailable".into());
        }
        entry.supports("runtime")?;
        Ok(entry)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GraphExpression {
    Constant {
        value: canon::ExactF64,
    },
    Column {
        column: ColumnRef,
    },
    Unary {
        op: UnOp,
        argument: Box<GraphExpression>,
    },
    Binary {
        op: BinOp,
        left: Box<GraphExpression>,
        right: Box<GraphExpression>,
    },
    Call {
        function: Func,
        arguments: Vec<GraphExpression>,
    },
}
/// Compatibility name retained for callers using the original graph IR API.
pub type Expression = GraphExpression;

impl GraphExpression {
    pub fn to_ast(&self, schema: &InputSchema, limits: &Constraints) -> Result<Expr> {
        let mut pending = vec![(self, 1)];
        let mut count = 0;
        while let Some((node, depth)) = pending.pop() {
            count += 1;
            if count > limits.max_nodes || depth > limits.max_depth {
                return Err("expression node/depth bound".into());
            }
            match node {
                Self::Constant { value } => {
                    value.value()?;
                }
                Self::Column { column } => column.validate(schema)?,
                Self::Unary { argument, .. } => pending.push((argument, depth + 1)),
                Self::Binary { left, right, .. } => {
                    pending.push((right, depth + 1));
                    pending.push((left, depth + 1));
                }
                Self::Call { arguments, .. } => {
                    if arguments.len() > 3 {
                        return Err("expression arity bound".into());
                    }
                    pending.extend(arguments.iter().map(|a| (a, depth + 1)));
                }
            }
        }
        fn lower(e: &GraphExpression) -> Result<Expr> {
            Ok(match e {
                GraphExpression::Constant { value } => Expr::Const(value.value()?),
                GraphExpression::Column { column } => Expr::Var(Var::Col(column.index)),
                GraphExpression::Unary { op, argument } => {
                    Expr::Unary(*op, Box::new(lower(argument)?))
                }
                GraphExpression::Binary { op, left, right } => {
                    Expr::Binary(*op, Box::new(lower(left)?), Box::new(lower(right)?))
                }
                GraphExpression::Call {
                    function,
                    arguments,
                } => Expr::Call(
                    *function,
                    arguments.iter().map(lower).collect::<Result<_>>()?,
                ),
            })
        }
        let ast = lower(self)?;
        MathExpr::from_ast(ast.clone()).map_err(|e| e.to_string())?;
        Ok(ast)
    }
    pub fn from_ast(ast: &Expr, schema: &InputSchema) -> Result<Self> {
        Ok(match ast {
            Expr::Const(v) => Self::Constant {
                value: canon::ExactF64::new(*v)?,
            },
            Expr::Var(Var::Col(i)) => Self::Column {
                column: schema.column(*i)?,
            },
            Expr::Var(_) => {
                return Err("legacy row-reduction/element variable remains opaque".into())
            }
            Expr::Unary(op, a) => Self::Unary {
                op: *op,
                argument: Box::new(Self::from_ast(a, schema)?),
            },
            Expr::Binary(op, a, b) => Self::Binary {
                op: *op,
                left: Box::new(Self::from_ast(a, schema)?),
                right: Box::new(Self::from_ast(b, schema)?),
            },
            Expr::Call(f, args) => Self::Call {
                function: *f,
                arguments: args
                    .iter()
                    .map(|a| Self::from_ast(a, schema))
                    .collect::<Result<_>>()?,
            },
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Node {
    Input,
    Columns {
        input: usize,
        columns: Vec<ColumnRef>,
    },
    Operator {
        input: usize,
        operator: OperatorRef,
        parameters: BTreeMap<String, Value>,
    },
    Scalar {
        input: usize,
        expression: GraphExpression,
        affine: bool,
    },
    Stack {
        members: Vec<usize>,
    },
    Residual {
        baseline: usize,
        correction: usize,
    },
}
impl Node {
    pub fn dependencies(&self) -> Vec<usize> {
        match self {
            Self::Input => vec![],
            Self::Columns { input, .. }
            | Self::Operator { input, .. }
            | Self::Scalar { input, .. } => vec![*input],
            Self::Stack { members } => members.clone(),
            Self::Residual {
                baseline,
                correction,
            } => vec![*baseline, *correction],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Constraints {
    pub max_nodes: usize,
    pub max_edges: usize,
    pub max_depth: usize,
    pub max_width: usize,
    pub max_bytes: usize,
}
impl Default for Constraints {
    fn default() -> Self {
        Self {
            max_nodes: 64,
            max_edges: 128,
            max_depth: 32,
            max_width: 512,
            max_bytes: 262_144,
        }
    }
}
impl Constraints {
    pub fn validate(self) -> Result<()> {
        if self.max_nodes == 0
            || self.max_nodes > 64
            || self.max_edges > 128
            || self.max_depth == 0
            || self.max_depth > 32
            || self.max_width == 0
            || self.max_width > 512
            || self.max_bytes == 0
            || self.max_bytes > canon::MAX_BYTES
        {
            return Err("graph constraints exceed ABI hard limits".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramGraph {
    pub(super) schema: String,
    pub(super) version: u32,
    pub(super) input_schema: InputSchema,
    pub(super) profile_id: String,
    pub(super) profile_revision: u32,
    pub(super) registry_sha256: String,
    pub(super) nodes: Vec<Node>,
    pub(super) output: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Port {
    Matrix(usize),
    Prediction,
}
#[derive(Clone, Debug)]
pub(crate) struct Lowered {
    pub genome: GenomeIr,
    pub columns: Vec<usize>,
    pub effects: Effects,
}

pub(crate) fn supported_head(name: &str) -> bool {
    matches!(name, "ridge" | "ridge_gcv" | "elasticnet")
}
pub(crate) fn stage(operator: &OperatorRef, parameters: &BTreeMap<String, Value>) -> Result<Stage> {
    operator.entry()?;
    let mut raw = Stage::new(&operator.name);
    for (key, value) in parameters {
        raw.params.insert(key.clone(), value.to_legacy()?);
    }
    let task =
        crate::TaskSpec::regression(crate::Objective::SquaredError).map_err(|e| e.to_string())?;
    registry_v2::RegistryProfile {
        dialect: "legacy_recipe",
    }
    .validate_stage(&raw, &task)
}

mod legacy;
mod pure_feature_dag;

#[cfg(test)]
mod pure_dag_tests;
