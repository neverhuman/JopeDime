use super::{canon, Effects, Result, Value};
use crate::dsl::{DslExpr, Stage};
use crate::kernel::registry_v2;
use crate::math_expr::{BinOp, Expr, Func, MathExpr, UnOp, Var};
use crate::registry::{self, ShapeRule};
use crate::GenomeIr;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const GRAPH_SCHEMA: &str = "jope.selfplay.graph-effects/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowSemantics {
    Iid,
    Ordered,
}

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
pub enum Expression {
    Constant {
        value: canon::ExactF64,
    },
    Column {
        column: ColumnRef,
    },
    Unary {
        op: UnOp,
        argument: Box<Expression>,
    },
    Binary {
        op: BinOp,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    Call {
        function: Func,
        arguments: Vec<Expression>,
    },
}
impl Expression {
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
        fn lower(e: &Expression) -> Result<Expr> {
            Ok(match e {
                Expression::Constant { value } => Expr::Const(value.value()?),
                Expression::Column { column } => Expr::Var(Var::Col(column.index)),
                Expression::Unary { op, argument } => Expr::Unary(*op, Box::new(lower(argument)?)),
                Expression::Binary { op, left, right } => {
                    Expr::Binary(*op, Box::new(lower(left)?), Box::new(lower(right)?))
                }
                Expression::Call {
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
        expression: Expression,
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

impl ProgramGraph {
    pub fn new(
        input_schema: InputSchema,
        nodes: Vec<Node>,
        output: usize,
        limits: &Constraints,
    ) -> Result<Self> {
        let graph = Self {
            schema: GRAPH_SCHEMA.into(),
            version: 1,
            input_schema,
            profile_id: registry_v2::PROFILE_ID.into(),
            profile_revision: registry_v2::REVISION,
            registry_sha256: registry_v2::source_sha256(),
            nodes,
            output,
        };
        graph.validate(limits)?;
        Ok(graph)
    }
    pub fn input_schema(&self) -> &InputSchema {
        &self.input_schema
    }
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    pub fn output(&self) -> usize {
        self.output
    }
    pub fn validate(&self, limits: &Constraints) -> Result<Effects> {
        Ok(self.lower(limits)?.effects)
    }
    pub fn to_bytes(&self, limits: &Constraints) -> Result<Vec<u8>> {
        self.validate(limits)?;
        canon::encode(self)
    }
    pub fn from_bytes(bytes: &[u8], limits: &Constraints) -> Result<Self> {
        limits.validate()?;
        if bytes.len() > limits.max_bytes {
            return Err("graph byte bound".into());
        }
        let graph: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if graph.to_bytes(limits)? != bytes {
            return Err("noncanonical graph bytes, duplicate key or unknown representation".into());
        }
        Ok(graph)
    }
    pub fn identity(&self, limits: &Constraints) -> Result<canon::ProgramIdV2> {
        canon::ProgramIdV2::compute(GRAPH_SCHEMA, &self.to_bytes(limits)?)
    }
    pub(crate) fn lower(&self, limits: &Constraints) -> Result<Lowered> {
        limits.validate()?;
        self.input_schema.validate()?;
        if self.schema != GRAPH_SCHEMA
            || self.version != 1
            || self.profile_id != registry_v2::PROFILE_ID
            || self.profile_revision != registry_v2::REVISION
            || self.registry_sha256 != registry_v2::source_sha256()
        {
            return Err("graph version/profile/registry identity mismatch".into());
        }
        if self.nodes.len() < 2
            || self.nodes.len() > limits.max_nodes
            || self.output != self.nodes.len() - 1
            || !matches!(self.nodes.first(), Some(Node::Input))
        {
            return Err("graph node/output bound or missing input".into());
        }
        let mut ports = Vec::new();
        let mut effects = Vec::new();
        let mut depths = Vec::new();
        let mut edges = 0usize;
        let mut stages = Vec::new();
        let mut columns: Vec<usize> = (0..self.input_schema.columns.len()).collect();
        let mut symbolic = None;
        for (index, node) in self.nodes.iter().enumerate() {
            let deps = node.dependencies();
            edges = edges.checked_add(deps.len()).ok_or("edge overflow")?;
            if edges > limits.max_edges || deps.iter().any(|d| *d >= index) {
                return Err("cycle, forward/unknown node reference or edge bound".into());
            }
            let depth = deps.iter().map(|d| depths[*d]).max().unwrap_or(0) + 1;
            if depth > limits.max_depth {
                return Err("graph depth bound".into());
            }
            let mut effect = deps
                .iter()
                .fold(Effects::default(), |a, d| a.union(effects[*d]));
            let port=match node {
                Node::Input if index==0=>Port::Matrix(self.input_schema.columns.len()),
                Node::Input=>return Err("multiple graph inputs".into()),
                Node::Columns{input,columns:refs}=>{
                    if *input!=0||index!=1||refs.is_empty()||refs.len()>limits.max_width{return Err("column projection must directly follow input within width bound".into());}
                    for column in refs{column.validate(&self.input_schema)?;}
                    columns=refs.iter().map(|c|c.index).collect();Port::Matrix(columns.len())
                },
                Node::Operator{input,operator,parameters}=>{
                    if *input!=index-1{return Err("executable adapter requires a linear stage chain".into());}
                    let Port::Matrix(width)=ports[*input]else{return Err("matrix input required".into());};
                    if width>limits.max_width{return Err("fit width bound; select schema columns before native fitting".into());}
                    let entry=operator.entry()?;let facts=entry.facts();let materialized=stage(operator,parameters)?;
                    effect.fit_x|=facts.fitted.ok_or("unknown fitting effect")?;effect.fit_y|=facts.requires_labels;effect.target_derived|=facts.requires_labels;
                    effect.ordered_rows|=entry.axis=="ordered_rows";
                    effect.seeded|=materialized.params.contains_key("seed");
                    let port=if facts.role=="head"{
                        if !supported_head(facts.canonical){return Err(format!("{}: head/composition adapter unavailable; nested cross-fitting not proven",operator.name));}
                        Port::Prediction
                    }else if facts.role=="transform"&&entry.inherit=="registry_v1"{
                        let row=registry::operator(&operator.name).ok_or("inherited shape missing")?;
                        let width=match row.shape{
                            ShapeRule::Preserve=>width,
                            ShapeRule::Parameter(key)=>materialized.get_usize(key,0),
                            ShapeRule::Fixed(n)=>n,
                            _=>return Err(format!("{}: exact width adapter unavailable",operator.name)),
                        };
                        let width=if matches!(materialized.params.get("append"),Some(crate::dsl::ParamValue::Bool(true))){width.checked_add(match ports[*input]{Port::Matrix(n)=>n,_=>0}).ok_or("width overflow")?}else{width};
                        if width==0||width>limits.max_width{return Err("operator output width bound".into());}Port::Matrix(width)
                    }else{return Err("target/directive/expansion layout adapter unavailable".into());};
                    stages.push(materialized);port
                },
                Node::Scalar{input,expression,affine}=>{
                    if *input!=0||index!=1{return Err("scalar adapter requires direct schema input".into());}
                    let expr=MathExpr::from_ast(expression.to_ast(&self.input_schema,limits)?).map_err(|e|e.to_string())?;
                    effect.fit_x|=*affine;effect.fit_y|=*affine;effect.target_derived|=*affine;
                    symbolic=Some(GenomeIr::Symbolic{source:expr.to_source().map_err(|e|e.to_string())?,expr,affine:*affine});Port::Prediction
                },
                Node::Stack{..}|Node::Residual{..}=>return Err("nested cross-fitting adapter unavailable for stack/residual; target-derived paths cannot be admitted".into()),
            };
            ports.push(port);
            effects.push(effect);
            depths.push(depth);
        }
        if ports[self.output] != Port::Prediction {
            return Err("graph output must be a prediction port".into());
        }
        let mut live = std::collections::BTreeSet::new();
        let mut pending = vec![self.output];
        while let Some(i) = pending.pop() {
            if live.insert(i) {
                pending.extend(self.nodes[i].dependencies());
            }
        }
        if live.len() != self.nodes.len() {
            return Err("unreachable graph nodes".into());
        }
        if canon::encode(self)?.len() > limits.max_bytes {
            return Err("graph byte bound".into());
        }
        Ok(Lowered {
            genome: symbolic.unwrap_or(GenomeIr::Pipeline(DslExpr { stages })),
            columns,
            effects: effects[self.output],
        })
    }
}

impl InputSchema {
    /// Opt-in semantic-key schema. The union schema sorts keys canonically, so its id is
    /// permutation-invariant. `identity()` above is unchanged and stays order-sensitive.
    pub fn semantic_key_schema(
        &self,
    ) -> std::result::Result<crate::combinators::union::Schema, crate::combinators::union::GraphError>
    {
        crate::combinators::union::Schema::new(self.columns.clone())
    }

    /// Opt-in binding by semantic key, for use with `combinators::union::Program`.
    pub fn bind_semantic_column(
        &self,
        key: &str,
    ) -> std::result::Result<
        crate::combinators::union::ColumnRef,
        crate::combinators::union::GraphError,
    > {
        let schema = self.semantic_key_schema()?;
        crate::combinators::union::ColumnRef::bind(&schema, key)
    }
}
