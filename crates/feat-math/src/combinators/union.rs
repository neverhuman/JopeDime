//! Bounded typed DAG representation for schema-bound wide-column compositions.
//!
//! Routing is eager: both branches are evaluated once per graph invocation, and
//! the validator charges both branches against width, depth, allocation and fuel
//! limits before evaluation begins.

use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

use super::residual::FitRole;
use super::route::route_rows;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphError {
    InvalidSchema(String),
    InvalidGraph(String),
    Cycle(usize),
    Limit(String),
    Input(String),
    Evaluation(String),
    UnsupportedEmitter(String),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for GraphError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schema {
    id: String,
    columns: Vec<String>,
}

impl Schema {
    /// Schema identity binds the set of semantic column keys, not their physical order.
    /// Thus a shuffled view of the same unique keys retains the same identity.
    pub fn new(columns: Vec<String>) -> Result<Self, GraphError> {
        if columns.is_empty() || columns.iter().any(|key| key.trim().is_empty()) {
            return Err(GraphError::InvalidSchema(
                "schema needs nonempty semantic column keys".into(),
            ));
        }
        let unique: BTreeSet<_> = columns.iter().collect();
        if unique.len() != columns.len() {
            return Err(GraphError::InvalidSchema(
                "schema column keys must be unique".into(),
            ));
        }
        let mut canonical: Vec<_> = columns.iter().map(String::as_str).collect();
        canonical.sort_unstable();
        let mut hash = Sha256::new();
        hash.update(b"jope-dag-schema-v1\0");
        for key in canonical {
            hash.update((key.len() as u64).to_le_bytes());
            hash.update(key.as_bytes());
        }
        let id = format!("schema-v1:{:x}", hash.finalize());
        Ok(Self { id, columns })
    }

    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn columns(&self) -> &[String] {
        &self.columns
    }
    fn index_of(&self, key: &str) -> Option<usize> {
        self.columns.iter().position(|column| column == key)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnRef {
    schema_id: String,
    key: String,
}

impl ColumnRef {
    pub fn bind(schema: &Schema, key: impl Into<String>) -> Result<Self, GraphError> {
        let key = key.into();
        if schema.index_of(&key).is_none() {
            return Err(GraphError::InvalidSchema(format!(
                "column {key:?} is absent from schema"
            )));
        }
        Ok(Self {
            schema_id: schema.id.clone(),
            key,
        })
    }
    pub fn key(&self) -> &str {
        &self.key
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FitEffect {
    /// A stable identity for one target-aware fit. Repeated IDs may share only an
    /// exactly identical target, role, and training-row closure.
    pub fit_id: String,
    pub target_id: String,
    pub role: FitRole,
    pub train_row_ids: Vec<u64>,
    pub fuel: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    Column(ColumnRef),
    Union(Vec<usize>),
    Route {
        predicate: usize,
        when_true: usize,
        when_false: usize,
    },
    /// Binds an upstream node. Consumers share the bound node, which is computed once.
    Let(usize),
    /// Projects one column from a multi-column input (width 1).
    SelectFeature {
        input: usize,
        column_index: usize,
    },
    /// Concatenates feature matrices in the given order.
    ConcatFeatures(Vec<usize>),
    /// Row-wise finite affine transform of every value in the input.
    Affine {
        input: usize,
        scale: f64,
        offset: f64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub action: Action,
    pub fit_effect: Option<FitEffect>,
}

impl Node {
    pub fn new(action: Action) -> Self {
        Self {
            action,
            fit_effect: None,
        }
    }
    pub fn with_fit_effect(mut self, effect: FitEffect) -> Self {
        self.fit_effect = Some(effect);
        self
    }
    fn inputs(&self) -> Vec<usize> {
        match &self.action {
            Action::Column(_) => Vec::new(),
            Action::Union(inputs) => inputs.clone(),
            Action::Route {
                predicate,
                when_true,
                when_false,
            } => vec![*predicate, *when_true, *when_false],
            Action::Let(input)
            | Action::SelectFeature { input, .. }
            | Action::Affine { input, .. } => vec![*input],
            Action::ConcatFeatures(inputs) => inputs.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_nodes: usize,
    pub max_width: usize,
    pub max_depth: usize,
    pub max_rows: usize,
    pub max_fuel: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub width: usize,
    pub values: Vec<f64>,
}

impl Matrix {
    pub fn get(&self, row: usize, column: usize) -> f64 {
        self.values[row * self.width + column]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DenseInput {
    pub schema: Schema,
    pub values: Vec<Vec<f64>>,
}

impl DenseInput {
    pub fn new(columns: Vec<String>, values: Vec<Vec<f64>>) -> Result<Self, GraphError> {
        let schema = Schema::new(columns)?;
        if values.is_empty()
            || values.iter().any(|row| {
                row.len() != schema.columns.len() || row.iter().any(|value| !value.is_finite())
            })
        {
            return Err(GraphError::Input(
                "input rows must be nonempty, finite, and match the full schema width".into(),
            ));
        }
        Ok(Self { schema, values })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Validation {
    pub node_widths: Vec<usize>,
    pub node_depths: Vec<usize>,
    pub worst_case_fuel: u64,
    pub eager_route_branches: usize,
    pub shared_fit_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub abi: u16,
    pub nodes: Vec<Node>,
    pub output: usize,
    pub limits: Limits,
}

impl Program {
    pub const ABI: u16 = 1;

    pub fn validate(&self, schema: &Schema, rows: usize) -> Result<Validation, GraphError> {
        if self.abi != Self::ABI
            || self.nodes.is_empty()
            || self.output >= self.nodes.len()
            || self.nodes.len() > self.limits.max_nodes
            || rows == 0
            || rows > self.limits.max_rows
            || self.limits.max_width == 0
            || self.limits.max_depth == 0
            || self.limits.max_fuel == 0
        {
            return Err(GraphError::Limit(
                "invalid ABI, graph shape, or declared limit".into(),
            ));
        }
        let mut widths = vec![None; self.nodes.len()];
        let mut depths = vec![None; self.nodes.len()];
        let mut state = vec![0u8; self.nodes.len()];
        let mut reachable = BTreeSet::new();
        let mut route_count = 0usize;
        let mut fuel = 0u64;
        let mut fits: BTreeMap<&str, &FitEffect> = BTreeMap::new();
        let mut shared_fit_ids = BTreeSet::new();
        let output_width = self.measure(
            self.output,
            schema,
            rows,
            &mut widths,
            &mut depths,
            &mut state,
            &mut reachable,
            &mut route_count,
            &mut fuel,
            &mut fits,
            &mut shared_fit_ids,
        )?;
        if output_width == 0 || reachable.len() != self.nodes.len() {
            return Err(GraphError::InvalidGraph(
                "every node must be reachable from a nonempty output".into(),
            ));
        }
        if fuel > self.limits.max_fuel {
            return Err(GraphError::Limit(format!(
                "worst-case eager fuel {fuel} exceeds {}",
                self.limits.max_fuel
            )));
        }
        Ok(Validation {
            node_widths: widths.into_iter().map(Option::unwrap).collect(),
            node_depths: depths.into_iter().map(Option::unwrap).collect(),
            worst_case_fuel: fuel,
            eager_route_branches: route_count,
            shared_fit_ids: shared_fit_ids.into_iter().map(str::to_owned).collect(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn measure<'a>(
        &'a self,
        id: usize,
        schema: &Schema,
        rows: usize,
        widths: &mut [Option<usize>],
        depths: &mut [Option<usize>],
        state: &mut [u8],
        reachable: &mut BTreeSet<usize>,
        route_count: &mut usize,
        fuel: &mut u64,
        fits: &mut BTreeMap<&'a str, &'a FitEffect>,
        shared_fit_ids: &mut BTreeSet<&'a str>,
    ) -> Result<usize, GraphError> {
        if id >= self.nodes.len() {
            return Err(GraphError::InvalidGraph(format!(
                "node reference {id} is absent"
            )));
        }
        if state[id] == 1 {
            return Err(GraphError::Cycle(id));
        }
        if let Some(width) = widths[id] {
            return Ok(width);
        }
        state[id] = 1;
        let node = &self.nodes[id];
        reachable.insert(id);
        let children = node.inputs();
        let mut child_widths = Vec::with_capacity(children.len());
        let mut child_depth = 0usize;
        for child in children {
            let width = self.measure(
                child,
                schema,
                rows,
                widths,
                depths,
                state,
                reachable,
                route_count,
                fuel,
                fits,
                shared_fit_ids,
            )?;
            child_widths.push(width);
            child_depth = child_depth.max(depths[child].unwrap());
        }
        let width = match &node.action {
            Action::Column(reference) => {
                if reference.schema_id != schema.id {
                    return Err(GraphError::InvalidSchema(
                        "column reference belongs to a different schema".into(),
                    ));
                }
                if schema.index_of(&reference.key).is_none() {
                    return Err(GraphError::InvalidSchema(format!(
                        "bound column {:?} is not in this schema",
                        reference.key
                    )));
                }
                1
            }
            Action::Union(inputs) => {
                if inputs.is_empty() {
                    return Err(GraphError::InvalidGraph("union must have inputs".into()));
                }
                child_widths.iter().try_fold(0usize, |sum, next| {
                    sum.checked_add(*next)
                        .ok_or_else(|| GraphError::Limit("union width overflow".into()))
                })?
            }
            Action::Route { .. } => {
                *route_count = route_count.saturating_add(1);
                if child_widths.len() != 3 || child_widths[0] != 1 {
                    return Err(GraphError::InvalidGraph(
                        "route predicate must have width one".into(),
                    ));
                }
                if child_widths[1] != child_widths[2] {
                    return Err(GraphError::InvalidGraph(
                        "eager route branches must have equal widths".into(),
                    ));
                }
                child_widths[1]
            }
            Action::Let(_) => child_widths[0],
            Action::SelectFeature {
                column_index,
                input: _,
            } => {
                if *column_index >= child_widths[0] {
                    return Err(GraphError::InvalidGraph(format!(
                        "node {id} selects column {column_index} of width {}",
                        child_widths[0]
                    )));
                }
                1
            }
            Action::ConcatFeatures(inputs) => {
                if inputs.is_empty() {
                    return Err(GraphError::InvalidGraph(
                        "concatenation must have inputs".into(),
                    ));
                }
                child_widths.iter().try_fold(0usize, |sum, next| {
                    sum.checked_add(*next)
                        .ok_or_else(|| GraphError::Limit("concatenation width overflow".into()))
                })?
            }
            Action::Affine { scale, offset, .. } => {
                if !scale.is_finite() || !offset.is_finite() {
                    return Err(GraphError::InvalidGraph(format!(
                        "node {id} affine parameters must be finite"
                    )));
                }
                child_widths[0]
            }
        };
        if width == 0 || width > self.limits.max_width {
            return Err(GraphError::Limit(format!(
                "node {id} width {width} exceeds {}",
                self.limits.max_width
            )));
        }
        let depth = child_depth.saturating_add(1);
        if depth > self.limits.max_depth {
            return Err(GraphError::Limit(format!(
                "node {id} depth {depth} exceeds {}",
                self.limits.max_depth
            )));
        }
        let node_fuel = (rows as u64)
            .checked_mul(width as u64)
            .ok_or_else(|| GraphError::Limit("fuel arithmetic overflow".into()))?;
        *fuel = fuel
            .checked_add(node_fuel)
            .ok_or_else(|| GraphError::Limit("fuel arithmetic overflow".into()))?;
        if let Some(effect) = &node.fit_effect {
            if effect.fit_id.trim().is_empty()
                || effect.target_id.trim().is_empty()
                || effect.train_row_ids.is_empty()
                || effect
                    .train_row_ids
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .len()
                    != effect.train_row_ids.len()
            {
                return Err(GraphError::InvalidGraph(
                    "incomplete target-aware fit effect".into(),
                ));
            }
            let first_fit = if let Some(previous) = fits.get(effect.fit_id.as_str()) {
                if *previous != effect {
                    return Err(GraphError::InvalidGraph(format!(
                        "shared fit identity {:?} crosses incompatible target, role, or rows",
                        effect.fit_id
                    )));
                }
                shared_fit_ids.insert(effect.fit_id.as_str());
                false
            } else {
                fits.insert(effect.fit_id.as_str(), effect);
                true
            };
            if first_fit {
                *fuel = fuel
                    .checked_add(effect.fuel)
                    .ok_or_else(|| GraphError::Limit("fit fuel arithmetic overflow".into()))?;
            }
        }
        if *fuel > self.limits.max_fuel {
            return Err(GraphError::Limit(format!(
                "worst-case eager fuel exceeds {}",
                self.limits.max_fuel
            )));
        }
        widths[id] = Some(width);
        depths[id] = Some(depth);
        state[id] = 2;
        Ok(width)
    }

    /// No portable backend is registered for this draft representation. Refuse
    /// explicitly instead of relabeling interpreter execution as emitted output.
    pub fn emit_for_target(&self, target: &str) -> Result<String, GraphError> {
        Err(GraphError::UnsupportedEmitter(format!(
            "DAG ABI {} has no registered emitter for target {target:?}",
            self.abi
        )))
    }

    pub fn evaluate(&self, input: &DenseInput) -> Result<(Matrix, Validation), GraphError> {
        let (output, validation, _counts) = self.evaluate_counted(input)?;
        Ok((output, validation))
    }

    /// Evaluate and report how many times each node was computed. A node shared by several
    /// consumers is cached after its first computation, so its count is 1.
    pub fn evaluate_counted(
        &self,
        input: &DenseInput,
    ) -> Result<(Matrix, Validation, Vec<usize>), GraphError> {
        if self.nodes.iter().any(|node| node.fit_effect.is_some()) {
            return Err(GraphError::InvalidGraph(
                "pure feature DAG evaluation refuses fit_effect nodes; fit/label capabilities require scoped fit execution"
                    .into(),
            ));
        }
        let validation = self.validate(&input.schema, input.values.len())?;
        if input.values.iter().any(|row| {
            row.len() != input.schema.columns.len() || row.iter().any(|value| !value.is_finite())
        }) {
            return Err(GraphError::Input(
                "invalid dense input shape or value".into(),
            ));
        }
        let mut cache = vec![None; self.nodes.len()];
        let mut counts = vec![0usize; self.nodes.len()];
        let output =
            self.evaluate_node(self.output, input, &validation, &mut cache, &mut counts)?;
        Ok((output, validation, counts))
    }

    fn evaluate_node(
        &self,
        id: usize,
        input: &DenseInput,
        validation: &Validation,
        cache: &mut [Option<Matrix>],
        counts: &mut [usize],
    ) -> Result<Matrix, GraphError> {
        if let Some(matrix) = &cache[id] {
            return Ok(matrix.clone());
        }
        counts[id] += 1;
        let node = &self.nodes[id];
        let rows = input.values.len();
        let matrix = match &node.action {
            Action::Column(reference) => {
                if reference.schema_id != input.schema.id {
                    return Err(GraphError::InvalidSchema("schema identity changed".into()));
                }
                let column = input
                    .schema
                    .index_of(&reference.key)
                    .ok_or_else(|| GraphError::InvalidSchema("column key missing".into()))?;
                Matrix {
                    rows,
                    width: 1,
                    values: input.values.iter().map(|row| row[column]).collect(),
                }
            }
            Action::Union(children) => {
                let matrices = children
                    .iter()
                    .map(|child| self.evaluate_node(*child, input, validation, cache, counts))
                    .collect::<Result<Vec<_>, _>>()?;
                let width = validation.node_widths[id];
                let mut values = Vec::with_capacity(rows * width);
                for row in 0..rows {
                    for matrix in &matrices {
                        let start = row * matrix.width;
                        values.extend_from_slice(&matrix.values[start..start + matrix.width]);
                    }
                }
                Matrix {
                    rows,
                    width,
                    values,
                }
            }
            Action::Route {
                predicate,
                when_true,
                when_false,
            } => {
                // Both branches are deliberately evaluated; validation charged the
                // complete eager branch DAG before any output allocation.
                let predicate = self.evaluate_node(*predicate, input, validation, cache, counts)?;
                let yes = self.evaluate_node(*when_true, input, validation, cache, counts)?;
                let no = self.evaluate_node(*when_false, input, validation, cache, counts)?;
                route_rows(&predicate, &yes, &no)?
            }
            Action::Let(bound) => self.evaluate_node(*bound, input, validation, cache, counts)?,
            Action::SelectFeature {
                input: source,
                column_index,
            } => {
                let source = self.evaluate_node(*source, input, validation, cache, counts)?;
                Matrix {
                    rows,
                    width: 1,
                    values: (0..rows)
                        .map(|row| source.values[row * source.width + column_index])
                        .collect(),
                }
            }
            Action::ConcatFeatures(children) => {
                let matrices = children
                    .iter()
                    .map(|child| self.evaluate_node(*child, input, validation, cache, counts))
                    .collect::<Result<Vec<_>, _>>()?;
                let width = validation.node_widths[id];
                let mut values = Vec::with_capacity(rows * width);
                for row in 0..rows {
                    for matrix in &matrices {
                        let start = row * matrix.width;
                        values.extend_from_slice(&matrix.values[start..start + matrix.width]);
                    }
                }
                Matrix {
                    rows,
                    width,
                    values,
                }
            }
            Action::Affine {
                input: source,
                scale,
                offset,
            } => {
                let source = self.evaluate_node(*source, input, validation, cache, counts)?;
                Matrix {
                    rows: source.rows,
                    width: source.width,
                    values: source.values.iter().map(|v| v * scale + offset).collect(),
                }
            }
        };
        if matrix.width != validation.node_widths[id]
            || matrix.values.len() != matrix.rows.saturating_mul(matrix.width)
            || matrix.values.iter().any(|value| !value.is_finite())
        {
            return Err(GraphError::Evaluation(
                "node returned malformed matrix".into(),
            ));
        }
        cache[id] = Some(matrix.clone());
        Ok(matrix)
    }
}
