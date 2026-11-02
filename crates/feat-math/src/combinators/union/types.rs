use super::*;

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
    pub(super) id: String,
    pub(super) columns: Vec<String>,
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
    pub(super) fn index_of(&self, key: &str) -> Option<usize> {
        self.columns.iter().position(|column| column == key)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnRef {
    pub(super) schema_id: String,
    pub(super) key: String,
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

/// An exact, `Eq` finite scalar for affine parameters. It stores the IEEE-754 bit pattern, so
/// equality is exact and no value is quantized. Non-finite values are refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactScalar {
    bits: u64,
}

impl ExactScalar {
    pub fn new(value: f64) -> Result<Self, GraphError> {
        if !value.is_finite() {
            return Err(GraphError::InvalidGraph(
                "affine parameter must be finite".into(),
            ));
        }
        Ok(Self {
            bits: value.to_bits(),
        })
    }

    /// Build from raw bits without checking. The value is validated when the node is measured.
    pub fn from_bits(bits: u64) -> Self {
        Self { bits }
    }

    pub fn value(&self) -> Result<f64, GraphError> {
        let value = f64::from_bits(self.bits);
        if value.is_finite() {
            Ok(value)
        } else {
            Err(GraphError::InvalidGraph(
                "affine parameter must be finite".into(),
            ))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnionAction {
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
        scale: ExactScalar,
        offset: ExactScalar,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
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
    pub(super) fn inputs(&self) -> Vec<usize> {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub abi: u16,
    pub nodes: Vec<Node>,
    pub output: usize,
    pub limits: Limits,
}

/// Compatibility alias for the domain-specific `UnionAction` type.
pub type Action = UnionAction;
