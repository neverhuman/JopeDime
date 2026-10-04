//! Bounded scalar ingress for versioned functions. Decoding performs no
//! evaluation; standalone execution is explicit. Models, export and shipping
//! generation remain held. MathExpr and its exact codec remain v1.
use super::codec::{Reader, Writer};
use super::inverse_links::InverseLink;
use super::numerical_policy::NumericalPolicy;
use super::wire_v2::{VersionError, VersionResult, MAGIC_V2, READER_VERSION};
use crate::math_expr::{BinOp, Expr, Func, MathExpr, UnOp, Var};

const SCALAR_VARIANT: u8 = 1;
const INVERSE_LINK_FEATURE: u64 = 2;
const STABLE_EXPONENTIAL_FEATURE: u64 = 4;
const NUMERICAL_POLICY_FEATURE: u64 = 8;

// Explicit IDs reproduce the frozen v1 prefix, independently of enum order.
const V1_FUNCTION_IDS: &[(u16, Func)] = &[
    (0, Func::Abs),
    (1, Func::Sqrt),
    (2, Func::Log1p),
    (3, Func::Sin),
    (4, Func::Cos),
    (5, Func::Tanh),
    (6, Func::Sigmoid),
    (7, Func::Min),
    (8, Func::Max),
    (9, Func::Clip),
    (10, Func::Step),
    (11, Func::Exp),
    (12, Func::Log),
    (13, Func::Asinh),
    (14, Func::Atan),
    (15, Func::Erf),
    (16, Func::Sign),
    (17, Func::Floor),
    (18, Func::Round),
];
const VARIABLE_IDS: &[(u8, Var)] = &[
    (0, Var::X),
    (1, Var::RowMean),
    (2, Var::RowSum),
    (3, Var::RowL1),
    (4, Var::RowL2),
    (5, Var::RowMin),
    (6, Var::RowMax),
    (7, Var::RowStd),
];
const BINARY_IDS: &[(u8, BinOp)] = &[
    (0, BinOp::Add),
    (1, BinOp::Sub),
    (2, BinOp::Mul),
    (3, BinOp::Div),
    (4, BinOp::Pow),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarFunction {
    V1(Func),
    InverseLink(InverseLink),
    Expm1,
    Log1mexp,
}

impl ScalarFunction {
    pub fn from_versioned_id(version: u8, id: u16) -> VersionResult<Self> {
        if version != READER_VERSION {
            return Err(VersionError::UnsupportedVersion(format!(
                "scalar function codec version {version}"
            )));
        }
        if let Some((_, func)) = V1_FUNCTION_IDS.iter().find(|(tag, _)| *tag == id) {
            return Ok(Self::V1(*func));
        }
        match id {
            21 => Ok(Self::Expm1),
            22 => Ok(Self::Log1mexp),
            _ => InverseLink::from_versioned_id(version, id).map(Self::InverseLink),
        }
    }

    pub fn id(self) -> u16 {
        match self {
            Self::V1(func) => {
                V1_FUNCTION_IDS
                    .iter()
                    .find(|(_, value)| *value == func)
                    .unwrap()
                    .0
            }
            Self::InverseLink(link) => link.id(),
            Self::Expm1 => 21,
            Self::Log1mexp => 22,
        }
    }

    fn additional_features(self) -> u64 {
        match self {
            Self::Expm1 | Self::Log1mexp => STABLE_EXPONENTIAL_FEATURE,
            _ => 0,
        }
    }

    fn check_arity(self, count: usize) -> VersionResult<()> {
        let expected = match self {
            Self::V1(Func::Min | Func::Max | Func::Step) => 2,
            Self::V1(Func::Clip) => 3,
            _ => 1,
        };
        if count != expected {
            return Err(VersionError::Malformed(format!(
                "scalar function {} requires {expected} arguments, got {count}",
                self.id()
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScalarNode {
    Const(f64),
    Var(Var),
    Unary(UnOp, Box<ScalarNode>),
    Binary(BinOp, Box<ScalarNode>, Box<ScalarNode>),
    Call(ScalarFunction, Vec<ScalarNode>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarExpressionProgram {
    root: ScalarNode,
    numerical_policy: NumericalPolicy,
}

impl ScalarExpressionProgram {
    pub fn new(root: ScalarNode) -> VersionResult<Self> {
        // Use the same bounded structural checks as serialization; no evaluator
        // runs during construction, even for nonfinite literal payloads.
        encode_node(&root, &mut Writer::default(), 1, &mut NodeBudget::default())?;
        Ok(Self {
            root,
            numerical_policy: NumericalPolicy::LegacyV1,
        })
    }

    pub fn ast(&self) -> &ScalarNode {
        &self.root
    }

    pub fn numerical_policy(&self) -> NumericalPolicy {
        self.numerical_policy
    }

    /// Select a contract explicitly, without running an evaluator or admitting a model.
    /// Selecting v1 restores the original scalar envelope, with no selector.
    #[must_use]
    pub fn with_numerical_policy(mut self, policy: NumericalPolicy) -> Self {
        self.numerical_policy = policy;
        self
    }

    /// Import typed v1 nodes without reparsing their lossy source printer.
    pub fn from_v1(expression: &MathExpr) -> VersionResult<Self> {
        Ok(Self {
            root: import_node(expression.ast(), 1, &mut NodeBudget::default())?,
            numerical_policy: NumericalPolicy::LegacyV1,
        })
    }

    pub fn to_bytes(&self) -> VersionResult<Vec<u8>> {
        let mut body = Writer::default();
        let mut budget = NodeBudget::default();
        encode_node(&self.root, &mut body, 1, &mut budget)?;
        let mut writer = Writer::default();
        writer.raw(MAGIC_V2)?;
        writer.byte(SCALAR_VARIANT)?;
        writer.byte(READER_VERSION)?;
        let explicit_policy = self.numerical_policy == NumericalPolicy::StableV2;
        writer.word(
            budget.required_features
                | if explicit_policy {
                    NUMERICAL_POLICY_FEATURE
                } else {
                    0
                },
        )?;
        if explicit_policy {
            writer.byte(2)?;
        }
        writer.raw(&body.finish())?;
        Ok(writer.finish())
    }

    pub fn from_bytes(bytes: &[u8]) -> VersionResult<Self> {
        let mut reader = Reader::new(bytes)?;
        let magic = reader.take(MAGIC_V2.len())?;
        if magic[..7] != MAGIC_V2[..7] {
            return Err(VersionError::Malformed("program magic".into()));
        }
        if magic[7] != READER_VERSION {
            return Err(VersionError::UnsupportedVersion(
                "program envelope version".into(),
            ));
        }
        if reader.byte()? != SCALAR_VARIANT {
            return Err(VersionError::UnsupportedVersion(
                "program v2 variant".into(),
            ));
        }
        if reader.byte()? != READER_VERSION {
            return Err(VersionError::UnsupportedVersion(
                "minimum reader version".into(),
            ));
        }
        let features = reader.word()?;
        let supported =
            INVERSE_LINK_FEATURE | STABLE_EXPONENTIAL_FEATURE | NUMERICAL_POLICY_FEATURE;
        if features & INVERSE_LINK_FEATURE == 0 || features & !supported != 0 {
            return Err(VersionError::UnsupportedVersion(
                "program v2 variant requires inverse-link feature set".into(),
            ));
        }
        let numerical_policy = if features & NUMERICAL_POLICY_FEATURE != 0 {
            let policy = reader.byte()?;
            if policy != 2 {
                return Err(VersionError::UnsupportedVersion(format!(
                    "numerical policy {policy}"
                )));
            }
            NumericalPolicy::StableV2
        } else {
            NumericalPolicy::LegacyV1
        };
        let mut budget = NodeBudget::default();
        let root = decode_node(&mut reader, 1, &mut budget)?;
        reader.finish()?;
        let declared_body_features = features & !NUMERICAL_POLICY_FEATURE;
        if declared_body_features != budget.required_features {
            return Err(VersionError::Malformed(
                "scalar required features do not match body".into(),
            ));
        }
        Ok(Self {
            root,
            numerical_policy,
        })
    }
}

struct NodeBudget {
    remaining: usize,
    required_features: u64,
}

impl Default for NodeBudget {
    fn default() -> Self {
        Self {
            remaining: 64,
            required_features: INVERSE_LINK_FEATURE,
        }
    }
}

impl NodeBudget {
    fn visit(&mut self, depth: usize) -> VersionResult<()> {
        if depth > 16 || self.remaining == 0 {
            return Err(VersionError::Malformed(
                "scalar AST node/depth limit".into(),
            ));
        }
        self.remaining -= 1;
        Ok(())
    }
}

fn encode_node(
    node: &ScalarNode,
    writer: &mut Writer,
    depth: usize,
    budget: &mut NodeBudget,
) -> VersionResult<()> {
    budget.visit(depth)?;
    match node {
        ScalarNode::Const(value) => {
            writer.byte(0)?;
            writer.word(value.to_bits())?;
        }
        ScalarNode::Var(Var::Col(index)) => {
            writer.byte(1)?;
            writer.word(*index as u64)?;
        }
        ScalarNode::Var(var) => {
            writer.byte(2)?;
            writer.byte(
                VARIABLE_IDS
                    .iter()
                    .find(|(_, value)| value == var)
                    .unwrap()
                    .0,
            )?;
        }
        ScalarNode::Unary(UnOp::Neg, child) => {
            writer.byte(3)?;
            encode_node(child, writer, depth + 1, budget)?;
        }
        ScalarNode::Binary(op, left, right) => {
            writer.byte(4)?;
            writer.byte(BINARY_IDS.iter().find(|(_, value)| value == op).unwrap().0)?;
            encode_node(left, writer, depth + 1, budget)?;
            encode_node(right, writer, depth + 1, budget)?;
        }
        ScalarNode::Call(func, arguments) => {
            func.check_arity(arguments.len())?;
            budget.required_features |= func.additional_features();
            writer.byte(5)?;
            writer.byte(func.id() as u8)?;
            writer.word(arguments.len() as u64)?;
            for child in arguments {
                encode_node(child, writer, depth + 1, budget)?;
            }
        }
    }
    Ok(())
}

fn decode_node(
    reader: &mut Reader<'_>,
    depth: usize,
    budget: &mut NodeBudget,
) -> VersionResult<ScalarNode> {
    budget.visit(depth)?;
    Ok(match reader.byte()? {
        0 => ScalarNode::Const(f64::from_bits(reader.word()?)),
        1 => ScalarNode::Var(Var::Col(
            usize::try_from(reader.word()?)
                .map_err(|_| VersionError::Malformed("column index".into()))?,
        )),
        2 => {
            let id = reader.byte()?;
            let var = VARIABLE_IDS
                .iter()
                .find(|(tag, _)| *tag == id)
                .ok_or_else(|| VersionError::UnsupportedVersion(format!("scalar variable {id}")))?
                .1;
            ScalarNode::Var(var)
        }
        3 => ScalarNode::Unary(UnOp::Neg, Box::new(decode_node(reader, depth + 1, budget)?)),
        4 => {
            let id = reader.byte()?;
            let op = BINARY_IDS
                .iter()
                .find(|(tag, _)| *tag == id)
                .ok_or_else(|| {
                    VersionError::UnsupportedVersion(format!("scalar binary operator {id}"))
                })?
                .1;
            let left = decode_node(reader, depth + 1, budget)?;
            let right = decode_node(reader, depth + 1, budget)?;
            ScalarNode::Binary(op, Box::new(left), Box::new(right))
        }
        5 => {
            let func =
                ScalarFunction::from_versioned_id(READER_VERSION, u16::from(reader.byte()?))?;
            let count = reader.length()?;
            func.check_arity(count)?;
            budget.required_features |= func.additional_features();
            let arguments = (0..count)
                .map(|_| decode_node(reader, depth + 1, budget))
                .collect::<VersionResult<Vec<_>>>()?;
            ScalarNode::Call(func, arguments)
        }
        tag => {
            return Err(VersionError::UnsupportedVersion(format!(
                "scalar node {tag}"
            )))
        }
    })
}

fn import_node(node: &Expr, depth: usize, budget: &mut NodeBudget) -> VersionResult<ScalarNode> {
    budget.visit(depth)?;
    Ok(match node {
        Expr::Const(value) => ScalarNode::Const(*value),
        Expr::Var(var) => ScalarNode::Var(*var),
        Expr::Unary(op, child) => {
            ScalarNode::Unary(*op, Box::new(import_node(child, depth + 1, budget)?))
        }
        Expr::Binary(op, left, right) => ScalarNode::Binary(
            *op,
            Box::new(import_node(left, depth + 1, budget)?),
            Box::new(import_node(right, depth + 1, budget)?),
        ),
        Expr::Call(func, arguments) => {
            let func = ScalarFunction::V1(*func);
            func.check_arity(arguments.len())?;
            ScalarNode::Call(
                func,
                arguments
                    .iter()
                    .map(|child| import_node(child, depth + 1, budget))
                    .collect::<VersionResult<Vec<_>>>()?,
            )
        }
    })
}
